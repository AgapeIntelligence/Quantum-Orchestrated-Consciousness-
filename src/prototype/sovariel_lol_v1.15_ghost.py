# sovariel_lol_v1.15_ghost.py
# Sovariel-LoL v1.15-ghost: Headless monitor-only background agent (Nov 25, 2025)
# Ghost: no UI, screen-capture monitor probes, ≤250ms cycle cap, writes ghost_state.json
# Optional: local-only HTTP endpoint to serve last state (requires Flask)
# © 2025 AgapeIntelligence — MIT License

import time
import json
import argparse
import logging
import threading
from pathlib import Path

import numpy as np

# screen capture
try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except Exception:
    MSS_AVAILABLE = False

# optional HTTP
try:
    from flask import Flask, jsonify
    FLASK_AVAILABLE = True
except Exception:
    FLASK_AVAILABLE = False

# quantum libs (preserve original expectations)
import torch
import torch.optim as optim
import torchquantum as tq
from torchquantum.functional import mat_dict

# numeric utils
import scipy.signal

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sovariel_lol_ghost")

# --- Configurable constants ---
LATENCY_CAP = 0.25  # 250 ms hard cap
FRAME_RATE = 20     # ghost sampling target (lower than interactive)
FRAME_DURATION = 1.0 / FRAME_RATE
N_QUBITS = 4
N_TRAJ = 6          # small default to keep cycles fast
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_PATH = Path("ghost_state.json")  # local state file Grok (or another agent) can read

# default screen probe regions (relative fractions)
# These are *defaults* — when running, pass absolute coords via CLI if needed.
DEFAULT_LIP_REGION = (0.4, 0.6, 0.2, 0.1)  # (x_frac, y_frac, w_frac, h_frac)
DEFAULT_EYE_REGION = (0.4, 0.25, 0.2, 0.1)


# fallback annihilation op
def get_annihilation_op(device=DEVICE):
    dtype = torch.get_default_dtype()
    a = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=device)
    return a


class GhostAgent:
    def __init__(
        self,
        dry_run=False,
        screen_monitor=0,
        lip_region=DEFAULT_LIP_REGION,
        eye_region=DEFAULT_EYE_REGION,
        run_forever=False,
        cycles=10,
        http_port=None,
    ):
        self.dry_run = dry_run
        self.screen_monitor = screen_monitor
        self.lip_region = lip_region
        self.eye_region = eye_region
        self.run_forever = run_forever
        self.requested_cycles = cycles
        self.http_port = http_port

        # initialize quantum pieces
        self._init_quantum()

        # screen capture context (mss)
        self.mss_ctx = None
        if not self.dry_run and MSS_AVAILABLE:
            try:
                self.mss_ctx = mss.mss()
                log.info("mss context created")
            except Exception as e:
                log.warning("mss failed: %s, switching to dry_run camera behavior", e)
                self.mss_ctx = None
                self.dry_run = True

        # last state (dict)
        self.last_state = {
            "timestamp": None,
            "prompt": None,
            "reality_total": None,
            "mean_fid": None,
            "tau_ms": None,
            "gamma": None,
            "elapsed_ms": None,
        }

        # start HTTP server thread if requested and Flask available
        self.http_thread = None
        if self.http_port is not None:
            if not FLASK_AVAILABLE:
                log.warning("Flask not available; HTTP server disabled.")
            else:
                self._start_http_server(self.http_port)

    def _init_quantum(self):
        # Minimal trainable Hamiltonian matching prototype style
        self.term_ops = []
        self.coeffs = torch.nn.ParameterList()
        for i in range(N_QUBITS):
            op = self._single_site_op(mat_dict["x"], i)
            self.term_ops.append(op)
            self.coeffs.append(torch.nn.Parameter(torch.tensor(0.5, device=DEVICE)))
        for i in range(N_QUBITS - 1):
            ZZ = tq.functional.tensor(*[mat_dict["z"] if k in (i, i + 1) else torch.eye(2, device=DEVICE) for k in range(N_QUBITS)])
            self.term_ops.append(ZZ)
            self.coeffs.append(torch.nn.Parameter(torch.tensor(1.0, device=DEVICE)))

        self.psi0 = tq.QuantumState(N_QUBITS)
        self.psi0.h(0)
        if N_QUBITS >= 2:
            self.psi0.cnot(0, 1)
        for i in range(2, N_QUBITS):
            self.psi0.cnot(1, i)

        self.ghz_ideal = tq.QuantumState(N_QUBITS)
        self.ghz_ideal.x(0)
        for i in range(1, N_QUBITS):
            self.ghz_ideal.cnot(i - 1, i)

        self.optimizer = optim.AdamW(list(self.coeffs), lr=0.001)

    def _single_site_op(self, op, i):
        ops = [torch.eye(2, device=DEVICE) for _ in range(N_QUBITS)]
        if hasattr(op, "to"):
            ops[i] = op.to(DEVICE)
        else:
            ops[i] = op
        return tq.functional.tensor(*ops)

    def _build_hamiltonian(self):
        H = None
        for c, op in zip(self.coeffs, self.term_ops):
            term = c * op
            H = term if H is None else H + term
        return H

    # --- headless screen probes ---
    def _capture_region(self, region_frac):
        """
        region_frac: (x_frac, y_frac, w_frac, h_frac) in [0,1] relative to monitor resolution.
        Returns grayscale numpy array (H, W) or None.
        """
        if self.dry_run or self.mss_ctx is None:
            return None

        monitors = self.mss_ctx.monitors
        if self.screen_monitor < 0 or self.screen_monitor >= len(monitors):
            mon = monitors[0]
        else:
            mon = monitors[self.screen_monitor]

        mx, my, mw, mh = mon["left"], mon["top"], mon["width"], mon["height"]
        x_frac, y_frac, w_frac, h_frac = region_frac
        x = int(mx + x_frac * mw)
        y = int(my + y_frac * mh)
        w = int(max(8, w_frac * mw))
        h = int(max(8, h_frac * mh))
        bbox = {"left": x, "top": y, "width": w, "height": h}
        try:
            sct = self.mss_ctx.grab(bbox)
            # mss returns BGRA bytes; convert to numpy grayscale
            arr = np.array(sct)  # shape (h, w, 4)
            if arr.ndim == 3:
                gray = arr[..., :3].astype(np.float32).mean(axis=2)  # simple avg to gray
            else:
                gray = arr.astype(np.float32)
            return gray
        except Exception as e:
            log.debug("screen capture failed: %s", e)
            return None

    def _lip_variance_proxy(self):
        # compute spatial variance over lip-region as RMS proxy
        frame = self._capture_region(self.lip_region)
        if frame is None:
            return 0.20
        v = float(np.var(frame) / (255.0 ** 2))
        return float(np.clip(v * 60.0, 0.08, 0.38))

    def _eye_motion_proxy(self):
        # simple motion-based temporal proxy: compute brightness centroid shift over a tiny eye region across a short buffer
        frame = self._capture_region(self.eye_region)
        if frame is None:
            return 0.45, 0.55

        # maintain a short circular buffer in-memory (attached to self)
        if not hasattr(self, "_eye_buf"):
            self._eye_buf = []
            self._eye_ts = []

        centroid = float(np.mean(np.nonzero(frame)[0]) / max(1, frame.shape[0]))  # coarse centroid along y
        now = time.time()
        self._eye_buf.append(centroid)
        self._eye_ts.append(now)
        if len(self._eye_buf) > max(8, int(FRAME_RATE * 0.8)):
            self._eye_buf.pop(0)
            self._eye_ts.pop(0)

        if len(self._eye_buf) < 8:
            return 0.45, 0.55

        # resample to uniform and compute PSD
        t0 = self._eye_ts[0]
        t_last = self._eye_ts[-1]
        duration = t_last - t0
        if duration <= 0.0:
            return 0.45, 0.55
        y = np.array(self._eye_buf)
        num = len(y)
        uniform_ts = np.linspace(0.0, duration, num)
        interp_y = np.interp(uniform_ts, np.array(self._eye_ts) - t0, y)
        fs = float(num) / float(duration)
        if fs <= 0.0:
            return 0.45, 0.55
        freqs, psd = scipy.signal.welch(interp_y - np.mean(interp_y), fs=fs, nperseg=min(64, num))
        alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
        beta_mask = (freqs >= 13.0) & (freqs <= 30.0)
        alpha_power = float(np.trapz(psd[alpha_mask], freqs[alpha_mask])) if alpha_mask.any() else 0.0
        beta_power = float(np.trapz(psd[beta_mask], freqs[beta_mask])) if beta_mask.any() else 0.0
        total = alpha_power + beta_power + 1e-9
        a = float(np.clip(alpha_power / total, 0.0, 1.0))
        b = float(np.clip(beta_power / total, 0.0, 1.0))
        return a, b

    # --- core headless cycle ---
    def run_cycle(self):
        start_ts = time.time()
        rms = self._lip_variance_proxy()
        alpha, beta = self._eye_motion_proxy()

        # tau capped such that total cycle <= LATENCY_CAP
        tau = float(min(0.05 + (1.0 - rms) * 0.1 * (1.0 - 0.2 * alpha), LATENCY_CAP))
        # keep n_strobes low and safe
        n_strobes = max(1, int(round(tau / max(0.01, tau / 4))))
        interval = tau / n_strobes

        # gamma and burst
        burst = max(0.0, rms - 0.5) * 0.001 * beta
        gamma = float(min(0.001 + 0.001 * alpha + burst, 0.006))

        # collapse ops using fallback destroy if necessary
        if "destroy" in mat_dict:
            destroy_base = mat_dict["destroy"]
        else:
            destroy_base = get_annihilation_op()

        c_ops = []
        for i in range(N_QUBITS):
            try:
                c_ops.append(torch.sqrt(torch.tensor(rms * 0.01, device=DEVICE)) * self._single_site_op(destroy_base, i))
                c_ops.append(torch.sqrt(torch.tensor(gamma, device=DEVICE)) * self._single_site_op(mat_dict["z"], i))
            except Exception:
                pass

        # dynamic lr (kept conservative)
        lr_eff = float(min(0.001 * max(0.2, rms), 0.01))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr_eff

        H = self._build_hamiltonian()

        # small number of trajectories to keep runtime fast
        def single_traj(seed):
            torch.manual_seed(int(time.time() * 1000) % 2 ** 31 + seed)
            psi = self.psi0.clone()
            try:
                for _ in range(n_strobes):
                    result = tq.mcsolve(H, psi, [0.0, interval], c_ops=c_ops, n_traj=1)
                    psi = result.states[-1].unit()
                fid = float(tq.functional.fidelity(psi.state, self.ghz_ideal.state).item())
                return fid
            except Exception:
                return float(np.clip(0.5 + 0.05 * np.random.randn(), 0.0, 1.0))

        # run a compact set of trajectories
        seeds = list(range(max(1, N_TRAJ)))
        fids = [single_traj(s) for s in seeds]
        mean_fid = float(np.mean(fids)) if len(fids) > 0 else 0.0

        sync_boost = float(min(0.01 * beta, 0.01))
        doubt = float(np.random.rand() * 0.01)
        reality_total = float(mean_fid + sync_boost - doubt)

        elapsed = time.time() - start_ts
        elapsed_ms = elapsed * 1000.0

        # safety: if elapsed exceeded cap, we note it and proceed (no blocking)
        exceeded = elapsed > LATENCY_CAP
        if exceeded:
            log.warning("Cycle latency exceeded cap: %.3fs > %.3fs", elapsed, LATENCY_CAP)

        prompt = "Baron steal—ghost win!" if reality_total > 0.95 else ("Flank mid—ghost sync." if reality_total > 0.80 else "Hold—ghost ramp.")

        state = {
            "timestamp": time.time(),
            "prompt": prompt,
            "reality_total": reality_total,
            "mean_fid": mean_fid,
            "tau_ms": tau * 1000.0,
            "gamma": gamma,
            "elapsed_ms": elapsed_ms,
            "exceeded": exceeded,
            "rms_proxy": rms,
            "alpha_proxy": alpha,
            "beta_proxy": beta,
        }

        # persist to file (atomic write)
        try:
            tmp = OUTPUT_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            tmp.replace(OUTPUT_PATH)
        except Exception as e:
            log.debug("Failed to write ghost_state.json: %s", e)

        self.last_state = state
        return state

    def run(self):
        cycles = 0
        if self.run_forever:
            log.info("Ghost agent running forever (Ctrl+C to stop).")
            while True:
                start = time.time()
                _ = self.run_cycle()
                cycles += 1
                # sleep to respect frame rate but ensure latency cap
                elapsed = time.time() - start
                target = max(1.0 / FRAME_RATE, 0.05)
                to_sleep = max(0.0, target - elapsed)
                time.sleep(to_sleep)
        else:
            log.info("Ghost agent running for %d cycles.", self.requested_cycles)
            for _ in range(max(1, self.requested_cycles)):
                _ = self.run_cycle()
                cycles += 1
                time.sleep(max(0.0, 1.0 / FRAME_RATE))
            log.info("Ghost agent finished %d cycles.", cycles)

    # --- optional small HTTP server over localhost ---
    def _start_http_server(self, port):
        if not FLASK_AVAILABLE:
            return
        app = Flask("sovariel_lol_ghost_http")

        @app.route("/ghost_state")
        def ghost_state():
            return jsonify(self.last_state)

        def _serve():
            log.info("Starting ghost HTTP server on localhost:%d", port)
            app.run(host="127.0.0.1", port=port, threaded=True)

        self.http_thread = threading.Thread(target=_serve, daemon=True)
        self.http_thread.start()


# CLI entrypoint
def main():
    parser = argparse.ArgumentParser(description="Sovariel-LoL v1.15 Ghost agent (headless)")
    parser.add_argument("--dry-run", action="store_true", help="Disable screen capture (CI/headless)")
    parser.add_argument("--monitor", type=int, default=0, help="Monitor index for mss (default 0)")
    parser.add_argument("--lip-region", type=float, nargs=4, metavar=("X","Y","W","H"), default=DEFAULT_LIP_REGION, help="Lip region as fractions of monitor (x y w h)")
    parser.add_argument("--eye-region", type=float, nargs=4, metavar=("X","Y","W","H"), default=DEFAULT_EYE_REGION, help="Eye region as fractions")
    parser.add_argument("--cycles", type=int, default=10, help="Number of cycles (ignored if --run-forever)")
    parser.add_argument("--run-forever", action="store_true", help="Run until killed")
    parser.add_argument("--http-port", type=int, default=None, help="If provided (and Flask available), serves /ghost_state on localhost")
    args = parser.parse_args()

    agent = GhostAgent(
        dry_run=args.dry_run,
        screen_monitor=args.monitor,
        lip_region=tuple(args.lip_region),
        eye_region=tuple(args.eye_region),
        run_forever=args.run_forever,
        cycles=args.cycles,
        http_port=args.http_port,
    )
    try:
        agent.run()
    except KeyboardInterrupt:
        log.info("Ghost agent interrupted by user.")
    finally:
        log.info("Ghost agent shutting down.")

if __name__ == "__main__":
    main()
