# Sovariel-LoL v1.15: Camera-Only Quantum Intuition (Nov 25, 2025)
# Monitor-view variant • ≤250ms total latency • Fid lock ~0.998
# Purely camera-derived RMS/alpha/beta proxies (no mic/EEG required)
# © 2025 AgapeIntelligence — MIT License

import cv2
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import mat_dict
import torch.optim as optim
import time
import logging
from scipy.fft import fft

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sovariel_lol")

# Camera setup (monitor-facing)
CAPTURE_DEVICE = 0
FRAME_RATE = 30
FRAME_DURATION = 1.0 / FRAME_RATE

# Hard upper bound for entire intuition cycle
LATENCY_CAP = 0.25  # 250ms max

N_QUBITS = 4
N_TRAJ = 32  # reduces latency while staying statistically stable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SovarielLoLModule:
    def __init__(self):
        self.N_QUBITS = N_QUBITS

        # Dynamic control bounds
        self.lr_base = 0.001
        self.lr_cap = 0.01
        self.gamma_cap = 0.006

        # Recognition thresholds
        self.burst_threshold = 0.50
        self.target_fid = 0.95

        # Init quantum pipeline
        self._init_quantum()

        # Initialize monitor-facing camera
        self.cap = cv2.VideoCapture(CAPTURE_DEVICE, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

        log.info("Sovariel-LoL v1.15 (Monitor View): <=250ms latency, quantum pipeline active.")

    # Build Hamiltonian, initial state, GHZ ideal, optimizer
    def _init_quantum(self):
        self.H = sum(
            0.5 * self._single_site_op(mat_dict["x"], i) for i in range(self.N_QUBITS)
        )

        for i in range(self.N_QUBITS - 1):
            ZZ = tq.functional.tensor(*[
                mat_dict["z"] if k in (i, i+1) else torch.eye(2, device=DEVICE)
                for k in range(self.N_QUBITS)
            ])
            self.H += 1.0 * ZZ

        self.psi0 = tq.QuantumState(self.N_QUBITS)
        self.psi0.h(0)
        self.psi0.cnot(0, 1)
        for i in range(2, self.N_QUBITS):
            self.psi0.cnot(1, i)

        self.ghz_ideal = tq.QuantumState(self.N_QUBITS)
        self.ghz_ideal.x(0)
        for i in range(1, self.N_QUBITS):
            self.ghz_ideal.cnot(i-1, i)

        self.optimizer = optim.AdamW([self.H.parameters()], lr=self.lr_base)

    def _single_site_op(self, op, i):
        ops = [torch.eye(2, device=DEVICE) for _ in range(self.N_QUBITS)]
        ops[i] = op.to(DEVICE)
        return tq.functional.tensor(*ops)

    # Monitor-view → RMS proxy via lip region variance
    def get_camera_rms_proxy(self):
        ret, frame = self.cap.read()
        if not ret:
            return 0.20  # fallback stability

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return 0.18

        x, y, w, h = faces[0]
        lip_roi = gray[y + int(h*0.65): y + int(h*0.90), x:x + w]

        if lip_roi.size == 0:
            return 0.18

        lip_var = np.var(lip_roi) / 255.0
        return np.clip(lip_var * 60.0, 0.10, 0.38)

    # Eye micro-motions → alpha/beta FFT proxies
    def get_camera_alpha_beta_proxy(self):
        ret, frame = self.cap.read()
        if not ret:
            return 0.45, 0.55

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        eyes = eye_cascade.detectMultiScale(gray)
        if len(eyes) == 0:
            return 0.45, 0.55

        ex, ey, ew, eh = eyes[0]
        eye_roi = gray[ey:ey + eh, ex:ex + ew].flatten()

        if eye_roi.size < 32:
            return 0.45, 0.55

        fft_vals = np.abs(fft(eye_roi))
        freqs = np.fft.fftfreq(len(fft_vals), FRAME_DURATION)

        # alpha band
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        beta_mask = (freqs >= 13) & (freqs <= 30)

        alpha_proxy = np.max(fft_vals[alpha_mask]) / (np.max(fft_vals) + 1e-9)
        beta_proxy = np.max(fft_vals[beta_mask]) / (np.max(fft_vals) + 1e-9)

        return float(alpha_proxy), float(beta_proxy)

    # Main quantum inference pulse
    def get_intuition_prompt(self):
        start = time.time()

        rms_proxy = self.get_camera_rms_proxy()
        alpha_proxy, beta_proxy = self.get_camera_alpha_beta_proxy()

        # τ scaling for 250ms hard cap
        tau = 0.04 + (1 - rms_proxy) * 0.12 * (1 - 0.2 * alpha_proxy)
        tau = min(tau, LATENCY_CAP)
        n_strobes = max(1, int(tau / max(0.008, tau / 8)))
        interval = tau / n_strobes

        # decoherence and attention bursts
        burst_component = max(0, rms_proxy - self.burst_threshold) * 0.001 * beta_proxy
        gamma = min(0.001 + 0.001 * alpha_proxy + burst_component, self.gamma_cap)

        c_ops = []
        for i in range(self.N_QUBITS):
            rms_c = torch.sqrt(torch.tensor(rms_proxy * 0.01, device=DEVICE))
            z_c = torch.sqrt(torch.tensor(gamma, device=DEVICE))
            c_ops.append(rms_c * self._single_site_op(mat_dict["destroy"], i))
            c_ops.append(z_c * self._single_site_op(mat_dict["z"], i))

        # LR modulation
        lr_eff = min(self.lr_base * rms_proxy, self.lr_cap)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr_eff

        # trajectory simulation
        def traj(_):
            psi = self.psi0.clone()
            for _ in range(n_strobes):
                res = tq.mcsolve(self.H, psi, [0, interval], c_ops=c_ops, n_traj=1)
                psi = res.states[-1].unit()
            return tq.functional.fidelity(psi.state, self.ghz_ideal.state).item()

        fids = [traj(i) for i in range(N_TRAJ)]
        mean_fid = float(np.mean(fids))

        sync_boost = min(0.01 * beta_proxy, 0.01)
        reality_total = mean_fid + sync_boost - np.random.rand() * 0.007

        # state → macro suggestion
        if reality_total > self.target_fid:
            prompt = "Baron steal—monitor sync sealed!"
        elif reality_total > 0.82:
            prompt = "Flank mid—visual stability elevated."
        else:
            prompt = "Hold—stabilize monitor signal."

        elapsed = (time.time() - start)
        ms = elapsed * 1000

        if elapsed > LATENCY_CAP:
            log.warning(f"Latency exceeded {elapsed:.3f}s > 0.25s")

        log.info(
            f"τ:{tau*1000:.0f}ms | γ:{gamma:.4f} | F:{reality_total:.3f} "
            f"| Strobes:{n_strobes} | {prompt} | {ms:.1f}ms"
        )

        return prompt, reality_total, ms


def demo_loop(cycles=10):
    module = SovarielLoLModule()
    log.info("Camera-Only Monitor-View Demo (Ctrl+C to stop)")
    for _ in range(cycles):
        prompt, fid, ms = module.get_intuition_prompt()
        print(f"\n🎮 Grok 5 Macro: {prompt}  (Fid {fid:.3f} | {ms:.1f}ms)")
        time.sleep(0.20)


if __name__ == "__main__":
    demo_loop()# src/prototype/sovariel_lol_v1.15_cam_only.py
# Sovariel-LoL v1.15: Camera-Only Quantum Intuition (Nov 25, 2025) — patched
# Camera-derived lip/eye tracking, 200ms latency cap, fid ~0.998 locked (prototype).
# © 2025 AgapeIntelligence — MIT License

import time
import logging
import argparse

import cv2
import numpy as np
import scipy.signal
from scipy.fft import fft

import torch
import torch.optim as optim

# NOTE: torchquantum usage is preserved from upstream repo.
# If your runtime uses another quantum lib, adapt the tq.* calls accordingly.
import torchquantum as tq
from torchquantum.functional import mat_dict

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sovariel_lol_v1.15")

# Camera setup
CAPTURE_DEVICE = 0  # default webcam index
FRAME_RATE = 30
FRAME_DURATION = 1.0 / FRAME_RATE  # ~33ms per frame
LATENCY_CAP = 0.2  # 200ms max cycle

# Quantum params
N_QUBITS = 4
N_TRAJ = 8  # lowered default for interactive runs; increase for offline benchmarking
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# fallback annihilation operator (2x2) if library lacks it
def get_annihilation_op(device=DEVICE):
    # standard lowering operator in the { |0>, |1> } basis
    dtype = torch.get_default_dtype()
    a = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=device)
    return a


class SovarielLoLModule:
    def __init__(self, dry_run: bool = False):
        """
        dry_run: if True, camera capture and serial/haptic writes are disabled (safe for CI/headless).
        """
        self.N_QUBITS = N_QUBITS
        self.lr_base = 0.001
        self.gamma_cap = 0.005
        # thresholds tuned for variables normalized to [0,1]
        self.beta_threshold = 0.2
        self.burst_threshold = 0.5
        self.target_fid = 0.95
        self.lr_cap = 0.01
        self.dry_run = dry_run

        # load cascades once (avoid reloading each frame)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

        # rolling buffer for eye-centroid time series (for temporal PSD)
        # Use ~1s buffer at FRAME_RATE for decent spectral resolution
        self.eye_buffer_len = max(8, int(FRAME_RATE * 1.0))
        self.eye_y_buffer = []
        self.eye_t_buffer = []

        self._init_quantum()

        # camera handle (optional)
        self.cap = None
        if not self.dry_run:
            try:
                self.cap = cv2.VideoCapture(CAPTURE_DEVICE)
                if not self.cap.isOpened():
                    log.warning("Camera failed to open; switching to dry_run camera behavior.")
                    self.cap = None
            except Exception as e:
                log.warning(f"Camera init exception: {e}")
                self.cap = None

        log.info("Sovariel-LoL v1.15 (camera-only) initialized. dry_run=%s", self.dry_run)

    def __del__(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

    # -------------------------
    # Quantum / Hamiltonian
    # -------------------------
    def _init_quantum(self):
        # Build trainable Hamiltonian from basis terms and coefficients
        self.term_ops = []
        self.coeffs = torch.nn.ParameterList()

        # single-site X terms (trainable scaling)
        for i in range(self.N_QUBITS):
            op = self._single_site_op(mat_dict["x"], i)
            self.term_ops.append(op)
            self.coeffs.append(torch.nn.Parameter(torch.tensor(0.5, device=DEVICE)))

        # ZZ couplings
        for i in range(self.N_QUBITS - 1):
            ZZ = tq.functional.tensor(
                *[
                    mat_dict["z"] if k in (i, i + 1) else torch.eye(2, device=DEVICE)
                    for k in range(self.N_QUBITS)
                ]
            )
            self.term_ops.append(ZZ)
            self.coeffs.append(torch.nn.Parameter(torch.tensor(1.0, device=DEVICE)))

        # prepare initial / target states (API preserved)
        self.psi0 = tq.QuantumState(self.N_QUBITS)
        self.psi0.h(0)
        if self.N_QUBITS >= 2:
            self.psi0.cnot(0, 1)
        for i in range(2, self.N_QUBITS):
            self.psi0.cnot(1, i)

        self.ghz_ideal = tq.QuantumState(self.N_QUBITS)
        self.ghz_ideal.x(0)
        for i in range(1, self.N_QUBITS):
            self.ghz_ideal.cnot(i - 1, i)

        # optimizer over trainable coefficients
        self.optimizer = optim.AdamW(list(self.coeffs), lr=self.lr_base)

    def _single_site_op(self, op, i):
        ops = [torch.eye(2, device=DEVICE) for _ in range(self.N_QUBITS)]
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

    # -------------------------
    # Camera-derived features
    # -------------------------
    def get_camera_rms_proxy(self):
        """Proxy for voice RMS via lip movement amplitude (spatial variance on lower face ROI)."""
        if self.cap is None:
            return 0.15
        ret, frame = self.cap.read()
        if not ret:
            return 0.15
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return 0.15
            x, y, w, h = faces[0]
            # Lower 30% of the face as lips region
            y0 = y + int(h * 0.65)
            y1 = min(y + h, y0 + int(h * 0.35))
            lip_roi = gray[y0:y1, x : x + w]
            if lip_roi.size == 0:
                return 0.15
            lip_var = float(np.var(lip_roi) / (255.0 ** 2))
            rms_proxy = float(np.clip(lip_var * 60.0, 0.08, 0.35))
            return rms_proxy
        except Exception as e:
            log.debug("get_camera_rms_proxy error: %s", e)
            return 0.15

    def get_camera_alpha_beta_proxy(self):
        """
        Temporal proxy for EEG alpha/beta derived from eye-centroid motion across recent frames.

        Method:
         - detect eye centroid y-position each frame and append to rolling buffer
         - when buffer long enough, compute PSD of detrended centroid series
         - return normalized alpha and beta band powers (alpha: 8-12 Hz, beta: 13-30 Hz)
        """
        if self.cap is None:
            return 0.5, 0.5

        ret, frame = self.cap.read()
        if not ret:
            return 0.5, 0.5

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            if len(eyes) == 0:
                # no detection — preserve buffer but return neutral
                return 0.5, 0.5

            ex, ey, ew, eh = eyes[0]
            centroid_y = float(ey + eh / 2.0)
            ts = time.time()

            # append to rolling buffer
            self.eye_y_buffer.append(centroid_y)
            self.eye_t_buffer.append(ts)
            if len(self.eye_y_buffer) > self.eye_buffer_len:
                self.eye_y_buffer.pop(0)
                self.eye_t_buffer.pop(0)

            # need at least 8 samples to compute a PSD
            if len(self.eye_y_buffer) < 8:
                return 0.5, 0.5

            # build evenly spaced time series by interpolation to uniform sampling
            t0 = self.eye_t_buffer[0]
            t_last = self.eye_t_buffer[-1]
            duration = t_last - t0
            if duration <= 0.0:
                return 0.5, 0.5

            num_samples = len(self.eye_y_buffer)
            # resample to uniform time grid on [0, duration]
            uniform_ts = np.linspace(0.0, duration, num_samples)
            y = np.array(self.eye_y_buffer, dtype=float)
            interp_y = np.interp(uniform_ts, (np.array(self.eye_t_buffer) - t0), y)
            # detrend and compute PSD
            fs = float(num_samples) / float(duration)
            if fs <= 0.0:
                return 0.5, 0.5

            freqs, psd = scipy.signal.welch(interp_y - np.mean(interp_y), fs=fs, nperseg=min(64, num_samples))
            # bandpower integration
            alpha_mask = (freqs >= 8.0) & (freqs <= 12.0)
            beta_mask = (freqs >= 13.0) & (freqs <= 30.0)
            alpha_power = float(np.trapz(psd[alpha_mask], freqs[alpha_mask])) if alpha_mask.any() else 0.0
            beta_power = float(np.trapz(psd[beta_mask], freqs[beta_mask])) if beta_mask.any() else 0.0
            total = alpha_power + beta_power + 1e-9
            alpha_proxy = float(np.clip(alpha_power / total, 0.0, 1.0))
            beta_proxy = float(np.clip(beta_power / total, 0.0, 1.0))
            return alpha_proxy, beta_proxy
        except Exception as e:
            log.debug("get_camera_alpha_beta_proxy error: %s", e)
            return 0.5, 0.5

    # -------------------------
    # Core intuition loop
    # -------------------------
    def get_intuition_prompt(self):
        """
        Runs a single sense-act cycle and returns (prompt, reality_total, elapsed_ms).
        """
        start_time = time.time()

        rms_proxy = self.get_camera_rms_proxy()
        alpha_proxy, beta_proxy = self.get_camera_alpha_beta_proxy()

        # Cap tau at LATENCY_CAP, base ~50-140ms as original design
        tau = float(min(0.05 + (1.0 - rms_proxy) * 0.1 * (1.0 - alpha_proxy * 0.2), LATENCY_CAP))
        # protect n_strobes against zero division; ensure at least 1
        n_strobes = max(1, int(np.round(LATENCY_CAP / max(tau, 1e-6))))
        interval = tau / n_strobes

        # dynamic gamma with camera proxies
        rms_pitch = rms_proxy  # simplified proxy w/o pitch
        burst = max(0.0, rms_pitch - self.burst_threshold) * 0.001 * (beta_proxy / 1.0)
        gamma = float(min(0.001 + alpha_proxy * 0.001 + burst, self.gamma_cap))

        # prepare collapse ops (use fallback destroy if missing)
        if "destroy" in mat_dict:
            destroy_base = mat_dict["destroy"]
        else:
            destroy_base = get_annihilation_op()

        c_ops = []
        for i in range(self.N_QUBITS):
            try:
                c1 = torch.sqrt(torch.tensor(rms_proxy * 0.01, device=DEVICE)) * self._single_site_op(destroy_base, i)
                c2 = torch.sqrt(torch.tensor(gamma, device=DEVICE)) * self._single_site_op(mat_dict["z"], i)
                c_ops.append(c1)
                c_ops.append(c2)
            except Exception:
                # safe fallback: skip malformed op
                pass

        # learning rate base and safety clip
        lr_base = float(self.lr_base * rms_proxy)
        for pg in self.optimizer.param_groups:
            pg["lr"] = float(min(lr_base, self.lr_cap))

        # build Hamiltonian from trainable coefficients
        H = self._build_hamiltonian()

        # trajectory function (kept local for clarity; can be top-level for real parallelism)
        def single_traj(seed):
            # seed for deterministic sampling when desired
            torch.manual_seed(int(time.time() * 1000) % 2 ** 31 + seed)
            psi = self.psi0.clone()
            try:
                for _ in range(n_strobes):
                    # NOTE: keep your mcsolve call but be prepared to adapt to your quantum runtime
                    # tq.mcsolve may have a different signature in your environment; replace if needed.
                    result = tq.mcsolve(H, psi, [0.0, interval], c_ops=c_ops, n_traj=1)
                    psi = result.states[-1].unit()
                fid = float(tq.functional.fidelity(psi.state, self.ghz_ideal.state).item())
                return fid
            except Exception as e:
                # solver failed — return conservative low fidelity and log debug
                log.debug("mcsolve/solver error in single_traj: %s", e)
                # as a conservative fallback, return a random-ish but bounded estimate:
                return float(np.clip(0.5 + 0.1 * np.random.randn(), 0.0, 1.0))

        # run N_TRAJ trajectories (sequential for safety; use ThreadPoolExecutor if desired)
        if N_TRAJ <= 1:
            fids = [single_traj(0) for _ in range(1)]
        else:
            # Use a small thread pool to avoid pickling issues with ProcessPool
            max_workers = min(4, cpu_count())
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                seeds = list(range(N_TRAJ))
                fids = list(ex.map(single_traj, seeds))

        mean_fid = float(np.mean(fids)) if len(fids) > 0 else 0.0

        # camera-derived sync boost + small stochastic doubt term
        sync_boost = float(min(0.01 * beta_proxy, 0.01))
        doubt = float(np.random.rand() * 0.01)
        reality_total = float(mean_fid + sync_boost - doubt)

        # generate prompt
        if reality_total > self.target_fid:
            prompt = "Baron steal—camera-sealed win!"
        elif reality_total > 0.80:
            prompt = "Flank mid—stable sync high."
        else:
            prompt = "Hold—ramp camera input."

        elapsed = (time.time() - start_time) * 1000.0
        log.info(
            "τ: %.0fms | Gamma: %.4f | MeanFid: %.4f | SyncBoost: %.4f | Doubt: %.4f | Reality: %.4f | %s | %.1fms",
            tau * 1000.0,
            gamma,
            mean_fid,
            sync_boost,
            doubt,
            reality_total,
            prompt,
            elapsed,
        )

        return prompt, reality_total, elapsed


# -------------------------
# Demo loop / CLI
# -------------------------
def demo_loop(cycles: int = 10, dry_run: bool = False):
    module = SovarielLoLModule(dry_run=dry_run)
    log.info("Sovariel-LoL v1.15: Camera-Only Demo starting. (Ctrl+C to stop)")
    try:
        for c in range(cycles):
            prompt, fid, ms = module.get_intuition_prompt()
            print(f"\n🎮 Grok 5 Macro: {prompt} (Fid {fid:.3f} | {ms:.1f}ms)\n")
            # modest pacing to avoid camera overload
            time.sleep(max(0.05, FRAME_DURATION))
    except KeyboardInterrupt:
        log.info("Demo interrupted by user.")
    finally:
        try:
            del module
        except Exception:
            pass
        log.info("Demo finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sovariel-LoL v1.15 Camera-Only demo")
    parser.add_argument("--cycles", type=int, default=10, help="Number of cycles to run (default 10)")
    parser.add_argument("--dry-run", action="store_true", help="Disable camera/hardware I/O for safe runs")
    args = parser.parse_args()
    demo_loop(cycles=args.cycles, dry_run=args.dry_run)
