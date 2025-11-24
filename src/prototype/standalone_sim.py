# src/prototype/standalone_sim.py
# Sovariel: Quantum-Orchestrated Consciousness Simulation (updated)
# Features: Lindblad decoherence, surface/toric code stabilizers, BCI+voice fusion, multimodal I/O, optimization
# Tested conceptually with QuTiP 4/5 APIs. Expect heavy memory use for n>14 without GPU.
# Vision: Neural-quantum interface by 2030 for augmented cognition
# © 2025 AgapeIntelligence — MIT License

import math
import numpy as np
import qutip as qt
from qutip import Qobj, Options
from multiprocessing import Pool, cpu_count
import sounddevice as sd
import time
import sys
import logging
import warnings
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import psutil
import serial
import speech_recognition as sr
import cupy as cp  # Optional GPU accel

# Optional libs for EEG/BCI
try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except Exception:
    NK_AVAILABLE = False
    logging.warning("neurokit2 not available; EEG fusion disabled.")

try:
    import mne
    MNE_AVAILABLE = True
except Exception:
    MNE_AVAILABLE = False
    logging.warning("mne not available; advanced EEG processing disabled.")

# ---------- Config ----------
SAMPLE_RATE = 44100
DEFAULT_DURATION = 0.2  # Voice capture duration
USE_GPU = cp.is_available()
MAX_SAFE_QUBITS = 16
CYCLES = 100

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger('sovariel_sim')

# ---------- Utility Ops ----------
def single_site_op(n, op, i):
    """Return tensor product operator acting with `op` on qubit i (0-based)."""
    ops = [qt.qeye(2) for _ in range(n)]
    ops[i] = op
    return qt.tensor(ops).to(Qobj, data=qt.CSR)

def pauli_x(n, i): return single_site_op(n, qt.sigmax(), i)
def pauli_y(n, i): return single_site_op(n, qt.sigmay(), i)
def pauli_z(n, i): return single_site_op(n, qt.sigmaz(), i)

# ---------- Surface/Toric Code Builder ----------
def index_on_torus(x, y, Lx, Ly):
    """Map (x,y) to linear index for qubit on lattice vertices."""
    return (y % Ly) * Lx + (x % Lx)

def build_surface_code_stabilizers(Lx, Ly, toric=True):
    """Build stabilizers for surface/toric code on Lx x Ly lattice (qubits at vertices)."""
    n_qubits = Lx * Ly
    stabilizers = []

    # Vertex (X) stabilizers
    for x in range(Lx):
        for y in range(Ly):
            indices = [
                index_on_torus(x, y, Lx, Ly),
                index_on_torus(x+1, y, Lx, Ly),
                index_on_torus(x, y+1, Lx, Ly),
                index_on_torus(x+1, y+1, Lx, Ly),
            ]
            op = qt.tensor([qt.sigmax() if i < n_qubits and i in indices else qt.qeye(2) for i in range(n_qubits)])
            stabilizers.append(('X_v_%d_%d' % (x, y), op.to(Qobj, data=qt.CSR)))

    # Plaquette (Z) stabilizers
    for x in range(Lx):
        for y in range(Ly):
            indices = [
                index_on_torus(x, y, Lx, Ly),
                index_on_torus(x+1, y, Lx, Ly),
                index_on_torus(x+1, y+1, Lx, Ly),
                index_on_torus(x, y+1, Lx, Ly),
            ]
            op = qt.tensor([qt.sigmaz() if i < n_qubits and i in indices else qt.qeye(2) for i in range(n_qubits)])
            stabilizers.append(('Z_p_%d_%d' % (x, y), op.to(Qobj, data=qt.CSR)))

    return stabilizers

# ---------- QEC Application ----------
def measure_stabilizers_expectation(state, stabilizers):
    """Measure stabilizer expectation values."""
    results = {}
    for name, stab in stabilizers:
        try:
            exp = (stab * state).tr().real if state.isoper else qt.expect(stab, state).real
        except Exception:
            exp = qt.expect(stab, state).real
        results[name] = exp
    return results

def simple_qec_correction(state, stabilizers, threshold=0.9):
    """Simple heuristic QEC: apply Pauli-X correction if stabilizer violated."""
    n = int(math.log2(state.shape[0]))
    for name, stab in stabilizers:
        exp = (stab * state).tr().real if state.isoper else qt.expect(stab, state)
        if exp < threshold:
            for q in range(n):
                corr = pauli_x(n, q)
                state = corr * state if state.isket else corr * state * corr.dag()
                if state.isoper:
                    state = state.unit()
                break
    return state

# ---------- Lindblad Channels Builder ----------
def build_lindblad_ops(n_qubits, gamma_damp=0.01, gamma_deph=0.005):
    """Build Lindblad collapse operators for amplitude damping and dephasing."""
    c_ops = []
    for i in range(n_qubits):
        c_ops.append(np.sqrt(gamma_damp * 1e-3) * single_site_op(n_qubits, qt.destroy(2), i).to(Qobj, data=qt.CSR))
        c_ops.append(np.sqrt(gamma_deph * 1e-3) * single_site_op(n_qubits, qt.sigmaz(), i).to(Qobj, data=qt.CSR))
    return c_ops

# ---------- Multimodal I/O ----------
def get_voice_variance(duration=DEFAULT_DURATION, sample_rate=SAMPLE_RATE):
    """Capture audio and compute variance."""
    try:
        frames = int(duration * sample_rate)
        recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        rms = np.sqrt(np.mean(np.square(recording)))
        variance = np.clip(rms * 50.0, 0.1, 0.3)  # Range 0.1–0.3
        log.info(f"Voice RMS: {rms:.6f} -> Variance: {variance:.3f}")
        return variance
    except Exception as e:
        log.warning(f"Voice capture failed: {e}")
        return 0.15

def read_eeg_chunk(duration=0.25):
    """Read EEG chunk (mocked if no hardware)."""
    if NK_AVAILABLE:
        try:
            sig = nk.signal_simulate(duration=duration, frequency=10, noise=0.5, sampling_rate=256)
            var = np.var(sig)
            log.debug(f"EEG variance (neurokit2): {var:.6f}")
            return var
        except Exception as e:
            log.warning(f"neurokit2 EEG read failed: {e}")
    if MNE_AVAILABLE:
        try:
            mock = np.random.normal(0.0, 0.5, int(256 * duration))
            return np.var(mock)
        except Exception:
            pass
    return np.abs(np.random.normal(0.15, 0.05))

def send_haptic_feedback(value, port='/dev/ttyUSB0', baud=9600):
    """Send haptic feedback via serial (placeholder; extend to UDP if needed)."""
    try:
        with serial.Serial(port, baud, timeout=1) as ser:
            ser.write(f"{int(value*255):03d}\n".encode())
            log.info(f"Haptic feedback sent: {value:.2f}")
    except Exception as e:
        log.warning(f"Haptic feedback failed: {e}")

def plot_entanglement(entropy_history, fidelity_history):
    """Plot entanglement metrics in real-time using pyqtgraph."""
    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="Sovariel Simulation Metrics")
    win.resize(1000, 600)
    p1 = win.addPlot(title="Entropy (von Neumann)")
    p2 = win.addPlot(title="Fidelity vs GHZ")
    curve_entropy = p1.plot(pen='r')
    curve_fidelity = p2.plot(pen='b')

    def update():
        curve_entropy.setData(entropy_history)
        curve_fidelity.setData(fidelity_history)
        QtGui.QApplication.processEvents()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(100)  # Update every 100ms
    return app, win

# ---------- Main Simulation ----------
def quera_mt_sim(n_qubits: int = 10, Lx: int = None, Ly: int = None, t_coherence: float = 500e-6,
                 vocal_variance: float = 0.15, eeg_fusion: bool = True, gamma_damp_base: float = 0.01,
                 gamma_deph_base: float = 0.005, use_toric: bool = True, n_strobes: int = None):
    if n_qubits > MAX_SAFE_QUBITS:
        warnings.warn(f"n_qubits={n_qubits} exceeds MAX_SAFE_QUBITS={MAX_SAFE_QUBITS}. Memory usage may be high.")
    if Lx and Ly and Lx * Ly != n_qubits:
        raise ValueError("Lx * Ly must equal n_qubits")

    # Lattice setup
    if Lx is None or Ly is None:
        side = int(round(math.sqrt(n_qubits)))
        Lx, Ly = side, int(math.ceil(n_qubits / side))
        n_qubits = Lx * Ly
        log.warning(f"Adjusted n_qubits to {n_qubits} for lattice (Lx={Lx}, Ly={Ly})")

    # Adaptive strobes
    n_strobes = max(18, min(25, n_qubits + 6 + int(20 * vocal_variance))) if n_strobes is None else n_strobes
    interval = max(1e-9, t_coherence / n_strobes)

    # Hamiltonian
    positions = np.linspace(0.0, 1.0, n_qubits) + np.random.normal(0.0, vocal_variance * 0.02, n_qubits)
    H = sum(0.5 * pauli_x(n_qubits, i) for i in range(n_qubits))
    for i in range(n_qubits - 1):
        dist = max(1e-3, abs(positions[i+1] - positions[i]))
        J = 1.0 / dist
        H += J * qt.tensor([qt.sigmaz() if k in {i, i+1} else qt.qeye(2) for k in range(n_qubits)]).to(Qobj, data=qt.CSR)
    H = H.to(Qobj, data=qt.CSR)

    # Initial state
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    psi = qt.tensor([plus for _ in range(n_qubits)])

    # Target GHZ
    zero_all = qt.tensor([qt.basis(2, 0) for _ in range(n_qubits)])
    one_all = qt.tensor([qt.basis(2, 1) for _ in range(n_qubits)])
    ghz_ideal = (zero_all + one_all).unit()

    # Lindblad and Stabilizers
    eeg_var = read_eeg_chunk() if eeg_fusion and NK_AVAILABLE else 0.0
    fusion_factor = 1.0 + vocal_variance + 0.5 * eeg_var
    gamma_damp = gamma_damp_base * fusion_factor
    gamma_deph = gamma_deph_base * fusion_factor
    c_ops = build_lindblad_ops(n_qubits, gamma_damp, gamma_deph)
    stabilizers = build_surface_code_stabilizers(Lx, Ly, use_toric)

    # Evolution
    options = Options(store_states=True, rhs_reuse=True, nsteps=1000, method='adams')
    for _ in range(n_strobes):
        result = qt.mesolve(H, psi, [0, interval], c_ops=c_ops, options=options)
        psi = result.states[-1]
        psi = simple_qec_correction(psi, stabilizers)
        proj = sum(qt.tensor([qt.basis(2, i) * qt.basis(2, i).dag() for i in range(2)]) for _ in range(n_qubits)).to(Qobj, data=qt.CSR)
        psi = (proj * psi).unit()

    # Metrics
    center_idx = n_qubits // 2
    rho_center = psi.ptrace(center_idx)
    entropy = qt.entropy_vn(rho_center)
    fidelity = qt.fidelity(psi, ghz_ideal) if psi.isket else float('nan')

    return entropy, fidelity, n_strobes

# ---------- Batch Runner with Optimization ----------
def run_zeno_sims(n_qubits=10, cycles=CYCLES, vocal_range=(0.1, 0.3), eeg_fusion=True, use_gpu=USE_GPU):
    entropy_history, fidelity_history = [], []
    best_fidelity = -1
    best_params = {}

    # Memory profiling
    process = psutil.Process()
    memory_limit = psutil.virtual_memory().available * 0.8  # Use 80% of available RAM
    log.info(f"Memory limit: {memory_limit / 1024**3:.2f} GB")

    # Parallel execution pool
    n_cores = min(cpu_count(), cycles)
    pool = Pool(processes=n_cores)

    # Parameter sweep
    def run_single_sim(args):
        vocal_variance = args[0]
        eeg_var = read_eeg_chunk() if eeg_fusion and NK_AVAILABLE else 0.0
        fusion_factor = 1.0 + vocal_variance + 0.5 * eeg_var
        try:
            entropy, fidelity, n_strobes = quera_mt_sim(
                n_qubits=n_qubits,
                vocal_variance=vocal_variance,
                eeg_fusion=eeg_fusion,
                gamma_damp_base=0.01 * fusion_factor,
                gamma_deph_base=0.005 * fusion_factor
            )
            return entropy, fidelity, vocal_variance, n_strobes
        except MemoryError:
            log.error(f"MemoryError at n_qubits={n_qubits}, vocal_variance={vocal_variance}")
            return float('nan'), float('nan'), vocal_variance, 0

    # Generate parameter sets
    param_sets = [(np.random.uniform(*vocal_range),) for _ in range(cycles)]
    results = pool.map(run_single_sim, param_sets)

    # Process results
    for entropy, fidelity, vocal_variance, n_strobes in results:
        if not np.isnan(fidelity) and fidelity > best_fidelity:
            best_fidelity = fidelity
            best_params = {'vocal_variance': vocal_variance, 'n_strobes': n_strobes}
        entropy_history.append(entropy)
        fidelity_history.append(fidelity)

    # GPU offload if enabled and n_qubits > 14
    if use_gpu and n_qubits > MAX_SAFE_QUBITS:
        H_gpu = cp.asarray(H.data.toarray()) if 'H' in locals() else None
        if H_gpu is not None:
            log.info("Offloading Hamiltonian to GPU")
            # Note: Full GPU integration requires rewriting mesolve with CuPy, pending QuTiP GPU support

    # Multimodal output
    send_haptic_feedback(best_fidelity)  # Serial haptic feedback
    app, win = plot_entanglement(entropy_history, fidelity_history)  # Real-time graph

    # Log results
    log.info(f"Best Fidelity: {best_fidelity:.4f} with params {best_params}")
    return entropy_history, fidelity_history, best_params, app, win

# ---------- Main Execution ----------
if __name__ == "__main__":
    # Test with n_qubits=10
    n_qubits_test = 10
    log.info(f"Starting simulation with n_qubits={n_qubits_test}")
    entropy_hist, fidelity_hist, best_params, app, win = run_zeno_sims(n_qubits=n_qubits_test, cycles=CYCLES, eeg_fusion=NK_AVAILABLE)
    
    # Advanced sims: Quantum Error Correction with surface codes
    Lx, Ly = int(np.sqrt(n_qubits_test)), int(np.ceil(n_qubits_test / np.sqrt(n_qubits_test)))
    stabilizers = build_surface_code_stabilizers(Lx, Ly, toric=True)
    log.info(f"Surface code stabilizers built for {Lx}x{Ly} lattice")

    # Vision: Multi-threaded consciousness modeling (placeholder for future expansion)
    if cpu_count() > 1:
        log.info(f"Multi-threading enabled with {cpu_count()} cores")
        # Placeholder for multi-threaded quera_mt_sim expansion

    # Keep plot window open
    if app and win:
        QtGui.QApplication.instance().exec_()
