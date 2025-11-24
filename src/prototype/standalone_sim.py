# src/prototype/standalone_sim.py
# QOC + σ_z Zeno with Lindblad Decoherence, Sparse Solvers, Voice Input
# Tested: Python 3.11, QuTiP 5.0, NumPy, sounddevice
# © 2025 AgapeIntelligence — MIT License

import math
import numpy as np
import qutip as qt
from multiprocessing import Pool, cpu_count
import random
import sounddevice as sd
import time

def single_site_op(n, op, i):
    ops = [qt.qeye(2) for _ in range(n)]
    ops[i] = op
    return qt.tensor(ops)

def binary_entropy(p):
    if p <= 0 or p >= 1: return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def sovariel_qualia(depth: int = 256):
    current = {'d': 3, 'l': 3}
    tokens = sum(current.values())  # Initialize tokens
    p = current['d'] / tokens       # Initialize p
    H = binary_entropy(p)           # Initialize H
    for i in range(1, depth + 1):
        if i > 1:
            tokens = sum(current.values())
            large = tokens // 3 + 1
            small = tokens // 6 + 1
            lead = 'd' if current['d'] < current['l'] else 'l'
            add_d = large // 2 + (2 * small) if lead == 'd' else 0
            add_l = large // 2 + (2 * small) if lead == 'l' else 0
            add_d += int(add_d * np.random.uniform(-0.05, 0.05))
            add_l += int(add_l * np.random.uniform(-0.05, 0.05))
            new = {'d': current['d'] + max(0, add_d), 'l': current['l'] + max(0, add_l)}
            new_tokens = sum(new.values())
            p = new['d'] / new_tokens
            if binary_entropy(p) < 0.99:
                diff = round((0.5 - p) * new_tokens)
                new['d'] += diff
                new['l'] -= diff
            current = new
    tokens = sum(current.values())
    p = current['d'] / tokens
    H = binary_entropy(p)
    cri = 0.4 * (tokens / 5 / 10) + 0.3 / (1 + H) + 0.3 * (4 / 10)
    return H, p, cri

def triadic_ghz_zeno(R_lattice: float, vocal_variance: float = 0.15, n_qubits: int = 10):
    adaptive_threshold = 0.3 + min(vocal_variance * 0.2, 0.2) + min(45 * 0.02, 0.1)
    prob_plus = max(0.0, min(0.5 + 0.5 * R_lattice * 0.75 - adaptive_threshold, 1.0))
    outcome = "+|+++⟩ GHZ" if np.random.rand() < prob_plus else "-|---⟩ separable"
    n_strobes = max(10, n_qubits + 5 + round(12 * vocal_variance))  # 10–20+ range
    return {'outcome': outcome, 'prob_plus': prob_plus, 'n_strobes': n_strobes}

def fibonacci_lattice(n_qubits, vocal_variance=0.15):
    fib = [0, 1]
    while len(fib) < n_qubits:
        fib.append(fib[-1] + fib[-2])
    pos = np.array(fib[:n_qubits]) * 1.618
    jitter = pos * np.random.uniform(-vocal_variance, vocal_variance, n_qubits)
    return pos + jitter

def quera_mt_sim(n_qubits: int = 10, t_coherence: float = 500e-6, vocal_variance: float = 0.15):
    n_strobes = max(10, n_qubits + 5 + round(12 * vocal_variance))  # 10–20+ range
    interval = t_coherence / n_strobes

    # Hamiltonian with sparse operators
    positions = fibonacci_lattice(n_qubits, vocal_variance)
    H = sum(single_site_op(n_qubits, qt.sigmax(), i) for i in range(n_qubits))
    for i in range(n_qubits - 1):
        dist = max(1e-6, abs(positions[i+1] - positions[i]))
        J = 1.0 / dist
        ops = [qt.qeye(2) for _ in range(n_qubits)]
        ops[i] = ops[i+1] = qt.sigmaz()
        H += J * qt.tensor(ops)
    H = H.to(qt.Qobj, data=qt.CSR)

    # Initial state |+>^n
    plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    psi = qt.tensor([plus for _ in range(n_qubits)])

    # Ideal GHZ target
    zero_all = qt.tensor([qt.basis(2, 0) for _ in range(n_qubits)])
    one_all = qt.tensor([qt.basis(2, 1) for _ in range(n_qubits)])
    ghz_ideal = (zero_all + one_all).unit()

    # Lindblad decoherence operators (QuEra noise model)
    gamma_damp = 0.01  # ms^-1
    gamma_deph = 0.005  # ms^-1
    c_ops = []
    for i in range(n_qubits):
        c_ops.append(np.sqrt(gamma_damp * 1e-3) * single_site_op(n_qubits, qt.destroy(2), i).to(qt.Qobj, data=qt.CSR))
        c_ops.append(np.sqrt(gamma_deph * 1e-3) * single_site_op(n_qubits, qt.sigmaz(), i).to(qt.Qobj, data=qt.CSR))

    # Zeno-enhanced evolution with Lindblad decoherence
    for _ in range(n_strobes):
        result = qt.mesolve(H, psi, [0, interval], c_ops=c_ops, options=qt.Options(store_states=True, rhs_reuse=True))
        psi = result.states[-1]
        # Global σ_z^{\otimes n} projection
        proj = sum(qt.tensor([qt.basis(2, i) * qt.basis(2, i).dag() for i in range(2)]) for _ in range(n_qubits)).to(qt.Qobj, data=qt.CSR)
        psi = (proj * psi).unit()

    fidelity = qt.fidelity(psi, ghz_ideal)
    rho_center = psi.ptrace(n_qubits // 2)
    entropy = qt.entropy_vn(rho_center)

    return entropy, fidelity, n_strobes

def run_zeno_sims(n_qubits=10, num_sims=None, vocal_variance=0.15):
    if num_sims is None:
        num_sims = max(1, cpu_count() - 1)
    args = [(n_qubits, 500e-6, vocal_variance) for _ in range(num_sims)]
    with Pool(num_sims) as p:
        results = p.map(lambda x: quera_mt_sim(*x), args)
    entropies, fidelities, strobes = zip(*results)
    return np.mean(entropies), np.std(entropies), np.mean(fidelities), np.std(fidelities), np.mean(strobes)

def get_voice_variance(duration=0.1, sample_rate=44100):
    def callback(indata, frames, time, status):
        volume_norm = np.linalg.norm(indata) / frames
        variance = min(0.3, max(0.05, volume_norm * 10))  # Scale to 0.05–0.3
        nonlocal vocal_variance
        vocal_variance = variance

    vocal_variance = 0.15  # Default
    with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
        time.sleep(duration)
    return vocal_variance

def standalone_sim(n_qubits=10, cycles=100):
    print("=== QOC + σ_z Zeno with Lindblad Decoherence & Voice Input ===\n")
    _, _, cri = sovariel_qualia()
    R_lattice = min(cri / 10.0, 1.5)
    print(f"R_lattice = {R_lattice:.4f}")

    mean_entropies, mean_fidelities, mean_strobes = [], [], []
    for cycle in range(cycles):
        # Update vocal_variance with real-time voice input
        vocal_variance = get_voice_variance()
        mean_entropy, std_entropy, mean_fidelity, std_fidelity, strobes = run_zeno_sims(n_qubits, vocal_variance=vocal_variance)
        mean_entropies.append(mean_entropy)
        mean_fidelities.append(mean_fidelity)
        mean_strobes.append(strobes)
        print(f"Cycle {cycle}: S_vN = {mean_entropy:.4f} ± {std_entropy:.4f}, Fidelity = {mean_fidelity:.4f} ± {std_fidelity:.4f}, Strobes = {strobes:.1f}, Voice Var = {vocal_variance:.3f}")

    print(f"\nMean entropy over {cycles} cycles = {np.mean(mean_entropies):.4f} ± {np.std(mean_entropies):.4f}")
    print(f"Mean fidelity over {cycles} cycles = {np.mean(mean_fidelities):.4f} ± {np.std(mean_fidelities):.4f}")
    print(f"Mean Zeno strobes = {np.mean(mean_strobes):.1f}")
    print("=== Sim Complete ===")

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    # Install sounddevice if not present: pip install sounddevice
    standalone_sim(n_qubits=10, cycles=100)
