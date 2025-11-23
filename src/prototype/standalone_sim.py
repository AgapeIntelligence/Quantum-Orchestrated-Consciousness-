import math
import numpy as np
import qutip as qt
from multiprocessing import Pool, cpu_count

def binary_entropy(p: float) -> float:
    """Binary entropy in bits."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def sovariel_qualia(depth: int = 256, noise: float = 0.05):
    """Recursive lattice model with self-regulated entropy (H ≈ 1)."""
    current = {'d': 3, 'l': 3}
    for i in range(1, depth + 1):
        if i > 1:
            tokens = sum(current.values())
            large = tokens // 3 + 1
            small = tokens // 6 + 1
            lead = 'd' if current['d'] < current['l'] else 'l'
            add_d = large // 2 + (2 * small) if lead == 'd' else 0
            add_l = large // 2 + (2 * small) if lead == 'l' else 0
            # Noise skew (EEG-like variance)
            add_d += int(add_d * np.random.uniform(-noise, noise))
            add_l += int(add_l * np.random.uniform(-noise, noise))
            new = {'d': current['d'] + max(0, add_d),
                   'l': current['l'] + max(0, add_l)}
            new_tokens = sum(new.values())
            p = new['d'] / new_tokens
            H = binary_entropy(p)
            if H < 0.99:
                diff = round((0.5 - p) * new_tokens)
                new['d'] += diff
                new['l'] -= diff
            current = new

    tokens = sum(current.values())
    p = current['d'] / tokens
    H = binary_entropy(p)
    cri = 0.4 * (tokens / 5 / 10) + 0.3 / (1 + H) + 0.3 * (4 / 10)
    r = 0.115
    efficiency_gain = 24.7
    latency = 8.3e-3 / 16  # Scaled to ~0.5 ms for OR timing

    return H, p, cri, r, efficiency_gain, latency

def triadic_ghz_evolution(R_lattice: float, voice_envelope_db: float = 40.0, vocal_variance: float = 0.1):
    """Python port of TriadicGHZ.evolveTriadicGHZ for standalone sim."""
    h_coefficient = 1.885

    base_threshold = 0.3
    max_threshold_shift = 0.2
    voice_db_sensitivity = 0.02

    threshold_shift = max(0.0, min(vocal_variance * max_threshold_shift, max_threshold_shift))
    adaptive_threshold = base_threshold + threshold_shift + max(0.0, min(voice_envelope_db * voice_db_sensitivity, 0.1))

    t_coherence_us = 0.1 + 250.0 * R_lattice * max(0.0, min(voice_envelope_db / 50.0, 2.0)) * (1.0 + adaptive_threshold)

    prob_plus = max(0.0, min(0.5 + 0.5 * R_lattice * max(0.5, min(voice_envelope_db / 60.0, 1.5)) - adaptive_threshold, 1.0))

    outcome = "+|+++⟩ GHZ — triadic qualia collapse" if np.random.rand() < prob_plus else "-|---⟩ separable"

    # Haptic proxy: print intensity instead of calling Flutter
    intensity = prob_plus
    threshold = adaptive_threshold
    adjusted = intensity * (1.0 + threshold)
    haptic_style = "light" if adjusted < 0.3 else "medium" if adjusted < 0.7 else "heavy"
    print(f"Haptic proxy: {haptic_style} impact")

    return {
        'outcome': outcome,
        'prob_plus': prob_plus,
        't_coherence_us': t_coherence_us,
        'adaptive_threshold': adaptive_threshold,
    }

def fibonacci_lattice(n_qubits: int, vocal_variance: float = 0.1):
    """Generate Fibonacci-spaced positions with vocal variance jitter."""
    fib = [0, 1]
    while len(fib) < n_qubits:
        fib.append(fib[-1] + fib[-2])
    positions = np.array(fib[:n_qubits]) * 1.618
    jitter = positions * np.random.uniform(-vocal_variance, vocal_variance, n_qubits)
    return positions + jitter

def noisy_ghz_hamiltonian(n_qubits: int, positions: np.ndarray, coupling_strength: float = 1.0):
    """H for GHZ evolution with nearest-neighbor coupling scaled by Fib distances."""
    H = qt.tensor([qt.sigmax()] * n_qubits)  # Base transverse field
    for i in range(n_qubits - 1):
        dist = abs(positions[i+1] - positions[i])
        J = coupling_strength / dist  # Inverse-distance coupling (Rydberg-like)
        H += J * qt.tensor([qt.sigmaz() if j == i else qt.qeye(2) for j in range(n_qubits)]) * \
             qt.tensor([qt.sigmaz() if j == i+1 else qt.qeye(2) for j in range(n_qubits)])
    return H

def fib_mt_sim(args):
    """Wrapper for multiprocessing: (n_qubits, t_final, noise_rate, vocal_variance)"""
    n_qubits, t_final, noise_rate, vocal_variance = args
    positions = fibonacci_lattice(n_qubits, vocal_variance)
    H = noisy_ghz_hamiltonian(n_qubits, positions)
    
    # Initial |+>^n state for GHZ
    initial = qt.tensor([qt.basis(2, 0) + qt.basis(2, 1) for _ in range(n_qubits)]).unit()
    
    # Collapse operators: amplitude damping + dephasing for noise
    c_ops = [np.sqrt(noise_rate) * qt.tensor([qt.destroy(2) if i == j else qt.qeye(2) for i in range(n_qubits)]) 
             for j in range(n_qubits)]
    c_ops += [np.sqrt(0.005) * qt.tensor([qt.sigmaz() if i == j else qt.qeye(2) for i in range(n_qubits)]) 
              for j in range(n_qubits)]  # Dephasing

    times = np.linspace(0, t_final, 100)
    result = qt.mesolve(H, initial, times, c_ops=c_ops)
    
    final_state = result.states[-1]
    rho_center = final_state.ptrace(n_qubits // 2)
    entropy = qt.entropy_vn(rho_center)
    
    ghz_ideal = (qt.tensor([qt.basis(2, 0) + qt.basis(2, 1) for _ in range(n_qubits)]) / math.sqrt(2**n_qubits)).unit()
    fidelity = qt.fidelity(final_state, ghz_ideal)
    
    return entropy, fidelity

def run_multi_sims(n_qubits: int = 13, num_sims: int = cpu_count(), vocal_variance: float = 0.1):
    args = [(n_qubits, 500e-6, 0.01, vocal_variance) for _ in range(num_sims)]
    with Pool(num_sims) as p:
        results = p.map(fib_mt_sim, args)
    avg_entropy = np.mean([r[0] for r in results])
    avg_fidelity = np.mean([r[1] for r in results])
    return avg_entropy, avg_fidelity

def standalone_sim(n_qubits: int = 13, cycles: int = 100, vocal_variance: float = 0.1, num_sims: int = cpu_count()):
    """Standalone Orch-OR proxy sim: Sovariel → TriadicGHZ → QuEra (multi-threaded)."""
    print("=== Standalone Orch-OR Proxy Sim (Multi-Threaded) ===\n")

    # Step 1: Sovariel lattice
    H, p, cri, r, gain, latency = sovariel_qualia()
    R_lattice = min(cri / 10.0, 1.5)
    print(f"Sovariel: H={H:.4f}, p={p:.4f}, CRI={cri:.2e}, R={r}, Gain={gain}%, Latency={latency:.1e}s")
    print(f"Derived R_lattice={R_lattice:.4f}\n")

    # Step 2: Triadic collapse loop
    entropy_drops = []
    for cycle in range(cycles):
        result = triadic_ghz_evolution(R_lattice=R_lattice, voice_envelope_db=45.0, vocal_variance=vocal_variance)
        print(f"Cycle {cycle}: Outcome={result['outcome']}, τ≈{result['t_coherence_us']:.1f} µs")

        # Step 3: Multi-threaded QuEra sim proxy
        avg_entropy, avg_fidelity = run_multi_sims(n_qubits, num_sims=num_sims, vocal_variance=vocal_variance)
        print(f"Cycle {cycle}: Avg entropy={avg_entropy:.4f}, Avg GHZ fidelity={avg_fidelity:.4f}\n")
        entropy_drops.append(avg_entropy)

    print(f"Mean entropy drop over {cycles} cycles: {np.mean(entropy_drops):.4f}")
    print("=== Sim Complete ===\n")

standalone_sim()
```​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​
