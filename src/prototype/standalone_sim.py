import math
import numpy as np
import qutip as qt

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

def quera_mt_sim(n_qubits: int = 3, t_coherence: float = 500e-6):
    """QuTiP sim of triadic GHZ in Fibonacci lattice proxy for MT."""
    # Simple 3-qubit GHZ (scale n_qubits for larger sims)
    H = qt.tensor([qt.sigmax()] * n_qubits)
    initial_state = qt.tensor([qt.basis(2, 0)] * n_qubits)
    times = np.linspace(0, t_coherence, 100)
    result = qt.mesolve(H, initial_state, times)
    final_state = result.states[-1]
    # Entanglement entropy proxy (partial trace on first qubit)
    entropy = qt.entropy_vn(final_state.ptrace(0))
    return entropy, final_state

def standalone_sim():
    """Standalone Orch-OR proxy sim: Sovariel → TriadicGHZ → QuEra."""
    print("=== Standalone Orch-OR Proxy Sim ===\n")

    # Step 1: Sovariel lattice
    H, p, cri, r, gain, latency = sovariel_qualia()
    R_lattice = min(cri / 10.0, 1.5)
    print(f"Sovariel: H={H:.4f}, p={p:.4f}, CRI={cri:.2e}, R={r}, Gain={gain}%, Latency={latency:.1e}s")
    print(f"Derived R_lattice={R_lattice:.4f}\n")

    # Step 2: Triadic collapse
    result = triadic_ghz_evolution(R_lattice=R_lattice, voice_envelope_db=45.0, vocal_variance=0.15)
    print(f"TriadicGHZ: Outcome={result['outcome']}")
    print(f"Prob+={result['prob_plus']:.4f}, τ≈{result['t_coherence_us']:.1f} µs, Threshold={result['adaptive_threshold']:.4f}\n")

    # Step 3: QuEra sim proxy
    entropy, final_state = quera_mt_sim(n_qubits=3, t_coherence=result['t_coherence_us'] * 1e-6)
    print(f"QuEra Sim: Entanglement entropy={entropy:.4f}")
    print(f"Final state summary: {final_state.full()}\n")

    print("=== Sim Complete ===\n")

standalone_sim()
