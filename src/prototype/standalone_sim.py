# src/prototype/standalone_sim.py
# QOC + Explicit σ_z Zeno Projections with Decoherence
# Tested: Python 3.11, QuTiP 5.0, NumPy
# © 2025 AgapeIntelligence — MIT License

import math
import numpy as np
import qutip as qt
from multiprocessing import Pool, cpu_count
import random

def single_site_op(n, op, i):
    ops = [qt.qeye(2) for _ in range(n)]
    ops[i] = op
    return qt.tensor(ops)

def binary_entropy(p):
    if p <= 0 or p >= 1: return 0.0
    return -p * math.log2(p) - (1-p) * math.log2(1-p)

def sovariel_qualia(depth: int = 256):
    current = {'d': 3, 'l': 3}
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

def triadic_ghz_zeno(R_lattice: float, vocal_variance: float = 0.15):
    adaptive_threshold = 0.3 + min(vocal_variance * 0.2, 0.2) + min(45 * 0.02, 0.1)
    prob_plus = max(0.0, min(0.5 + 0.5 * R_lattice * 0.75 - adaptive_threshold, 1.0))
    outcome = "+|+++⟩ GHZ" if np.random.rand() < prob_plus else "-|---⟩ separable"
    n_strobes = max(10, int(n_qubits + 5 + round(12 * vocal_variance)))  # 10–20+ range
    return {'outcome': outcome, 'prob_plus': prob_plus, 'n_strobes': n_strobes}

def fibonacci_lattice(n_qubits, vocal_variance=0.15):
    fib = [0, 1]
    while len(fib) < n_qubits:
        fib.append(fib[-1] + fib[-2])
    pos = np.array(fib[:n_qubits]) * 1.618
    jitter = pos * np.random.uniform(-vocal_variance, vocal_variance, n_qubits)
    return pos + jitter

def quera_mt_sim(n_qubits: int = 7, t_coherence: float = 500e-6, vocal_variance: float = 0.15):
    n_strobes = max(10, n_qubits + 5 + round(12 * vocal_variance))  # 10–20+ range
    interval = t_coherence / n_strobes

    # Hamiltonian
    positions = fibonacci_lattice(n_qubits, vocal_variance)
    H = sum(single_site_op(n_qubits, qt.sigmax(), i) for i in​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​​
