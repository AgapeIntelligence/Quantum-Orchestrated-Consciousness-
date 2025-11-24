def standalone_sim(n_qubits=7, cycles=100, vocal_variance=0.15):
    print("=== QOC + Explicit σ_z Zeno Projections ===\n")
    _, _, cri = sovariel_qualia()
    R_lattice = min(cri / 10.0, 1.5)
    print(f"R_lattice = {R_lattice:.4f}")

    mean_entropies, mean_fidelities, mean_strobes = [], [], []
    for cycle in range(cycles):
        entropy, fidelity, strobes = quera_mt_sim(n_qubits, 500e-6, vocal_variance)
        mean_entropies.append(entropy)
        mean_fidelities.append(fidelity)
        mean_strobes.append(strobes)
        print(f"Cycle {cycle}: S_vN = {entropy:.4f}, Fidelity = {fidelity:.4f}, Strobes = {strobes}")

    print(f"\nMean entropy over {cycles} cycles = {np.mean(mean_entropies):.4f} ± {np.std(mean_entropies):.4f}")
    print(f"Mean fidelity over {cycles} cycles = {np.mean(mean_fidelities):.4f} ± {np.std(mean_fidelities):.4f}")
    print(f"Mean Zeno strobes = {np.mean(mean_strobes):.1f}")
    print("=== Sim Complete ===")
