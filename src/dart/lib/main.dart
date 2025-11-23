// src/dart/lib/main.dart
// Minimal entrypoint — proves flutter run works instantly

import 'package:flutter/material.dart';
import 'core/triadic_ghz_full_evolution.dart';

void main() {
  runApp(const QOCApp());
}

class QOCApp extends StatelessWidget {
  const QOCApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'QOC Framework',
      home: Scaffold(
        appBar: AppBar(title: const Text('Quantum-Orchestrated Consciousness')),
        body: const Center(child: TriadicDemo()),
      ),
    );
  }
}

class TriadicDemo extends StatelessWidget {
  const TriadicDemo({super.key});

  @override
  Widget build(BuildContext context) {
    final result = TriadicGHZ.evolveTriadicGHZ(
      R_lattice: 1.12,
      voiceEnvelopeDb: 48.0,
      vocalVariance: 0.15,
      haptic: true,
    );

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(result['outcome'], style: const TextStyle(fontSize: 20)),
        const SizedBox(height: 16),
        Text("R_lattice used: 1.12 | τ ≈ ${result['tCoherenceUs'].toStringAsFixed(1)} µs"),
        const SizedBox(height: 32),
        const Text("QOC Framework v0.1.0 — Ready for QuEra 2026–2027"),
      ],
    );
  }
}
