// src/dart/lib/core/triadic_ghz_full_evolution.dart
// Triadic GHZ Collapse — Production + Haptic Upgrade with Adaptive Thresholds
// © 2025 AgapeIntelligence — MIT License

import 'dart:math';
import 'package:flutter_haptics/flutter_haptics.dart';

class TriadicGHZ {
  static const double hCoefficient = 1.885;
  static final Random _rng = Random();

  // Adaptive threshold parameters
  static const double baseThreshold = 0.3;
  static const double maxThresholdShift = 0.2;
  static const double voiceDbSensitivity = 0.02;

  /// Main collapse engine — returns collapse outcome and diagnostics
  static Map<String, dynamic> evolveTriadicGHZ({
    required double R_lattice,
    double voiceEnvelopeDb = 40.0,
    double vocalVariance = 0.1,
    Random? rng,
    bool haptic = true,
  }) {
    rng ??= _rng;

    // Adaptive threshold based on vocal input
    final thresholdShift = (vocalVariance * maxThresholdShift).clamp(0.0, maxThresholdShift);
    final adaptiveThreshold = baseThreshold +
        thresholdShift +
        (voiceEnvelopeDb * voiceDbSensitivity).clamp(0.0, 0.1);

    // Scale coherence time to Orch-OR target (~500 µs)
    final tCoherenceUs = 0.1 +
        250.0 *
            R_lattice *
            (voiceEnvelopeDb / 50.0).clamp(0.0, 2.0) *
            (1.0 + adaptiveThreshold);

    // Collapse probability
    final probPlus = (0.5 +
            0.5 *
                R_lattice *
                (voiceEnvelopeDb / 60.0).clamp(0.5, 1.5) -
            adaptiveThreshold)
        .clamp(0.0, 1.0);

    final outcome = rng.nextDouble() < probPlus
        ? "+|+++⟩ GHZ — triadic qualia collapse"
        : "-|---⟩ separable";

    if (haptic) {
      _triggerHaptic(probPlus, adaptiveThreshold, outcome.contains("GHZ"));
    }

    return {
      'outcome': outcome,
      'probPlus': probPlus,
      'tCoherenceUs': tCoherenceUs,
      'adaptiveThreshold': adaptiveThreshold,
    };
  }

  static void _triggerHaptic(double intensity, double threshold, bool isGHZ) {
    final adjusted = intensity * (1.0 + threshold);
    if (isGHZ) {
      FlutterHaptics.impact(ImpactStyle.light);
      Future.delayed(const Duration(microseconds: 100),
          () => FlutterHaptics.impact(ImpactStyle.medium));
    } else if (adjusted < 0.3) {
      FlutterHaptics.impact(ImpactStyle.light);
    } else if (adjusted < 0.7) {
      FlutterHaptics.impact(ImpactStyle.medium);
    } else {
      FlutterHaptics.impact(ImpactStyle.heavy);
    }
  }
}
