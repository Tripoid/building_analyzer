import 'package:flutter/material.dart';

/// Catalog of built-in references the user can pick from. Values are the
/// midpoints of the prior range accepted by `backend/calibration.py`.
enum ReferenceType {
  door(
    apiKey: 'door',
    labelRu: 'Дверной проём',
    defaultWidthM: 0.9,
    minWidthM: 0.7,
    maxWidthM: 1.2,
  ),
  window(
    apiKey: 'window',
    labelRu: 'Окно',
    defaultWidthM: 1.2,
    minWidthM: 0.8,
    maxWidthM: 2.0,
  ),
  brick(
    apiKey: 'brick',
    labelRu: 'Кирпич (длинная сторона)',
    defaultWidthM: 0.25,
    minWidthM: 0.2,
    maxWidthM: 0.3,
  ),
  custom(
    apiKey: 'custom',
    labelRu: 'Произвольный размер',
    defaultWidthM: 1.0,
    minWidthM: 0.05,
    maxWidthM: 50.0,
  );

  const ReferenceType({
    required this.apiKey,
    required this.labelRu,
    required this.defaultWidthM,
    required this.minWidthM,
    required this.maxWidthM,
  });

  final String apiKey;
  final String labelRu;
  final double defaultWidthM;
  final double minWidthM;
  final double maxWidthM;
}

/// Result of the two-tap or rectangle-drawing calibration widget.
class CalibrationInput {
  const CalibrationInput({
    required this.type,
    required this.widthM,
    required this.imageWidthPx,
    required this.imageHeightPx,
    this.heightM,
    this.p1,
    this.p2,
    this.bbox,
  }) : assert(
          (p1 != null && p2 != null) || bbox != null,
          'either two points or a bbox must be set',
        );

  final ReferenceType type;
  final double widthM;
  final double? heightM;
  final int imageWidthPx;
  final int imageHeightPx;
  final Offset? p1;
  final Offset? p2;
  final Rect? bbox;

  Map<String, dynamic> toApiPayload() {
    final m = <String, dynamic>{
      'reference_type': type.apiKey,
      'reference_width_m': widthM,
      if (heightM != null) 'reference_height_m': heightM,
      'image_width_px': imageWidthPx,
      'image_height_px': imageHeightPx,
    };
    if (p1 != null && p2 != null) {
      m['p1'] = [p1!.dx, p1!.dy];
      m['p2'] = [p2!.dx, p2!.dy];
    } else if (bbox != null) {
      m['bbox'] = [bbox!.left, bbox!.top, bbox!.right, bbox!.bottom];
    }
    return m;
  }
}

/// Response from POST /api/calibrate.
class CalibrationResult {
  const CalibrationResult({
    required this.calibrationId,
    required this.pxPerM,
    required this.m2PerPx,
    required this.warnings,
  });

  factory CalibrationResult.fromJson(Map<String, dynamic> json) =>
      CalibrationResult(
        calibrationId: json['calibration_id'] as String,
        pxPerM: (json['px_per_m'] as num).toDouble(),
        m2PerPx: (json['m2_per_px'] as num).toDouble(),
        warnings:
            (json['warnings'] as List?)?.map((e) => e.toString()).toList() ?? const [],
      );

  final String calibrationId;
  final double pxPerM;
  final double m2PerPx;
  final List<String> warnings;
}
