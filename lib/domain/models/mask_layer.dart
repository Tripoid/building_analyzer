import 'package:flutter/material.dart';

/// A single togglable layer in the photoshop-style image viewer.
///
/// Each layer points to a PNG URL produced by the backend (masks are saved
/// as 8-bit palette PNGs alongside the analysis). Tint is applied via a
/// ColorFilter on top of the base image, not baked in server-side.
class MaskLayer {
  const MaskLayer({
    required this.id,
    required this.group,
    required this.nameRu,
    required this.url,
    required this.tint,
    this.visible = true,
  });

  final String id;
  final MaskGroup group;
  final String nameRu;
  final String url;
  final Color tint;
  final bool visible;

  MaskLayer copyWith({bool? visible}) => MaskLayer(
        id: id,
        group: group,
        nameRu: nameRu,
        url: url,
        tint: tint,
        visible: visible ?? this.visible,
      );
}

enum MaskGroup {
  geometry('Геометрия'),
  materials('Материалы'),
  defects('Дефекты');

  const MaskGroup(this.labelRu);
  final String labelRu;
}
