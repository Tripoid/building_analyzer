import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';
import 'package:photo_view/photo_view.dart';

import '../../domain/models/mask_layer.dart';

/// Photoshop-style layered viewer:
///     base image + stacked masks (each tinted via ColorFilter, toggled via
///     MaskLayer.visible). Pinch-zoom + pan via [PhotoView].
///
/// All mask PNGs are served with `ngrok-skip-browser-warning` headers through
/// [cached_network_image]. If you change the header scheme, update the map here.
class MaskLayerViewer extends StatelessWidget {
  const MaskLayerViewer({
    super.key,
    required this.baseImageUrl,
    required this.layers,
    this.aspectRatio = 4 / 3,
  });

  final String baseImageUrl;
  final List<MaskLayer> layers;
  final double aspectRatio;

  static const Map<String, String> _headers = {
    'ngrok-skip-browser-warning': 'true',
  };

  @override
  Widget build(BuildContext context) {
    return AspectRatio(
      aspectRatio: aspectRatio,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: Container(
          color: Colors.black,
          child: PhotoView.customChild(
            minScale: PhotoViewComputedScale.contained,
            maxScale: PhotoViewComputedScale.covered * 4,
            backgroundDecoration: const BoxDecoration(color: Colors.black),
            childSize: null,
            child: Stack(
              fit: StackFit.expand,
              children: [
                if (baseImageUrl.isNotEmpty)
                  CachedNetworkImage(
                    imageUrl: baseImageUrl,
                    httpHeaders: _headers,
                    fit: BoxFit.contain,
                    placeholder: (_, __) => const Center(
                      child: CircularProgressIndicator(),
                    ),
                    errorWidget: (_, __, ___) => const Icon(
                      Icons.broken_image_rounded,
                      size: 64,
                      color: Colors.white30,
                    ),
                  ),
                for (final layer in layers.where((l) => l.visible))
                  IgnorePointer(
                    child: ColorFiltered(
                      colorFilter: _maskTint(layer.tint),
                      child: CachedNetworkImage(
                        imageUrl: layer.url,
                        httpHeaders: _headers,
                        fit: BoxFit.contain,
                        fadeInDuration: const Duration(milliseconds: 120),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  /// Tint-through: the mask is binary (0/255); we keep only the bright pixels
  /// and recolour them with the requested tint (alpha from the provided color).
  ColorFilter _maskTint(Color color) {
    return ColorFilter.matrix(<double>[
      0, 0, 0, 0, color.red.toDouble(),
      0, 0, 0, 0, color.green.toDouble(),
      0, 0, 0, 0, color.blue.toDouble(),
      // keep alpha modulated by luminance of the mask (so black → transparent)
      0.299, 0.587, 0.114, 0, -255 * (1 - color.opacity),
    ]);
  }
}
