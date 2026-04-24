import 'dart:io';

import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../theme/app_theme.dart';

/// Lightweight preview → "на калибровку / пропустить и сразу в анализ".
///
/// When [image] is null we fall back to a placeholder illustration (used only
/// in mock mode, e.g. running the app on a device without a camera).
class PhotoPreviewScreen extends StatelessWidget {
  const PhotoPreviewScreen({super.key, this.image});
  final File? image;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.primaryDark,
      appBar: AppBar(title: const Text('Предпросмотр')),
      body: SafeArea(
        child: LayoutBuilder(
          builder: (ctx, cons) => Column(
            children: [
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Container(
                    decoration: BoxDecoration(
                      color: AppTheme.surfaceCard,
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(
                        color: AppTheme.surfaceLight.withOpacity(0.3),
                      ),
                    ),
                    clipBehavior: Clip.antiAlias,
                    child: image != null
                        ? InteractiveViewer(
                            child: Image.file(image!, fit: BoxFit.contain),
                          )
                        : const Center(
                            child: Icon(
                              Icons.apartment_rounded,
                              size: 120,
                              color: AppTheme.textSecondary,
                            ),
                          ),
                  ),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    ElevatedButton.icon(
                      onPressed: image == null
                          ? null
                          : () => context.go('/calibrate', extra: image),
                      icon: const Icon(Icons.straighten_rounded),
                      label: const Text('Указать масштаб и проанализировать'),
                    ),
                    const SizedBox(height: 8),
                    OutlinedButton.icon(
                      onPressed: () => context.pop(),
                      icon: const Icon(Icons.close_rounded),
                      label: const Text('Отмена'),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
