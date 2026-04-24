import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';

import '../app/app_config.dart';
import '../theme/app_theme.dart';

/// Home screen — launches camera or gallery picker, routes to preview.
/// Fully responsive: no fixed heights, scrolls safely on small phones.
class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final cfg = ref.watch(appConfigProvider);
    return Scaffold(
      backgroundColor: AppTheme.primaryDark,
      appBar: AppBar(
        title: const Text('AlegroCode'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_rounded),
            onPressed: () => context.go('/settings'),
          ),
        ],
      ),
      body: SafeArea(
        child: LayoutBuilder(
          builder: (ctx, cons) {
            return SingleChildScrollView(
              physics: const BouncingScrollPhysics(),
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _Header(serverConnected: cfg.serverUrl.isNotEmpty || cfg.useMock),
                  const SizedBox(height: 24),
                  _ActionGrid(
                    onCamera: () => _pickAndPreview(context, ImageSource.camera),
                    onGallery: () => _pickAndPreview(context, ImageSource.gallery),
                  ),
                  const SizedBox(height: 32),
                  Text(
                    'Как это работает',
                    style: Theme.of(context).textTheme.titleLarge,
                  ),
                  const SizedBox(height: 12),
                  const _Step(
                    index: 1,
                    title: 'Сфотографируйте фасад',
                    description: 'Один прямой кадр, фасад должен быть виден целиком.',
                  ),
                  const _Step(
                    index: 2,
                    title: 'Укажите эталон масштаба',
                    description: 'Дверь, окно или кирпич — чтобы площадь считалась в м².',
                  ),
                  const _Step(
                    index: 3,
                    title: 'Получите смету в ₽',
                    description: 'Выделяем дефекты, рассчитываем ремонт, показываем ИИ-реставрацию.',
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  Future<void> _pickAndPreview(BuildContext context, ImageSource source) async {
    final picker = ImagePicker();
    final XFile? picked = await picker.pickImage(
      source: source,
      maxWidth: 4000,
      imageQuality: 92,
    );
    if (!context.mounted || picked == null) return;
    context.go('/preview', extra: File(picked.path));
  }
}

class _Header extends StatelessWidget {
  const _Header({required this.serverConnected});
  final bool serverConnected;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Добро пожаловать',
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        const SizedBox(height: 4),
        Row(
          children: [
            Container(
              width: 8,
              height: 8,
              decoration: BoxDecoration(
                color: serverConnected ? AppTheme.success : AppTheme.warning,
                shape: BoxShape.circle,
              ),
            ),
            const SizedBox(width: 6),
            Flexible(
              child: Text(
                serverConnected ? 'Сервер подключён' : 'Укажите адрес сервера в настройках',
                style: Theme.of(context).textTheme.bodySmall,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
      ],
    );
  }
}

class _ActionGrid extends StatelessWidget {
  const _ActionGrid({required this.onCamera, required this.onGallery});
  final VoidCallback onCamera;
  final VoidCallback onGallery;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (ctx, cons) {
        final wide = cons.maxWidth >= 480;
        final children = [
          _ActionCard(
            icon: Icons.photo_camera_rounded,
            title: 'Снять фото',
            subtitle: 'Сфотографировать фасад на камеру',
            onTap: onCamera,
          ),
          _ActionCard(
            icon: Icons.photo_library_rounded,
            title: 'Из галереи',
            subtitle: 'Выбрать готовое фото фасада',
            onTap: onGallery,
          ),
        ];
        return wide
            ? Row(children: children.map((c) => Expanded(child: Padding(
                  padding: const EdgeInsets.all(4),
                  child: c,
                ))).toList())
            : Column(children: children.map((c) => Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: c,
                )).toList());
      },
    );
  }
}

class _ActionCard extends StatelessWidget {
  const _ActionCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: AppTheme.surfaceCard,
      borderRadius: BorderRadius.circular(16),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              CircleAvatar(
                backgroundColor: AppTheme.accent.withOpacity(0.15),
                foregroundColor: AppTheme.accent,
                child: Icon(icon),
              ),
              const SizedBox(height: 12),
              Text(
                title,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 4),
              Text(
                subtitle,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Step extends StatelessWidget {
  const _Step({
    required this.index,
    required this.title,
    required this.description,
  });
  final int index;
  final String title;
  final String description;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          CircleAvatar(
            radius: 18,
            backgroundColor: AppTheme.accent.withOpacity(0.15),
            foregroundColor: AppTheme.accent,
            child: Text('$index',
                style: const TextStyle(fontWeight: FontWeight.w700)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: Theme.of(context).textTheme.titleSmall),
                const SizedBox(height: 2),
                Text(description, style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
