import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../app/providers.dart';
import '../../app/router.dart';
import '../../data/api/api_exceptions.dart';
import '../../domain/models/upload_progress.dart';
import '../../theme/app_theme.dart';

/// Drives the actual upload + analyse call and surfaces typed errors. Replaces
/// the old timer-based pseudo-loader that never hit the network.
class RealAnalysisLoadingScreen extends ConsumerStatefulWidget {
  const RealAnalysisLoadingScreen({super.key, this.args});
  final AnalyzeArgs? args;

  @override
  ConsumerState<RealAnalysisLoadingScreen> createState() =>
      _RealAnalysisLoadingScreenState();
}

class _RealAnalysisLoadingScreenState
    extends ConsumerState<RealAnalysisLoadingScreen> {
  final _cancel = CancelToken();
  ApiException? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _run());
  }

  Future<void> _run() async {
    final args = widget.args;
    if (args == null) {
      context.go('/');
      return;
    }
    final repo = ref.read(analysisRepositoryProvider);
    try {
      final result = await repo.analyze(
        image: args.image,
        calibrationId: args.calibrationId,
        fallbackTotalAreaM2: args.fallbackTotalAreaM2,
        cancelToken: _cancel,
      );
      if (!mounted) return;
      context.go('/results', extra: result);
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() => _error = e);
    }
  }

  @override
  void dispose() {
    if (!_cancel.isCancelled) _cancel.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final asyncProgress = ref.watch(uploadProgressProvider);
    return Scaffold(
      backgroundColor: AppTheme.primaryDark,
      appBar: AppBar(
        automaticallyImplyLeading: false,
        title: const Text('Анализ фасада'),
      ),
      body: SafeArea(
        child: _error != null
            ? _ErrorView(
                error: _error!,
                onRetry: () {
                  setState(() => _error = null);
                  _run();
                },
                onCancel: () => context.go('/'),
              )
            : Padding(
                padding: const EdgeInsets.all(24),
                child: asyncProgress.when(
                  data: (p) => _ProgressView(progress: p, onCancel: _cancelTap),
                  loading: () => _ProgressView(
                    progress: UploadProgress.initial,
                    onCancel: _cancelTap,
                  ),
                  error: (_, __) => _ProgressView(
                    progress: UploadProgress.initial,
                    onCancel: _cancelTap,
                  ),
                ),
              ),
      ),
    );
  }

  void _cancelTap() {
    _cancel.cancel('user');
    context.go('/');
  }
}

class _ProgressView extends StatelessWidget {
  const _ProgressView({required this.progress, required this.onCancel});
  final UploadProgress progress;
  final VoidCallback onCancel;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Spacer(),
        const Center(
          child: SizedBox(
            width: 64,
            height: 64,
            child: CircularProgressIndicator(strokeWidth: 5),
          ),
        ),
        const SizedBox(height: 24),
        Center(
          child: Text(
            progress.stage.label,
            style: Theme.of(context).textTheme.titleMedium,
          ),
        ),
        const SizedBox(height: 24),
        LinearProgressIndicator(
          value: progress.total > 0 ? progress.ratio.clamp(0.0, 1.0) : null,
          minHeight: 6,
        ),
        const SizedBox(height: 8),
        Text(
          progress.total > 0
              ? '${(progress.ratio * 100).toStringAsFixed(0)} %  '
                  '(${_human(progress.sent)} / ${_human(progress.total)})'
              : 'Готовим данные…',
          textAlign: TextAlign.center,
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const Spacer(),
        OutlinedButton.icon(
          onPressed: onCancel,
          icon: const Icon(Icons.close_rounded),
          label: const Text('Отменить'),
        ),
      ],
    );
  }

  String _human(int bytes) {
    if (bytes < 1024) return '$bytes Б';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(0)} КБ';
    return '${(bytes / 1024 / 1024).toStringAsFixed(1)} МБ';
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({
    required this.error,
    required this.onRetry,
    required this.onCancel,
  });
  final ApiException error;
  final VoidCallback onRetry;
  final VoidCallback onCancel;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline, size: 64, color: AppTheme.danger),
            const SizedBox(height: 16),
            Text(
              _title(error),
              style: Theme.of(context).textTheme.titleLarge,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 12),
            Text(
              error.message,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 24),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                OutlinedButton.icon(
                  onPressed: onCancel,
                  icon: const Icon(Icons.arrow_back_rounded),
                  label: const Text('Назад'),
                ),
                const SizedBox(width: 12),
                ElevatedButton.icon(
                  onPressed: onRetry,
                  icon: const Icon(Icons.refresh_rounded),
                  label: const Text('Повторить'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  String _title(ApiException e) => switch (e) {
        NgrokInterstitialException() => 'Туннель показал заглушку',
        ValidationException() => 'Неверные параметры запроса',
        RequestTimeoutException() => 'Сервер не успел ответить',
        NetworkException() => 'Нет связи с сервером',
        CancelledException() => 'Отменено',
        ClientException() => 'Ошибка запроса',
        ServerException() => 'Ошибка на сервере',
      };
}
