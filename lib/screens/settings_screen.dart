import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../app/app_config.dart';
import '../app/providers.dart';
import '../data/api/api_exceptions.dart';
import '../theme/app_theme.dart';

class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  late TextEditingController _urlCtrl;
  _HealthStatus _health = _HealthStatus.idle();

  @override
  void initState() {
    super.initState();
    _urlCtrl = TextEditingController(
      text: ref.read(appConfigProvider).serverUrl,
    );
  }

  @override
  void dispose() {
    _urlCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cfg = ref.watch(appConfigProvider);
    return Scaffold(
      backgroundColor: AppTheme.primaryDark,
      appBar: AppBar(
        title: const Text('Настройки'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded),
          onPressed: () {
            if (context.canPop()) {
              context.pop();
            } else {
              context.go('/');
            }
          },
        ),
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            SwitchListTile(
              value: cfg.useMock,
              onChanged: (v) =>
                  ref.read(appConfigProvider.notifier).setUseMock(v),
              title: const Text('Режим мок-данных'),
              subtitle: const Text('Сервер не нужен — вернутся демо-данные'),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _urlCtrl,
              enabled: !cfg.useMock,
              decoration: const InputDecoration(
                labelText: 'URL сервера',
                hintText: 'https://your-tunnel.ngrok-free.app',
                prefixIcon: Icon(Icons.link_rounded),
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.url,
              onChanged: (v) =>
                  ref.read(appConfigProvider.notifier).setServerUrl(v),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: cfg.useMock ? null : _checkConnection,
                    icon: const Icon(Icons.network_check_rounded),
                    label: const Text('Проверить подключение'),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            _HealthBanner(status: _health),
            const SizedBox(height: 16),
            SwitchListTile(
              value: cfg.verboseLogging,
              onChanged: (v) =>
                  ref.read(appConfigProvider.notifier).setVerbose(v),
              title: const Text('Подробные логи'),
              subtitle: const Text('Сетевые запросы/ответы в консоли'),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _checkConnection() async {
    setState(() => _health = _HealthStatus.checking());
    try {
      final dio = ref.read(dioProvider);
      final resp = await dio
          .get<Map<String, dynamic>>('/api/health')
          .timeout(const Duration(seconds: 10));
      if (resp.data != null && resp.statusCode == 200) {
        setState(
          () => _health = _HealthStatus.ok(
            version: resp.data!['version']?.toString() ?? '?',
            device: resp.data!['device']?.toString() ?? '?',
            modelsLoaded: resp.data!['models_loaded'] == true,
          ),
        );
      } else {
        setState(
          () => _health = _HealthStatus.error('HTTP ${resp.statusCode ?? 0}'),
        );
      }
    } on DioException catch (e) {
      final typed = mapDioException(e);
      setState(() => _health = _HealthStatus.error(typed.message));
    } catch (e) {
      setState(() => _health = _HealthStatus.error(e.toString()));
    }
  }
}

class _HealthStatus {
  _HealthStatus._(
    this.state, {
    this.message,
    this.version,
    this.device,
    this.modelsLoaded,
  });
  factory _HealthStatus.idle() => _HealthStatus._(_State.idle);
  factory _HealthStatus.checking() => _HealthStatus._(_State.checking);
  factory _HealthStatus.ok({
    required String version,
    required String device,
    required bool modelsLoaded,
  }) => _HealthStatus._(
    _State.ok,
    version: version,
    device: device,
    modelsLoaded: modelsLoaded,
  );
  factory _HealthStatus.error(String message) =>
      _HealthStatus._(_State.error, message: message);

  final _State state;
  final String? message;
  final String? version;
  final String? device;
  final bool? modelsLoaded;
}

enum _State { idle, checking, ok, error }

class _HealthBanner extends StatelessWidget {
  const _HealthBanner({required this.status});
  final _HealthStatus status;

  @override
  Widget build(BuildContext context) {
    switch (status.state) {
      case _State.idle:
        return const SizedBox.shrink();
      case _State.checking:
        return const ListTile(
          leading: SizedBox(
            width: 24,
            height: 24,
            child: CircularProgressIndicator(strokeWidth: 2.5),
          ),
          title: Text('Проверка подключения…'),
        );
      case _State.ok:
        return Card(
          color: AppTheme.success.withOpacity(0.12),
          child: ListTile(
            leading: const Icon(
              Icons.check_circle_rounded,
              color: AppTheme.success,
            ),
            title: Text('Сервер доступен · v${status.version}'),
            subtitle: Text(
              'device: ${status.device} · models: ${status.modelsLoaded == true ? 'loaded' : 'loading'}',
            ),
          ),
        );
      case _State.error:
        return Card(
          color: AppTheme.danger.withOpacity(0.12),
          child: ListTile(
            leading: const Icon(Icons.error_rounded, color: AppTheme.danger),
            title: const Text('Ошибка подключения'),
            subtitle: Text(status.message ?? ''),
          ),
        );
    }
  }
}
