import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../theme/app_theme.dart';
import '../services/app_config.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _urlController = TextEditingController();
  final _config = AppConfig();
  bool _isCheckingConnection = false;
  String? _connectionStatus;

  @override
  void initState() {
    super.initState();
    _urlController.text = _config.serverUrl;
  }

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _checkConnection() async {
    setState(() {
      _isCheckingConnection = true;
      _connectionStatus = null;
    });

    try {
      final url = Uri.parse('${_urlController.text}/api/health');
      final response = await http.get(url).timeout(
        const Duration(seconds: 10),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body) as Map<String, dynamic>;
        if (data['status'] == 'ok') {
          setState(() {
            _connectionStatus = 'Подключено (${data['device'] ?? 'unknown'})';
          });
        } else {
          setState(() {
            _connectionStatus = 'Сервер ответил, но статус не OK';
          });
        }
      } else {
        setState(() {
          _connectionStatus = 'Ошибка HTTP: ${response.statusCode}';
        });
      }
    } catch (e) {
      setState(() {
        _connectionStatus = 'Ошибка: ${e.toString().split('\n').first}';
      });
    } finally {
      setState(() {
        _isCheckingConnection = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.primaryDark,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        leading: IconButton(
          onPressed: () => Navigator.pop(context),
          icon: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: AppTheme.surfaceCard.withValues(alpha: 0.6),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Icon(Icons.arrow_back_rounded, size: 20),
          ),
        ),
        title: const Text('Настройки'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(20),
        children: [
          // Mode toggle
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: AppTheme.surfaceCard,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: AppTheme.surfaceLight.withValues(alpha: 0.3),
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Row(
                  children: [
                    Icon(Icons.science_rounded, color: AppTheme.accent, size: 22),
                    SizedBox(width: 10),
                    Text(
                      'Режим работы',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                _buildModeOption(
                  title: 'Демо-режим (моки)',
                  subtitle: 'Тестовые данные без обращения к серверу',
                  icon: Icons.play_circle_outline_rounded,
                  isSelected: _config.useMock,
                  onTap: () => setState(() => _config.setUseMock(true)),
                ),
                const SizedBox(height: 10),
                _buildModeOption(
                  title: 'Рабочий режим (сервер)',
                  subtitle: 'Реальный ML-анализ на удалённом сервере',
                  icon: Icons.cloud_rounded,
                  isSelected: !_config.useMock,
                  onTap: () => setState(() => _config.setUseMock(false)),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),

          // Server URL
          AnimatedOpacity(
            duration: const Duration(milliseconds: 300),
            opacity: _config.useMock ? 0.4 : 1.0,
            child: Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: AppTheme.surfaceCard,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: AppTheme.surfaceLight.withValues(alpha: 0.3),
                ),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Row(
                    children: [
                      Icon(Icons.link_rounded, color: AppTheme.info, size: 22),
                      SizedBox(width: 10),
                      Text(
                        'URL сервера',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w700,
                          color: AppTheme.textPrimary,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  TextField(
                    controller: _urlController,
                    enabled: !_config.useMock,
                    style: const TextStyle(color: AppTheme.textPrimary, fontSize: 14),
                    decoration: InputDecoration(
                      hintText: 'https://xxxx.ngrok-free.app',
                      hintStyle: TextStyle(color: AppTheme.textSecondary.withValues(alpha: 0.5)),
                      filled: true,
                      fillColor: AppTheme.primaryLight,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: BorderSide.none,
                      ),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                    ),
                    onChanged: (value) {
                      _config.setServerUrl(value);
                    },
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: _config.useMock || _isCheckingConnection
                          ? null
                          : _checkConnection,
                      icon: _isCheckingConnection
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                color: Colors.white,
                              ),
                            )
                          : const Icon(Icons.wifi_find_rounded, size: 18),
                      label: Text(
                        _isCheckingConnection ? 'Проверка...' : 'Проверить соединение',
                      ),
                    ),
                  ),
                  if (_connectionStatus != null) ...[
                    const SizedBox(height: 10),
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: _connectionStatus!.startsWith('Подключено')
                            ? AppTheme.success.withValues(alpha: 0.1)
                            : AppTheme.danger.withValues(alpha: 0.1),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(
                          color: _connectionStatus!.startsWith('Подключено')
                              ? AppTheme.success.withValues(alpha: 0.3)
                              : AppTheme.danger.withValues(alpha: 0.3),
                        ),
                      ),
                      child: Row(
                        children: [
                          Icon(
                            _connectionStatus!.startsWith('Подключено')
                                ? Icons.check_circle_rounded
                                : Icons.error_rounded,
                            size: 18,
                            color: _connectionStatus!.startsWith('Подключено')
                                ? AppTheme.success
                                : AppTheme.danger,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              _connectionStatus!,
                              style: TextStyle(
                                fontSize: 12,
                                color: _connectionStatus!.startsWith('Подключено')
                                    ? AppTheme.success
                                    : AppTheme.danger,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Info
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AppTheme.info.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppTheme.info.withValues(alpha: 0.2)),
            ),
            child: const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.help_outline_rounded, color: AppTheme.info, size: 18),
                    SizedBox(width: 8),
                    Text('Как запустить сервер?', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: AppTheme.info)),
                  ],
                ),
                SizedBox(height: 10),
                Text(
                  '1. Откройте colab_server.ipynb в Google Colab\n'
                  '2. Включите GPU: Runtime → Change runtime type → T4\n'
                  '3. Запустите все ячейки по порядку\n'
                  '4. Скопируйте ngrok URL в поле выше',
                  style: TextStyle(fontSize: 12, color: AppTheme.textSecondary, height: 1.6),
                ),
              ],
            ),
          ),

          const SizedBox(height: 24),
          // Version info
          Center(
            child: Text(
              'Facade Analyzer v1.0.0',
              style: TextStyle(
                fontSize: 12,
                color: AppTheme.textSecondary.withValues(alpha: 0.5),
              ),
            ),
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }

  Widget _buildModeOption({
    required String title,
    required String subtitle,
    required IconData icon,
    required bool isSelected,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: isSelected
              ? AppTheme.accent.withValues(alpha: 0.1)
              : AppTheme.primaryLight.withValues(alpha: 0.5),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isSelected ? AppTheme.accent : Colors.transparent,
            width: isSelected ? 1.5 : 1,
          ),
        ),
        child: Row(
          children: [
            Icon(icon, color: isSelected ? AppTheme.accent : AppTheme.textSecondary, size: 24),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: isSelected ? AppTheme.textPrimary : AppTheme.textSecondary,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    style: const TextStyle(fontSize: 11, color: AppTheme.textSecondary),
                  ),
                ],
              ),
            ),
            Icon(
              isSelected ? Icons.radio_button_checked_rounded : Icons.radio_button_off_rounded,
              color: isSelected ? AppTheme.accent : AppTheme.textSecondary,
              size: 22,
            ),
          ],
        ),
      ),
    );
  }
}
