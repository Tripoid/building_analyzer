import 'package:flutter/material.dart';

/// Global app configuration — mock/live mode toggle and server URL
class AppConfig extends ChangeNotifier {
  // Singleton
  static final AppConfig _instance = AppConfig._internal();
  factory AppConfig() => _instance;
  AppConfig._internal();

  bool _useMock = true;
  String _serverUrl = 'https://your-ngrok-url.ngrok-free.app';

  bool get useMock => _useMock;
  String get serverUrl => _serverUrl;

  void setUseMock(bool value) {
    _useMock = value;
    notifyListeners();
  }

  void setServerUrl(String url) {
    _serverUrl = url.trim();
    if (_serverUrl.endsWith('/')) {
      _serverUrl = _serverUrl.substring(0, _serverUrl.length - 1);
    }
    notifyListeners();
  }
}
