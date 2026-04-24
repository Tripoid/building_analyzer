import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Persistent app settings — kept simple (in-memory now, but trivial to back
/// with SharedPreferences later). Riverpod exposes it as a NotifierProvider so
/// the settings screen rebuilds automatically when values change.
class AppConfig {
  const AppConfig({
    required this.useMock,
    required this.serverUrl,
    required this.verboseLogging,
  });

  final bool useMock;
  final String serverUrl;
  final bool verboseLogging;

  static AppConfig get initial => AppConfig(
        useMock: false,
        serverUrl: '',
        verboseLogging: kDebugMode,
      );

  AppConfig copyWith({
    bool? useMock,
    String? serverUrl,
    bool? verboseLogging,
  }) =>
      AppConfig(
        useMock: useMock ?? this.useMock,
        serverUrl: serverUrl ?? this.serverUrl,
        verboseLogging: verboseLogging ?? this.verboseLogging,
      );
}

class AppConfigNotifier extends Notifier<AppConfig> {
  @override
  AppConfig build() => AppConfig.initial;

  void setUseMock(bool v) => state = state.copyWith(useMock: v);
  void setServerUrl(String url) => state = state.copyWith(serverUrl: url.trim());
  void setVerbose(bool v) => state = state.copyWith(verboseLogging: v);
}

final appConfigProvider =
    NotifierProvider<AppConfigNotifier, AppConfig>(AppConfigNotifier.new);
