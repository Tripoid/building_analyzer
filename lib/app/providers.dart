import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../data/api/dio_client.dart';
import '../data/repositories/analysis_repository.dart';
import '../data/repositories/calibration_repository.dart';
import '../domain/models/upload_progress.dart';
import 'app_config.dart';

/// A Dio client that is rebuilt automatically whenever the base URL or
/// verbose flag change. Widgets consuming the repositories re-bind to a fresh
/// client as a consequence — no manual reload needed.
final dioProvider = Provider<Dio>((ref) {
  final cfg = ref.watch(appConfigProvider);
  final dio = buildDioClient(
    baseUrl: cfg.serverUrl,
    verbose: cfg.verboseLogging,
  );
  ref.onDispose(dio.close);
  return dio;
});

final analysisRepositoryProvider = Provider<AnalysisRepository>((ref) {
  final cfg = ref.watch(appConfigProvider);
  if (cfg.useMock || cfg.serverUrl.isEmpty) {
    return MockAnalysisRepository();
  }
  final dio = ref.watch(dioProvider);
  return DioAnalysisRepository(dio);
});

final calibrationRepositoryProvider = Provider<CalibrationRepository>((ref) {
  final cfg = ref.watch(appConfigProvider);
  if (cfg.useMock || cfg.serverUrl.isEmpty) {
    return MockCalibrationRepository();
  }
  final dio = ref.watch(dioProvider);
  return DioCalibrationRepository(dio);
});

/// Exposes real upload progress to the loading screen.
final uploadProgressProvider = StreamProvider<UploadProgress>((ref) {
  final repo = ref.watch(analysisRepositoryProvider);
  return repo.progress;
});
