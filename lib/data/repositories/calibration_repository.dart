import 'package:dio/dio.dart';

import '../../domain/models/calibration_input.dart';
import '../api/api_exceptions.dart';

abstract class CalibrationRepository {
  Future<CalibrationResult> calibrate(CalibrationInput input);
}

class DioCalibrationRepository implements CalibrationRepository {
  DioCalibrationRepository(this._dio);
  final Dio _dio;

  @override
  Future<CalibrationResult> calibrate(CalibrationInput input) async {
    try {
      final resp = await _dio.post<Map<String, dynamic>>(
        '/api/calibrate',
        data: input.toApiPayload(),
      );
      if (resp.data == null) {
        throw const ServerException('Пустой ответ /api/calibrate', statusCode: 500);
      }
      return CalibrationResult.fromJson(resp.data!);
    } on DioException catch (e) {
      throw mapDioException(e);
    }
  }
}

class MockCalibrationRepository implements CalibrationRepository {
  @override
  Future<CalibrationResult> calibrate(CalibrationInput input) async {
    await Future<void>.delayed(const Duration(milliseconds: 250));
    const pxPerM = 200.0;
    return CalibrationResult(
      calibrationId: 'mock-${DateTime.now().millisecondsSinceEpoch}',
      pxPerM: pxPerM,
      m2PerPx: 1 / (pxPerM * pxPerM),
      warnings: const [],
    );
  }
}
