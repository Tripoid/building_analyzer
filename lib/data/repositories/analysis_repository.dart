import 'dart:async';
import 'dart:io';

import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

import '../../domain/models/upload_progress.dart';
import '../../models/analysis_result.dart';
import '../api/api_exceptions.dart';

/// Contract: one repository = one screen flow. Both the real Dio-backed
/// implementation and the mock are interchangeable via Riverpod overrides.
abstract class AnalysisRepository {
  Stream<UploadProgress> get progress;

  Future<AnalysisResult> analyze({
    required File image,
    String? calibrationId,
    double? fallbackTotalAreaM2,
    CancelToken? cancelToken,
  });

  Future<AnalysisResult?> fetchStored(String analysisId);

  Future<AnalysisResult> requestRestoration({
    required String analysisId,
    String quality = 'fast',
    String? prompt,
  });
}

class DioAnalysisRepository implements AnalysisRepository {
  DioAnalysisRepository(this._dio);
  final Dio _dio;

  final StreamController<UploadProgress> _progressCtrl =
      StreamController<UploadProgress>.broadcast();

  @override
  Stream<UploadProgress> get progress => _progressCtrl.stream;

  @override
  Future<AnalysisResult> analyze({
    required File image,
    String? calibrationId,
    double? fallbackTotalAreaM2,
    CancelToken? cancelToken,
  }) async {
    _progressCtrl.add(UploadProgress.initial.copyWith(
      total: await image.length(),
      stage: AnalysisStage.preparing,
    ));

    final form = FormData.fromMap({
      if (calibrationId != null) 'calibration_id': calibrationId,
      if (fallbackTotalAreaM2 != null) 'total_area_m2': fallbackTotalAreaM2,
      'file': await MultipartFile.fromFile(
        image.path,
        filename: image.uri.pathSegments.last,
      ),
    });

    try {
      final totalBytes = await image.length();
      _progressCtrl.add(UploadProgress(
        sent: 0,
        total: totalBytes,
        stage: AnalysisStage.uploading,
      ));

      final resp = await _dio.post<Map<String, dynamic>>(
        '/api/analyze',
        data: form,
        cancelToken: cancelToken,
        onSendProgress: (sent, total) {
          final t = total <= 0 ? totalBytes : total;
          final stage = sent >= t && t > 0
              ? AnalysisStage.analyzing
              : AnalysisStage.uploading;
          _progressCtrl.add(UploadProgress(sent: sent, total: t, stage: stage));
        },
        options: Options(
          headers: {Headers.contentTypeHeader: 'multipart/form-data'},
        ),
      );

      _progressCtrl.add(UploadProgress(
        sent: totalBytes,
        total: totalBytes,
        stage: AnalysisStage.estimating,
      ));

      final body = resp.data;
      if (body == null) {
        throw const ServerException('Пустой ответ сервера', statusCode: 500);
      }
      if (resp.statusCode == 400 || resp.statusCode == 413) {
        throw ClientException(
          body['detail']?.toString() ?? 'Некорректный запрос',
          statusCode: resp.statusCode ?? 400,
        );
      }
      if (resp.statusCode != null && resp.statusCode! >= 400) {
        throw ServerException(
          body['detail']?.toString() ?? 'Ошибка сервера',
          statusCode: resp.statusCode!,
        );
      }

      _progressCtrl.add(UploadProgress(
        sent: totalBytes,
        total: totalBytes,
        stage: AnalysisStage.done,
      ));
      return AnalysisResult.fromJson(body);
    } on DioException catch (e) {
      throw mapDioException(e);
    } catch (e) {
      if (e is ApiException) rethrow;
      if (kDebugMode) debugPrint('analyze failure: $e');
      throw ServerException(e.toString(), statusCode: 500, cause: e);
    }
  }

  @override
  Future<AnalysisResult?> fetchStored(String analysisId) async {
    try {
      final resp = await _dio.get<Map<String, dynamic>>(
        '/api/results/$analysisId',
      );
      return resp.data == null ? null : AnalysisResult.fromJson(resp.data!);
    } on DioException catch (e) {
      if (e.response?.statusCode == 404) return null;
      throw mapDioException(e);
    }
  }

  @override
  Future<AnalysisResult> requestRestoration({
    required String analysisId,
    String quality = 'fast',
    String? prompt,
  }) async {
    try {
      final resp = await _dio.post<Map<String, dynamic>>(
        '/api/restore/$analysisId',
        data: {'quality': quality, if (prompt != null) 'prompt': prompt},
      );
      final updated = await fetchStored(analysisId);
      if (updated == null) {
        throw const ServerException('Не удалось получить восстановленное фото', statusCode: 500);
      }
      return updated.copyWithRestored(resp.data?['restored_url']?.toString());
    } on DioException catch (e) {
      throw mapDioException(e);
    }
  }

  void dispose() => _progressCtrl.close();
}

extension on AnalysisResult {
  AnalysisResult copyWithRestored(String? url) => AnalysisResult(
        id: id,
        overallScore: overallScore,
        overallCondition: overallCondition,
        damages: damages,
        materials: materials,
        costs: costs,
        processedImages: processedImages,
        totalArea: totalArea,
        damagedArea: damagedArea,
        repairEstimate: repairEstimate,
        masks: masks,
        priceSnapshotDate: priceSnapshotDate,
        priceSource: priceSource,
        restoredUrl: url ?? restoredUrl,
        calibrationWarnings: calibrationWarnings,
      );
}

/// Mock — served when AppConfig.useMock is true. Produces the same contract
/// as the real one so the UI never branches on "isMock".
class MockAnalysisRepository implements AnalysisRepository {
  final _ctrl = StreamController<UploadProgress>.broadcast();

  @override
  Stream<UploadProgress> get progress => _ctrl.stream;

  @override
  Future<AnalysisResult> analyze({
    required File image,
    String? calibrationId,
    double? fallbackTotalAreaM2,
    CancelToken? cancelToken,
  }) async {
    final total = 2_000_000;
    _ctrl.add(UploadProgress(sent: 0, total: total, stage: AnalysisStage.preparing));
    for (final stage in [
      AnalysisStage.uploading,
      AnalysisStage.analyzing,
      AnalysisStage.estimating,
    ]) {
      for (var i = 1; i <= 10; i++) {
        if (cancelToken?.isCancelled ?? false) throw const CancelledException();
        await Future<void>.delayed(const Duration(milliseconds: 120));
        _ctrl.add(UploadProgress(
          sent: (total * i / 10).round(),
          total: total,
          stage: stage,
        ));
      }
    }
    _ctrl.add(UploadProgress(sent: total, total: total, stage: AnalysisStage.done));
    return AnalysisResult.mock();
  }

  @override
  Future<AnalysisResult?> fetchStored(String analysisId) async =>
      AnalysisResult.mock();

  @override
  Future<AnalysisResult> requestRestoration({
    required String analysisId,
    String quality = 'fast',
    String? prompt,
  }) async {
    await Future<void>.delayed(const Duration(seconds: 2));
    return AnalysisResult.mock();
  }
}
