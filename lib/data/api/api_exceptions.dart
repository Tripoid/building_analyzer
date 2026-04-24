import 'package:dio/dio.dart';

/// Typed exceptions so the UI can react per-error instead of reading strings.
sealed class ApiException implements Exception {
  const ApiException(this.message, {this.cause, this.statusCode});
  final String message;
  final Object? cause;
  final int? statusCode;

  @override
  String toString() => '$runtimeType($message)';
}

class NetworkException extends ApiException {
  const NetworkException(super.message, {super.cause}) : super(statusCode: null);
}

class RequestTimeoutException extends ApiException {
  const RequestTimeoutException(super.message, {super.cause});
}

class CancelledException extends ApiException {
  const CancelledException() : super('Операция отменена');
}

class ValidationException extends ApiException {
  const ValidationException(super.message, {super.cause, this.fields})
      : super(statusCode: 422);
  final Map<String, Object?>? fields;
}

class ClientException extends ApiException {
  const ClientException(super.message, {required int super.statusCode, super.cause});
}

class ServerException extends ApiException {
  const ServerException(super.message, {required int super.statusCode, super.cause});
}

/// Server sent HTML instead of JSON — almost certainly the ngrok-free interstitial.
/// The Dio header `ngrok-skip-browser-warning: true` should prevent this, but if a
/// reverse-proxy strips headers we surface a dedicated error so the UI can explain.
class NgrokInterstitialException extends ApiException {
  const NgrokInterstitialException()
      : super(
          'Сервер вернул HTML-заглушку туннеля. Откройте публичный URL '
          'в браузере один раз и повторите попытку.',
          statusCode: 200,
        );
}

/// Bridge a DioException into a typed [ApiException].
ApiException mapDioException(DioException e) {
  switch (e.type) {
    case DioExceptionType.connectionTimeout:
    case DioExceptionType.sendTimeout:
    case DioExceptionType.receiveTimeout:
      return RequestTimeoutException(
        'Сервер отвечает слишком долго. Попробуйте ещё раз.',
        cause: e,
      );
    case DioExceptionType.cancel:
      return const CancelledException();
    case DioExceptionType.connectionError:
    case DioExceptionType.unknown:
      return NetworkException(
        'Нет связи с сервером. Проверьте адрес в Настройках.',
        cause: e,
      );
    case DioExceptionType.badCertificate:
      return NetworkException('Проблема с SSL-сертификатом сервера.', cause: e);
    case DioExceptionType.badResponse:
      final status = e.response?.statusCode ?? 0;
      final detail = _detailFrom(e.response?.data) ?? e.message ?? 'Ошибка $status';
      if (status == 422) {
        return ValidationException(detail, cause: e, fields: _fieldsFrom(e.response?.data));
      }
      if (_looksLikeNgrokInterstitial(e.response?.data)) {
        return const NgrokInterstitialException();
      }
      if (status >= 400 && status < 500) {
        return ClientException(detail, statusCode: status, cause: e);
      }
      return ServerException(detail, statusCode: status, cause: e);
  }
}

bool _looksLikeNgrokInterstitial(Object? body) {
  if (body is String) {
    final lower = body.toLowerCase();
    return lower.contains('<!doctype html') &&
        (lower.contains('ngrok') || lower.contains('visit this site'));
  }
  return false;
}

String? _detailFrom(Object? body) {
  if (body is Map) {
    final d = body['detail'];
    if (d is String) return d;
    if (d is List && d.isNotEmpty) return d.first.toString();
  }
  if (body is String && body.isNotEmpty) {
    return body.length > 280 ? '${body.substring(0, 280)}…' : body;
  }
  return null;
}

Map<String, Object?>? _fieldsFrom(Object? body) {
  if (body is Map && body['detail'] is List) {
    final m = <String, Object?>{};
    for (final item in body['detail'] as List) {
      if (item is Map && item['loc'] is List) {
        m[(item['loc'] as List).join('.')] = item['msg'];
      }
    }
    return m.isEmpty ? null : m;
  }
  return null;
}
