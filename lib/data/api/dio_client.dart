import 'package:dio/dio.dart';
import 'package:dio_smart_retry/dio_smart_retry.dart';
import 'package:pretty_dio_logger/pretty_dio_logger.dart';

import 'interceptors/auth_interceptor.dart';

/// Builds a preconfigured [Dio] for talking to the AlegroCode backend.
///
/// Timeouts are tuned for a GPU-bound ML endpoint:
///   - connect: 15s  (fast — network should be reachable quickly)
///   - send:    120s (large photo upload on slow mobile networks)
///   - receive: 180s (analysis itself can take a minute or two)
///
/// We retry only safe methods (GET/HEAD) automatically. `/analyze` and
/// `/restore` are NOT retried because they are not idempotent and can cost
/// a user real GPU time; the Flutter UI offers an explicit "повторить" button.
Dio buildDioClient({
  required String baseUrl,
  bool verbose = false,
}) {
  final dio = Dio(
    BaseOptions(
      baseUrl: _normalizeBaseUrl(baseUrl),
      connectTimeout: const Duration(seconds: 15),
      sendTimeout: const Duration(seconds: 120),
      receiveTimeout: const Duration(seconds: 180),
      responseType: ResponseType.json,
      validateStatus: (code) => code != null && code < 500,
    ),
  );

  dio.interceptors.add(NgrokHeaderInterceptor());
  dio.interceptors.add(
    RetryInterceptor(
      dio: dio,
      retries: 3,
      retryDelays: const [
        Duration(milliseconds: 300),
        Duration(seconds: 1),
        Duration(seconds: 3),
      ],
      // Never retry non-idempotent calls unless the caller opts in per-request.
      retryEvaluator: (error, attempt) {
        const retryableStatuses = {502, 503, 504};
        final status = error.response?.statusCode;
        if (status != null && !retryableStatuses.contains(status)) {
          return false;
        }

        final method = error.requestOptions.method.toUpperCase();
        if (method != 'GET' && method != 'HEAD') {
          return error.requestOptions.extra['retryable'] == true;
        }
        return true;
      },
    ),
  );
  if (verbose) {
    dio.interceptors.add(PrettyDioLogger(
      requestHeader: true,
      requestBody: false,
      responseHeader: false,
      responseBody: true,
      maxWidth: 120,
      compact: true,
    ));
  }

  return dio;
}

String _normalizeBaseUrl(String url) {
  var u = url.trim();
  if (u.isEmpty) return u;
  if (!u.startsWith('http://') && !u.startsWith('https://')) {
    u = 'https://$u';
  }
  if (u.endsWith('/')) u = u.substring(0, u.length - 1);
  return u;
}
