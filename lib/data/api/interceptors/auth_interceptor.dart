import 'package:dio/dio.dart';

/// Adds `ngrok-skip-browser-warning` to every request so the ngrok-free
/// tunnel never serves the browser interstitial (HTML in place of JSON / images).
class NgrokHeaderInterceptor extends Interceptor {
  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) {
    options.headers['ngrok-skip-browser-warning'] = 'true';
    options.headers.putIfAbsent('User-Agent', () => 'AlegroCodeApp/2.0');
    handler.next(options);
  }
}
