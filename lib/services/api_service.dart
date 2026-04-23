import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import '../models/analysis_result.dart';

/// HTTP client for FastAPI backend
class ApiService {
  final String baseUrl;

  ApiService({required this.baseUrl});

  /// Health check
  Future<Map<String, dynamic>> healthCheck() async {
    final client = HttpClient();
    try {
      final request = await client.getUrl(Uri.parse('$baseUrl/api/health'));
      final response = await request.close();
      final body = await response.transform(utf8.decoder).join();
      return json.decode(body) as Map<String, dynamic>;
    } finally {
      client.close();
    }
  }

  /// Analyze facade image
  Future<AnalysisResult> analyzeImage(Uint8List imageBytes, {
    String filename = 'facade.jpg',
    double totalAreaM2 = 450.0,
  }) async {
    final client = HttpClient();
    try {
      final uri = Uri.parse('$baseUrl/api/analyze?total_area_m2=$totalAreaM2');
      final request = await client.postUrl(uri);

      final boundary = '----FormBoundary${DateTime.now().millisecondsSinceEpoch}';
      request.headers.set('Content-Type', 'multipart/form-data; boundary=$boundary');

      final body = StringBuffer();
      body.write('--$boundary\r\n');
      body.write('Content-Disposition: form-data; name="file"; filename="$filename"\r\n');
      body.write('Content-Type: image/jpeg\r\n');
      body.write('\r\n');

      final header = utf8.encode(body.toString());
      final footer = utf8.encode('\r\n--$boundary--\r\n');

      request.contentLength = header.length + imageBytes.length + footer.length;
      request.add(header);
      request.add(imageBytes);
      request.add(footer);

      final response = await request.close();
      final responseBody = await response.transform(utf8.decoder).join();

      if (response.statusCode != 200) {
        throw Exception('Server error: ${response.statusCode} — $responseBody');
      }

      final jsonData = json.decode(responseBody) as Map<String, dynamic>;
      return AnalysisResult.fromJson(jsonData);
    } finally {
      client.close();
    }
  }

  /// Get processed image URL
  String getImageUrl(String analysisId, String imageType) {
    return '$baseUrl/api/results/$analysisId/images/$imageType';
  }
}
