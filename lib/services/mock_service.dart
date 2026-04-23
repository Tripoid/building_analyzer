import '../models/analysis_result.dart';

/// Mock service that returns sample data with a simulated delay
class MockService {
  Future<AnalysisResult> analyzeImage() async {
    // Simulate network + ML processing delay
    await Future.delayed(const Duration(seconds: 3));
    return AnalysisResult.mock();
  }
}
