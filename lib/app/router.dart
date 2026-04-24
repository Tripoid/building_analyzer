import 'dart:io';

import 'package:go_router/go_router.dart';

import '../features/calibration/physical_scale_input.dart';
import '../features/loading/real_analysis_loading.dart';
import '../models/analysis_result.dart';
import '../screens/home_screen.dart';
import '../screens/photo_preview_screen.dart';
import '../screens/results_screen.dart';
import '../screens/settings_screen.dart';

GoRouter buildRouter() => GoRouter(
  initialLocation: '/',
  routes: [
    GoRoute(path: '/', builder: (_, __) => const HomeScreen()),
    GoRoute(
      path: '/preview',
      builder: (ctx, state) {
        final image = state.extra as File?;
        return PhotoPreviewScreen(image: image);
      },
    ),
    GoRoute(
      path: '/calibrate',
      builder: (ctx, state) {
        final image = state.extra as File;
        return PhysicalScaleInputScreen(image: image);
      },
    ),
    GoRoute(
      path: '/loading',
      builder: (ctx, state) {
        final args = state.extra as AnalyzeArgs?;
        return RealAnalysisLoadingScreen(args: args);
      },
    ),
    GoRoute(
      path: '/results',
      builder: (ctx, state) {
        final result = state.extra as AnalysisResult?;
        return ResultsScreen(result: result);
      },
    ),
    GoRoute(path: '/settings', builder: (_, __) => const SettingsScreen()),
  ],
);

/// Container for everything the loading screen needs to run one analysis.
class AnalyzeArgs {
  const AnalyzeArgs({
    required this.image,
    this.calibrationId,
    this.fallbackTotalAreaM2,
  });
  final File image;
  final String? calibrationId;
  final double? fallbackTotalAreaM2;
}
