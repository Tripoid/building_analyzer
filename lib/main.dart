import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'theme/app_theme.dart';
import 'screens/home_screen.dart';
import 'screens/photo_preview_screen.dart';
import 'screens/analysis_loading_screen.dart';
import 'screens/results_screen.dart';
import 'screens/settings_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
      systemNavigationBarColor: AppTheme.primaryDark,
      systemNavigationBarIconBrightness: Brightness.light,
    ),
  );
  runApp(const FacadeAnalyzerApp());
}

class FacadeAnalyzerApp extends StatelessWidget {
  const FacadeAnalyzerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Анализ Фасадов',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkTheme,
      initialRoute: '/',
      routes: {
        '/': (context) => const HomeScreen(),
        '/preview': (context) => const PhotoPreviewScreen(),
        '/loading': (context) => const AnalysisLoadingScreen(),
        '/results': (context) => const ResultsScreen(),
        '/settings': (context) => const SettingsScreen(),
      },
    );
  }
}
