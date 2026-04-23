import 'dart:async';
import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

class AnalysisLoadingScreen extends StatefulWidget {
  const AnalysisLoadingScreen({super.key});

  @override
  State<AnalysisLoadingScreen> createState() => _AnalysisLoadingScreenState();
}

class _AnalysisLoadingScreenState extends State<AnalysisLoadingScreen>
    with TickerProviderStateMixin {
  late AnimationController _pulseController;
  late AnimationController _progressController;
  int _currentStep = 0;

  final List<Map<String, dynamic>> _steps = [
    {'label': 'Загрузка изображения...', 'icon': Icons.cloud_upload_rounded},
    {'label': 'Обнаружение фасада...', 'icon': Icons.search_rounded},
    {'label': 'Анализ повреждений...', 'icon': Icons.analytics_rounded},
    {'label': 'Сегментация материалов...', 'icon': Icons.layers_rounded},
    {'label': 'Расчёт стоимости...', 'icon': Icons.calculate_rounded},
    {'label': 'Формирование отчёта...', 'icon': Icons.description_rounded},
  ];

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat(reverse: true);

    _progressController = AnimationController(
      duration: const Duration(milliseconds: 500),
      vsync: this,
    );

    _startAnalysis();
  }

  void _startAnalysis() {
    Timer.periodic(const Duration(milliseconds: 600), (timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      if (_currentStep < _steps.length - 1) {
        setState(() {
          _currentStep++;
        });
      } else {
        timer.cancel();
        Future.delayed(const Duration(milliseconds: 500), () {
          if (mounted) {
            Navigator.pushReplacementNamed(context, '/results');
          }
        });
      }
    });
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _progressController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.primaryDark,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Animated pulse ring
            _buildPulseAnimation(),
            const SizedBox(height: 48),
            // Progress steps
            _buildProgressSteps(),
            const SizedBox(height: 40),
            // Progress bar
            _buildProgressBar(),
          ],
        ),
      ),
    );
  }

  Widget _buildPulseAnimation() {
    return AnimatedBuilder(
      animation: _pulseController,
      builder: (context, child) {
        return Stack(
          alignment: Alignment.center,
          children: [
            // Outer ring
            Container(
              width: 140 + _pulseController.value * 20,
              height: 140 + _pulseController.value * 20,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(
                  color: AppTheme.accent.withOpacity(
                    0.1 + _pulseController.value * 0.1,
                  ),
                  width: 2,
                ),
              ),
            ),
            // Middle ring
            Container(
              width: 110 + _pulseController.value * 10,
              height: 110 + _pulseController.value * 10,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(
                  color: AppTheme.accent.withOpacity(
                    0.2 + _pulseController.value * 0.1,
                  ),
                  width: 2,
                ),
              ),
            ),
            // Inner filled circle
            Container(
              width: 90,
              height: 90,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  colors: [
                    AppTheme.accent.withOpacity(0.3),
                    AppTheme.accent.withOpacity(0.1),
                  ],
                ),
                boxShadow: [
                  BoxShadow(
                    color: AppTheme.accent.withOpacity(
                      0.2 + _pulseController.value * 0.2,
                    ),
                    blurRadius: 24,
                    spreadRadius: 8,
                  ),
                ],
              ),
              child: Icon(
                _steps[_currentStep]['icon'] as IconData,
                color: AppTheme.accent,
                size: 36,
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _buildProgressSteps() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 48),
      child: Column(
        children: [
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 300),
            transitionBuilder: (child, animation) {
              return SlideTransition(
                position: Tween<Offset>(
                  begin: const Offset(0, 0.3),
                  end: Offset.zero,
                ).animate(animation),
                child: FadeTransition(
                  opacity: animation,
                  child: child,
                ),
              );
            },
            child: Text(
              _steps[_currentStep]['label'] as String,
              key: ValueKey(_currentStep),
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w600,
                color: AppTheme.textPrimary,
              ),
            ),
          ),
          const SizedBox(height: 12),
          Text(
            'Шаг ${_currentStep + 1} из ${_steps.length}',
            style: const TextStyle(
              fontSize: 13,
              color: AppTheme.textSecondary,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProgressBar() {
    final progress = (_currentStep + 1) / _steps.length;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 64),
      child: Column(
        children: [
          Container(
            height: 6,
            decoration: BoxDecoration(
              color: AppTheme.surfaceCard,
              borderRadius: BorderRadius.circular(3),
            ),
            child: LayoutBuilder(
              builder: (context, constraints) {
                return Stack(
                  children: [
                    AnimatedContainer(
                      duration: const Duration(milliseconds: 400),
                      curve: Curves.easeOutCubic,
                      height: 6,
                      width: constraints.maxWidth * progress,
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [AppTheme.accent, Color(0xFFFF8F00)],
                        ),
                        borderRadius: BorderRadius.circular(3),
                        boxShadow: [
                          BoxShadow(
                            color: AppTheme.accent.withOpacity(0.4),
                            blurRadius: 6,
                          ),
                        ],
                      ),
                    ),
                  ],
                );
              },
            ),
          ),
          const SizedBox(height: 12),
          Text(
            '${(progress * 100).toInt()}%',
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: AppTheme.accent,
            ),
          ),
        ],
      ),
    );
  }
}
