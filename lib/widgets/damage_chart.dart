import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

class DamageChart extends StatefulWidget {
  final List<DamageChartData> data;
  final double size;

  const DamageChart({
    super.key,
    required this.data,
    this.size = 200,
  });

  @override
  State<DamageChart> createState() => _DamageChartState();
}

class _DamageChartState extends State<DamageChart>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 1200),
      vsync: this,
    );
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    );
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return CustomPaint(
          size: Size(widget.size, widget.size),
          painter: _DamageChartPainter(
            data: widget.data,
            progress: _animation.value,
          ),
        );
      },
    );
  }
}

class _DamageChartPainter extends CustomPainter {
  final List<DamageChartData> data;
  final double progress;

  _DamageChartPainter({
    required this.data,
    required this.progress,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = math.min(size.width, size.height) / 2 - 8;
    final innerRadius = radius * 0.55;

    double startAngle = -math.pi / 2;
    final total = data.fold<double>(0, (sum, d) => sum + d.value);

    for (var item in data) {
      final sweepAngle = (item.value / total) * 2 * math.pi * progress;
      final paint = Paint()
        ..color = item.color
        ..style = PaintingStyle.stroke
        ..strokeWidth = radius - innerRadius
        ..strokeCap = StrokeCap.butt;

      canvas.drawArc(
        Rect.fromCircle(center: center, radius: (radius + innerRadius) / 2),
        startAngle,
        sweepAngle,
        false,
        paint,
      );

      startAngle += sweepAngle;
    }

    // Inner circle background
    final innerPaint = Paint()
      ..color = AppTheme.primaryDark
      ..style = PaintingStyle.fill;
    canvas.drawCircle(center, innerRadius - 2, innerPaint);

    // Center text
    final textPainter = TextPainter(
      text: TextSpan(
        children: [
          TextSpan(
            text: '${(data.first.value * progress).toStringAsFixed(0)}%\n',
            style: const TextStyle(
              color: AppTheme.textPrimary,
              fontSize: 28,
              fontWeight: FontWeight.w700,
              height: 1.2,
            ),
          ),
          TextSpan(
            text: data.first.label,
            style: const TextStyle(
              color: AppTheme.textSecondary,
              fontSize: 11,
              fontWeight: FontWeight.w400,
            ),
          ),
        ],
      ),
      textDirection: TextDirection.ltr,
      textAlign: TextAlign.center,
    );
    textPainter.layout();
    textPainter.paint(
      canvas,
      center - Offset(textPainter.width / 2, textPainter.height / 2),
    );
  }

  @override
  bool shouldRepaint(covariant _DamageChartPainter oldDelegate) {
    return oldDelegate.progress != progress;
  }
}

class DamageChartData {
  final String label;
  final double value;
  final Color color;

  const DamageChartData({
    required this.label,
    required this.value,
    required this.color,
  });
}
