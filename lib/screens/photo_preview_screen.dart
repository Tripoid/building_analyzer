import 'package:flutter/material.dart';
import '../theme/app_theme.dart';

class PhotoPreviewScreen extends StatefulWidget {
  const PhotoPreviewScreen({super.key});

  @override
  State<PhotoPreviewScreen> createState() => _PhotoPreviewScreenState();
}

class _PhotoPreviewScreenState extends State<PhotoPreviewScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );
    _scaleAnimation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutBack,
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
    return Scaffold(
      backgroundColor: AppTheme.primaryDark,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        leading: IconButton(
          onPressed: () => Navigator.pop(context),
          icon: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: AppTheme.surfaceCard.withOpacity(0.6),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Icon(Icons.arrow_back_rounded, size: 20),
          ),
        ),
        title: const Text('Предпросмотр'),
      ),
      body: ScaleTransition(
        scale: _scaleAnimation,
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              // Photo preview area
              Expanded(
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: AppTheme.surfaceCard,
                    borderRadius: BorderRadius.circular(24),
                    border: Border.all(
                      color: AppTheme.surfaceLight.withOpacity(0.3),
                      width: 1,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.3),
                        blurRadius: 24,
                        offset: const Offset(0, 8),
                      ),
                    ],
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(24),
                    child: Stack(
                      fit: StackFit.expand,
                      children: [
                        // Placeholder building illustration
                        _buildPlaceholderImage(),
                        // Overlay gradient
                        Positioned(
                          bottom: 0,
                          left: 0,
                          right: 0,
                          child: Container(
                            height: 120,
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                begin: Alignment.topCenter,
                                end: Alignment.bottomCenter,
                                colors: [
                                  Colors.transparent,
                                  AppTheme.primaryDark.withOpacity(0.8),
                                ],
                              ),
                            ),
                          ),
                        ),
                        // Photo info
                        Positioned(
                          bottom: 16,
                          left: 16,
                          right: 16,
                          child: Row(
                            children: [
                              Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 12,
                                  vertical: 6,
                                ),
                                decoration: BoxDecoration(
                                  color: AppTheme.surfaceCard.withOpacity(0.8),
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                child: const Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(
                                      Icons.photo_size_select_actual_rounded,
                                      color: AppTheme.textSecondary,
                                      size: 14,
                                    ),
                                    SizedBox(width: 6),
                                    Text(
                                      '4032 × 3024',
                                      style: TextStyle(
                                        color: AppTheme.textSecondary,
                                        fontSize: 12,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              const Spacer(),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 12,
                                  vertical: 6,
                                ),
                                decoration: BoxDecoration(
                                  color: AppTheme.surfaceCard.withOpacity(0.8),
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                child: const Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(
                                      Icons.straighten_rounded,
                                      color: AppTheme.textSecondary,
                                      size: 14,
                                    ),
                                    SizedBox(width: 6),
                                    Text(
                                      '5.2 MB',
                                      style: TextStyle(
                                        color: AppTheme.textSecondary,
                                        fontSize: 12,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 24),
              // Action buttons
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.refresh_rounded),
                      label: const Text('Переснять'),
                    ),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    flex: 2,
                    child: ElevatedButton.icon(
                      onPressed: () =>
                          Navigator.pushNamed(context, '/loading'),
                      icon: const Icon(Icons.auto_awesome_rounded),
                      label: const Text('Анализировать'),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 16),
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              // Tips
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: AppTheme.info.withOpacity(0.08),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: AppTheme.info.withOpacity(0.2),
                  ),
                ),
                child: const Row(
                  children: [
                    Icon(
                      Icons.lightbulb_outline_rounded,
                      color: AppTheme.info,
                      size: 20,
                    ),
                    SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'Для лучших результатов фотографируйте фасад прямо, при хорошем освещении',
                        style: TextStyle(
                          color: AppTheme.textSecondary,
                          fontSize: 12,
                          height: 1.4,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildPlaceholderImage() {
    return CustomPaint(
      painter: _BuildingPainter(),
    );
  }
}

class _BuildingPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    // Sky gradient
    final skyPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          const Color(0xFF1A237E),
          const Color(0xFF283593),
          const Color(0xFF3949AB),
        ],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height * 0.4));
    canvas.drawRect(
        Rect.fromLTWH(0, 0, size.width, size.height * 0.4), skyPaint);

    // Building body
    final buildingPaint = Paint()..color = const Color(0xFF78909C);
    final buildingRect = Rect.fromLTWH(
      size.width * 0.15,
      size.height * 0.15,
      size.width * 0.7,
      size.height * 0.85,
    );
    canvas.drawRRect(
      RRect.fromRectAndRadius(buildingRect, const Radius.circular(4)),
      buildingPaint,
    );

    // Building shadow
    final shadowPaint = Paint()..color = const Color(0xFF546E7A);
    canvas.drawRect(
      Rect.fromLTWH(
        size.width * 0.55,
        size.height * 0.15,
        size.width * 0.3,
        size.height * 0.85,
      ),
      shadowPaint,
    );

    // Windows
    final windowPaint = Paint()..color = const Color(0xFFFFD54F).withOpacity(0.7);
    final darkWindowPaint = Paint()..color = const Color(0xFF37474F);

    final windowW = size.width * 0.08;
    final windowH = size.height * 0.055;
    final startX = size.width * 0.22;
    final startY = size.height * 0.2;

    for (int row = 0; row < 8; row++) {
      for (int col = 0; col < 5; col++) {
        final x = startX + col * (windowW + size.width * 0.04);
        final y = startY + row * (windowH + size.height * 0.04);
        final isLit = (row + col) % 3 != 0;
        canvas.drawRRect(
          RRect.fromRectAndRadius(
            Rect.fromLTWH(x, y, windowW, windowH),
            const Radius.circular(2),
          ),
          isLit ? windowPaint : darkWindowPaint,
        );
      }
    }

    // Damage marks (red areas)
    final damagePaint = Paint()
      ..color = const Color(0xFFEF5350).withOpacity(0.3)
      ..style = PaintingStyle.fill;

    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(
            size.width * 0.2, size.height * 0.4, size.width * 0.2, size.height * 0.15),
        const Radius.circular(4),
      ),
      damagePaint,
    );

    canvas.drawRRect(
      RRect.fromRectAndRadius(
        Rect.fromLTWH(
            size.width * 0.5, size.height * 0.6, size.width * 0.25, size.height * 0.1),
        const Radius.circular(4),
      ),
      damagePaint,
    );

    // Crack lines
    final crackPaint = Paint()
      ..color = const Color(0xFFEF5350).withOpacity(0.5)
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    final path = Path();
    path.moveTo(size.width * 0.3, size.height * 0.45);
    path.lineTo(size.width * 0.32, size.height * 0.48);
    path.lineTo(size.width * 0.28, size.height * 0.52);
    path.lineTo(size.width * 0.33, size.height * 0.55);
    canvas.drawPath(path, crackPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
