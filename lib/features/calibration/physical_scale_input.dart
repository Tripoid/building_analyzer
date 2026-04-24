import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../app/providers.dart';
import '../../app/router.dart';
import '../../data/api/api_exceptions.dart';
import '../../domain/models/calibration_input.dart';
import '../../theme/app_theme.dart';

/// Two-tap calibration screen: user taps both ends of a known reference
/// dimension (usually a door). Fallback: single-tap-and-draw rectangle mode.
class PhysicalScaleInputScreen extends ConsumerStatefulWidget {
  const PhysicalScaleInputScreen({super.key, required this.image});
  final File image;

  @override
  ConsumerState<PhysicalScaleInputScreen> createState() =>
      _PhysicalScaleInputScreenState();
}

class _PhysicalScaleInputScreenState
    extends ConsumerState<PhysicalScaleInputScreen> {
  ReferenceType _type = ReferenceType.door;
  late double _widthM;
  double? _heightM;
  _Mode _mode = _Mode.twoTap;
  Offset? _p1;
  Offset? _p2;
  Rect? _bbox;
  Offset? _bboxStart;
  Size? _imageSize;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _widthM = _type.defaultWidthM;
  }

  Future<void> _loadImageSize() async {
    if (_imageSize != null) return;
    final bytes = await widget.image.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    _imageSize = Size(frame.image.width.toDouble(), frame.image.height.toDouble());
    if (mounted) setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    _loadImageSize();
    return Scaffold(
      backgroundColor: AppTheme.primaryDark,
      appBar: AppBar(title: const Text('Масштаб')),
      body: SafeArea(
        child: LayoutBuilder(
          builder: (ctx, constraints) {
            return Column(
              children: [
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: _ReferenceEditor(
                    type: _type,
                    widthM: _widthM,
                    heightM: _heightM,
                    mode: _mode,
                    onTypeChanged: (t) => setState(() {
                      _type = t;
                      _widthM = t.defaultWidthM;
                      _heightM = null;
                    }),
                    onWidthChanged: (v) => setState(() => _widthM = v),
                    onHeightChanged: (v) => setState(() => _heightM = v),
                    onModeChanged: (m) => setState(() {
                      _mode = m;
                      _p1 = _p2 = null;
                      _bbox = null;
                      _bboxStart = null;
                    }),
                  ),
                ),
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: AspectRatio(
                      aspectRatio: 4 / 3,
                      child: _InteractiveCanvas(
                        image: widget.image,
                        mode: _mode,
                        p1: _p1,
                        p2: _p2,
                        bbox: _bbox,
                        onTap: _handleTap,
                        onPanStart: _handlePanStart,
                        onPanUpdate: _handlePanUpdate,
                      ),
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _busy ? null : _skip,
                          icon: const Icon(Icons.skip_next_rounded),
                          label: const Text('Пропустить'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        flex: 2,
                        child: ElevatedButton.icon(
                          onPressed: _busy || !_canSubmit ? null : _submit,
                          icon: _busy
                              ? const SizedBox(
                                  width: 18,
                                  height: 18,
                                  child: CircularProgressIndicator(strokeWidth: 2),
                                )
                              : const Icon(Icons.check_rounded),
                          label: const Text('Отправить на анализ'),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  bool get _canSubmit =>
      _mode == _Mode.twoTap ? (_p1 != null && _p2 != null) : (_bbox != null);

  void _handleTap(Offset normalized) {
    if (_mode != _Mode.twoTap) return;
    setState(() {
      if (_p1 == null) {
        _p1 = normalized;
      } else if (_p2 == null) {
        _p2 = normalized;
      } else {
        _p1 = normalized;
        _p2 = null;
      }
    });
  }

  void _handlePanStart(Offset normalized) {
    if (_mode != _Mode.rectangle) return;
    setState(() {
      _bboxStart = normalized;
      _bbox = Rect.fromPoints(normalized, normalized);
    });
  }

  void _handlePanUpdate(Offset normalized) {
    if (_mode != _Mode.rectangle || _bboxStart == null) return;
    setState(() => _bbox = Rect.fromPoints(_bboxStart!, normalized));
  }

  void _skip() {
    context.go('/loading', extra: AnalyzeArgs(image: widget.image));
  }

  Future<void> _submit() async {
    if (_imageSize == null) return;
    final w = _imageSize!.width.toInt();
    final h = _imageSize!.height.toInt();
    final input = _mode == _Mode.twoTap
        ? CalibrationInput(
            type: _type,
            widthM: _widthM,
            heightM: _heightM,
            imageWidthPx: w,
            imageHeightPx: h,
            p1: Offset(_p1!.dx * w, _p1!.dy * h),
            p2: Offset(_p2!.dx * w, _p2!.dy * h),
          )
        : CalibrationInput(
            type: _type,
            widthM: _widthM,
            heightM: _heightM,
            imageWidthPx: w,
            imageHeightPx: h,
            bbox: Rect.fromLTRB(
              _bbox!.left * w,
              _bbox!.top * h,
              _bbox!.right * w,
              _bbox!.bottom * h,
            ),
          );

    setState(() => _busy = true);
    final repo = ref.read(calibrationRepositoryProvider);
    try {
      final result = await repo.calibrate(input);
      if (!mounted) return;
      if (result.warnings.isNotEmpty) {
        await showDialog<void>(
          context: context,
          builder: (_) => AlertDialog(
            title: const Text('Внимание'),
            content: Text(result.warnings.join('\n')),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('OK'),
              ),
            ],
          ),
        );
      }
      if (!mounted) return;
      context.go(
        '/loading',
        extra: AnalyzeArgs(image: widget.image, calibrationId: result.calibrationId),
      );
    } on ApiException catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(e.message)),
      );
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }
}

enum _Mode { twoTap, rectangle }

class _ReferenceEditor extends StatelessWidget {
  const _ReferenceEditor({
    required this.type,
    required this.widthM,
    required this.heightM,
    required this.mode,
    required this.onTypeChanged,
    required this.onWidthChanged,
    required this.onHeightChanged,
    required this.onModeChanged,
  });
  final ReferenceType type;
  final double widthM;
  final double? heightM;
  final _Mode mode;
  final ValueChanged<ReferenceType> onTypeChanged;
  final ValueChanged<double> onWidthChanged;
  final ValueChanged<double?> onHeightChanged;
  final ValueChanged<_Mode> onModeChanged;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Эталонный объект', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: ReferenceType.values
              .map((t) => ChoiceChip(
                    label: Text(t.labelRu),
                    selected: t == type,
                    onSelected: (v) => v ? onTypeChanged(t) : null,
                  ))
              .toList(),
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: TextFormField(
                key: ValueKey('width-${type.name}'),
                initialValue: widthM.toStringAsFixed(2),
                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                decoration: const InputDecoration(
                  labelText: 'Ширина, м',
                  border: OutlineInputBorder(),
                ),
                onChanged: (v) => onWidthChanged(double.tryParse(v.replaceAll(',', '.')) ?? widthM),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: TextFormField(
                key: ValueKey('height-${type.name}'),
                initialValue: heightM?.toStringAsFixed(2) ?? '',
                keyboardType: const TextInputType.numberWithOptions(decimal: true),
                decoration: const InputDecoration(
                  labelText: 'Высота, м (необяз.)',
                  border: OutlineInputBorder(),
                ),
                onChanged: (v) {
                  final p = double.tryParse(v.replaceAll(',', '.'));
                  onHeightChanged((p == null || p <= 0) ? null : p);
                },
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        SegmentedButton<_Mode>(
          segments: const [
            ButtonSegment(
              value: _Mode.twoTap,
              label: Text('Две точки'),
              icon: Icon(Icons.touch_app_rounded),
            ),
            ButtonSegment(
              value: _Mode.rectangle,
              label: Text('Прямоугольник'),
              icon: Icon(Icons.crop_square_rounded),
            ),
          ],
          selected: {mode},
          onSelectionChanged: (s) => onModeChanged(s.first),
        ),
      ],
    );
  }
}

class _InteractiveCanvas extends StatelessWidget {
  const _InteractiveCanvas({
    required this.image,
    required this.mode,
    required this.p1,
    required this.p2,
    required this.bbox,
    required this.onTap,
    required this.onPanStart,
    required this.onPanUpdate,
  });
  final File image;
  final _Mode mode;
  final Offset? p1;
  final Offset? p2;
  final Rect? bbox;
  final ValueChanged<Offset> onTap;
  final ValueChanged<Offset> onPanStart;
  final ValueChanged<Offset> onPanUpdate;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: Container(
        color: Colors.black,
        child: LayoutBuilder(
          builder: (ctx, cons) {
            return GestureDetector(
              onTapDown: (d) => onTap(Offset(
                (d.localPosition.dx / cons.maxWidth).clamp(0.0, 1.0),
                (d.localPosition.dy / cons.maxHeight).clamp(0.0, 1.0),
              )),
              onPanStart: (d) => onPanStart(Offset(
                (d.localPosition.dx / cons.maxWidth).clamp(0.0, 1.0),
                (d.localPosition.dy / cons.maxHeight).clamp(0.0, 1.0),
              )),
              onPanUpdate: (d) => onPanUpdate(Offset(
                (d.localPosition.dx / cons.maxWidth).clamp(0.0, 1.0),
                (d.localPosition.dy / cons.maxHeight).clamp(0.0, 1.0),
              )),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Image.file(image, fit: BoxFit.contain),
                  CustomPaint(
                    painter: _OverlayPainter(mode: mode, p1: p1, p2: p2, bbox: bbox),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }
}

class _OverlayPainter extends CustomPainter {
  _OverlayPainter({required this.mode, this.p1, this.p2, this.bbox});
  final _Mode mode;
  final Offset? p1;
  final Offset? p2;
  final Rect? bbox;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = AppTheme.accent
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    if (mode == _Mode.twoTap) {
      if (p1 != null) {
        canvas.drawCircle(_abs(p1!, size), 8, paint..style = PaintingStyle.stroke);
      }
      if (p2 != null) {
        canvas.drawCircle(_abs(p2!, size), 8, paint);
      }
      if (p1 != null && p2 != null) {
        canvas.drawLine(_abs(p1!, size), _abs(p2!, size), paint);
      }
    } else if (bbox != null) {
      final rect = Rect.fromLTRB(
        bbox!.left * size.width,
        bbox!.top * size.height,
        bbox!.right * size.width,
        bbox!.bottom * size.height,
      );
      canvas.drawRect(rect, paint);
    }
  }

  Offset _abs(Offset o, Size s) => Offset(o.dx * s.width, o.dy * s.height);

  @override
  bool shouldRepaint(covariant _OverlayPainter old) =>
      old.mode != mode || old.p1 != p1 || old.p2 != p2 || old.bbox != bbox;
}
