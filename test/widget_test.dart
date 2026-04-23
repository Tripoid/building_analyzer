import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:facade_analyzer/main.dart';

void main() {
  testWidgets('App renders home screen', (WidgetTester tester) async {
    await tester.pumpWidget(const FacadeAnalyzerApp());
    expect(find.text('Анализ\nФасадов'), findsOneWidget);
  });
}
