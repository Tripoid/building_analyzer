import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:facade_analyzer/main.dart';

void main() {
  testWidgets('App renders AlegroCode home', (tester) async {
    await tester.pumpWidget(const ProviderScope(child: AlegroCodeApp()));
    await tester.pump();
    expect(find.text('AlegroCode'), findsWidgets);
  });
}
