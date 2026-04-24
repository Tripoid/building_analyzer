import 'package:cached_network_image/cached_network_image.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:intl/intl.dart';

import '../app/providers.dart';
import '../domain/models/mask_layer.dart';
import '../features/results/layer_panel.dart';
import '../features/results/mask_layer_viewer.dart';
import '../models/analysis_result.dart';
import '../theme/app_theme.dart';
import '../widgets/cost_breakdown_card.dart';
import '../widgets/damage_chart.dart';
import '../widgets/stat_card.dart';

/// Responsive 5-tab results screen. Mobile: vertical stack. Tablet/landscape:
/// mask viewer + layer panel side-by-side.
class ResultsScreen extends ConsumerStatefulWidget {
  const ResultsScreen({super.key, this.result});
  final AnalysisResult? result;

  @override
  ConsumerState<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends ConsumerState<ResultsScreen>
    with TickerProviderStateMixin {
  late TabController _tabs;
  late AnalysisResult _result;
  late List<MaskLayer> _layers;
  bool _restoring = false;

  static final _rub =
      NumberFormat.currency(locale: 'ru', symbol: '₽', decimalDigits: 0);

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 5, vsync: this);
    _result = widget.result ?? AnalysisResult.mock();
    _layers = _result.masks.toLayers();
  }

  @override
  void dispose() {
    _tabs.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.primaryDark,
      body: SafeArea(
        child: NestedScrollView(
          headerSliverBuilder: (ctx, inner) => [
            SliverAppBar(
              pinned: true,
              expandedHeight: 220,
              leading: IconButton(
                onPressed: () => context.go('/'),
                icon: const Icon(Icons.arrow_back_rounded),
              ),
              flexibleSpace: FlexibleSpaceBar(
                collapseMode: CollapseMode.pin,
                background: _Header(result: _result),
              ),
              bottom: TabBar(
                controller: _tabs,
                isScrollable: true,
                tabs: const [
                  Tab(text: 'Фото'),
                  Tab(text: 'Дефекты'),
                  Tab(text: 'Материалы'),
                  Tab(text: 'Смета'),
                  Tab(text: 'Ведомость'),
                ],
              ),
            ),
          ],
          body: TabBarView(
            controller: _tabs,
            children: [
              _PhotoTab(
                result: _result,
                layers: _layers,
                onToggle: _toggleLayer,
                onPreset: _applyPreset,
                onRestore: _restoring ? null : _requestRestoration,
                restoring: _restoring,
              ),
              _DefectsTab(result: _result),
              _MaterialsTab(result: _result),
              _EstimateTab(result: _result, rub: _rub),
              _BillTab(result: _result, rub: _rub),
            ],
          ),
        ),
      ),
    );
  }

  void _toggleLayer(String id, bool visible) {
    setState(() {
      _layers = [
        for (final l in _layers) l.id == id ? l.copyWith(visible: visible) : l,
      ];
    });
  }

  void _applyPreset(LayerPreset preset) {
    setState(() {
      _layers = [
        for (final l in _layers)
          l.copyWith(
            visible: switch (preset) {
              LayerPreset.all => true,
              LayerPreset.none => false,
              LayerPreset.defectsOnly => l.group == MaskGroup.defects,
              LayerPreset.materialsOnly => l.group == MaskGroup.materials,
            },
          ),
      ];
    });
  }

  Future<void> _requestRestoration() async {
    final id = _result.id;
    if (id == null) return;
    setState(() => _restoring = true);
    final repo = ref.read(analysisRepositoryProvider);
    try {
      final updated = await repo.requestRestoration(analysisId: id);
      if (!mounted) return;
      setState(() {
        _result = updated;
        _layers = updated.masks.toLayers();
      });
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Не удалось восстановить: $e')),
      );
    } finally {
      if (mounted) setState(() => _restoring = false);
    }
  }
}

// ──────────────────────── Header ────────────────────────

class _Header extends StatelessWidget {
  const _Header({required this.result});
  final AnalysisResult result;

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [AppTheme.primaryDark, AppTheme.primaryMid],
        ),
      ),
      padding: const EdgeInsets.fromLTRB(16, 48, 16, 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(
            width: 96,
            height: 96,
            child: CircularProgressIndicator(
              value: (result.overallScore / 100).clamp(0.0, 1.0),
              strokeWidth: 8,
              valueColor:
                  AlwaysStoppedAnimation(_scoreColor(result.overallScore)),
              backgroundColor: AppTheme.surfaceLight,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  '${result.overallScore.toStringAsFixed(1)} / 100',
                  style: Theme.of(context).textTheme.headlineSmall,
                  overflow: TextOverflow.ellipsis,
                ),
                Text(
                  result.overallCondition,
                  style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: AppTheme.textSecondary,
                      ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
                if (result.priceSnapshotDate != null || result.pricesAreStale)
                  Padding(
                    padding: const EdgeInsets.only(top: 4),
                    child: _PriceBanner(result: result),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  static Color _scoreColor(double s) {
    if (s >= 80) return AppTheme.success;
    if (s >= 60) return AppTheme.warning;
    if (s >= 30) return AppTheme.accent;
    return AppTheme.danger;
  }
}

class _PriceBanner extends StatelessWidget {
  const _PriceBanner({required this.result});
  final AnalysisResult result;

  @override
  Widget build(BuildContext context) {
    final stale = result.pricesAreStale;
    final label = result.priceSnapshotDate != null
        ? 'Цены от ${_shortDate(result.priceSnapshotDate!)}'
        : 'Цены: базовый прайс';
    return Text(
      stale ? '$label · обновление прайса не удалось' : label,
      style: TextStyle(
        fontSize: 11,
        color: stale ? AppTheme.warning : AppTheme.textSecondary,
      ),
      maxLines: 2,
      overflow: TextOverflow.ellipsis,
    );
  }

  String _shortDate(String iso) {
    final parsed = DateTime.tryParse(iso);
    if (parsed == null) return iso;
    return DateFormat('d MMM y', 'ru').format(parsed);
  }
}

// ──────────────────────── Photo tab ────────────────────────

class _PhotoTab extends StatelessWidget {
  const _PhotoTab({
    required this.result,
    required this.layers,
    required this.onToggle,
    required this.onPreset,
    required this.onRestore,
    required this.restoring,
  });
  final AnalysisResult result;
  final List<MaskLayer> layers;
  final void Function(String, bool) onToggle;
  final void Function(LayerPreset) onPreset;
  final VoidCallback? onRestore;
  final bool restoring;

  static const _headers = {'ngrok-skip-browser-warning': 'true'};

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (ctx, cons) {
        final wide = cons.maxWidth >= 720;
        final viewer = MaskLayerViewer(
          baseImageUrl: result.masks.baseImageUrl,
          layers: layers,
        );
        final panel = LayerPanel(
          layers: layers,
          onToggle: onToggle,
          onPreset: onPreset,
        );
        final restoredPreview = result.restoredUrl != null
            ? Card(
                margin: const EdgeInsets.only(top: 12),
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Text('ИИ-реставрация',
                          style: Theme.of(context).textTheme.titleSmall),
                      const SizedBox(height: 8),
                      AspectRatio(
                        aspectRatio: 4 / 3,
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(12),
                          child: CachedNetworkImage(
                            imageUrl: _absoluteUrl(result.restoredUrl!),
                            httpHeaders: _headers,
                            fit: BoxFit.contain,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              )
            : const SizedBox.shrink();
        final restoreBtn = Padding(
          padding: const EdgeInsets.only(top: 12),
          child: ElevatedButton.icon(
            onPressed: onRestore,
            icon: restoring
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.auto_fix_high_rounded),
            label: Text(result.restoredUrl == null
                ? 'ИИ-реставрация фасада'
                : 'Перезапустить реставрацию'),
          ),
        );

        return SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          physics: const BouncingScrollPhysics(),
          child: wide
              ? Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      flex: 2,
                      child: Column(
                        children: [viewer, restoreBtn, restoredPreview],
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(child: panel),
                  ],
                )
              : Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    viewer,
                    const SizedBox(height: 12),
                    panel,
                    restoreBtn,
                    restoredPreview,
                  ],
                ),
        );
      },
    );
  }

  String _absoluteUrl(String url) {
    if (url.startsWith('http')) return url;
    final base = result.masks.baseImageUrl;
    final idx = base.indexOf('/api/');
    if (idx < 0) return url;
    return '${base.substring(0, idx)}$url';
  }
}

// ──────────────────────── Defects tab ────────────────────────

class _DefectsTab extends StatelessWidget {
  const _DefectsTab({required this.result});
  final AnalysisResult result;

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      physics: const BouncingScrollPhysics(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          LayoutBuilder(
            builder: (ctx, cons) => Center(
              child: DamageChart(
                data: [
                  for (final d in result.damages)
                    DamageChartData(
                      label: d.type,
                      value: d.percentage,
                      color: _damageColor(d.severity),
                    ),
                ],
                size: cons.maxWidth.clamp(180.0, 260.0),
              ),
            ),
          ),
          const SizedBox(height: 16),
          for (final d in result.damages)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            d.type,
                            style: Theme.of(context).textTheme.titleSmall,
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        _Severity(severity: d.severity),
                      ],
                    ),
                    const SizedBox(height: 6),
                    LinearProgressIndicator(
                      value: (d.percentage / 100).clamp(0.0, 1.0),
                      minHeight: 6,
                      color: _damageColor(d.severity),
                      backgroundColor: AppTheme.surfaceLight,
                    ),
                    const SizedBox(height: 6),
                    Text(
                      '${d.percentage.toStringAsFixed(1)}% · ${d.areaM2.toStringAsFixed(1)} м²',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                    if (d.affectedLayers.isNotEmpty)
                      Padding(
                        padding: const EdgeInsets.only(top: 6),
                        child: Wrap(
                          spacing: 4,
                          runSpacing: 4,
                          children: [
                            for (final l in d.affectedLayers)
                              Chip(
                                label: Text(_layerName(l)),
                                visualDensity: VisualDensity.compact,
                              ),
                          ],
                        ),
                      ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }

  static Color _damageColor(String severity) {
    final s = severity.toLowerCase();
    if (s.contains('выс')) return AppTheme.danger;
    if (s.contains('сред')) return AppTheme.warning;
    return AppTheme.info;
  }

  static String _layerName(String k) => switch (k) {
        'finish' => 'финиш',
        'base_plaster' => 'штукатурка',
        'insulation' => 'утеплитель',
        'structural' => 'несущий',
        _ => k,
      };
}

class _Severity extends StatelessWidget {
  const _Severity({required this.severity});
  final String severity;
  @override
  Widget build(BuildContext context) {
    final s = severity.toLowerCase();
    final color = s.contains('выс')
        ? AppTheme.danger
        : s.contains('сред')
            ? AppTheme.warning
            : AppTheme.info;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Text(severity, style: TextStyle(color: color, fontSize: 11)),
    );
  }
}

// ──────────────────────── Materials tab ────────────────────────

class _MaterialsTab extends StatelessWidget {
  const _MaterialsTab({required this.result});
  final AnalysisResult result;

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      padding: const EdgeInsets.all(16),
      physics: const BouncingScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
        maxCrossAxisExtent: 220,
        mainAxisSpacing: 12,
        crossAxisSpacing: 12,
        childAspectRatio: 1.2,
      ),
      itemCount: result.materials.length,
      itemBuilder: (ctx, i) {
        final m = result.materials[i];
        return Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(m.iconData, color: AppTheme.accent),
                const SizedBox(height: 8),
                Text(
                  m.name,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                const Spacer(),
                Text(
                  '${m.percentage.toStringAsFixed(1)}%',
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                Text(
                  '${m.areaM2.toStringAsFixed(1)} м²',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
        );
      },
    );
  }
}

// ──────────────────────── Estimate tab ────────────────────────

class _EstimateTab extends StatelessWidget {
  const _EstimateTab({required this.result, required this.rub});
  final AnalysisResult result;
  final NumberFormat rub;

  @override
  Widget build(BuildContext context) {
    final e = result.repairEstimate;
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      physics: const BouncingScrollPhysics(),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          LayoutBuilder(
            builder: (ctx, cons) => Wrap(
              spacing: 12,
              runSpacing: 12,
              children: [
                SizedBox(
                  width: cons.maxWidth >= 480
                      ? (cons.maxWidth - 12) / 2
                      : cons.maxWidth,
                  child: StatCard(
                    icon: Icons.account_balance_wallet_rounded,
                    label: 'Итого к оплате',
                    value: rub.format(e.grandTotal),
                    color: AppTheme.accent,
                  ),
                ),
                SizedBox(
                  width: cons.maxWidth >= 480
                      ? (cons.maxWidth - 12) / 2
                      : cons.maxWidth,
                  child: StatCard(
                    icon: Icons.schedule_rounded,
                    label: 'Срок работ',
                    value: '${e.estimatedDays} дн.',
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          CostBreakdownCard(items: result.costs),
        ],
      ),
    );
  }
}

// ──────────────────────── Bill tab ────────────────────────

class _BillTab extends StatelessWidget {
  const _BillTab({required this.result, required this.rub});
  final AnalysisResult result;
  final NumberFormat rub;

  @override
  Widget build(BuildContext context) {
    final e = result.repairEstimate;
    return ListView(
      padding: const EdgeInsets.all(16),
      physics: const BouncingScrollPhysics(),
      children: [
        Text('Материалы', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        for (final m in e.materials)
          ListTile(
            dense: true,
            title: Text(m.name, maxLines: 2, overflow: TextOverflow.ellipsis),
            subtitle: Text(
              '${m.quantity.toStringAsFixed(2)} ${m.unit} × ${rub.format(m.pricePerUnit)}',
            ),
            trailing: Text(
              rub.format(m.totalCost),
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
          ),
        const Divider(),
        Text('Работы', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        for (final l in e.labor)
          ListTile(
            dense: true,
            title: Text(l.name, maxLines: 2, overflow: TextOverflow.ellipsis),
            subtitle: Text(
              '${l.quantity.toStringAsFixed(2)} ${l.unit} · ${l.normHours.toStringAsFixed(1)} ч',
            ),
            trailing: Text(
              rub.format(l.totalCost),
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
          ),
        const Divider(),
        ListTile(
          title: const Text('Леса'),
          trailing: Text(
            rub.format(e.scaffoldingTotal),
            style: const TextStyle(fontWeight: FontWeight.w600),
          ),
        ),
        ListTile(
          title: Text('НДС (${(e.vatRate * 100).toStringAsFixed(0)}%)'),
          trailing: Text(rub.format(e.vatAmount)),
        ),
        Container(
          margin: const EdgeInsets.symmetric(vertical: 16),
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [AppTheme.accent, AppTheme.accentLight],
            ),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Row(
            children: [
              const Icon(Icons.payments_rounded, color: Colors.white),
              const SizedBox(width: 12),
              const Expanded(
                child: Text(
                  'Итого',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              Text(
                rub.format(e.grandTotal),
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 22,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
