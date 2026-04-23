import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../models/analysis_result.dart';
import '../widgets/stat_card.dart';
import '../widgets/damage_chart.dart';
import '../widgets/cost_breakdown_card.dart';

class ResultsScreen extends StatefulWidget {
  const ResultsScreen({super.key});

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen>
    with TickerProviderStateMixin {
  late TabController _tabController;
  late AnimationController _fadeController;
  final AnalysisResult _result = AnalysisResult.mock();
  int _currentImageIndex = 0;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 5, vsync: this);
    _fadeController = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );
    _fadeController.forward();
  }

  @override
  void dispose() {
    _tabController.dispose();
    _fadeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.primaryDark,
      body: NestedScrollView(
        headerSliverBuilder: (context, innerBoxIsScrolled) {
          return [
            SliverAppBar(
              expandedHeight: 260,
              pinned: true,
              backgroundColor: AppTheme.primaryMid,
              leading: IconButton(
                onPressed: () =>
                    Navigator.popUntil(context, (route) => route.isFirst),
                icon: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: AppTheme.surfaceCard.withValues(alpha: 0.6),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Icon(Icons.arrow_back_rounded, size: 20),
                ),
              ),
              actions: [
                IconButton(
                  onPressed: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Поделиться отчётом...')),
                    );
                  },
                  icon: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: AppTheme.surfaceCard.withValues(alpha: 0.6),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Icon(Icons.share_rounded, size: 20),
                  ),
                ),
                IconButton(
                  onPressed: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Скачивание PDF отчёта...')),
                    );
                  },
                  icon: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: AppTheme.surfaceCard.withValues(alpha: 0.6),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Icon(Icons.download_rounded, size: 20),
                  ),
                ),
                const SizedBox(width: 8),
              ],
              flexibleSpace: FlexibleSpaceBar(
                background: _buildScoreHeader(),
              ),
              bottom: PreferredSize(
                preferredSize: const Size.fromHeight(48),
                child: Container(
                  color: AppTheme.primaryMid,
                  child: TabBar(
                    controller: _tabController,
                    isScrollable: true,
                    tabAlignment: TabAlignment.start,
                    padding: const EdgeInsets.symmetric(horizontal: 4),
                    indicatorSize: TabBarIndicatorSize.label,
                    dividerColor: Colors.transparent,
                    tabs: const [
                      Tab(icon: Icon(Icons.image_rounded, size: 16), text: 'Снимки'),
                      Tab(icon: Icon(Icons.bar_chart_rounded, size: 16), text: 'Дефекты'),
                      Tab(icon: Icon(Icons.layers_rounded, size: 16), text: 'Материалы'),
                      Tab(icon: Icon(Icons.payments_rounded, size: 16), text: 'Смета'),
                      Tab(icon: Icon(Icons.build_rounded, size: 16), text: 'Ведомость'),
                    ],
                  ),
                ),
              ),
            ),
          ];
        },
        body: FadeTransition(
          opacity: _fadeController,
          child: TabBarView(
            controller: _tabController,
            children: [
              _buildImagesTab(),
              _buildDamageTab(),
              _buildMaterialsTab(),
              _buildCostTab(),
              _buildRepairMaterialsTab(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildScoreHeader() {
    Color scoreColor;
    if (_result.overallScore >= 80) {
      scoreColor = AppTheme.success;
    } else if (_result.overallScore >= 60) {
      scoreColor = AppTheme.warning;
    } else {
      scoreColor = AppTheme.danger;
    }

    return Container(
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            AppTheme.primaryDark,
            AppTheme.primaryMid,
          ],
        ),
      ),
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 48, 24, 56),
          child: Row(
            children: [
              SizedBox(
                width: 100,
                height: 100,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    SizedBox(
                      width: 100,
                      height: 100,
                      child: CircularProgressIndicator(
                        value: _result.overallScore / 100,
                        strokeWidth: 8,
                        backgroundColor: AppTheme.surfaceLight.withValues(alpha: 0.3),
                        valueColor: AlwaysStoppedAnimation<Color>(scoreColor),
                        strokeCap: StrokeCap.round,
                      ),
                    ),
                    Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          _result.overallScore.toStringAsFixed(0),
                          style: TextStyle(
                            fontSize: 32,
                            fontWeight: FontWeight.w800,
                            color: scoreColor,
                            height: 1,
                          ),
                        ),
                        Text(
                          'баллов',
                          style: TextStyle(
                            fontSize: 11,
                            color: scoreColor.withValues(alpha: 0.7),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 20),
              Expanded(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: scoreColor.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        _result.overallCondition,
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: scoreColor,
                        ),
                      ),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      'Площадь: ${_result.totalArea.toStringAsFixed(0)} м²',
                      style: const TextStyle(fontSize: 13, color: AppTheme.textSecondary),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Повреждено: ${_result.damagedArea.toStringAsFixed(0)} м² (${(_result.damagedArea / _result.totalArea * 100).toStringAsFixed(0)}%)',
                      style: const TextStyle(fontSize: 13, color: AppTheme.textSecondary),
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

  // ── Tab 1: Processed Images ──

  Widget _buildImagesTab() {
    return ListView(
      padding: const EdgeInsets.all(20),
      physics: const BouncingScrollPhysics(),
      children: [
        SizedBox(
          height: 280,
          child: PageView.builder(
            itemCount: _result.processedImages.length,
            onPageChanged: (i) => setState(() => _currentImageIndex = i),
            itemBuilder: (context, index) {
              final img = _result.processedImages[index];
              return Container(
                margin: const EdgeInsets.symmetric(horizontal: 4),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: AppTheme.surfaceLight.withValues(alpha: 0.3)),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      CustomPaint(painter: _ProcessedImagePainter(type: img.type)),
                      Positioned(
                        bottom: 0, left: 0, right: 0,
                        child: Container(
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              begin: Alignment.topCenter,
                              end: Alignment.bottomCenter,
                              colors: [Colors.transparent, AppTheme.primaryDark.withValues(alpha: 0.9)],
                            ),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(img.title, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
                              const SizedBox(height: 4),
                              Text(img.description, style: const TextStyle(fontSize: 12, color: AppTheme.textSecondary)),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
        const SizedBox(height: 12),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: List.generate(
            _result.processedImages.length,
            (i) => AnimatedContainer(
              duration: const Duration(milliseconds: 300),
              margin: const EdgeInsets.symmetric(horizontal: 3),
              width: i == _currentImageIndex ? 24 : 8,
              height: 8,
              decoration: BoxDecoration(
                color: i == _currentImageIndex ? AppTheme.accent : AppTheme.surfaceLight,
                borderRadius: BorderRadius.circular(4),
              ),
            ),
          ),
        ),
        const SizedBox(height: 24),
        GridView.count(
          crossAxisCount: 2,
          childAspectRatio: 1.3,
          mainAxisSpacing: 12,
          crossAxisSpacing: 12,
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          children: [
            StatCard(icon: Icons.image_rounded, value: '${_result.processedImages.length}', label: 'Обработанных\nснимков', color: AppTheme.info),
            StatCard(icon: Icons.bug_report_rounded, value: '${_result.damages.length}', label: 'Типов\nдефектов', color: AppTheme.danger),
            StatCard(icon: Icons.square_foot_rounded, value: '${_result.totalArea.toStringAsFixed(0)} м²', label: 'Общая\nплощадь', color: AppTheme.success),
            StatCard(icon: Icons.warning_rounded, value: '${(_result.damagedArea / _result.totalArea * 100).toStringAsFixed(0)}%', label: 'Площадь\nповреждений', color: AppTheme.warning),
          ],
        ),
      ],
    );
  }

  // ── Tab 2: Damage Statistics ──

  Widget _buildDamageTab() {
    final chartColors = [AppTheme.danger, AppTheme.warning, AppTheme.info, AppTheme.success, AppTheme.accentLight];
    return ListView(
      padding: const EdgeInsets.all(20),
      physics: const BouncingScrollPhysics(),
      children: [
        Center(
          child: DamageChart(
            size: 200,
            data: _result.damages.asMap().entries.map((e) {
              return DamageChartData(label: e.value.type, value: e.value.percentage, color: chartColors[e.key % chartColors.length]);
            }).toList(),
          ),
        ),
        const SizedBox(height: 24),
        ...List.generate(_result.damages.length, (i) {
          final d = _result.damages[i];
          return _buildDamageItem(d, chartColors[i % chartColors.length]);
        }),
      ],
    );
  }

  Widget _buildDamageItem(DamageInfo damage, Color color) {
    Color severityColor;
    switch (damage.severity) {
      case 'Высокая':
        severityColor = AppTheme.danger;
        break;
      case 'Средняя':
        severityColor = AppTheme.warning;
        break;
      default:
        severityColor = AppTheme.success;
    }

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.surfaceCard,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withValues(alpha: 0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(width: 12, height: 12, decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(3))),
              const SizedBox(width: 10),
              Expanded(child: Text(damage.type, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: AppTheme.textPrimary))),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(color: severityColor.withValues(alpha: 0.15), borderRadius: BorderRadius.circular(6)),
                child: Text(damage.severity, style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: severityColor)),
              ),
              const SizedBox(width: 8),
              Text('${damage.percentage.toStringAsFixed(0)}%', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: color)),
            ],
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(3),
            child: LinearProgressIndicator(value: damage.percentage / 100, backgroundColor: AppTheme.primaryLight, valueColor: AlwaysStoppedAnimation<Color>(color), minHeight: 4),
          ),
          const SizedBox(height: 8),
          Text(damage.description, style: const TextStyle(fontSize: 12, color: AppTheme.textSecondary, height: 1.4)),
          if (damage.affectedLayers.isNotEmpty) ...[
            const SizedBox(height: 8),
            Wrap(
              spacing: 6,
              children: damage.affectedLayers.map((layer) {
                return Chip(
                  label: Text(_layerName(layer), style: const TextStyle(fontSize: 10, color: AppTheme.textSecondary)),
                  backgroundColor: AppTheme.primaryLight,
                  padding: EdgeInsets.zero,
                  materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  visualDensity: VisualDensity.compact,
                );
              }).toList(),
            ),
          ],
        ],
      ),
    );
  }

  String _layerName(String key) {
    switch (key) {
      case 'finish': return 'Финиш';
      case 'base_plaster': return 'Штукатурка';
      case 'insulation': return 'Утеплитель';
      case 'structural': return 'Несущий';
      default: return key;
    }
  }

  // ── Tab 3: Materials ──

  Widget _buildMaterialsTab() {
    return ListView(
      padding: const EdgeInsets.all(20),
      physics: const BouncingScrollPhysics(),
      children: [
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [AppTheme.surfaceCard, AppTheme.surfaceLight.withValues(alpha: 0.5)],
            ),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: AppTheme.surfaceLight.withValues(alpha: 0.3)),
          ),
          child: Column(
            children: [
              const Row(
                children: [
                  Icon(Icons.layers_rounded, color: AppTheme.accent, size: 22),
                  SizedBox(width: 10),
                  Text('Состав фасада', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
                ],
              ),
              const SizedBox(height: 16),
              ..._result.materials.map((m) => _buildMaterialBar(m)),
            ],
          ),
        ),
        const SizedBox(height: 16),
        ..._result.materials.map((m) => _buildMaterialDetailCard(m)),
      ],
    );
  }

  Widget _buildMaterialBar(MaterialInfo material) {
    Color barColor;
    switch (material.condition) {
      case 'Хорошее':
        barColor = AppTheme.success;
        break;
      case 'Удовлетворительное':
        barColor = AppTheme.warning;
        break;
      default:
        barColor = AppTheme.danger;
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Icon(material.iconData, size: 16, color: AppTheme.textSecondary),
                  const SizedBox(width: 6),
                  Text(material.name, style: const TextStyle(fontSize: 13, color: AppTheme.textPrimary)),
                ],
              ),
              Text('${material.percentage.toStringAsFixed(0)}%', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: barColor)),
            ],
          ),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(3),
            child: LinearProgressIndicator(value: material.percentage / 100, backgroundColor: AppTheme.primaryLight, valueColor: AlwaysStoppedAnimation<Color>(barColor), minHeight: 6),
          ),
        ],
      ),
    );
  }

  Widget _buildMaterialDetailCard(MaterialInfo material) {
    Color conditionColor;
    IconData conditionIcon;
    switch (material.condition) {
      case 'Хорошее':
        conditionColor = AppTheme.success;
        conditionIcon = Icons.check_circle_rounded;
        break;
      case 'Удовлетворительное':
        conditionColor = AppTheme.warning;
        conditionIcon = Icons.info_rounded;
        break;
      default:
        conditionColor = AppTheme.danger;
        conditionIcon = Icons.warning_rounded;
    }

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.surfaceCard,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: conditionColor.withValues(alpha: 0.15)),
      ),
      child: Row(
        children: [
          Icon(material.iconData, size: 28, color: conditionColor),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(material.name, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: AppTheme.textPrimary)),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Icon(conditionIcon, color: conditionColor, size: 14),
                    const SizedBox(width: 4),
                    Text(material.condition, style: TextStyle(fontSize: 12, color: conditionColor, fontWeight: FontWeight.w500)),
                  ],
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(color: AppTheme.primaryLight, borderRadius: BorderRadius.circular(8)),
            child: Text('${material.percentage.toStringAsFixed(0)}%', style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
          ),
        ],
      ),
    );
  }

  // ── Tab 4: Cost Estimation ──

  Widget _buildCostTab() {
    return ListView(
      padding: const EdgeInsets.all(20),
      physics: const BouncingScrollPhysics(),
      children: [
        // Summary cards
        Row(
          children: [
            Expanded(
              child: StatCard(
                icon: Icons.access_time_rounded,
                value: '${_result.repairEstimate.estimatedDays}',
                label: 'Дней\nработы',
                color: AppTheme.info,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: StatCard(
                icon: Icons.engineering_rounded,
                value: '${_result.repairEstimate.totalWorkHours.toStringAsFixed(0)}',
                label: 'Нормо-\nчасов',
                color: AppTheme.warning,
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),
        // Disclaimer
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: AppTheme.warning.withValues(alpha: 0.08),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: AppTheme.warning.withValues(alpha: 0.2)),
          ),
          child: const Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(Icons.info_outline_rounded, color: AppTheme.warning, size: 18),
              SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Расчёт является предварительным и основан на средних рыночных ценах в регионе',
                  style: TextStyle(color: AppTheme.textSecondary, fontSize: 12, height: 1.4),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),
        CostBreakdownCard(items: _result.costs),
        const SizedBox(height: 16),
        Row(
          children: [
            Expanded(
              child: OutlinedButton.icon(
                onPressed: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Генерация PDF отчёта...')),
                  );
                },
                icon: const Icon(Icons.picture_as_pdf_rounded, size: 18),
                label: const Text('PDF отчёт'),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: ElevatedButton.icon(
                onPressed: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Отправка отчёта...')),
                  );
                },
                icon: const Icon(Icons.send_rounded, size: 18),
                label: const Text('Отправить'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 20),
      ],
    );
  }

  // ── Tab 5: Repair Materials (Ведомость) ──

  Widget _buildRepairMaterialsTab() {
    final estimate = _result.repairEstimate;
    return ListView(
      padding: const EdgeInsets.all(20),
      physics: const BouncingScrollPhysics(),
      children: [
        // Materials section
        const Row(
          children: [
            Icon(Icons.inventory_2_rounded, color: AppTheme.accent, size: 22),
            SizedBox(width: 10),
            Text('Необходимые материалы', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
          ],
        ),
        const SizedBox(height: 4),
        Text('${estimate.materials.length} наименований • запас 10%', style: const TextStyle(fontSize: 12, color: AppTheme.textSecondary)),
        const SizedBox(height: 12),
        ...estimate.materials.map((m) => _buildRepairMaterialCard(m)),

        const SizedBox(height: 24),
        // Labor section
        const Row(
          children: [
            Icon(Icons.engineering_rounded, color: AppTheme.info, size: 22),
            SizedBox(width: 10),
            Text('Работы', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
          ],
        ),
        const SizedBox(height: 12),
        ...estimate.labor.map((l) => _buildLaborCard(l)),

        const SizedBox(height: 24),
        // Grand total
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            gradient: const LinearGradient(colors: [AppTheme.accent, Color(0xFFFF8F00)]),
            borderRadius: BorderRadius.circular(14),
            boxShadow: [BoxShadow(color: AppTheme.accent.withValues(alpha: 0.3), blurRadius: 12, offset: const Offset(0, 4))],
          ),
          child: Column(
            children: [
              _summaryRow('Материалы', estimate.materialsTotal),
              _summaryRow('Работы', estimate.laborTotal),
              _summaryRow('Леса', estimate.scaffoldingTotal),
              _summaryRow('НДС 12%', estimate.vatAmount),
              const Divider(color: Colors.white24, height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text('Итого', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: Colors.white)),
                  Text(_formatCurrency(estimate.grandTotal), style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w800, color: Colors.white)),
                ],
              ),
            ],
          ),
        ),
        const SizedBox(height: 20),
      ],
    );
  }

  Widget _summaryRow(String label, double amount) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(fontSize: 13, color: Colors.white.withValues(alpha: 0.8))),
          Text(_formatCurrency(amount), style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: Colors.white.withValues(alpha: 0.9))),
        ],
      ),
    );
  }

  Widget _buildRepairMaterialCard(RepairMaterialItem item) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surfaceCard,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.surfaceLight.withValues(alpha: 0.2)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(item.name, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: AppTheme.textPrimary)),
                const SizedBox(height: 4),
                Text(
                  '${item.quantity.toStringAsFixed(1)} ${item.unit} × ${_formatCurrency(item.pricePerUnit)}',
                  style: const TextStyle(fontSize: 11, color: AppTheme.textSecondary),
                ),
              ],
            ),
          ),
          Text(
            _formatCurrency(item.totalCost),
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: AppTheme.accent),
          ),
        ],
      ),
    );
  }

  Widget _buildLaborCard(LaborItem item) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.surfaceCard,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.info.withValues(alpha: 0.15)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(item.name, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: AppTheme.textPrimary)),
                const SizedBox(height: 4),
                Text(
                  '${item.quantity} ${item.unit} • ${item.normHours.toStringAsFixed(0)} ч',
                  style: const TextStyle(fontSize: 11, color: AppTheme.textSecondary),
                ),
              ],
            ),
          ),
          Text(
            _formatCurrency(item.totalCost),
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: AppTheme.info),
          ),
        ],
      ),
    );
  }

  String _formatCurrency(double amount) {
    final formatted = amount.toStringAsFixed(0);
    final buffer = StringBuffer();
    int count = 0;
    for (int i = formatted.length - 1; i >= 0; i--) {
      buffer.write(formatted[i]);
      count++;
      if (count % 3 == 0 && i > 0) buffer.write(' ');
    }
    return '${buffer.toString().split('').reversed.join()} Р';
  }
}

// Custom painter for processed image previews
class _ProcessedImagePainter extends CustomPainter {
  final String type;
  _ProcessedImagePainter({required this.type});

  @override
  void paint(Canvas canvas, Size size) {
    final rng = math.Random(type.hashCode);
    final bgPaint = Paint()
      ..shader = LinearGradient(
        begin: Alignment.topLeft, end: Alignment.bottomRight,
        colors: [const Color(0xFF1A237E), const Color(0xFF283593)],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), bgPaint);

    final buildingPaint = Paint()..color = const Color(0xFF78909C);
    canvas.drawRect(Rect.fromLTWH(size.width * 0.1, size.height * 0.15, size.width * 0.8, size.height * 0.85), buildingPaint);

    switch (type) {
      case 'heatmap': _paintHeatmap(canvas, size, rng); break;
      case 'defects': _paintDefects(canvas, size, rng); break;
      case 'segments': _paintSegments(canvas, size, rng); break;
      case 'overlay': _paintOverlay(canvas, size, rng); break;
    }
  }

  void _paintHeatmap(Canvas canvas, Size size, math.Random rng) {
    for (int i = 0; i < 30; i++) {
      final x = size.width * 0.1 + rng.nextDouble() * size.width * 0.8;
      final y = size.height * 0.15 + rng.nextDouble() * size.height * 0.75;
      final r = 15 + rng.nextDouble() * 40;
      final severity = rng.nextDouble();
      final color = Color.lerp(Colors.green.withValues(alpha: 0.3), Colors.red.withValues(alpha: 0.5), severity)!;
      canvas.drawCircle(Offset(x, y), r, Paint()..color = color);
    }
  }

  void _paintDefects(Canvas canvas, Size size, math.Random rng) {
    final defectPaint = Paint()..color = Colors.red..style = PaintingStyle.stroke..strokeWidth = 2;
    for (int i = 0; i < 8; i++) {
      final x = size.width * 0.15 + rng.nextDouble() * size.width * 0.65;
      final y = size.height * 0.2 + rng.nextDouble() * size.height * 0.6;
      final w = 20 + rng.nextDouble() * 50;
      final h = 15 + rng.nextDouble() * 40;
      canvas.drawRRect(RRect.fromRectAndRadius(Rect.fromLTWH(x, y, w, h), const Radius.circular(4)), defectPaint);
      canvas.drawRRect(RRect.fromRectAndRadius(Rect.fromLTWH(x, y, w, h), const Radius.circular(4)), Paint()..color = Colors.red.withValues(alpha: 0.15));
    }
  }

  void _paintSegments(Canvas canvas, Size size, math.Random rng) {
    final colors = [Colors.blue.withValues(alpha: 0.3), Colors.green.withValues(alpha: 0.3), Colors.orange.withValues(alpha: 0.3), Colors.purple.withValues(alpha: 0.3)];
    final segH = size.height * 0.85 / colors.length;
    for (int i = 0; i < colors.length; i++) {
      canvas.drawRect(Rect.fromLTWH(size.width * 0.1, size.height * 0.15 + i * segH, size.width * 0.8, segH), Paint()..color = colors[i]);
    }
  }

  void _paintOverlay(Canvas canvas, Size size, math.Random rng) {
    final zones = [
      Rect.fromLTWH(size.width * 0.15, size.height * 0.2, size.width * 0.3, size.height * 0.25),
      Rect.fromLTWH(size.width * 0.5, size.height * 0.5, size.width * 0.35, size.height * 0.2),
    ];
    for (final z in zones) {
      canvas.drawRRect(RRect.fromRectAndRadius(z, const Radius.circular(8)), Paint()..color = Colors.amber.withValues(alpha: 0.3));
      canvas.drawRRect(RRect.fromRectAndRadius(z, const Radius.circular(8)), Paint()..color = Colors.amber..style = PaintingStyle.stroke..strokeWidth = 2);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
