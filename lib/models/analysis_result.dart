import 'package:flutter/material.dart';

import '../domain/models/mask_layer.dart';

class DamageInfo {
  final String type;
  final String rawType;
  final double percentage;
  final String severity;
  final String description;
  final double areaM2;
  final List<String> affectedLayers;
  final String? crackDepth;

  const DamageInfo({
    required this.type,
    required this.rawType,
    required this.percentage,
    required this.severity,
    required this.description,
    this.areaM2 = 0,
    this.affectedLayers = const ['finish'],
    this.crackDepth,
  });

  factory DamageInfo.fromJson(Map<String, dynamic> json) {
    return DamageInfo(
      type: json['type_display'] ?? json['type'] ?? '',
      rawType: (json['type'] ?? '').toString(),
      percentage: (json['percentage'] ?? 0).toDouble(),
      severity: json['severity_display'] ?? json['severity'] ?? '',
      description: json['description'] ?? '',
      areaM2: (json['area_m2'] ?? 0).toDouble(),
      affectedLayers: json['affected_layers'] != null
          ? List<String>.from(json['affected_layers'])
          : const ['finish'],
      crackDepth: json['crack_depth'],
    );
  }
}

class MaterialInfo {
  final String name;
  final String rawName;
  final double percentage;
  final double areaM2;
  final String condition;
  final IconData iconData;

  const MaterialInfo({
    required this.name,
    required this.rawName,
    required this.percentage,
    required this.condition,
    required this.iconData,
    this.areaM2 = 0,
  });

  factory MaterialInfo.fromJson(Map<String, dynamic> json) {
    return MaterialInfo(
      name: json['name_display'] ?? json['name'] ?? '',
      rawName: (json['name'] ?? '').toString(),
      percentage: (json['percentage'] ?? 0).toDouble(),
      areaM2: (json['area_m2'] ?? 0).toDouble(),
      condition: json['condition'] ?? '',
      iconData: _iconForMaterial(json['name'] ?? ''),
    );
  }

  static IconData _iconForMaterial(String name) {
    final n = name.toLowerCase();
    if (n.contains('brick')) return Icons.grid_view_rounded;
    if (n.contains('concrete')) return Icons.square_rounded;
    if (n.contains('plaster') || n.contains('штукатурка')) return Icons.format_paint_rounded;
    if (n.contains('metal') || n.contains('металл')) return Icons.settings_rounded;
    if (n.contains('wood') || n.contains('дерево')) return Icons.park_rounded;
    if (n.contains('glass') || n.contains('стекло')) return Icons.window_rounded;
    if (n.contains('molding') || n.contains('лепнина')) return Icons.architecture_rounded;
    if (n.contains('tile') || n.contains('плитка')) return Icons.dashboard_rounded;
    if (n.contains('paint') || n.contains('краска')) return Icons.brush_rounded;
    return Icons.category_rounded;
  }
}

class CostItem {
  final String category;
  final String description;
  final double cost;
  final String unit;

  const CostItem({
    required this.category,
    required this.description,
    required this.cost,
    required this.unit,
  });

  factory CostItem.fromJson(Map<String, dynamic> json) {
    return CostItem(
      category: json['category'] ?? '',
      description: json['description'] ?? '',
      cost: (json['cost'] ?? 0).toDouble(),
      unit: json['unit'] ?? '₽',
    );
  }
}

class RepairMaterialItem {
  final String name;
  final String unit;
  final double quantity;
  final double pricePerUnit;
  final double totalCost;

  const RepairMaterialItem({
    required this.name,
    required this.unit,
    required this.quantity,
    required this.pricePerUnit,
    required this.totalCost,
  });

  factory RepairMaterialItem.fromJson(Map<String, dynamic> json) {
    return RepairMaterialItem(
      name: json['display'] ?? json['name_display'] ?? json['name'] ?? '',
      unit: json['unit'] ?? '',
      quantity: (json['quantity'] ?? 0).toDouble(),
      pricePerUnit: (json['price_per_unit'] ?? 0).toDouble(),
      totalCost: (json['total_cost'] ?? 0).toDouble(),
    );
  }
}

class LaborItem {
  final String name;
  final String unit;
  final double quantity;
  final double pricePerUnit;
  final double totalCost;
  final double normHours;

  const LaborItem({
    required this.name,
    required this.unit,
    required this.quantity,
    required this.pricePerUnit,
    required this.totalCost,
    required this.normHours,
  });

  factory LaborItem.fromJson(Map<String, dynamic> json) {
    return LaborItem(
      name: json['display'] ?? json['name_display'] ?? json['name'] ?? '',
      unit: json['unit'] ?? '',
      quantity: (json['quantity'] ?? 0).toDouble(),
      pricePerUnit: (json['price_per_unit'] ?? 0).toDouble(),
      totalCost: (json['total_cost'] ?? 0).toDouble(),
      normHours: (json['norm_hours'] ?? 0).toDouble(),
    );
  }
}

class RepairEstimate {
  final String currencySymbol;
  final List<RepairMaterialItem> materials;
  final List<LaborItem> labor;
  final double materialsTotal;
  final double laborTotal;
  final double scaffoldingTotal;
  final double subtotal;
  final double vatAmount;
  final double vatRate;
  final double grandTotal;
  final double totalWorkHours;
  final int estimatedDays;
  final double wasteFactor;
  final List<CostItem> costsForFlutter;

  const RepairEstimate({
    required this.currencySymbol,
    required this.materials,
    required this.labor,
    required this.materialsTotal,
    required this.laborTotal,
    required this.scaffoldingTotal,
    required this.subtotal,
    required this.vatAmount,
    required this.vatRate,
    required this.grandTotal,
    required this.totalWorkHours,
    required this.estimatedDays,
    required this.wasteFactor,
    required this.costsForFlutter,
  });

  factory RepairEstimate.fromJson(Map<String, dynamic> json) {
    final summary = json['summary'] ?? const {};
    return RepairEstimate(
      currencySymbol: json['currency_symbol'] ?? '₽',
      materials: (json['materials'] as List? ?? [])
          .map((m) => RepairMaterialItem.fromJson(m))
          .toList(),
      labor: (json['labor'] as List? ?? [])
          .map((l) => LaborItem.fromJson(l))
          .toList(),
      materialsTotal: (summary['materials_total'] ?? 0).toDouble(),
      laborTotal: (summary['labor_total'] ?? 0).toDouble(),
      scaffoldingTotal: (summary['scaffolding_total'] ?? 0).toDouble(),
      subtotal: (summary['subtotal'] ?? 0).toDouble(),
      vatAmount: (summary['vat_amount'] ?? 0).toDouble(),
      vatRate: (summary['vat_rate'] ?? 0.20).toDouble(),
      grandTotal: (summary['grand_total'] ?? 0).toDouble(),
      totalWorkHours: (summary['total_hours'] ?? summary['total_work_hours'] ?? 0).toDouble(),
      estimatedDays: (summary['estimated_days'] ?? 1).toInt(),
      wasteFactor: (summary['waste_factor'] ?? 1.10).toDouble(),
      costsForFlutter: (json['costs_for_flutter'] as List? ?? [])
          .map((c) => CostItem.fromJson(c))
          .toList(),
    );
  }

  static RepairEstimate mock() {
    return const RepairEstimate(
      currencySymbol: '₽',
      materials: [
        RepairMaterialItem(name: 'Штукатурка фасадная', unit: 'кг', quantity: 528.0, pricePerUnit: 48, totalCost: 25344),
        RepairMaterialItem(name: 'Сетка армирующая', unit: 'м²', quantity: 36.3, pricePerUnit: 130, totalCost: 4719),
        RepairMaterialItem(name: 'Грунтовка фасадная', unit: 'л', quantity: 9.9, pricePerUnit: 380, totalCost: 3762),
        RepairMaterialItem(name: 'Шпатлёвка фасадная', unit: 'кг', quantity: 29.7, pricePerUnit: 65, totalCost: 1931),
        RepairMaterialItem(name: 'Краска фасадная', unit: 'л', quantity: 24.75, pricePerUnit: 520, totalCost: 12870),
      ],
      labor: [
        LaborItem(name: 'Восстановление штукатурного слоя', unit: 'м²', quantity: 30.0, pricePerUnit: 620, totalCost: 18600, normHours: 60.0),
        LaborItem(name: 'Заделка поверхностных трещин', unit: 'м', quantity: 42.0, pricePerUnit: 520, totalCost: 21840, normHours: 33.6),
        LaborItem(name: 'Биоцидная обработка', unit: 'м²', quantity: 18.0, pricePerUnit: 220, totalCost: 3960, normHours: 5.4),
      ],
      materialsTotal: 48626,
      laborTotal: 44400,
      scaffoldingTotal: 72000,
      subtotal: 165026,
      vatAmount: 33005,
      vatRate: 0.20,
      grandTotal: 198031,
      totalWorkHours: 99.0,
      estimatedDays: 13,
      wasteFactor: 1.10,
      costsForFlutter: [
        CostItem(category: 'Строительные материалы', description: '5 наименований с учётом запаса 10%', cost: 48626, unit: '₽'),
        CostItem(category: 'Восстановление штукатурного слоя', description: '30.0 м²', cost: 18600, unit: '₽'),
        CostItem(category: 'Заделка поверхностных трещин', description: '42.0 м', cost: 21840, unit: '₽'),
        CostItem(category: 'Биоцидная обработка', description: '18.0 м²', cost: 3960, unit: '₽'),
        CostItem(category: 'Леса и оборудование', description: 'Монтаж/демонтаж — 4 эт.', cost: 72000, unit: '₽'),
        CostItem(category: 'НДС 20%', description: 'Налог на добавленную стоимость', cost: 33005, unit: '₽'),
      ],
    );
  }
}

class ProcessedImage {
  final String title;
  final String description;
  final String type;
  final String? url;

  const ProcessedImage({
    required this.title,
    required this.description,
    required this.type,
    this.url,
  });
}

class MaskCatalog {
  final String baseImageUrl;
  final Map<String, String> geometry;
  final Map<String, String> materials;
  final Map<String, String> defects;
  final Map<String, String> visualizations;

  const MaskCatalog({
    required this.baseImageUrl,
    this.geometry = const {},
    this.materials = const {},
    this.defects = const {},
    this.visualizations = const {},
  });

  factory MaskCatalog.fromJson(Map<String, dynamic>? json) {
    if (json == null) {
      return const MaskCatalog(baseImageUrl: '');
    }
    Map<String, String> asMap(dynamic raw) => (raw as Map?)?.map(
          (k, v) => MapEntry(k.toString(), v.toString()),
        ) ?? const {};
    return MaskCatalog(
      baseImageUrl: (json['base_image'] ?? '').toString(),
      geometry: asMap(json['geometry']),
      materials: asMap(json['materials']),
      defects: asMap(json['defects']),
      visualizations: asMap(json['visualizations']),
    );
  }

  List<MaskLayer> toLayers() {
    final layers = <MaskLayer>[];
    geometry.forEach((k, url) => layers.add(MaskLayer(
          id: 'geom_$k',
          group: MaskGroup.geometry,
          nameRu: _geomLabel(k),
          url: url,
          tint: _geomTint(k),
          visible: false,
        )));
    materials.forEach((k, url) => layers.add(MaskLayer(
          id: 'mat_$k',
          group: MaskGroup.materials,
          nameRu: _materialLabel(k),
          url: url,
          tint: _materialTint(k),
          visible: false,
        )));
    defects.forEach((k, url) => layers.add(MaskLayer(
          id: 'def_$k',
          group: MaskGroup.defects,
          nameRu: _defectLabel(k),
          url: url,
          tint: _defectTint(k),
          visible: true,
        )));
    return layers;
  }

  static String _geomLabel(String k) => switch (k) {
        'window' => 'Окна',
        'door' => 'Двери',
        'balcony' => 'Балконы',
        'roof' => 'Крыша',
        'foundation' => 'Фундамент',
        'pipe' => 'Водостоки',
        'chimney' => 'Дымоходы',
        'fence' => 'Ограждения',
        'ac_unit' => 'Кондиционеры',
        'molding' => 'Лепнина',
        _ => k,
      };
  static String _materialLabel(String k) => switch (k) {
        'brick' => 'Кирпич',
        'concrete' => 'Бетон',
        'cement_plaster' => 'Штукатурка',
        'decorative_plaster' => 'Декоративная штукатурка',
        'wood' => 'Дерево',
        'metal' => 'Металл',
        'glass' => 'Стекло',
        'painted_surface' => 'Окрашенная поверхность',
        'ceramic_tile' => 'Плитка',
        'molding' => 'Лепнина',
        _ => k,
      };
  static String _defectLabel(String k) => switch (k) {
        'crack' => 'Трещины',
        'peeling' => 'Отслоение',
        'exposed_brick' => 'Оголённый кирпич',
        'water_damage' => 'Протечки',
        'rust' => 'Ржавчина',
        'rust_stain' => 'Ржавые подтёки',
        'moss' => 'Мох',
        'efflorescence' => 'Высолы',
        'spalling' => 'Разрушение бетона',
        'wood_rot' => 'Гниение дерева',
        'mold' => 'Плесень / протечки',
        'glass_crack' => 'Трещины стекла',
        'broken_glass' => 'Разбитое стекло',
        'damaged_wood' => 'Повреждённое дерево',
        'rusty_metal' => 'Ржавый металл',
        'damaged_railing' => 'Повреждения перил',
        _ => k,
      };
  static Color _geomTint(String k) => const Color(0x664a90e2);
  static Color _materialTint(String k) => switch (k) {
        'brick' => const Color(0xAAD35400),
        'wood' => const Color(0xAA8B5A2B),
        'metal' => const Color(0xAAA9A9BE),
        'glass' => const Color(0xAA87CEFA),
        _ => const Color(0xAA7F8C8D),
      };
  static Color _defectTint(String k) => switch (k) {
        'crack' => const Color(0xAAFF4136),
        'peeling' => const Color(0xAAF1C40F),
        'exposed_brick' => const Color(0xAAE67E22),
        'mold' => const Color(0xAA4B0082),
        'wood_rot' => const Color(0xAA5C3317),
        'rust_stain' => const Color(0xAAB7410E),
        'glass_crack' => const Color(0xAA64C8FF),
        _ => const Color(0xAADC143C),
      };
}

class AnalysisResult {
  final String? id;
  final double overallScore;
  final String overallCondition;
  final List<DamageInfo> damages;
  final List<MaterialInfo> materials;
  final List<CostItem> costs;
  final List<ProcessedImage> processedImages;
  final double totalArea;
  final double damagedArea;
  final RepairEstimate repairEstimate;
  final MaskCatalog masks;
  final String? priceSnapshotDate;
  final String? priceSource;
  final String? restoredUrl;
  final List<String> calibrationWarnings;

  const AnalysisResult({
    this.id,
    required this.overallScore,
    required this.overallCondition,
    required this.damages,
    required this.materials,
    required this.costs,
    required this.processedImages,
    required this.totalArea,
    required this.damagedArea,
    required this.repairEstimate,
    required this.masks,
    this.priceSnapshotDate,
    this.priceSource,
    this.restoredUrl,
    this.calibrationWarnings = const [],
  });

  double get totalCost => repairEstimate.grandTotal;
  bool get pricesAreStale => priceSource == 'yaml_fallback';

  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    final repair = json['repair_estimate'] is Map<String, dynamic>
        ? RepairEstimate.fromJson(json['repair_estimate'] as Map<String, dynamic>)
        : RepairEstimate.mock();

    final masks = MaskCatalog.fromJson(json['masks'] as Map<String, dynamic>?);

    final vizImages = masks.visualizations.entries.map((e) {
      return ProcessedImage(
        title: _vizTitle(e.key),
        description: _vizDescription(e.key),
        type: e.key,
        url: e.value,
      );
    }).toList();

    return AnalysisResult(
      id: json['id']?.toString(),
      overallScore: (json['overall_score'] ?? 0).toDouble(),
      overallCondition: json['overall_condition'] ?? '',
      totalArea: (json['total_area_m2'] ?? 0).toDouble(),
      damagedArea: (json['damaged_area_m2'] ?? 0).toDouble(),
      damages: (json['damages'] as List? ?? [])
          .map((d) => DamageInfo.fromJson(d as Map<String, dynamic>))
          .toList(),
      materials: (json['materials'] as List? ?? [])
          .map((m) => MaterialInfo.fromJson(m as Map<String, dynamic>))
          .toList(),
      costs: repair.costsForFlutter,
      processedImages: vizImages.isNotEmpty ? vizImages : _defaultVizDescriptions,
      repairEstimate: repair,
      masks: masks,
      priceSnapshotDate: json['price_snapshot_date']?.toString(),
      priceSource: json['price_source']?.toString(),
      restoredUrl: json['restored_url']?.toString(),
      calibrationWarnings: (json['calibration_warnings'] as List?)
              ?.map((e) => e.toString())
              .toList() ??
          const [],
    );
  }

  static AnalysisResult mock() {
    return AnalysisResult(
      overallScore: 67.5,
      overallCondition: 'Удовлетворительное',
      totalArea: 180.0,
      damagedArea: 42.5,
      damages: const [
        DamageInfo(
          type: 'Трещины',
          rawType: 'crack',
          percentage: 35.0,
          severity: 'Средняя',
          description: 'Микро- и макротрещины в штукатурке',
          areaM2: 14.9,
          affectedLayers: ['finish', 'base_plaster'],
          crackDepth: 'surface',
        ),
        DamageInfo(
          type: 'Отслоение штукатурки',
          rawType: 'peeling',
          percentage: 25.0,
          severity: 'Высокая',
          description: 'Отслоение финишного слоя',
          areaM2: 10.6,
          affectedLayers: ['finish', 'base_plaster'],
        ),
        DamageInfo(
          type: 'Высолы',
          rawType: 'efflorescence',
          percentage: 20.0,
          severity: 'Низкая',
          description: 'Белые солевые отложения',
          areaM2: 8.5,
          affectedLayers: ['finish'],
        ),
      ],
      materials: const [
        MaterialInfo(name: 'Кирпич', rawName: 'brick', percentage: 45.0, condition: 'Хорошее', iconData: Icons.grid_view_rounded),
        MaterialInfo(name: 'Штукатурка', rawName: 'cement_plaster', percentage: 30.0, condition: 'Удовлетворительное', iconData: Icons.format_paint_rounded),
        MaterialInfo(name: 'Бетон', rawName: 'concrete', percentage: 15.0, condition: 'Хорошее', iconData: Icons.square_rounded),
      ],
      costs: RepairEstimate.mock().costsForFlutter,
      processedImages: _defaultVizDescriptions,
      repairEstimate: RepairEstimate.mock(),
      masks: const MaskCatalog(baseImageUrl: ''),
      priceSnapshotDate: null,
      priceSource: 'yaml_fallback',
    );
  }

  static const _defaultVizDescriptions = <ProcessedImage>[
    ProcessedImage(title: 'Тепловая карта повреждений', description: 'Визуализация плотности дефектов', type: 'heatmap'),
    ProcessedImage(title: 'Выделенные дефекты', description: 'Обнаружение и маркировка дефектов', type: 'defects'),
    ProcessedImage(title: 'Сегментация материалов', description: 'Разбивка фасада по типам материалов', type: 'segments'),
    ProcessedImage(title: 'Зоны ремонта', description: 'Рекомендуемые области для ремонта', type: 'overlay'),
  ];

  static String _vizTitle(String type) => switch (type) {
        'heatmap' => 'Тепловая карта повреждений',
        'defects' => 'Выделенные дефекты',
        'segments' => 'Сегментация материалов',
        'overlay' => 'Зоны ремонта',
        _ => type,
      };

  static String _vizDescription(String type) => switch (type) {
        'heatmap' => 'Визуализация плотности дефектов',
        'defects' => 'Обнаружение и маркировка дефектов',
        'segments' => 'Разбивка фасада по типам материалов',
        'overlay' => 'Рекомендуемые области для ремонта',
        _ => '',
      };
}
