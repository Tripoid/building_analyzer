import 'package:flutter/material.dart';

class DamageInfo {
  final String type;
  final double percentage;
  final String severity;
  final String description;
  final List<String> affectedLayers;
  final String? crackDepth;

  const DamageInfo({
    required this.type,
    required this.percentage,
    required this.severity,
    required this.description,
    this.affectedLayers = const ['finish'],
    this.crackDepth,
  });

  factory DamageInfo.fromJson(Map<String, dynamic> json) {
    return DamageInfo(
      type: json['type_display'] ?? json['type'] ?? '',
      percentage: (json['percentage'] ?? 0).toDouble(),
      severity: json['severity_display'] ?? json['severity'] ?? '',
      description: json['description'] ?? '',
      affectedLayers: json['affected_layers'] != null
          ? List<String>.from(json['affected_layers'])
          : const ['finish'],
      crackDepth: json['crack_depth'],
    );
  }
}

class MaterialInfo {
  final String name;
  final double percentage;
  final String condition;
  final IconData iconData;

  const MaterialInfo({
    required this.name,
    required this.percentage,
    required this.condition,
    required this.iconData,
  });

  factory MaterialInfo.fromJson(Map<String, dynamic> json) {
    return MaterialInfo(
      name: json['name_display'] ?? json['name'] ?? '',
      percentage: (json['percentage'] ?? 0).toDouble(),
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
      unit: json['unit'] ?? '₸',
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
      name: json['name_display'] ?? json['name'] ?? '',
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
      name: json['name_display'] ?? json['name'] ?? '',
      unit: json['unit'] ?? '',
      quantity: (json['quantity'] ?? 0).toDouble(),
      pricePerUnit: (json['price_per_unit'] ?? 0).toDouble(),
      totalCost: (json['total_cost'] ?? 0).toDouble(),
      normHours: (json['norm_hours'] ?? 0).toDouble(),
    );
  }
}

class RepairEstimate {
  final List<RepairMaterialItem> materials;
  final List<LaborItem> labor;
  final double materialsTotal;
  final double laborTotal;
  final double scaffoldingTotal;
  final double subtotal;
  final double vatAmount;
  final double grandTotal;
  final double totalWorkHours;
  final int estimatedDays;
  final List<CostItem> costsForFlutter;

  const RepairEstimate({
    required this.materials,
    required this.labor,
    required this.materialsTotal,
    required this.laborTotal,
    required this.scaffoldingTotal,
    required this.subtotal,
    required this.vatAmount,
    required this.grandTotal,
    required this.totalWorkHours,
    required this.estimatedDays,
    required this.costsForFlutter,
  });

  factory RepairEstimate.fromJson(Map<String, dynamic> json) {
    final summary = json['summary'] ?? {};
    return RepairEstimate(
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
      grandTotal: (summary['grand_total'] ?? 0).toDouble(),
      totalWorkHours: (summary['total_work_hours'] ?? 0).toDouble(),
      estimatedDays: (summary['estimated_days'] ?? 1).toInt(),
      costsForFlutter: (json['costs_for_flutter'] as List? ?? [])
          .map((c) => CostItem.fromJson(c))
          .toList(),
    );
  }

  static RepairEstimate mock() {
    return const RepairEstimate(
      materials: [
        RepairMaterialItem(name: 'Штукатурка цементная фасадная', unit: 'кг', quantity: 528.0, pricePerUnit: 320, totalCost: 168960),
        RepairMaterialItem(name: 'Армирующая сетка фасадная', unit: 'м²', quantity: 36.3, pricePerUnit: 650, totalCost: 23595),
        RepairMaterialItem(name: 'Грунтовка глубокого проникновения', unit: 'л', quantity: 9.9, pricePerUnit: 1200, totalCost: 11880),
        RepairMaterialItem(name: 'Шпатлёвка фасадная', unit: 'кг', quantity: 29.7, pricePerUnit: 850, totalCost: 25245),
        RepairMaterialItem(name: 'Краска фасадная', unit: 'л', quantity: 24.75, pricePerUnit: 3500, totalCost: 86625),
        RepairMaterialItem(name: 'Антисоль (очиститель)', unit: 'л', quantity: 5.94, pricePerUnit: 2200, totalCost: 13068),
        RepairMaterialItem(name: 'Гидрофобизатор фасадный', unit: 'л', quantity: 4.95, pricePerUnit: 3800, totalCost: 18810),
        RepairMaterialItem(name: 'Антисептик фасадный', unit: 'л', quantity: 3.56, pricePerUnit: 1800, totalCost: 6413),
      ],
      labor: [
        LaborItem(name: 'Восстановление штукатурного слоя', unit: 'м²', quantity: 30.0, pricePerUnit: 4500, totalCost: 135000, normHours: 60.0),
        LaborItem(name: 'Заделка поверхностных трещин', unit: 'м.п.', quantity: 42.0, pricePerUnit: 1500, totalCost: 63000, normHours: 21.0),
        LaborItem(name: 'Удаление высолов и гидрофобизация', unit: 'м²', quantity: 18.0, pricePerUnit: 2000, totalCost: 36000, normHours: 14.4),
        LaborItem(name: 'Биоочистка и антисептирование', unit: 'м²', quantity: 10.8, pricePerUnit: 1800, totalCost: 19440, normHours: 7.56),
      ],
      materialsTotal: 354596,
      laborTotal: 253440,
      scaffoldingTotal: 340000,
      subtotal: 948036,
      vatAmount: 113764,
      grandTotal: 1061800,
      totalWorkHours: 134.96,
      estimatedDays: 17,
      costsForFlutter: [
        CostItem(category: 'Строительные материалы', description: '8 наименований с учётом запаса 10%', cost: 354596, unit: 'Р'),
        CostItem(category: 'Восстановление штукатурного слоя', description: '30.0 м²', cost: 135000, unit: 'Р'),
        CostItem(category: 'Заделка поверхностных трещин', description: '42.0 м.п.', cost: 63000, unit: 'Р'),
        CostItem(category: 'Удаление высолов и гидрофобизация', description: '18.0 м²', cost: 36000, unit: 'Р'),
        CostItem(category: 'Биоочистка и антисептирование', description: '10.8 м²', cost: 19440, unit: 'Р'),
        CostItem(category: 'Леса и оборудование', description: 'Монтаж/демонтаж строительных лесов', cost: 340000, unit: 'Р'),
        CostItem(category: 'НДС (12%)', description: 'Налог на добавленную стоимость', cost: 113764, unit: 'Р'),
      ],
    );
  }
}

class ProcessedImage {
  final String title;
  final String description;
  final String type;

  const ProcessedImage({
    required this.title,
    required this.description,
    required this.type,
  });
}

class AnalysisResult {
  final double overallScore;
  final String overallCondition;
  final List<DamageInfo> damages;
  final List<MaterialInfo> materials;
  final List<CostItem> costs;
  final List<ProcessedImage> processedImages;
  final double totalArea;
  final double damagedArea;
  final RepairEstimate repairEstimate;

  const AnalysisResult({
    required this.overallScore,
    required this.overallCondition,
    required this.damages,
    required this.materials,
    required this.costs,
    required this.processedImages,
    required this.totalArea,
    required this.damagedArea,
    required this.repairEstimate,
  });

  double get totalCost => costs.fold(0, (sum, item) => sum + item.cost);

  factory AnalysisResult.fromJson(Map<String, dynamic> json) {
    final repair = json['repair_estimate'] != null
        ? RepairEstimate.fromJson(json['repair_estimate'])
        : RepairEstimate.mock();

    return AnalysisResult(
      overallScore: (json['overall_score'] ?? 0).toDouble(),
      overallCondition: json['overall_condition'] ?? '',
      totalArea: (json['total_area_m2'] ?? 450.0).toDouble(),
      damagedArea: (json['damaged_area_m2'] ?? 0).toDouble(),
      damages: (json['damages'] as List? ?? [])
          .map((d) => DamageInfo.fromJson(d))
          .toList(),
      materials: (json['materials'] as List? ?? [])
          .map((m) => MaterialInfo.fromJson(m))
          .toList(),
      costs: repair.costsForFlutter,
      processedImages: const [
        ProcessedImage(title: 'Тепловая карта повреждений', description: 'Визуализация плотности дефектов на фасаде', type: 'heatmap'),
        ProcessedImage(title: 'Выделенные дефекты', description: 'Автоматическое обнаружение и маркировка дефектов', type: 'defects'),
        ProcessedImage(title: 'Сегментация материалов', description: 'Разбивка фасада по типам материалов', type: 'segments'),
        ProcessedImage(title: 'Зоны ремонта', description: 'Рекомендуемые области для первоочередного ремонта', type: 'overlay'),
      ],
      repairEstimate: repair,
    );
  }

  static AnalysisResult mock() {
    return AnalysisResult(
      overallScore: 67.5,
      overallCondition: 'Удовлетворительное',
      totalArea: 450.0,
      damagedArea: 146.25,
      damages: const [
        DamageInfo(
          type: 'Трещины',
          percentage: 35.0,
          severity: 'Средняя',
          description: 'Микро- и макротрещины в штукатурке, преимущественно в зоне окон',
          affectedLayers: ['finish', 'base_plaster'],
          crackDepth: 'surface',
        ),
        DamageInfo(
          type: 'Отслоение штукатурки',
          percentage: 25.0,
          severity: 'Высокая',
          description: 'Отслоение финишного слоя на 2-4 этажах',
          affectedLayers: ['finish', 'base_plaster'],
        ),
        DamageInfo(
          type: 'Высолы',
          percentage: 20.0,
          severity: 'Низкая',
          description: 'Белые солевые отложения в нижней части фасада',
          affectedLayers: ['finish'],
        ),
        DamageInfo(
          type: 'Биопоражение',
          percentage: 12.0,
          severity: 'Средняя',
          description: 'Мох и грибок на северной стороне',
          affectedLayers: ['finish'],
        ),
        DamageInfo(
          type: 'Механические повреждения',
          percentage: 8.0,
          severity: 'Низкая',
          description: 'Сколы и вмятины на цокольном уровне',
          affectedLayers: ['finish', 'base_plaster', 'structural'],
        ),
      ],
      materials: const [
        MaterialInfo(name: 'Кирпич керамический', percentage: 45.0, condition: 'Хорошее', iconData: Icons.grid_view_rounded),
        MaterialInfo(name: 'Штукатурка цементная', percentage: 30.0, condition: 'Удовлетворительное', iconData: Icons.format_paint_rounded),
        MaterialInfo(name: 'Бетон', percentage: 15.0, condition: 'Хорошее', iconData: Icons.square_rounded),
        MaterialInfo(name: 'Металл (элементы)', percentage: 7.0, condition: 'Требует покраски', iconData: Icons.settings_rounded),
        MaterialInfo(name: 'Дерево (оконные рамы)', percentage: 3.0, condition: 'Плохое', iconData: Icons.park_rounded),
      ],
      costs: RepairEstimate.mock().costsForFlutter,
      processedImages: const [
        ProcessedImage(title: 'Тепловая карта повреждений', description: 'Визуализация плотности дефектов на фасаде', type: 'heatmap'),
        ProcessedImage(title: 'Выделенные дефекты', description: 'Автоматическое обнаружение и маркировка дефектов', type: 'defects'),
        ProcessedImage(title: 'Сегментация материалов', description: 'Разбивка фасада по типам материалов', type: 'segments'),
        ProcessedImage(title: 'Зоны ремонта', description: 'Рекомендуемые области для первоочередного ремонта', type: 'overlay'),
      ],
      repairEstimate: RepairEstimate.mock(),
    );
  }
}
