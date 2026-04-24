import 'package:flutter/material.dart';

import '../../domain/models/mask_layer.dart';
import '../../theme/app_theme.dart';

/// Compact grouped list of layer toggles shown next to the [MaskLayerViewer].
/// Adapts to tight viewports by wrapping to a bottom-sheet on narrow screens.
class LayerPanel extends StatelessWidget {
  const LayerPanel({
    super.key,
    required this.layers,
    required this.onToggle,
    required this.onPreset,
  });

  final List<MaskLayer> layers;
  final void Function(String id, bool visible) onToggle;
  final void Function(LayerPreset preset) onPreset;

  @override
  Widget build(BuildContext context) {
    final byGroup = <MaskGroup, List<MaskLayer>>{};
    for (final l in layers) {
      byGroup.putIfAbsent(l.group, () => []).add(l);
    }

    return Container(
      decoration: BoxDecoration(
        color: AppTheme.surfaceCard,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.surfaceLight.withOpacity(0.3)),
      ),
      child: ListView(
        shrinkWrap: true,
        padding: const EdgeInsets.all(12),
        children: [
          Text('Слои', style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 8),
          Wrap(
            spacing: 6,
            children: [
              _chip(context, 'Всё', LayerPreset.all),
              _chip(context, 'Только дефекты', LayerPreset.defectsOnly),
              _chip(context, 'Только материалы', LayerPreset.materialsOnly),
              _chip(context, 'Выключить', LayerPreset.none),
            ],
          ),
          const SizedBox(height: 8),
          for (final group in MaskGroup.values)
            if ((byGroup[group] ?? []).isNotEmpty) ...[
              Padding(
                padding: const EdgeInsets.only(top: 12, bottom: 4),
                child: Text(
                  group.labelRu,
                  style: Theme.of(context).textTheme.labelLarge?.copyWith(
                        color: AppTheme.textSecondary,
                      ),
                ),
              ),
              for (final layer in byGroup[group]!)
                _LayerTile(
                  layer: layer,
                  onToggle: (v) => onToggle(layer.id, v),
                ),
            ],
        ],
      ),
    );
  }

  Widget _chip(BuildContext context, String label, LayerPreset preset) {
    return ActionChip(
      label: Text(label),
      onPressed: () => onPreset(preset),
    );
  }
}

enum LayerPreset { all, defectsOnly, materialsOnly, none }

class _LayerTile extends StatelessWidget {
  const _LayerTile({required this.layer, required this.onToggle});
  final MaskLayer layer;
  final ValueChanged<bool> onToggle;

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () => onToggle(!layer.visible),
      borderRadius: BorderRadius.circular(8),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        child: Row(
          children: [
            Container(
              width: 14,
              height: 14,
              decoration: BoxDecoration(
                color: layer.tint,
                borderRadius: BorderRadius.circular(3),
                border: Border.all(color: Colors.white24),
              ),
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                layer.nameRu,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            Switch(
              value: layer.visible,
              onChanged: onToggle,
            ),
          ],
        ),
      ),
    );
  }
}
