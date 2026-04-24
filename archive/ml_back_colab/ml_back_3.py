# ====================================================================
# ЯЧЕЙКА 3: Z-INDEX АНАЛИЗ МАТЕРИАЛОВ ФАСАДА (AlegroCode 2.1)
# ====================================================================
# 1. Устанавливаем и импортируем нужные библиотеки (делаем ячейку автономной)
!pip install transformers accelerate git+https://github.com/facebookresearch/sam2.git -q
!pip install -U ultralytics "rembg[cpu]" -q

import cv2
import torch
import gc
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from rembg import remove

print("=== 1. ПОДГОТОВКА ИЗОЛИРОВАННОЙ СТЕНЫ ===")
torch.cuda.empty_cache()
gc.collect()

if 'img_norm_rgb' not in locals() or 'geom_masks' not in locals():
    raise ValueError("ОШИБКА: Запустите Ячейку 1 для получения базовых данных!")

original_size = img_norm_rgb.shape[:2]

# Восстанавливаем идеальный силуэт (если запускаем ячейку отдельно)
if 'true_silhouette' not in locals():
    print("-> Восстановление силуэта (U-2-Net)...")
    rgba_img = remove(img_norm_rgb)
    true_silhouette = rgba_img[:, :, 3] > 128 

# Вычитаем элементы, чтобы анализировать ТОЛЬКО стену
elements_mask = np.zeros(original_size, dtype=bool)
for key in ["window", "door", "balcony", "roof", "molding"]:
    if key in geom_masks:
        elements_mask = elements_mask | geom_masks[key]

# Голая стена = Идеальный Силуэт МИНУС элементы
bare_wall_mask = true_silhouette & ~elements_mask

print("=== 2. СКАНИРОВАНИЕ ЧИСТЫХ ТЕКСТУР (CLIPSeg) ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

# --- ИСПРАВЛЕННЫЕ ПРОМПТЫ: Чистые текстуры ---
# Промпты упрощены для устранения путаницы с Moldings
MATERIAL_PROMPTS = [
    "grey concrete stone stone wall base",     # Бетон/камень цоколя
    "terracotta red brick masonry wall",      # Оголенный красный кирпич
    "plain smooth plaster wall facade",       # Окрашенная гладкая штукатурка
    "architectural molding cash decoration"    # Декоративная лепнина/карнизы
]

CLASS_MAP_MATERIALS = {
    "concrete": ["grey concrete stone"],
    "brick": ["terracotta red brick"],
    "plaster": ["plain smooth plaster"],
    "molding": ["architectural molding"]
}

# Архитектурная палитра для чертежа (Пастельные тона)
COLORS_MATERIALS = {
    "concrete": [127, 140, 141], # Серый бетон
    "brick": [211, 84, 0],       # Терракотовый кирпичный
    "plaster": [243, 214, 115],  # Песочно-желтый
    "molding": [236, 240, 241]   # Светло-серый/Белый
}

def get_mat_class(raw_label):
    for base, synonyms in CLASS_MAP_MATERIALS.items():
        if any(syn in raw_label.lower() for syn in synonyms):
            return base
    return None

inputs = clipseg_processor(
    text=MATERIAL_PROMPTS, 
    images=[Image.fromarray(img_norm_rgb)] * len(MATERIAL_PROMPTS), 
    padding=True, 
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = clipseg_model(**inputs)

preds = torch.nn.functional.interpolate(
    outputs.logits.unsqueeze(1), size=original_size, mode="bilinear", align_corners=False
).squeeze(1)

# Тепловые вероятности материалов (0-1)
prob_materials = torch.sigmoid(preds).cpu().numpy() 

print("=== 3. НАРЕЗКА И ИДЕНТИФИКАЦИЯ ТЕКСТУР (SAM 2) ===")
weights_path_sam2 = "sam2_hiera_small.pt"
if not os.path.exists(weights_path_sam2):
    !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt

sam2_model = build_sam2("sam2_hiera_s.yaml", weights_path_sam2, device=device)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model, points_per_side=32, pred_iou_thresh=0.8, stability_score_thresh=0.8, min_mask_region_area=200
)

with torch.no_grad():
    masks_amg = mask_generator.generate(img_norm_rgb)

# Маски с чисто CLIPSeg-детекцией
raw_material_masks = {k: np.zeros(original_size, dtype=bool) for k in CLASS_MAP_MATERIALS.keys()}

print(f"-> Идентификация {len(masks_amg)} физических лоскутов...")
for ann in masks_amg:
    m = ann['segmentation']
    
    # Игнорируем лоскуты вне голой стены
    if not np.any(m & bare_wall_mask):
        continue
        
    avg_scores = [prob_materials[i][m].mean() for i in range(len(MATERIAL_PROMPTS))]
    best_idx = np.argmax(avg_scores)
    
    # Снизил порог до 0.18, чтобы нейросеть лучше цеплялась за цоколь и кирпич
    if avg_scores[best_idx] > 0.18:
        class_name = get_mat_class(MATERIAL_PROMPTS[best_idx])
        if class_name:
            raw_material_masks[class_name] |= (m & bare_wall_mask)

# Очистка памяти
del sam2_model, mask_generator, clipseg_model, clipseg_processor, inputs, outputs
torch.cuda.empty_cache()
gc.collect()

# === 4. УЛЬТИМАТИВНАЯ Z-INDEX СБОРКА И ОТЧЕТ (BIM Architecture) ===
print("=== 4. ИЕРАРХИЧЕСКАЯ СБОРКА BIM-КАРТЫ ===")
mask_only_canvas = np.zeros((*original_size, 3), dtype=np.uint8)
final_masks_materials = {k: np.zeros(original_size, dtype=bool) for k in CLASS_MAP_MATERIALS.keys()}
found_materials = []

# Сборка контекста (Окна/Двери)
context_mask = elements_mask & true_silhouette
mask_only_canvas[context_mask] = [40, 40, 40]

# --- УСИЛЕНИЕ КИРПИЧА (Гениальная идея) ---
# Мы берем маску оголенного кирпича из Ячейки 2 и объединяем с тем, что нашел CLIPSeg
mask_brick_reinforced = raw_material_masks["brick"] | final_wall_defect_masks.get("exposed_brick", np.zeros(original_size, dtype=bool))


# --- Z-INDEX СБОРКА (The Z-Stack) ---
# Слой 1: Бетон (Цоколь) - самый нижний слой
draw_order_materials = ["concrete"]
for mat in draw_order_materials:
    m = raw_material_masks[mat] & bare_wall_mask
    if np.any(m):
        found_materials.append(mat)
        final_masks_materials[mat] |= m
        mask_only_canvas[m] = np.array(COLORS_MATERIALS[mat], dtype=np.uint8)

# Слой 2: Штукатурка (Основной фон) - ложится на всё здание (на всё, что не бетон)
plaster_m = bare_wall_mask & ~final_masks_materials["concrete"]
if np.any(plaster_m):
    found_materials.append("plaster")
    final_masks_materials["plaster"] |= plaster_m
    mask_only_canvas[plaster_m] = np.array(COLORS_MATERIALS["plaster"], dtype=np.uint8)

# Слой 3: Кирпич (Обнаженный) - вытесняет штукатурку (Z-Index Кирпича < Z-Index Штукатурки)
brick_m = mask_brick_reinforced & plaster_m
if np.any(brick_m):
    found_materials.append("brick")
    final_masks_materials["brick"] |= brick_m
    mask_only_canvas[brick_m] = np.array(COLORS_MATERIALS["brick"], dtype=np.uint8)

# Слой 4: Лепнина (Карнизы/Наличники) - ложится ПОВЕРХ ВСЕГО (ПОВЕРХ кирпича и штукатурки)
# Это гарантирует, что лепнина наличников будет видна
draw_order_molding = ["molding"]
for mat in draw_order_molding:
    m = raw_material_masks[mat] & (plaster_m | brick_m)
    if np.any(m):
        found_materials.append(mat)
        final_masks_materials[mat] |= m
        mask_only_canvas[m] = np.array(COLORS_MATERIALS[mat], dtype=np.uint8)


overlay_canvas = img_norm_rgb.copy()
overlay_canvas = cv2.addWeighted(overlay_canvas, 0.3, mask_only_canvas, 0.7, 0)

fig, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].imshow(overlay_canvas)
axes[0].set_title("1. Наложение Карты Материалов (Z-Index Stack)", fontsize=16, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(mask_only_canvas)
axes[1].set_title("2. Идеальная BIM-Карта Материалов", fontsize=16, fontweight='bold')
axes[1].axis('off')

legend_elements = [Patch(facecolor=np.array(COLORS_MATERIALS[c])/255, label=c.title()) for c in reversed(found_materials)]
if np.any(context_mask): 
    legend_elements.append(Patch(facecolor=[40/255, 40/255, 40/255], label="Building Elements (Windows/Doors)"))
axes[1].legend(handles=legend_elements, loc='upper right', fontsize=12, facecolor='black', labelcolor='white')

plt.tight_layout()
plt.show()

# --- ОТЧЕТ ПО ПЛОЩАДЯМ МАТЕРИАЛОВ ---
print("\n=== 🧱 ВЕДОМОСТЬ ОТДЕЛКИ ФАСАДА (AlegroCode Material Report) ===")
total_wall_area = bare_wall_mask.sum()
for mat_name in found_materials:
    mat_area = final_masks_materials[mat_name].sum()
    percent = (mat_area / total_wall_area) * 100
    print(f"  • {mat_name.title()}: {mat_area} px ({percent:.1f}% от площади стены)")

print("\n✅ Ультимативный Z-Index анализ строительных материалов завершен!")