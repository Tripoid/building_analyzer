# ====================================================================
# ЯЧЕЙКА 2: ИЕРАРХИЧЕСКОЕ МАСКИРОВАНИЕ (AlegroCode 7.1 - Стабильный U-2-Net)
# ====================================================================
!apt-get install aria2 -y -q
!pip install transformers accelerate git+https://github.com/facebookresearch/sam2.git -q
!pip install -U ultralytics modelscope "rembg[cpu]" -q

import cv2
import torch
import gc
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch
from ultralytics.models.sam import SAM3SemanticPredictor
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from rembg import remove

print("=== 1. ХИРУРГИЧЕСКАЯ ИЗОЛЯЦИЯ ФАСАДА (U-2-Net) ===")
torch.cuda.empty_cache()
gc.collect()

if 'img_norm_rgb' not in locals() or 'geom_masks' not in locals():
    raise ValueError("ОШИБКА: Запустите Ячейку 1, чтобы получить нормализованное фото и маски!")

original_size = img_norm_rgb.shape[:2]

print("-> Генерация идеального силуэта (отсечение фона)...")
# rembg возвращает RGBA, где альфа-канал - это идеальная маска здания
rgba_img = remove(img_norm_rgb)
true_silhouette = rgba_img[:, :, 3] > 128 

print("-> Создание маски 'Голая стена'...")
elements_mask = np.zeros(original_size, dtype=bool)
for key in ["window", "door", "balcony", "roof", "molding"]:
    if key in geom_masks:
        elements_mask = elements_mask | geom_masks[key]

# Голая стена = Идеальный Силуэт МИНУС элементы
bare_wall_mask = true_silhouette & ~elements_mask

isolated_bare_wall_img = img_norm_rgb.copy()
# Абсолютно черная заливка всего, что не является стеной (фон + окна)
isolated_bare_wall_img[~bare_wall_mask] = [0, 0, 0]

cv2.imwrite("isolated_wall.jpg", cv2.cvtColor(isolated_bare_wall_img, cv2.COLOR_RGB2BGR))

print("=== 2. СКАНИРОВАНИЕ ДЕФЕКТОВ СТЕН (SAM 3) ===")
weights_path_sam3 = "sam3.pt"
if not os.path.exists(weights_path_sam3):
    print("-> Запуск aria2c (16 потоков)...")
    !aria2c -x 16 -s 16 -k 1M "https://modelscope.cn/api/v1/models/facebook/sam3/repo?Revision=master&FilePath=sam3.pt" -o sam3.pt

if 'sam3_predictor' not in locals():
    overrides_sam3 = dict(conf=0.18, task="segment", mode="predict", model=weights_path_sam3, half=True)
    sam3_predictor = SAM3SemanticPredictor(overrides=overrides_sam3)

sam3_predictor.set_image("isolated_wall.jpg")

WALL_DEFECT_PROMPTS = [
    "deep structural wall crack", 
    "damaged peeling plaster surface", 
    "exposed red brick from damaged wall"
]

CLASS_MAP_WALL_DEFECTS = {
    "crack": ["crack"],
    "peeling": ["peeling plaster"],
    "exposed_brick": ["exposed red brick"]
}

COLORS_WALL_DEFECTS = {
    "crack": [255, 0, 0],           
    "exposed_brick": [230, 126, 34],
    "peeling": [241, 196, 15]       
}

def get_wall_defect_class(raw_label):
    for base, synonyms in CLASS_MAP_WALL_DEFECTS.items():
        if any(syn in raw_label.lower() for syn in synonyms):
            return base
    return None

print("-> Тотальное сканирование патологий стен...")
wall_defect_results = sam3_predictor(text=WALL_DEFECT_PROMPTS)
final_wall_defect_masks = {k: np.zeros(original_size, dtype=bool) for k in COLORS_WALL_DEFECTS.keys()}

for result in wall_defect_results:
    if result.masks is not None:
        for i, mask_data in enumerate(result.masks.data.cpu().numpy()):
            class_idx = int(result.boxes.cls[i])
            base_defect = get_wall_defect_class(WALL_DEFECT_PROMPTS[class_idx])
            
            if base_defect:
                mask_uint8 = mask_data.astype(np.uint8)
                mask_resized = cv2.resize(mask_uint8, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                
                # ЖЕСТКИЙ РОУТИНГ: Дефект стены может быть только на голой стене
                final_wall_defect_masks[base_defect] |= (mask_resized & bare_wall_mask)

del sam3_predictor
torch.cuda.empty_cache()
gc.collect()

print("=== 3. СКАНИРОВАНИЕ ДЕФЕКТОВ ЭЛЕМЕНТОВ (Texture Fusion) ===")
device_clipseg = "cuda" if torch.cuda.is_available() else "cpu"
print("-> Загрузка CLIPSeg (Тепловизор)...")
clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device_clipseg)

ELEMENT_DEFECT_PROMPTS = [
    "broken shattered window glass pane",
    "damaged broken wooden door surface"
]

COLORS_ELEMENT_DEFECTS = {
    "broken_glass": [0, 255, 255],  
    "damaged_wood": [255, 20, 147], 
}

inputs_clipseg = clipseg_processor(
    text=ELEMENT_DEFECT_PROMPTS, 
    images=[Image.fromarray(img_norm_rgb)] * len(ELEMENT_DEFECT_PROMPTS), 
    padding=True, 
    return_tensors="pt"
).to(device_clipseg)

with torch.no_grad():
    outputs_clipseg = clipseg_model(**inputs_clipseg)

preds_clipseg = torch.nn.functional.interpolate(
    outputs_clipseg.logits.unsqueeze(1), size=original_size, mode="bilinear", align_corners=False
).squeeze(1)

probabilities = torch.sigmoid(preds_clipseg).cpu().numpy() 

print("-> Загрузка SAM 2 AMG для физической нарезки...")
weights_path_sam2 = "sam2_hiera_small.pt"
if not os.path.exists(weights_path_sam2):
    !wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt

sam2_model = build_sam2("sam2_hiera_s.yaml", weights_path_sam2, device=device_clipseg)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2_model, points_per_side=32, pred_iou_thresh=0.8, stability_score_thresh=0.8, min_mask_region_area=300
)

with torch.no_grad():
    masks_amg = mask_generator.generate(img_norm_rgb)

print(f"-> Слияние {len(masks_amg)} кластеров с тепловыми картами CLIPSeg...")
final_element_defect_masks = {
    "broken_glass": np.zeros(original_size, dtype=bool),
    "damaged_wood": np.zeros(original_size, dtype=bool)
}

CONFIDENCE_THRESHOLD = 0.35 

for ann in masks_amg:
    m = ann['segmentation']
    avg_scores = [probabilities[i][m].mean() for i in range(len(ELEMENT_DEFECT_PROMPTS))]
    best_idx = np.argmax(avg_scores)
    
    if avg_scores[best_idx] > CONFIDENCE_THRESHOLD:
        if best_idx == 0: final_element_defect_masks["broken_glass"] |= m
        elif best_idx == 1: final_element_defect_masks["damaged_wood"] |= m

del sam2_model, mask_generator, clipseg_model, clipseg_processor, inputs_clipseg, outputs_clipseg
torch.cuda.empty_cache()
gc.collect()

# ПРОСТРАНСТВЕННЫЙ РОУТИНГ ДЛЯ ЭЛЕМЕНТОВ
mask_windows = geom_masks.get("window", np.zeros(original_size, dtype=bool))
mask_doors = geom_masks.get("door", np.zeros(original_size, dtype=bool))

final_element_defect_masks["broken_glass"] &= mask_windows
final_element_defect_masks["damaged_wood"] &= mask_doors

# === 4. УЛЬТИМАТИВНАЯ ВИЗУАЛИЗАЦИЯ И ОТЧЕТ (Z-INDEX) ===
def calc_damage_report(building_area, final_masks, wall_defect_names, element_defect_names):
    report = []
    building_area_px = building_area.sum()
    for name in wall_defect_names:
        area_px = final_masks[0][name].sum()
        percent = (area_px / building_area_px) * 100
        report.append(f"  • {name.replace('_',' ').title()}: {area_px} px ({percent:.1f}% здания)")
    for name in element_defect_names:
        area_px = final_masks[1][name].sum()
        percent = (area_px / building_area_px) * 100
        report.append(f"  • {name.replace('_',' ').title()}: {area_px} px ({percent:.1f}% здания)")
    return "\n".join(report)

print("=== 4. СБОРКА КОМПЛЕКСНОЙ КАРТЫ ПАТОЛОГИЙ ===")
mask_only_canvas = np.zeros((*original_size, 3), dtype=np.uint8)
found_defects = []

# Стены
draw_order_wall = ["peeling", "exposed_brick", "crack"]
for defect_type in draw_order_wall:
    mask = final_wall_defect_masks[defect_type]
    if np.any(mask):
        found_defects.append(defect_type)
        mask_only_canvas[mask] = np.array(COLORS_WALL_DEFECTS[defect_type], dtype=np.uint8) 

# Элементы
draw_order_elements = ["damaged_wood", "broken_glass"]
for defect_type in draw_order_elements:
    mask = final_element_defect_masks[defect_type]
    if np.any(mask):
        found_defects.append(defect_type)
        mask_only_canvas[mask] = np.array(COLORS_ELEMENT_DEFECTS[defect_type], dtype=np.uint8) 

# Контекст
context_mask = np.zeros(original_size, dtype=bool)
for key, geom_mask in geom_masks.items():
    if key != "building": context_mask = context_mask | geom_mask

final_defects_map = {**COLORS_WALL_DEFECTS, **COLORS_ELEMENT_DEFECTS}

context_only_mask = context_mask & ~np.any(list(final_wall_defect_masks.values()) + list(final_element_defect_masks.values()), axis=0) & true_silhouette
mask_only_canvas[context_only_mask] = [30, 30, 30]

overlay_canvas = img_norm_rgb.copy()
overlay_canvas = cv2.addWeighted(overlay_canvas, 0.3, mask_only_canvas, 0.7, 0)

fig, axes = plt.subplots(1, 2, figsize=(24, 12))
axes[0].imshow(overlay_canvas)
axes[0].set_title("1. Наложение Комплексной Карты Патологий", fontsize=16, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(mask_only_canvas)
axes[1].set_title("2. Физика + Материалы (Иерархическое слияние)", fontsize=16, fontweight='bold')
axes[1].axis('off')

legend_elements = [Patch(facecolor=np.array(final_defects_map[c])/255, label=c.replace('_',' ').title()) for c in reversed(found_defects)]
if np.any(context_mask): legend_elements.append(Patch(facecolor=[30/255, 30/255, 30/255], label="Elements Context (DINO)"))
axes[1].legend(handles=legend_elements, loc='upper right', fontsize=12, facecolor='black', labelcolor='white')

plt.tight_layout()
plt.show()

print("\n=== 🛠️ СТРОИТЕЛЬНЫЙ ОТЧЕТ AlegroCode (Damage Report) ===")
print(calc_damage_report(true_silhouette, (final_wall_defect_masks, final_element_defect_masks), draw_order_wall, draw_order_elements))