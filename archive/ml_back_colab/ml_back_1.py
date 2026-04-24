import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.ops as ops
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
from google.colab import files

device = "cuda" if torch.cuda.is_available() else "cpu"

# === 1. КЭШИРОВАННАЯ ЗАГРУЗКА МОДЕЛЕЙ ===
if 'dino_model' not in locals():
    print("Загрузка DINO...")
    dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

if 'sam_model' not in locals():
    print("Загрузка SAM...")
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

# === 2. НАСТРОЙКИ КЛАССОВ И ЦВЕТОВ ===
TEXT_PROMPT = "window pane. entrance door. building facade. balcony."

CLASS_MAP = {
    "window": ["window", "pane", "glass"],
    "door": ["door", "entrance", "gate"],
    "balcony": ["balcony", "terrace"],
    "building": ["building", "facade", "wall"]
}

COLORS_HEX = {"window": "cyan", "door": "red", "balcony": "magenta", "building": "blue", "unknown": "yellow"}
COLORS_RGBA = {
    "window": np.array([0.0, 1.0, 1.0, 0.5]), 
    "door": np.array([1.0, 0.0, 0.0, 0.5]), 
    "balcony": np.array([1.0, 0.0, 1.0, 0.5]), 
    "building": np.array([0.0, 0.0, 1.0, 0.1]),
    "unknown": np.array([1.0, 1.0, 0.0, 0.5])
}
COLORS_RGB_SOLID = {
    "window": np.array([0, 255, 255]), 
    "door": np.array([255, 0, 0]), 
    "balcony": np.array([255, 0, 255]), 
    "building": np.array([0, 0, 255]),
    "unknown": np.array([255, 255, 0])
}

def get_base_class(raw_label):
    raw_label = raw_label.lower()
    for base, synonyms in CLASS_MAP.items():
        if any(syn in raw_label for syn in synonyms):
            return base
    return "unknown"

# === 3. ОБРАБОТКА ФОТОГРАФИИ ===
print("\nЗагрузите фотографию:")
uploaded = files.upload()

# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ ЯЧЕЙКИ 2
global img_norm_rgb, geom_masks 

for filename in uploaded.keys():
    print("-> Применение нормализации освещения (CLAHE)...")
    img_bgr = cv2.imread(filename)
    
    max_dim = 1024
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    img_clahe_bgr = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)
    img_norm_rgb = cv2.fastNlMeansDenoisingColored(cv2.cvtColor(img_clahe_bgr, cv2.COLOR_BGR2RGB), None, 5, 5, 7, 21)
    
    image = Image.fromarray(img_norm_rgb)
    original_size = image.size[::-1]

    # --- ИНИЦИАЛИЗАЦИЯ СЛОВАРЯ МАСОК ДЛЯ ЭКСПОРТА ---
    geom_masks = {k: np.zeros(original_size, dtype=bool) for k in CLASS_MAP.keys()}

    print("-> Сканирование геометрии (DINO)...")
    dino_inputs = dino_processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)

    with torch.no_grad():
        dino_outputs = dino_model(**dino_inputs)

    dino_results = dino_processor.post_process_grounded_object_detection(
        dino_outputs, dino_inputs.input_ids, threshold=0.25, text_threshold=0.25, target_sizes=[image.size[::-1]]
    )[0]

    all_scores = dino_results["scores"].cpu()
    valid_indices = all_scores > 0.25

    raw_boxes = dino_results["boxes"].cpu()[valid_indices]
    raw_scores = all_scores[valid_indices]
    labels_out = dino_results.get("text_labels", dino_results.get("labels"))
    raw_labels = [labels_out[i] for i, valid in enumerate(valid_indices) if valid]

    if len(raw_boxes) > 0:
        category_idxs = torch.tensor([list(CLASS_MAP.keys()).index(get_base_class(l)) if get_base_class(l) in CLASS_MAP else 99 for l in raw_labels])
        keep_indices = ops.batched_nms(raw_boxes, raw_scores, category_idxs, iou_threshold=0.5).numpy()

        boxes = raw_boxes[keep_indices].numpy()
        scores = raw_scores[keep_indices].numpy()
        labels = [raw_labels[i] for i in keep_indices]
        print(f"DINO нашел {len(raw_boxes)} объектов. После NMS осталось: {len(boxes)}.")
    else:
        boxes, scores, labels = [], [], []

    best_masks = []
    if len(boxes) > 0:
        print("-> Генерация пиксельных масок (SAM)...")
        input_boxes = [boxes.tolist()]
        sam_inputs = sam_processor(image, input_boxes=input_boxes, return_tensors="pt").to(device)

        with torch.no_grad():
            sam_outputs = sam_model(**sam_inputs)

        masks = sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu()
        )
        best_masks = masks[0][:, 0, :, :].numpy()

    print("-> Сборка архитектурного плана...")
    fig, axes = plt.subplots(1, 3, figsize=(28, 9))
    ax1, ax2, ax3 = axes

    ax1.imshow(img_norm_rgb); ax1.set_title("1. DINO Рамки + SAM Маски", fontsize=16, fontweight='bold'); ax1.axis('off')
    ax2.imshow(img_norm_rgb); ax2.set_title("2. Только SAM (Маски на фото)", fontsize=16, fontweight='bold'); ax2.axis('off')
    pure_mask = np.zeros((*original_size, 3), dtype=np.uint8)

    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    sorted_indices = np.argsort(areas)[::-1]

    for i in sorted_indices:
        box = boxes[i]
        score = scores[i]
        base_class = get_base_class(str(labels[i]))

        if len(best_masks) > 0:
            mask = best_masks[i]
            
            # --- СОХРАНЕНИЕ МАСКИ ДЛЯ ЯЧЕЙКИ 2 ---
            if base_class in geom_masks:
                geom_masks[base_class] = geom_masks[base_class] | mask
            
            color_rgba = COLORS_RGBA.get(base_class, COLORS_RGBA["unknown"])
            overlay = np.zeros((mask.shape[0], mask.shape[1], 4))
            overlay[mask == True] = color_rgba
            ax1.imshow(overlay); ax2.imshow(overlay)

            color_solid = COLORS_RGB_SOLID.get(base_class, COLORS_RGB_SOLID["unknown"])
            pure_mask[mask == True] = color_solid

        xmin, ymin, xmax, ymax = box
        width, height = xmax - xmin, ymax - ymin
        color_hex = COLORS_HEX.get(base_class, COLORS_HEX["unknown"])
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor=color_hex, facecolor='none')
        ax1.add_patch(rect)
        ax1.text(xmin, ymin - 5, f"{base_class} ({score:.2f})", color='black', fontsize=10, fontweight='bold', bbox=dict(facecolor=color_hex, alpha=0.8))

    ax3.imshow(pure_mask); ax3.set_title("3. Чистая маска (Архитектурный план)", fontsize=16, fontweight='bold'); ax3.axis('off')
    plt.tight_layout()
    plt.show()
    
    torch.cuda.empty_cache()
    print("✅ Этап 1 завершен. Данные (img_norm_rgb и geom_masks) готовы для Ячейки 2!")