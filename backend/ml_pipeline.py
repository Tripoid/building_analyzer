"""
FacadeAnalyzer — ML Pipeline for building facade analysis.
Consolidates 3 Colab prototype cells into a production-quality class.

Cell 1: Grounding DINO + SAM → geometry detection (windows, doors, balconies)
Cell 2: U-2-Net (rembg) + SAM3 + CLIPSeg + SAM2 → wall defect detection
Cell 3: CLIPSeg + SAM2 → material identification with z-index stacking
"""

import cv2
import torch
import gc
import os
import uuid
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import torchvision.ops as ops
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

GEOMETRY_PROMPT = (
    "window pane. entrance door. building facade wall. "
    "balcony terrace. roof rooftop. cornice molding decoration. "
    "drainpipe pipe. air conditioner unit."
)

CLASS_MAP_GEOMETRY = {
    "window": ["window", "pane", "glass"],
    "door": ["door", "entrance", "gate"],
    "balcony": ["balcony", "terrace"],
    "building": ["building", "facade", "wall"],
    "roof": ["roof", "rooftop"],
    "molding": ["cornice", "molding", "decoration"],
    "pipe": ["drainpipe", "pipe"],
    "ac_unit": ["air conditioner", "unit"],
}

WALL_DEFECT_PROMPTS = [
    "deep structural wall crack fracture",
    "damaged peeling plaster surface delamination",
    "exposed red brick from damaged crumbling wall",
    "water damage moisture stain on wall",
    "rust discoloration corrosion stain on facade",
    "green moss biological growth on wall surface",
    "white salt efflorescence deposit on wall",
    "spalling crumbling concrete surface damage",
]

CLASS_MAP_WALL_DEFECTS = {
    "crack": ["crack", "fracture"],
    "peeling": ["peeling plaster", "delamination"],
    "exposed_brick": ["exposed red brick", "crumbling wall"],
    "water_damage": ["water damage", "moisture stain"],
    "rust": ["rust", "corrosion"],
    "moss": ["moss", "biological growth"],
    "efflorescence": ["salt efflorescence", "deposit"],
    "spalling": ["spalling", "crumbling concrete"],
}

ELEMENT_DEFECT_PROMPTS = [
    "broken shattered cracked window glass pane",
    "damaged rotten wooden door frame surface",
    "rusty corroded metal element railing",
    "damaged cracked balcony railing surface",
]

CLASS_MAP_ELEMENT_DEFECTS = {
    "broken_glass": ["broken", "shattered", "cracked window"],
    "damaged_wood": ["damaged", "rotten wooden"],
    "rusty_metal": ["rusty", "corroded metal"],
    "damaged_railing": ["damaged", "cracked balcony"],
}

MATERIAL_PROMPTS = [
    "grey concrete stone wall base foundation",
    "terracotta red brick masonry wall surface",
    "plain smooth cement plaster wall facade",
    "decorative textured plaster facade finish",
    "architectural molding cornice decoration ornament",
    "ceramic tile cladding facade surface",
    "painted surface facade wall finish coating",
]

CLASS_MAP_MATERIALS = {
    "concrete": ["grey concrete stone"],
    "brick": ["terracotta red brick"],
    "cement_plaster": ["cement plaster"],
    "decorative_plaster": ["decorative textured plaster"],
    "molding": ["molding", "cornice", "decoration"],
    "ceramic_tile": ["ceramic tile"],
    "painted_surface": ["painted surface"],
}

COLORS_GEOMETRY = {
    "window": [0, 255, 255],
    "door": [255, 0, 0],
    "balcony": [255, 0, 255],
    "building": [0, 0, 255],
    "roof": [128, 128, 0],
    "molding": [200, 200, 200],
    "pipe": [128, 0, 128],
    "ac_unit": [0, 128, 128],
    "unknown": [255, 255, 0],
}

COLORS_WALL_DEFECTS = {
    "crack": [255, 0, 0],
    "peeling": [241, 196, 15],
    "exposed_brick": [230, 126, 34],
    "water_damage": [52, 152, 219],
    "rust": [192, 57, 43],
    "moss": [39, 174, 96],
    "efflorescence": [236, 240, 241],
    "spalling": [149, 165, 166],
}

COLORS_ELEMENT_DEFECTS = {
    "broken_glass": [0, 255, 255],
    "damaged_wood": [255, 20, 147],
    "rusty_metal": [205, 92, 0],
    "damaged_railing": [148, 103, 189],
}

COLORS_MATERIALS = {
    "concrete": [127, 140, 141],
    "brick": [211, 84, 0],
    "cement_plaster": [243, 214, 115],
    "decorative_plaster": [255, 235, 180],
    "molding": [236, 240, 241],
    "ceramic_tile": [46, 134, 193],
    "painted_surface": [175, 215, 160],
}

# Wall layer definitions for multi-layer analysis
WALL_LAYERS = {
    "finish": {"name": "Финишный слой", "depth_mm": 3, "materials": ["painted_surface", "decorative_plaster"]},
    "base_plaster": {"name": "Базовая штукатурка", "depth_mm": 20, "materials": ["cement_plaster"]},
    "insulation": {"name": "Утеплитель", "depth_mm": 50, "materials": []},
    "structural": {"name": "Несущий слой", "depth_mm": 200, "materials": ["brick", "concrete"]},
}

# Defect → affected wall layers mapping
DEFECT_LAYER_IMPACT = {
    "crack": {"surface": ["finish"], "deep": ["finish", "base_plaster", "structural"]},
    "peeling": ["finish", "base_plaster"],
    "exposed_brick": ["finish", "base_plaster"],
    "water_damage": ["finish", "base_plaster"],
    "rust": ["finish"],
    "moss": ["finish"],
    "efflorescence": ["finish"],
    "spalling": ["finish", "base_plaster", "structural"],
}

SEVERITY_LABELS = {
    "ru": {
        "high": "Высокая",
        "medium": "Средняя",
        "low": "Низкая",
    }
}

CONDITION_LABELS = {
    "ru": {
        "good": "Хорошее",
        "satisfactory": "Удовлетворительное",
        "needs_repair": "Требует ремонта",
        "critical": "Аварийное",
    }
}


def _get_base_class(raw_label: str, class_map: dict) -> Optional[str]:
    """Match a raw label to a base class using synonym lookup."""
    raw_label = raw_label.lower()
    for base, synonyms in class_map.items():
        if any(syn in raw_label for syn in synonyms):
            return base
    return None


def _adaptive_params(image_shape: Tuple[int, int]) -> dict:
    """Return adaptive ML parameters based on image resolution."""
    h, w = image_shape[:2]
    total_px = h * w
    if total_px > 2_000_000:  # >2MP
        return {"points_per_side": 48, "min_mask_area": 500, "dino_threshold": 0.20, "dino_text_threshold": 0.20}
    elif total_px > 800_000:  # >0.8MP
        return {"points_per_side": 32, "min_mask_area": 300, "dino_threshold": 0.25, "dino_text_threshold": 0.25}
    else:
        return {"points_per_side": 16, "min_mask_area": 150, "dino_threshold": 0.30, "dino_text_threshold": 0.30}


class FacadeAnalyzer:
    """
    Production ML pipeline for building facade analysis.
    Loads models once and caches them in memory.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models_loaded = False
        self._dino_processor = None
        self._dino_model = None
        self._sam_processor = None
        self._sam_model = None
        self._clipseg_processor = None
        self._clipseg_model = None

    def load_models(self):
        """Load and cache all ML models. Call once at startup."""
        if self.models_loaded:
            return

        logger.info("Loading Grounding DINO...")
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._dino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
        self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        ).to(self.device)

        logger.info("Loading SAM (vit-base)...")
        from transformers import SamModel, SamProcessor
        self._sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self._sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)

        logger.info("Loading CLIPSeg...")
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        self._clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self._clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(self.device)

        self.models_loaded = True
        logger.info("All models loaded successfully.")

    # ─────────────────────────────────────────────────
    # STEP 0: Preprocessing
    # ─────────────────────────────────────────────────

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """CLAHE normalization + adaptive denoising + resize."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize to max 1024px
        max_dim = 1024
        h, w = img_bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # CLAHE on L channel
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        img_clahe = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        # Adaptive denoising
        brightness = np.mean(cl)
        h_param = 3 if brightness > 120 else 7
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB)
        img_denoised = cv2.fastNlMeansDenoisingColored(img_rgb, None, h_param, h_param, 7, 21)

        return img_denoised

    # ─────────────────────────────────────────────────
    # STEP 1: Geometry Detection (DINO + SAM)
    # ─────────────────────────────────────────────────

    def detect_geometry(self, img_rgb: np.ndarray) -> Tuple[Dict[str, np.ndarray], list]:
        """Detect architectural elements using Grounding DINO + SAM."""
        image = Image.fromarray(img_rgb)
        original_size = image.size[::-1]  # (H, W)
        params = _adaptive_params(img_rgb.shape)

        # DINO detection
        dino_inputs = self._dino_processor(
            images=image, text=GEOMETRY_PROMPT, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            dino_outputs = self._dino_model(**dino_inputs)

        results = self._dino_processor.post_process_grounded_object_detection(
            dino_outputs, dino_inputs.input_ids,
            threshold=params["dino_threshold"],
            text_threshold=params["dino_text_threshold"],
            target_sizes=[original_size]
        )[0]

        all_scores = results["scores"].cpu()
        valid = all_scores > params["dino_threshold"]
        raw_boxes = results["boxes"].cpu()[valid]
        raw_scores = all_scores[valid]
        labels_out = results.get("text_labels", results.get("labels"))
        raw_labels = [labels_out[i] for i, v in enumerate(valid) if v]

        # NMS per category
        geom_masks = {k: np.zeros(original_size, dtype=bool) for k in CLASS_MAP_GEOMETRY}
        detections = []

        if len(raw_boxes) > 0:
            class_keys = list(CLASS_MAP_GEOMETRY.keys())
            category_idxs = torch.tensor([
                class_keys.index(_get_base_class(l, CLASS_MAP_GEOMETRY))
                if _get_base_class(l, CLASS_MAP_GEOMETRY) in class_keys else 99
                for l in raw_labels
            ])
            keep = ops.batched_nms(raw_boxes, raw_scores, category_idxs, iou_threshold=0.5).numpy()
            boxes = raw_boxes[keep].numpy()
            scores = raw_scores[keep].numpy()
            labels = [raw_labels[i] for i in keep]

            # SAM segmentation
            if len(boxes) > 0:
                sam_inputs = self._sam_processor(
                    image, input_boxes=[boxes.tolist()], return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    sam_outputs = self._sam_model(**sam_inputs)

                masks = self._sam_processor.image_processor.post_process_masks(
                    sam_outputs.pred_masks.cpu(),
                    sam_inputs["original_sizes"].cpu(),
                    sam_inputs["reshaped_input_sizes"].cpu()
                )
                best_masks = masks[0][:, 0, :, :].numpy()

                areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                sorted_indices = np.argsort(areas)[::-1]

                for i in sorted_indices:
                    base_class = _get_base_class(str(labels[i]), CLASS_MAP_GEOMETRY) or "unknown"
                    mask = best_masks[i] if len(best_masks) > i else None

                    if mask is not None and base_class in geom_masks:
                        geom_masks[base_class] |= mask

                    detections.append({
                        "class": base_class,
                        "score": float(scores[i]),
                        "box": boxes[i].tolist(),
                    })

        logger.info(f"Geometry: detected {len(detections)} elements")
        return geom_masks, detections

    # ─────────────────────────────────────────────────
    # STEP 2: Wall Defect Detection
    # ─────────────────────────────────────────────────

    def detect_defects(
        self, img_rgb: np.ndarray, geom_masks: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Detect wall and element defects using rembg + CLIPSeg + SAM2 AMG."""
        from rembg import remove
        original_size = img_rgb.shape[:2]
        params = _adaptive_params(img_rgb.shape)

        # 1. Isolate building silhouette
        logger.info("Generating building silhouette (rembg)...")
        rgba = remove(img_rgb)
        silhouette = rgba[:, :, 3] > 128

        # 2. Compute bare wall mask
        elements = np.zeros(original_size, dtype=bool)
        for key in ["window", "door", "balcony", "roof", "molding", "pipe", "ac_unit"]:
            if key in geom_masks:
                elements |= geom_masks[key]
        bare_wall = silhouette & ~elements

        # 3. Wall defects via CLIPSeg
        logger.info("Scanning wall defects (CLIPSeg)...")
        wall_defect_masks = self._clipseg_segment(
            img_rgb, WALL_DEFECT_PROMPTS, CLASS_MAP_WALL_DEFECTS,
            original_size, region_mask=bare_wall, threshold=0.30
        )

        # 4. Element defects via CLIPSeg
        logger.info("Scanning element defects (CLIPSeg)...")
        element_defect_masks = self._clipseg_segment(
            img_rgb, ELEMENT_DEFECT_PROMPTS, CLASS_MAP_ELEMENT_DEFECTS,
            original_size, threshold=0.35
        )

        # Spatial routing: element defects only within their geometry
        mask_windows = geom_masks.get("window", np.zeros(original_size, dtype=bool))
        mask_doors = geom_masks.get("door", np.zeros(original_size, dtype=bool))
        mask_balconies = geom_masks.get("balcony", np.zeros(original_size, dtype=bool))

        if "broken_glass" in element_defect_masks:
            element_defect_masks["broken_glass"] &= mask_windows
        if "damaged_wood" in element_defect_masks:
            element_defect_masks["damaged_wood"] &= mask_doors
        if "rusty_metal" in element_defect_masks:
            element_defect_masks["rusty_metal"] &= (mask_balconies | mask_windows)
        if "damaged_railing" in element_defect_masks:
            element_defect_masks["damaged_railing"] &= mask_balconies

        torch.cuda.empty_cache()
        gc.collect()

        return wall_defect_masks, element_defect_masks

    # ─────────────────────────────────────────────────
    # STEP 3: Material Identification
    # ─────────────────────────────────────────────────

    def analyze_materials(
        self, img_rgb: np.ndarray, geom_masks: Dict[str, np.ndarray],
        wall_defect_masks: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """Identify facade materials using CLIPSeg with z-index stacking."""
        from rembg import remove
        original_size = img_rgb.shape[:2]

        # Rebuild bare wall
        rgba = remove(img_rgb)
        silhouette = rgba[:, :, 3] > 128
        elements = np.zeros(original_size, dtype=bool)
        for key in ["window", "door", "balcony", "roof", "molding", "pipe", "ac_unit"]:
            if key in geom_masks:
                elements |= geom_masks[key]
        bare_wall = silhouette & ~elements

        # CLIPSeg material segmentation
        logger.info("Scanning materials (CLIPSeg)...")
        raw_material_masks = self._clipseg_segment(
            img_rgb, MATERIAL_PROMPTS, CLASS_MAP_MATERIALS,
            original_size, region_mask=bare_wall, threshold=0.18
        )

        # Z-INDEX stacking: structural → base → finish → decorative
        final_materials = {k: np.zeros(original_size, dtype=bool) for k in CLASS_MAP_MATERIALS}

        # Layer 1: Concrete (foundation/base)
        concrete_m = raw_material_masks.get("concrete", np.zeros(original_size, dtype=bool)) & bare_wall
        if np.any(concrete_m):
            final_materials["concrete"] = concrete_m

        # Layer 2: Cement plaster (default wall surface)
        plaster_m = bare_wall & ~final_materials["concrete"]
        if np.any(plaster_m):
            final_materials["cement_plaster"] = plaster_m

        # Layer 3: Brick (exposed, overrides plaster)
        brick_m = raw_material_masks.get("brick", np.zeros(original_size, dtype=bool))
        if wall_defect_masks and "exposed_brick" in wall_defect_masks:
            brick_m |= wall_defect_masks["exposed_brick"]
        brick_m &= plaster_m
        if np.any(brick_m):
            final_materials["brick"] = brick_m

        # Layer 4: Decorative plaster (overrides cement plaster)
        deco_m = raw_material_masks.get("decorative_plaster", np.zeros(original_size, dtype=bool)) & plaster_m
        if np.any(deco_m):
            final_materials["decorative_plaster"] = deco_m

        # Layer 5: Ceramic tile
        tile_m = raw_material_masks.get("ceramic_tile", np.zeros(original_size, dtype=bool)) & bare_wall
        if np.any(tile_m):
            final_materials["ceramic_tile"] = tile_m

        # Layer 6: Painted surface
        paint_m = raw_material_masks.get("painted_surface", np.zeros(original_size, dtype=bool)) & bare_wall
        if np.any(paint_m):
            final_materials["painted_surface"] = paint_m

        # Layer 7: Molding (decorative, top z-index)
        molding_m = raw_material_masks.get("molding", np.zeros(original_size, dtype=bool)) & bare_wall
        if np.any(molding_m):
            final_materials["molding"] = molding_m

        torch.cuda.empty_cache()
        gc.collect()

        logger.info("Material analysis complete.")
        return final_materials

    # ─────────────────────────────────────────────────
    # STEP 4: Wall Layer Classification
    # ─────────────────────────────────────────────────

    def classify_wall_layers(
        self, defect_masks: Dict[str, np.ndarray], material_masks: Dict[str, np.ndarray],
        original_size: Tuple[int, int]
    ) -> Dict[str, dict]:
        """Classify which wall layers each defect impacts."""
        layer_analysis = {}

        for defect_type, mask in defect_masks.items():
            if not np.any(mask):
                continue

            area_px = int(mask.sum())
            impact = DEFECT_LAYER_IMPACT.get(defect_type, ["finish"])

            # For cracks, determine depth
            if defect_type == "crack" and isinstance(impact, dict):
                # Heuristic: cracks overlapping exposed brick = deep
                brick_mask = material_masks.get("brick", np.zeros(original_size, dtype=bool))
                overlap = np.sum(mask & brick_mask)
                if overlap > area_px * 0.1:
                    affected_layers = impact["deep"]
                    crack_depth = "deep"
                else:
                    affected_layers = impact["surface"]
                    crack_depth = "surface"
                layer_analysis[defect_type] = {
                    "area_px": area_px,
                    "affected_layers": affected_layers,
                    "crack_depth": crack_depth,
                }
            else:
                if isinstance(impact, dict):
                    affected_layers = impact.get("surface", ["finish"])
                else:
                    affected_layers = impact
                layer_analysis[defect_type] = {
                    "area_px": area_px,
                    "affected_layers": affected_layers,
                }

        return layer_analysis

    # ─────────────────────────────────────────────────
    # VISUALIZATION
    # ─────────────────────────────────────────────────

    def generate_visualizations(
        self, img_rgb: np.ndarray,
        geom_masks: Dict[str, np.ndarray],
        wall_defect_masks: Dict[str, np.ndarray],
        element_defect_masks: Dict[str, np.ndarray],
        material_masks: Dict[str, np.ndarray],
        output_dir: str,
    ) -> Dict[str, str]:
        """Generate 4 visualization images and save to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        original_size = img_rgb.shape[:2]
        paths = {}

        # 1. Heatmap — defect density
        heatmap = np.zeros(original_size, dtype=np.float32)
        for mask in wall_defect_masks.values():
            heatmap += mask.astype(np.float32)
        for mask in element_defect_masks.values():
            heatmap += mask.astype(np.float32)
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = np.zeros(original_size, dtype=np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.5, heatmap_colored, 0.5, 0)
        path = os.path.join(output_dir, "heatmap.jpg")
        cv2.imwrite(path, overlay)
        paths["heatmap"] = path

        # 2. Defects — marked with colored boxes
        defects_canvas = img_rgb.copy()
        all_defects = {**COLORS_WALL_DEFECTS, **COLORS_ELEMENT_DEFECTS}
        all_masks = {**wall_defect_masks, **element_defect_masks}
        for dtype, color in all_defects.items():
            mask = all_masks.get(dtype)
            if mask is not None and np.any(mask):
                colored = np.zeros_like(defects_canvas)
                colored[mask] = color
                defects_canvas = cv2.addWeighted(defects_canvas, 1.0, colored, 0.5, 0)
        path = os.path.join(output_dir, "defects.jpg")
        cv2.imwrite(path, cv2.cvtColor(defects_canvas, cv2.COLOR_RGB2BGR))
        paths["defects"] = path

        # 3. Segments — material segmentation
        segments_canvas = np.zeros((*original_size, 3), dtype=np.uint8)
        for mat_name, color in COLORS_MATERIALS.items():
            mask = material_masks.get(mat_name)
            if mask is not None and np.any(mask):
                segments_canvas[mask] = color
        # Add element context
        for key in ["window", "door", "balcony"]:
            mask = geom_masks.get(key)
            if mask is not None and np.any(mask):
                segments_canvas[mask] = [40, 40, 40]
        path = os.path.join(output_dir, "segments.jpg")
        cv2.imwrite(path, cv2.cvtColor(segments_canvas, cv2.COLOR_RGB2BGR))
        paths["segments"] = path

        # 4. Repair overlay — priority repair zones
        repair_canvas = img_rgb.copy()
        priority_colors = {
            "crack": [255, 0, 0],
            "peeling": [255, 165, 0],
            "spalling": [255, 0, 0],
            "exposed_brick": [230, 126, 34],
            "water_damage": [52, 152, 219],
        }
        for dtype, color in priority_colors.items():
            mask = wall_defect_masks.get(dtype)
            if mask is not None and np.any(mask):
                # Draw border around repair zone
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                bgr_color = (color[2], color[1], color[0])
                cv2.drawContours(repair_canvas, contours, -1, bgr_color, 3)
                # Light fill
                colored = np.zeros_like(repair_canvas)
                colored[mask] = color
                repair_canvas = cv2.addWeighted(repair_canvas, 1.0, colored, 0.25, 0)
        path = os.path.join(output_dir, "overlay.jpg")
        cv2.imwrite(path, cv2.cvtColor(repair_canvas, cv2.COLOR_RGB2BGR))
        paths["overlay"] = path

        return paths

    # ─────────────────────────────────────────────────
    # FULL ANALYSIS PIPELINE
    # ─────────────────────────────────────────────────

    def analyze(self, image_bytes: bytes, output_dir: str = "/tmp/facade_results") -> dict:
        """Run the full analysis pipeline and return structured results."""
        analysis_id = str(uuid.uuid4())
        result_dir = os.path.join(output_dir, analysis_id)
        os.makedirs(result_dir, exist_ok=True)

        # Step 0
        logger.info("Step 0: Preprocessing...")
        img_rgb = self.preprocess(image_bytes)
        original_size = img_rgb.shape[:2]
        total_px = original_size[0] * original_size[1]

        # Save original preprocessed image
        cv2.imwrite(
            os.path.join(result_dir, "original.jpg"),
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        )

        # Step 1
        logger.info("Step 1: Geometry detection...")
        geom_masks, detections = self.detect_geometry(img_rgb)

        # Step 2
        logger.info("Step 2: Defect detection...")
        wall_defect_masks, element_defect_masks = self.detect_defects(img_rgb, geom_masks)

        # Step 3
        logger.info("Step 3: Material analysis...")
        material_masks = self.analyze_materials(img_rgb, geom_masks, wall_defect_masks)

        # Step 4: Wall layer classification
        logger.info("Step 4: Wall layer classification...")
        layer_analysis = self.classify_wall_layers(wall_defect_masks, material_masks, original_size)

        # Step 5: Visualizations
        logger.info("Step 5: Generating visualizations...")
        viz_paths = self.generate_visualizations(
            img_rgb, geom_masks, wall_defect_masks, element_defect_masks, material_masks, result_dir
        )

        # Build silhouette for area calculations
        from rembg import remove
        rgba = remove(img_rgb)
        silhouette = rgba[:, :, 3] > 128
        building_area_px = int(silhouette.sum())

        # Compile damage statistics
        damages = []
        all_defect_masks = {**wall_defect_masks, **element_defect_masks}
        total_damaged_px = 0
        for dtype, mask in all_defect_masks.items():
            if not np.any(mask):
                continue
            area_px = int(mask.sum())
            total_damaged_px += area_px
            pct = (area_px / building_area_px * 100) if building_area_px > 0 else 0

            # Severity classification
            if pct > 15:
                severity = "high"
            elif pct > 5:
                severity = "medium"
            else:
                severity = "low"

            layer_info = layer_analysis.get(dtype, {})

            damages.append({
                "type": dtype,
                "type_display": dtype.replace("_", " ").title(),
                "percentage": round(pct, 1),
                "area_px": area_px,
                "severity": severity,
                "severity_display": SEVERITY_LABELS["ru"][severity],
                "affected_layers": layer_info.get("affected_layers", ["finish"]),
                "crack_depth": layer_info.get("crack_depth"),
            })

        damages.sort(key=lambda d: d["percentage"], reverse=True)

        # Material statistics
        materials = []
        bare_wall_px = sum(m.sum() for m in material_masks.values())
        for mat_name, mask in material_masks.items():
            area_px = int(mask.sum())
            if area_px == 0:
                continue
            pct = (area_px / bare_wall_px * 100) if bare_wall_px > 0 else 0
            materials.append({
                "name": mat_name,
                "name_display": mat_name.replace("_", " ").title(),
                "percentage": round(pct, 1),
                "area_px": area_px,
            })
        materials.sort(key=lambda m: m["percentage"], reverse=True)

        # Overall score (0-100)
        damage_ratio = total_damaged_px / building_area_px if building_area_px > 0 else 0
        overall_score = max(0, min(100, round(100 * (1 - damage_ratio * 2.5), 1)))

        if overall_score >= 80:
            condition = "good"
        elif overall_score >= 60:
            condition = "satisfactory"
        elif overall_score >= 30:
            condition = "needs_repair"
        else:
            condition = "critical"

        return {
            "id": analysis_id,
            "overall_score": overall_score,
            "overall_condition": CONDITION_LABELS["ru"][condition],
            "total_area_px": building_area_px,
            "damaged_area_px": total_damaged_px,
            "damages": damages,
            "materials": materials,
            "layer_analysis": layer_analysis,
            "geometry_detections": detections,
            "processed_images": list(viz_paths.keys()),
            "image_paths": viz_paths,
        }

    # ─────────────────────────────────────────────────
    # HELPER: CLIPSeg segmentation
    # ─────────────────────────────────────────────────

    def _clipseg_segment(
        self, img_rgb: np.ndarray, prompts: List[str], class_map: dict,
        original_size: Tuple[int, int],
        region_mask: Optional[np.ndarray] = None,
        threshold: float = 0.30,
    ) -> Dict[str, np.ndarray]:
        """Generic CLIPSeg segmentation with optional region masking."""
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        params = _adaptive_params(original_size)

        image = Image.fromarray(img_rgb)
        inputs = self._clipseg_processor(
            text=prompts,
            images=[image] * len(prompts),
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self._clipseg_model(**inputs)

        preds = torch.nn.functional.interpolate(
            outputs.logits.unsqueeze(1), size=original_size,
            mode="bilinear", align_corners=False
        ).squeeze(1)
        probabilities = torch.sigmoid(preds).cpu().numpy()

        # SAM2 AMG for physical masks
        weights_path = "sam2_hiera_small.pt"
        if not os.path.exists(weights_path):
            import urllib.request
            logger.info("Downloading SAM2 weights...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
                weights_path
            )

        sam2_model = build_sam2("sam2_hiera_s.yaml", weights_path, device=self.device)
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=params["points_per_side"],
            pred_iou_thresh=0.8,
            stability_score_thresh=0.8,
            min_mask_region_area=params["min_mask_area"],
        )

        with torch.no_grad():
            masks_amg = mask_generator.generate(img_rgb)

        result_masks = {k: np.zeros(original_size, dtype=bool) for k in class_map}

        for ann in masks_amg:
            m = ann["segmentation"]
            if region_mask is not None and not np.any(m & region_mask):
                continue

            avg_scores = [probabilities[i][m].mean() for i in range(len(prompts))]
            best_idx = np.argmax(avg_scores)

            if avg_scores[best_idx] > threshold:
                class_name = _get_base_class(prompts[best_idx], class_map)
                if class_name:
                    final_mask = m & region_mask if region_mask is not None else m
                    result_masks[class_name] |= final_mask

        del sam2_model, mask_generator
        torch.cuda.empty_cache()
        gc.collect()

        return result_masks
