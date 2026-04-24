# AlegroCode

Production-ready фасадный аналитик: Flutter-приложение + Python-сервер на арендованной GPU.
Детекция дефектов и материалов, расчёт сметы в рублях и **локальная** ИИ-реставрация.

---

## Содержание

- [Обзор](#обзор)
- [Архитектура](#архитектура)
- [Запуск сервера в JupyterLab](#запуск-сервера-в-jupyterlab)
- [Сборка Flutter-приложения](#сборка-flutter-приложения)
- [Парсер цен](#парсер-цен)
- [API](#api)
- [Траблшутинг](#траблшутинг)
- [Лицензии моделей](#лицензии-моделей)

---

## Обзор

- **Цель:** выдать инженеру/заказчику смету в рублях по одному фото фасада.
- **Пайплайн:** Grounding DINO → SAM → U-2-Net (rembg) → CLIPSeg → SAM2 AMG.
- **Масштаб:** двухтаповая калибровка по известному размеру (дверь/окно/кирпич)
  превращает пиксели в м² и м.п.
- **Смета:** многослойный "пирог" — каждый дефект разворачивается в последовательность
  работ (например, *оголённый кирпич = восстановление кладки → грунтовка →
  штукатурка → шпаклёвка → покраска*). Цены берутся либо из ночного парсера
  (petrovich.ru, profi.ru), либо из YAML-baseline при сбое.
- **Реставрация:** локальный LaMa (default) или SD-Inpaint (`quality=high`) — не зависим
  от облачных API, арендованная T4 16 GB справляется.
- **Клиент:** Flutter (Android + iOS). Responsive верстка, Dio с ngrok-header'ом,
  Photoshop-подобный слоистый просмотрщик масок.

---

## Архитектура

```
┌─ Flutter (Android/iOS) ──────────────────────────────────────────┐
│ lib/                                                             │
│  ├─ app/          Riverpod + GoRouter                            │
│  ├─ data/api      Dio + interceptors (ngrok header, retry)       │
│  ├─ data/repos    AnalysisRepo · CalibrationRepo (+ mock)        │
│  ├─ features/     calibration · loading · results                │
│  └─ screens/      home · preview · settings · results            │
└──────────────┬───────────────────────────────────────────────────┘
               │ HTTPS (ngrok tunnel)
┌──────────────┴───────────────────────────────────────────────────┐
│  FastAPI uvicorn (non-blocking, nest_asyncio)                    │
│                                                                  │
│  POST /api/calibrate  ──► backend/calibration.py                 │
│  POST /api/analyze    ──► backend/ml_pipeline.py (DINO/SAM/CLIP) │
│  POST /api/estimate   ──► backend/estimator/ (recipes + prices)  │
│  POST /api/restore    ──► backend/restoration/ (LaMa / SD)       │
│  GET  /api/results/*  ──► backend/core/results_store.py (SQLite) │
└────────┬─────────────────────┬─────────────────┬─────────────────┘
         │                     │                 │
  backend/scraper/       default_prices_rub.yaml │
  (petrovich + profi)    ←  fallback baseline    │
  APScheduler cron                               │
         │                                       │
         ▼                                       ▼
    SQLite (prices)                       SQLite (results)
    backend/core/db.py                    backend/core/db.py
```

Все компоненты выполняются на одной GPU-машине:

- GPU-резидент: DINO / SAM / CLIPSeg / SAM2 / U-2-Net (~3 GB VRAM).
- LaMa (inpaint) загружается лениво при первом `/api/restore` и занимает 1–3 GB.
- Опциональный SD-inpaint (quality=high) живёт в `enable_sequential_cpu_offload`,
  3.5–5 GB VRAM в моменте. При OOM автоматически откат на LaMa.

---

## Запуск сервера в JupyterLab

### 1. Подготовка машины

Нужна GPU-машина с **≥ 12 GB VRAM** (T4 16 GB — минимум, A100 — comfort).

```bash
git clone <repo> /path/to/alegrocode
cd /path/to/alegrocode

pip install -r backend/requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git   # SAM2
```

### 2. Переменные окружения

```bash
export ALEGRO_NGROK_AUTHTOKEN=<ваш ngrok authtoken>
export ALEGRO_INPAINT_PROVIDER=lama         # или sd
export ALEGRO_DB_PATH=/path/to/alegrocode/data/alegrocode.db
# опционально: выключить автозапуск парсера
# export ALEGRO_SCRAPER_ENABLED=false
```

### 3. Открыть и выполнить тетрадку

`colab/colab_server_v2.ipynb` — выполнить ячейки **1 → 2 → 3 → 4 → 5** по порядку.

- Cell 1 — `nest_asyncio.apply()` + `pip install`
- Cell 2 — `FacadeAnalyzer().load_models()` (один раз; модели живут в kernel globals)
- Cell 3 — `asyncio.ensure_future(server.serve())` — **неблокирующий** старт
- Cell 4 — `GET /api/health` готовности
- Cell 5 — ngrok tunnel, печатает публичный URL
- Cell 6 — graceful stop (не убивает kernel — модели остаются в памяти)

Публичный URL → в Flutter → **Настройки → URL сервера**.

Подробности (fallback-вариант без nest_asyncio, траблшутинг) — в [`colab/README_colab.md`](colab/README_colab.md).

---

## Сборка Flutter-приложения

Требования: Flutter ≥ 3.11, Dart ≥ 3.5.

```bash
flutter pub get
flutter run                 # подключённое Android/iOS устройство
# или
flutter build apk --release
flutter build ipa --release
```

В первый запуск зайдите в **Настройки**, вставьте публичный URL сервера (из Cell 5),
нажмите **Проверить подключение**. Мок-режим (без сервера) включается тем же экраном.

Основной flow:

1. `HomeScreen` → «Снять фото» (image_picker) → `/preview`
2. `PhotoPreviewScreen` → «Указать масштаб» → `/calibrate`
3. `PhysicalScaleInputScreen` — выбор эталона, два тапа / прямоугольник
4. `RealAnalysisLoadingScreen` — реальный upload-progress через Riverpod `StreamProvider`
5. `ResultsScreen` — 5 табов: фото+маски, дефекты, материалы, смета, ведомость
6. Кнопка «ИИ-реставрация» отправляет `POST /api/restore/{id}` и обновляет вкладку «Фото»

---

## Парсер цен

Все цены хранятся в той же SQLite (`backend/core/db.py :: PriceSnapshot`). При сбое
или пустой базе калькулятор автоматически подставляет `backend/scraper/default_prices_rub.yaml`
и помечает ответ флагом `price_source="yaml_fallback"` — Flutter покажет
**баннер «Цены от …»**.

### Вариант A — APScheduler внутри uvicorn (по умолчанию)

Запускается автоматически при старте `create_app()`. Крон `0 3 * * *`.
Выключить: `ALEGRO_SCRAPER_ENABLED=false`.

### Вариант B — systemd + timer

```bash
sudo cp docs/systemd/alegrocode-scraper.service /etc/systemd/system/
sudo cp docs/systemd/alegrocode-scraper.timer   /etc/systemd/system/
sudo systemctl enable --now alegrocode-scraper.timer
```

### Вариант C — внешний cron

```cron
0 3 * * * cd /path/to/alegrocode && /usr/bin/env python -m backend.scraper.worker run-once --source all
```

### Ручной запуск / статус

```bash
python -m backend.scraper.worker run-once --source petrovich
python -m backend.scraper.worker status
```

**Avito** исключён из v1 — слишком агрессивный антибот, не гарантирует стабильность.

---

## API

| Метод      | Endpoint                                   | Назначение |
|------------|---------------------------------------------|------------|
| `GET`      | `/api/health`                              | liveness + `models_loaded` |
| `POST`     | `/api/calibrate`                           | pixel ↔ metre калибровка |
| `POST`     | `/api/analyze` (multipart: `file`, `calibration_id?`) | анализ фасада + смета |
| `POST`     | `/api/estimate`                            | пересчёт сметы для существующего анализа |
| `POST`     | `/api/restore/{analysis_id}`               | ИИ-реставрация (LaMa / SD) |
| `GET`      | `/api/results/{id}`                        | сохранённый JSON |
| `GET`      | `/api/results/{id}/image/{key}`            | base image, маски, восстановленное фото |

Swagger UI — `/docs`.

---

## Траблшутинг

**Ngrok показывает HTML-заглушку, а не JSON.** Flutter уже слепо добавляет заголовок
`ngrok-skip-browser-warning: true`; если не помогает — обновить `pyngrok ≥ 7.0`
и проверять, что туннель `http`, а не `tcp`.

**Сервер в Jupyter-ячейке «висит».** `nest_asyncio.apply()` — в Cell 1. Если он ничего не
делает (редкая конфигурация) — переключитесь на threading-вариант из
`colab/README_colab.md`.

**OOM на `quality=high`.** Система сама упадёт на LaMa. Чтобы форсировать только LaMa:
`ALEGRO_INPAINT_PROVIDER=lama`.

**Парсер 403 от profi.ru.** Сработал антибот. Через 3 неудачи подряд (внутри 24 ч) источник
уходит на 24-часовой circuit breaker; калькулятор продолжает работать на YAML.

**Долгий анализ.** Первый запрос тянет модели, до 2 минут. Последующие — 5–15 секунд на
T4, 1–3 секунды на A100.

---

## Лицензии моделей

| Модель | Лицензия |
|---|---|
| Grounding DINO | Apache 2.0 |
| SAM (vit-base) | Apache 2.0 |
| SAM 2 (hiera-small) | Apache 2.0 |
| CLIPSeg | MIT |
| U-2-Net (rembg) | Apache 2.0 |
| LaMa | Apache 2.0 |
| Stable Diffusion 2 Inpainting | CreativeML Open RAIL++-M |
