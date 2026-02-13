# Line Instance Grouping via Pixel Embeddings

Модель решает задачу instance-segmentation линий на графиках через пиксельные эмбеддинги и кластеризацию, без фиксированного числа линий.

## Задача

Есть изображение графика и пиксели-кандидаты линии. Нужно разнести эти пиксели по отдельным линиям (инстансам), даже если число линий заранее неизвестно.

## Идея подхода

Вместо прямого предсказания масок линий модель предсказывает эмбеддинг-вектор для каждого пикселя:

- пиксели одной линии имеют близкие эмбеддинги;
- пиксели разных линий имеют разные эмбеддинги.

После этого выполняется кластеризация эмбеддингов (например, DBSCAN), и кластеры интерпретируются как отдельные линии.

## Вход и выход модели

- Вход: `img` размера `3 x H x W` (RGB).
- Выход: `emb` размера `D x h x w` (в коде `emb_dim=8` по умолчанию).

Важно: из-за pooling/upsampling пространственный размер выхода может отличаться на 1 пиксель. В `main.py` это учитывается обрезкой `inst/valid` до размера `emb`.

## Архитектура

Используется компактный U-Net (`SmallUNet`):

- Encoder: блоки `Conv-BN-ReLU` + `MaxPool`, извлечение признаков и рост receptive field.
- Bottleneck: самый глубокий блок признаков.
- Decoder: `ConvTranspose` + skip-connections из encoder для восстановления деталей тонких линий.
- Head: `1x1 Conv` для проекции в эмбеддинг-пространство (`emb_dim` каналов).

## Формирование таргетов

Из COCO-like JSON формируются:

- `instance_mask (H x W)`: ID линии (`1..K`, `0` — фон).
- `valid_mask (H x W)`: валидные пиксели для лосса.

Полилинии растризуются через `cv2.polylines(...)` с настраиваемой толщиной (`--thickness`).  
Зоны overlap между линиями исключаются из `valid_mask`, чтобы не портить supervision неоднозначными пикселями.

## Loss: Discriminative Embedding Loss

Используется классическая discriminative loss для инстанс-эмбеддингов:

- Variance term: стягивает эмбеддинги внутри каждого инстанса к его центроиду.
- Distance term: раздвигает центроиды разных инстансов.
- Regularization term: ограничивает нормы центроидов.

Таким образом модель обучает геометрию эмбеддинг-пространства, а не фиксированные классы линий.

## Обучение

Скрипт обучения находится в `main.py`.  
Поддерживаются:

- CLI-параметры через `argparse`;
- валидация по `val_loss`;
- сохранение чекпоинтов `last/best/final`;
- возобновление обучения (`--resume`);
- ограничение шагов на эпоху (`--max_steps`) для быстрых экспериментов.

### Обязательные аргументы

- `--train_json`
- `--train_img_dir`
- `--val_json`
- `--val_img_dir`

### Аргументы по умолчанию

- `--epochs 10`
- `--batch_size 2`
- `--lr 1e-3`
- `--save_dir ./checkpoints`
- `--save_every 1`
- `--eval_every 1`
- `--resume ""`
- `--num_workers 2`
- `--thickness 2`
- `--line_category_id 2`
- `--max_steps 0` (`0` = полная эпоха)

### Пример запуска

```bash
python main.py \
  --train_json /path/to/annotations/train.json \
  --train_img_dir /path/to/images/train \
  --val_json /path/to/annotations/val.json \
  --val_img_dir /path/to/images/val \
  --epochs 20 \
  --batch_size 4 \
  --lr 1e-3 \
  --save_dir ./checkpoints
```

### Чекпоинты

В директории `--save_dir`:

- `ckpt_last.pt` — сохраняется каждые `--save_every` эпох;
- `ckpt_best.pt` — обновляется при улучшении `val_loss`;
- `ckpt_final.pt` — финальный сейв после завершения обучения.

## Инференс и группировка линий

Типовой пайплайн:

1. Прогнать `img` через модель и получить `emb`.
2. Взять эмбеддинги только в candidate/valid пикселях:
   `feats = emb[:, ys, xs].T` (`N x D`).
3. (Опционально) L2-нормализовать `feats`.
4. Кластеризовать `feats` (например, `DBSCAN`).
5. Собрать:
   - `pred_mask (h x w)` с ID линии;
   - словарь кластеров `cluster_id -> [(x, y), ...]`.

Ключевое преимущество: число линий определяется данными через кластеризацию, а не задается заранее.

## Зависимости

Основные библиотеки из `main.py`:

- `torch`
- `numpy`
- `opencv-python`
- `scikit-learn`
- `hdbscan` (опционально, если `clustering_method=hdbscan`)
- `matplotlib` (опционально для визуализации)

## API (FastAPI)

Добавлен API-сервис для инференса и кластеризации по входному изображению:

- `POST /v1/cluster`
- `GET /health`
- `GET /model/info`

### Что возвращает `POST /v1/cluster`

- `num_clusters` — количество кластеров;
- `clusters` — словарь `cluster_id -> [[x, y], ...]`;
- `cluster_pixel_values` — словарь `cluster_id -> [[r, g, b], ...]`;
- `noise` — точки, помеченные алгоритмом кластеризации как шум;
- `meta` — служебная информация и использованные параметры.

### Параметры метода

Параметры передаются в body как `multipart/form-data` вместе с файлом:

- файл: `file`
- параметры модели/кластеризации:
  - `clustering_method` (`dbscan` | `hdbscan`)
  - `eps`, `min_samples`, `l2_normalize`
  - `auto_eps`, `auto_eps_k`, `auto_eps_q`
  - `use_spatial`, `spatial_weight`
- параметры выбора candidate-пикселей:
  - `candidate_method` (`non_white` | `canny` | `all_pixels`)
  - `white_threshold`
  - `canny_threshold1`, `canny_threshold2`, `canny_aperture_size`, `canny_l2gradient`, `canny_dilate_iter`
  - `max_candidate_points`

### Установка зависимостей для API

```bash
pip install -r requirements_api.txt
```

### Конфигурация через `.env`

Добавьте/обновите `.env` в корне проекта:

```env
MODEL_CHECKPOINT_PATH=artifacts/ckpt_best.pt
MODEL_EMB_DIM=16
MODEL_DEVICE=
API_AUTH_ENABLED=true
API_AUTH_USERNAME=admin
API_AUTH_PASSWORD=change_me
MAX_UPLOAD_BYTES=10485760
```

### Запуск API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Пример запроса

```bash
curl -X POST "http://127.0.0.1:8000/v1/cluster" \
  -u "admin:change_me" \
  -F "file=@/path/to/image.png" \
  -F "clustering_method=dbscan" \
  -F "eps=0.3" \
  -F "min_samples=20" \
  -F "candidate_method=non_white" \
  -F "white_threshold=245"
```

Если `API_AUTH_ENABLED=true`, метод `POST /v1/cluster` защищен Basic Auth.
