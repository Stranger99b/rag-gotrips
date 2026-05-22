"""
Шаг 2: Создание эмбеддингов и загрузка в Qdrant.

Читает data/qa_pairs.jsonl, создаёт векторы для client_text,
загружает в коллекцию gotrips_dialogs.

Запуск: python3 2_index_qdrant.py
Поддерживает чекпоинты: если прервать и запустить снова — продолжит с места остановки.
"""

import json
import os
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from config import (
    DATA_DIR, QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_MODEL
)

BATCH_SIZE = 64
CHECKPOINT_EVERY = 10  # сохранять чекпоинт каждые N батчей
CHECKPOINT_FILE = os.path.join(DATA_DIR, "embeddings_checkpoint.npz")


def load_qa_pairs(path: str) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def encode_all(pairs: list[dict], model) -> list[list[float]]:
    """Кодирует все тексты с поддержкой чекпоинтов."""
    start_idx = 0
    all_vectors = []

    if os.path.exists(CHECKPOINT_FILE):
        print(f"Найден чекпоинт, загружаю...")
        data = np.load(CHECKPOINT_FILE)
        all_vectors = data["vectors"].tolist()
        start_idx = len(all_vectors)
        print(f"  Уже закодировано: {start_idx} из {len(pairs)}")

    remaining = pairs[start_idx:]
    if not remaining:
        print("  Все тексты уже закодированы.")
        return all_vectors

    batches = range(0, len(remaining), BATCH_SIZE)
    for batch_num, batch_start in enumerate(tqdm(batches, desc="Кодирую")):
        batch = remaining[batch_start: batch_start + BATCH_SIZE]
        texts = [p["client_text"] for p in batch]
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
        all_vectors.extend(vecs.tolist())

        if (batch_num + 1) % CHECKPOINT_EVERY == 0:
            np.savez(CHECKPOINT_FILE, vectors=np.array(all_vectors))

    np.savez(CHECKPOINT_FILE, vectors=np.array(all_vectors))
    print(f"  Закодировано итого: {len(all_vectors)}")
    return all_vectors


def upload_to_qdrant(client: QdrantClient, pairs: list[dict], vectors: list):
    """Загружает все точки в Qdrant батчами."""
    print(f"\nЗагружаю {len(vectors)} точек в Qdrant...")
    upload_batch = 256  # для загрузки можно крупнее

    for batch_start in tqdm(range(0, len(pairs), upload_batch), desc="Загружаю"):
        batch_pairs = pairs[batch_start: batch_start + upload_batch]
        batch_vecs = vectors[batch_start: batch_start + upload_batch]

        points = []
        for i, (pair, vec) in enumerate(zip(batch_pairs, batch_vecs)):
            points.append(PointStruct(
                id=batch_start + i,
                vector=vec,
                payload={
                    "client_text": pair["client_text"],
                    "manager_text": pair["manager_text"],
                    "context_before": pair["context_before"],
                    **pair["metadata"],
                },
            ))

        retries = 3
        for attempt in range(retries):
            try:
                client.upsert(collection_name=QDRANT_COLLECTION, points=points)
                break
            except Exception as e:
                if attempt < retries - 1:
                    print(f"\n  Ошибка загрузки (попытка {attempt+1}): {e}. Повтор...")
                    time.sleep(5)
                else:
                    raise


def main(reset: bool = True):
    qa_path = os.path.join(DATA_DIR, "qa_pairs.jsonl")
    if not os.path.exists(qa_path):
        print(f"Файл {qa_path} не найден. Сначала запустите 1_prepare_data.py")
        sys.exit(1)

    print(f"Загружаю Q&A пары...")
    pairs = load_qa_pairs(qa_path)
    print(f"  Всего пар: {len(pairs)}")

    print(f"\nЗагружаю модель {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    model.max_seq_length = 128
    dim = model.get_sentence_embedding_dimension()
    print(f"  Размерность векторов: {dim}")

    # Фаза 1: кодирование (долго, ~11 часов для RoBERTa)
    print(f"\n--- Фаза 1: кодирование ---")
    vectors = encode_all(pairs, model)

    # Фаза 2: загрузка в Qdrant (быстро)
    print(f"\n--- Фаза 2: загрузка в Qdrant ---")
    print(f"Подключаюсь к Qdrant {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)

    if reset and client.collection_exists(QDRANT_COLLECTION):
        print(f"  Удаляю старую коллекцию {QDRANT_COLLECTION}...")
        client.delete_collection(QDRANT_COLLECTION)

    if not client.collection_exists(QDRANT_COLLECTION):
        print(f"  Создаю коллекцию {QDRANT_COLLECTION}...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    upload_to_qdrant(client, pairs, vectors)

    info = client.get_collection(QDRANT_COLLECTION)
    print(f"\nГотово!")
    print(f"  Загружено точек: {info.points_count}")
    print(f"  Коллекция:       {QDRANT_COLLECTION}")

    # Чекпоинт больше не нужен
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"  Чекпоинт удалён.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reset", action="store_true", help="Не пересоздавать коллекцию")
    args = parser.parse_args()
    main(reset=not args.no_reset)
