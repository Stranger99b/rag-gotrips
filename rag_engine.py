"""
RAG движок: поиск похожих диалогов + генерация ответа через Anthropic SDK.
Использует OAuth токен из ~/.claude/.credentials.json (авторизация Claude Code).
"""

import json
import subprocess
import time
from pathlib import Path

import anthropic
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_MODEL,
)

CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
TOP_K = 3
EXAMPLE_MAX_CHARS = 300
# Обновлять токен если осталось меньше 10 минут
TOKEN_REFRESH_MARGIN_SEC = 600

SYSTEM_PROMPT = """Ты — менеджер туристической компании GoTrips (Беларусь).
Твоя задача — вести переписку с клиентами в Instagram/Telegram и помогать им выбрать и забронировать тур.

Стиль общения:
- Дружелюбный, но профессиональный
- Пишешь кратко и по делу, без лишней воды
- Используешь эмодзи умеренно (как в примерах)
- Отвечаешь на русском языке
- Если клиент спрашивает о конкретных датах/ценах — уточняешь детали, не выдумываешь
- НЕ пиши "Менеджер:" перед ответом — просто сам ответ

Ниже приведены примеры реальных переписок менеджеров с клиентами.
Используй их как образец стиля и содержания ответов."""


def _load_token() -> tuple[str, int]:
    """Возвращает (access_token, expires_at_ms)."""
    data = json.loads(CREDENTIALS_FILE.read_text())
    oauth = data["claudeAiOauth"]
    return oauth["accessToken"], oauth["expiresAt"]


def _refresh_token() -> str:
    """Запускает claude --print чтобы обновить OAuth токен, возвращает свежий токен."""
    subprocess.run(
        ["claude", "--print", "--model", "haiku"],
        input="ping",
        capture_output=True, text=True, timeout=30
    )
    token, _ = _load_token()
    return token


def _get_valid_token() -> str:
    """Возвращает актуальный токен, при необходимости обновляет."""
    token, expires_at_ms = _load_token()
    remaining = expires_at_ms / 1000 - time.time()
    if remaining < TOKEN_REFRESH_MARGIN_SEC:
        print(f"Токен истекает через {remaining:.0f}с, обновляю...")
        token = _refresh_token()
    return token


def _new_qdrant() -> QdrantClient:
    """Создаёт свежее соединение (избегает stale-connection после долгого encode)."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)


class RAGEngine:
    def __init__(self):
        print("Загружаю модель эмбеддингов...")
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        print("Прогреваю модель...")
        self._model.encode("прогрев", normalize_embeddings=True)
        self._model.encode("прогрев второй", normalize_embeddings=True)
        print("RAG движок готов.")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        vec = self._model.encode(query, normalize_embeddings=True).tolist()
        results = _new_qdrant().query_points(
            collection_name=QDRANT_COLLECTION,
            query=vec,
            limit=top_k,
            with_payload=True,
        )
        return [r.payload for r in results.points]

    def answer(
        self,
        user_message: str,
        chat_history: list[dict] | None = None,
        examples: list[dict] | None = None,
    ) -> str:
        """
        user_message: текущее сообщение клиента
        chat_history: [{"role": "user"|"assistant", "content": "..."}]
        examples: предварительно найденные примеры из search() — если None, ищет сам
        """
        similar = examples if examples is not None else self.search(user_message)

        examples_block = "\n\n".join(
            f"---\nКлиент: {ex['client_text'][:EXAMPLE_MAX_CHARS]}\n"
            f"Менеджер: {ex['manager_text'][:EXAMPLE_MAX_CHARS]}"
            for ex in similar
        )

        parts = [f"Примеры переписок (стиль и содержание):\n{examples_block}\n"]

        if chat_history:
            history_lines = []
            for msg in chat_history:
                role = "Клиент" if msg["role"] == "user" else "Менеджер"
                history_lines.append(f"{role}: {msg['content']}")
            parts.append("История текущего диалога:\n" + "\n".join(history_lines))

        parts.append(f"Теперь ответь клиенту:\nКлиент: {user_message}")
        full_prompt = "\n\n".join(parts)

        token = _get_valid_token()
        client = anthropic.Anthropic(api_key=token)

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": full_prompt}],
        )
        return response.content[0].text.strip()
