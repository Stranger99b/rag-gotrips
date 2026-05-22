"""
RAG движок: поиск похожих диалогов + генерация ответа через claude CLI.
"""

import subprocess
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import (
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_MODEL,
)

def _new_qdrant() -> QdrantClient:
    """Создаёт свежее соединение (избегает stale-connection после долгого encode)."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)

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

TOP_K = 5
CLAUDE_TIMEOUT = 60


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

    def answer(self, user_message: str, chat_history: list[dict] | None = None) -> str:
        """
        user_message: текущее сообщение клиента
        chat_history: [{"role": "user"|"assistant", "content": "..."}]
        """
        similar = self.search(user_message)

        examples_block = "\n\n".join(
            f"---\nКлиент: {ex['client_text']}\nМенеджер: {ex['manager_text']}"
            for ex in similar
        )

        # Собираем полный промпт: примеры + история + текущий вопрос
        parts = [f"Примеры переписок (стиль и содержание):\n{examples_block}\n"]

        if chat_history:
            history_lines = []
            for msg in chat_history:
                role = "Клиент" if msg["role"] == "user" else "Менеджер"
                history_lines.append(f"{role}: {msg['content']}")
            parts.append("История текущего диалога:\n" + "\n".join(history_lines))

        parts.append(f"Теперь ответь клиенту:\nКлиент: {user_message}")

        full_prompt = "\n\n".join(parts)

        result = subprocess.run(
            [
                "claude", "--print",
                "--model", "claude-sonnet-4-6",
                "--system-prompt", SYSTEM_PROMPT,
            ],
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=CLAUDE_TIMEOUT,
        )

        if result.returncode != 0:
            raise RuntimeError(f"claude CLI error: {result.stderr[:300]}")

        return result.stdout.strip()
