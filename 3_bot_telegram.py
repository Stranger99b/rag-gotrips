"""
Шаг 3: Telegram бот для тестирования RAG.

Запуск: TELEGRAM_BOT_TOKEN=xxx ANTHROPIC_API_KEY=xxx python3 3_bot_telegram.py

Команды:
  /start  — приветствие
  /reset  — сбросить историю переписки
  /debug  — показать последние найденные примеры из Qdrant
"""

import logging
import os
from telegram import Update
from telegram.error import TimedOut, NetworkError
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    ContextTypes, filters,
)
from rag_engine import RAGEngine
from config import TELEGRAM_TOKEN

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Глобальный движок (загружается один раз при старте)
engine: RAGEngine | None = None

# История чата: {chat_id: [{"role": ..., "content": ...}]}
# Храним только последние 10 ходов чтобы не раздувать контекст
MAX_HISTORY = 10
chat_histories: dict[int, list] = {}
# Последние найденные примеры для /debug
last_examples: dict[int, list] = {}


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    chat_histories[chat_id] = []
    await update.message.reply_text(
        "Привет! Я тестовый бот GoTrips. Напишите вопрос о туре — отвечу как менеджер.\n\n"
        "/reset — сбросить историю\n/debug — показать примеры из базы"
    )


async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    chat_histories[chat_id] = []
    await update.message.reply_text("История сброшена.")


async def cmd_debug(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    examples = last_examples.get(chat_id, [])
    if not examples:
        await update.message.reply_text("Нет данных. Сначала задайте вопрос.")
        return

    lines = []
    for i, ex in enumerate(examples, 1):
        date = ex.get("date", "?")
        channel = ex.get("channel", "?")
        lines.append(f"[{i}] {date} / {channel}")
        lines.append(f"К: {ex['client_text'][:120]}")
        lines.append(f"М: {ex['manager_text'][:120]}")
        lines.append("")

    await update.message.reply_text("\n".join(lines)[:4000])


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text.strip()

    if not user_text:
        return

    history = chat_histories.get(chat_id, [])

    await update.message.chat.send_action("typing")

    # Поиск один раз — результаты идут и в /debug, и в answer()
    examples = engine.search(user_text)
    last_examples[chat_id] = examples

    try:
        reply = engine.answer(user_text, chat_history=history if history else None, examples=examples)
    except Exception as e:
        logger.error(f"Ошибка RAG: {e}")
        await update.message.reply_text("Произошла ошибка, попробуйте ещё раз.")
        return

    # Обновляем историю (только сообщение клиента и ответ, без блока примеров)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    # Ограничиваем историю
    if len(history) > MAX_HISTORY * 2:
        history = history[-(MAX_HISTORY * 2):]
    chat_histories[chat_id] = history

    await update.message.reply_text(reply)


def main():
    global engine

    token = TELEGRAM_TOKEN or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        raise ValueError("Не задан TELEGRAM_BOT_TOKEN")

    logger.info("Инициализирую RAG движок...")
    engine = RAGEngine()

    app = Application.builder().token(token).connect_timeout(30).read_timeout(30).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("debug", cmd_debug))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        if isinstance(context.error, (TimedOut, NetworkError)):
            logger.warning(f"Telegram сетевая ошибка (авто-повтор): {context.error}")
            return
        logger.error(f"Ошибка: {context.error}", exc_info=context.error)

    app.add_error_handler(error_handler)
    logger.info("Бот запущен.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
