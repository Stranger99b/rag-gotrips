"""
Шаг 1: Очистка и разбивка диалогов на Q&A пары.

Результат: data/qa_pairs.jsonl
Каждая строка:
{
  "client_text": "...",
  "manager_text": "...",
  "context_before": "...",   // 1-2 реплики до вопроса (контекст)
  "metadata": {
    "date": "2026-04-15",
    "messenger": "Instagram",
    "source_type": "Органика",
    "dialog_id": 932137042
  }
}
"""

import json
import glob
import os
import re
from datetime import datetime
from tqdm import tqdm
from config import SALEBOT_DIALOG_DIR, DATA_DIR, MIN_CLIENT_MSG_LEN, MIN_MANAGER_MSG_LEN, MAX_MANAGER_MSGS_PER_QA

# Шаблонные фразы ботовых сообщений — такие Q&A пары не нужны
BOT_TEMPLATES = [
    "привет, это бот gotrips",
    "для начала нажмите привет",
    "ссылка для оплаты туристической услуги",
    "при удаленном бронировании тура договор будет подписан",
    "на ваш номер телефона будет отправлено смс",
    "подготовлю и отправлю вам договор",
    "оплата в течение 5 дней",
    "менеджер не найден. ожидайте подключения",
    "если хочешь посмотреть программы туров",
]

# Технические/шаблонные вопросы клиентов — тоже пропускаем
CLIENT_SKIP_PATTERNS = [
    r"^(привет|хорошо|спасибо|ок|окей|понятно|понял|ждем|ладно|договорились)[\s!?.]*$",
    r"^(да|нет)[\s!?.]*$",
    r"^\d{4,6}$",           # коды подтверждения
    r"^client_wall_reply",  # Instagram-уведомления о комментариях
    r"^https?://",          # сообщения только со ссылкой
    r"^\+$",                # просто плюс
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r\n", "\n", text)
    # Удаляем строки с системными событиями типа "change_responsible_client: ..."
    text = re.sub(r"^change_\w+:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def is_bot_template(text: str) -> bool:
    t = text.lower()
    return any(tmpl in t for tmpl in BOT_TEMPLATES)


def is_skip_client(text: str) -> bool:
    t = text.strip()
    for pattern in CLIENT_SKIP_PATTERNS:
        if re.match(pattern, t, re.IGNORECASE):
            return True
    return False


def extract_qa_pairs(history: list, date: str, messenger: str, source_type: str, dialog_id: int) -> list:
    """
    Из списка сообщений извлекает пары (вопрос клиента → ответ менеджера).
    history отсортирован по времени (старые сначала).
    """
    if not history:
        return []

    pairs = []
    i = 0
    context_window = []  # последние 2 реплики для контекста

    while i < len(history):
        msg = history[i]
        text = clean_text(msg.get("text") or "")

        if not text:
            i += 1
            continue

        # Клиентское сообщение(я) — могут идти несколько подряд
        if msg.get("client_replica"):
            client_msgs = [text]
            j = i + 1
            while j < len(history) and history[j].get("client_replica"):
                t = clean_text(history[j].get("text") or "")
                if t:
                    client_msgs.append(t)
                j += 1

            client_text = "\n".join(client_msgs)

            # Ищем ответы менеджера после
            manager_msgs = []
            k = j
            count = 0
            while k < len(history) and not history[k].get("client_replica") and count < MAX_MANAGER_MSGS_PER_QA:
                t = clean_text(history[k].get("text") or "")
                if t:
                    manager_msgs.append(t)
                    count += 1
                k += 1

            manager_text = "\n".join(manager_msgs)

            # Фильтры качества
            if (
                len(client_text) >= MIN_CLIENT_MSG_LEN
                and len(manager_text) >= MIN_MANAGER_MSG_LEN
                and not is_skip_client(client_text)
                and not is_bot_template(manager_text)
            ):
                context_str = "\n".join(
                    f"{'Клиент' if m.get('client_replica') else 'Менеджер'}: {clean_text(m.get('text') or '')}"
                    for m in context_window[-4:]  # последние 4 реплики = 2 хода
                    if clean_text(m.get("text") or "")
                )
                pairs.append({
                    "client_text": client_text,
                    "manager_text": manager_text,
                    "context_before": context_str,
                    "metadata": {
                        "date": date,
                        "messenger": messenger,
                        "source_type": source_type,
                        "dialog_id": dialog_id,
                    },
                })

            # Обновляем контекст
            context_window.extend(history[i:j])
            i = j

        else:
            context_window.append(msg)
            i += 1

    return pairs


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "qa_pairs.jsonl")

    all_files = sorted(glob.glob(f"{SALEBOT_DIALOG_DIR}/*/dialogs_*.json"))
    print(f"Найдено файлов: {len(all_files)}")

    total_dialogs = 0
    total_pairs = 0
    skipped_empty = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for fpath in tqdm(all_files, desc="Обрабатываю файлы"):
            date = os.path.basename(os.path.dirname(fpath))

            with open(fpath, encoding="utf-8") as f:
                dialogs = json.load(f)

            for dialog in dialogs:
                total_dialogs += 1
                history_raw = dialog.get("history_json", "[]")

                try:
                    history = json.loads(history_raw) if isinstance(history_raw, str) else history_raw
                except (json.JSONDecodeError, TypeError):
                    skipped_empty += 1
                    continue

                if not history:
                    skipped_empty += 1
                    continue

                # Сортируем по времени (старые сначала)
                history.sort(key=lambda m: m.get("created_at") or "")

                pairs = extract_qa_pairs(
                    history=history,
                    date=date,
                    messenger=dialog.get("messenger", ""),
                    source_type=dialog.get("source_type", ""),
                    dialog_id=dialog.get("client_id", 0),
                )

                for pair in pairs:
                    out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    total_pairs += 1

    print(f"\nГотово:")
    print(f"  Диалогов обработано:  {total_dialogs}")
    print(f"  Диалогов без истории: {skipped_empty}")
    print(f"  Q&A пар извлечено:    {total_pairs}")
    print(f"  Сохранено в:          {output_path}")

    # Статистика по мессенджерам
    from collections import Counter
    messengers = Counter()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            messengers[pair["metadata"]["messenger"]] += 1
    print(f"\nПо мессенджерам:")
    for m, c in messengers.most_common():
        print(f"  {m or 'неизвестно':20s}: {c}")


if __name__ == "__main__":
    main()
