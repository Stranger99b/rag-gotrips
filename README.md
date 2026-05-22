# RAG GoTrips — AI-менеджер туристической компании

RAG-система на базе реальных диалогов менеджеров GoTrips с клиентами.  
Бот автоматически отвечает на вопросы клиентов в стиле живого менеджера,  
используя примеры из базы знаний.

## Архитектура

```
Сообщение клиента
       ↓
Эмбеддинг запроса (ru-en-RoSBERTa)
       ↓
Поиск похожих диалогов (Qdrant, cosine similarity)
       ↓
Формирование промпта (5 примеров + история чата)
       ↓
Генерация ответа (Claude Sonnet через claude CLI)
       ↓
Ответ в Telegram
```

## Стек

| Компонент | Технология |
|-----------|-----------|
| Эмбеддинги | [ai-forever/ru-en-RoSBERTa](https://huggingface.co/ai-forever/ru-en-RoSBERTa) (1024-dim) |
| Векторная БД | [Qdrant](https://qdrant.tech/) (Docker, порт 6333) |
| LLM | Claude Sonnet 4.6 через `claude` CLI |
| Telegram-бот | python-telegram-bot v21+ |
| Данные | ~7 700 Q&A пар из реальных переписок (Instagram, Telegram, Viber) |

## Файлы проекта

```
config.py            — настройки (пути, Qdrant, модель, токены)
1_prepare_data.py    — шаг 1: очистка диалогов и извлечение Q&A пар
2_index_qdrant.py    — шаг 2: создание эмбеддингов и загрузка в Qdrant
rag_engine.py        — RAG движок (поиск + генерация ответа)
3_bot_telegram.py    — Telegram-бот (тестовый интерфейс)
```

## Установка

### 1. Зависимости

```bash
pip install -r requirements.txt
```

> **Важно:** `torch` CPU-версия (~2 GB) устанавливается вручную:
> ```bash
> # Скачать с https://download.pytorch.org/whl/cpu
> pip install torch-2.x.x+cpu-*.whl
> ```

### 2. Модель эмбеддингов

Скачать [ai-forever/ru-en-RoSBERTa](https://huggingface.co/ai-forever/ru-en-RoSBERTa) и положить в `~/models/ru-en-RoSBERTa/`.  
Путь настраивается в `config.py` → `EMBEDDING_MODEL`.

### 3. Qdrant

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -v ~/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

### 4. Переменные окружения

```bash
cp .env.example .env
# Заполнить TELEGRAM_BOT_TOKEN
```

`ANTHROPIC_API_KEY` не нужен — используется `claude` CLI (требуется активная подписка Claude).

## Запуск

### Шаг 1: Подготовка данных

```bash
# Указать путь к диалогам Salebot в config.py (SALEBOT_DIALOG_DIR)
python3 1_prepare_data.py
# → создаёт data/qa_pairs.jsonl
```

### Шаг 2: Индексация

```bash
python3 2_index_qdrant.py
# → создаёт коллекцию gotrips_dialogs в Qdrant
# Время: ~4 часа на CPU для 7700 пар (RoBERTa-large)
```

Поддерживаются чекпоинты: если прервать и запустить снова — продолжит с места остановки.

### Шаг 3: Запуск бота

```bash
source .env && python3 3_bot_telegram.py
```

### Команды бота

| Команда | Действие |
|---------|---------|
| `/start` | Начать новый диалог (сбросить историю) |
| `/reset` | То же что /start |
| `/debug` | Показать последние 5 найденных примеров из Qdrant |

## Структура данных

Каждая Q&A пара в `data/qa_pairs.jsonl`:

```json
{
  "client_text": "Здравствуйте, хочу узнать про туры в Турцию",
  "manager_text": "Добрый день! Есть отличные варианты на июль...",
  "context_before": "предыдущие сообщения диалога",
  "metadata": {
    "dialog_id": "123",
    "channel": "instagram",
    "date": "2024-01-15"
  }
}
```

## Источник данных

Диалоги из CRM Salebot (каналы: Instagram, Telegram, Viber, Online-чат).  
Перед индексацией данные проходят очистку:
- Убираются шаблонные бот-сообщения
- Убираются системные события (`change_responsible_client`)
- Убираются Instagram-уведомления о лайках/подписках
- Фильтруются слишком короткие реплики

## Дальнейшие планы

- [ ] Очистка персональных данных (ФИО, телефоны) из базы знаний
- [ ] Интеграция напрямую в Salebot (каналы Instagram / Telegram)
- [ ] A/B тест: ответы бота vs реальных менеджеров
