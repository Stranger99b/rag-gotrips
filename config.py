import os

SALEBOT_DIALOG_DIR = "/home/user/salebot_dialog"
DATA_DIR = "/home/user/rag_gotrips/data"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "gotrips_dialogs"

EMBEDDING_MODEL = "/home/user/models/ru-en-RoSBERTa"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Минимальная длина текста для включения в индекс
MIN_CLIENT_MSG_LEN = 15
MIN_MANAGER_MSG_LEN = 30

# Максимальное число сообщений менеджера в одной Q&A паре
MAX_MANAGER_MSGS_PER_QA = 5
