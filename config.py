import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # PostgreSQL
    DB_HOST     = os.getenv("DB_HOST", "localhost")
    DB_PORT     = os.getenv("DB_PORT", "5432")
    DB_NAME     = os.getenv("DB_NAME", "smartfolio-ai")
    DB_USER     = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "mahendra")

    DATABASE_URL = (
        f"postgresql://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod")
    DEBUG      = os.getenv("DEBUG", "true").lower() == "true"

    # LSTM model paths
    LSTM_MODEL_PATH    = os.getenv("LSTM_MODEL_PATH", "ai/saved_model/lstm_model.h5")
    TOKENIZER_PATH     = os.getenv("TOKENIZER_PATH", "ai/saved_model/tokenizer.pkl")

    # LSTM training settings
    LSTM_SEQUENCE_LEN  = 10
    LSTM_EPOCHS        = 50
    LSTM_BATCH_SIZE    = 64