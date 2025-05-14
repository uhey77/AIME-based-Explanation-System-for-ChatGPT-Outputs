# type: ignore
import os
from dotenv import load_dotenv
import openai

def load_environment():
    """.env から環境変数を読み込み、OpenAI APIキーを設定する"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI APIキーが設定されていません。.envファイルか環境変数で設定してください。")
    openai.api_key = api_key
    print("環境変数を読み込み、OpenAI APIキーを設定しました。")

# アプリ起動時に呼び出し
load_environment()

# --- 以下はデフォルト設定 ---
DEFAULT_DOC_DIRECTORY = os.getenv("DOC_DIRECTORY", "knowledge_base")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
DEFAULT_DB_TYPE = os.getenv("DB_TYPE", "chroma")

EVALUATION_DB_PATH = os.getenv("EVALUATION_DB_PATH", "explanation_metrics.db")

THINKING_START_TAG = "<think>"
THINKING_END_TAG   = "</think>"

APP_NAME        = "AIME-based Explanation System for ChatGPT Outputs"
APP_VERSION     = "1.1.0"