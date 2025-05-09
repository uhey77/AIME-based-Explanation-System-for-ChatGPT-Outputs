import os
from dotenv import load_dotenv
import openai


def load_environment():
    """環境変数を読み込み、OpenAI APIキーを設定する"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI APIキーが設定されていません。.envファイルを確認してください。")
    openai.api_key = api_key
    print("環境変数を読み込み、OpenAI APIキーを設定しました。")


# モジュール読み込み時に環境変数をロード
load_environment()

# 必要に応じて他の設定値もここに追加可能
DEFAULT_MODEL_NAME = "gpt-4.1-mini"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_DB_TYPE = "chroma"
DEFAULT_DOC_DIRECTORY = "knowledge_base"
EVALUATION_DB_PATH = "explanation_metrics.db"
