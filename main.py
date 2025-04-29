# main.py
import sys
import config  # 環境変数を最初に読み込む
from system import EnhancedAIExplanationSystem
from ui import create_gradio_interface


def main():
    """メイン関数"""
    print("Enhanced AIME-based Explanation System for ChatGPT Outputs を起動中...")

    try:
        # 設定値を config モジュールから取得してシステムを初期化
        system_instance = EnhancedAIExplanationSystem(
            doc_directory=config.DEFAULT_DOC_DIRECTORY,
            model_name=config.DEFAULT_MODEL_NAME,
            chunk_size=config.DEFAULT_CHUNK_SIZE,
            chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
            db_type=config.DEFAULT_DB_TYPE
        )

        # Gradioインターフェースを作成
        demo = create_gradio_interface(system_instance)

        # Gradioアプリケーションを起動
        # share=True はデプロイ環境に応じて設定
        print("Gradioインターフェースを起動します...")
        # share=True は外部アクセスを許可する際に使用。セキュリティに注意。
        demo.launch(share=False)  # ローカルでのみアクセスする場合は False

    except ValueError as ve:  # APIキーがない場合など
        print(f"設定エラー: {ve}")
        print("アプリケーションを起動できませんでした。設定を確認してください。")
        sys.exit(1)  # エラーで終了
    except ImportError as ie:
        print(f"依存関係エラー: {ie}")
        print("必要なライブラリがインストールされていない可能性があります。")
        print("requirements.txt を確認し、'pip install -r requirements.txt' を実行してください。")
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        # スタックトレースを表示するなど、デバッグ情報を追加可能
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
