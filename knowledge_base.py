# knowledge_base.py
import os
import PyPDF2
from typing import List, Optional
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStore

# modelsモジュールからDocumentクラスをインポート
from models import Document
# configモジュールから設定値をインポート
import config


def load_documents(doc_directory: str) -> List[Document]:
    """
    指定されたディレクトリからドキュメントを読み込む

    Parameters:
    -----------
    doc_directory : str
        ドキュメントが格納されているディレクトリのパス

    Returns:
    --------
    List[Document]
        読み込まれたドキュメントオブジェクトのリスト
    """
    documents = []

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(doc_directory):
        os.makedirs(doc_directory)
        print(f"ディレクトリ '{doc_directory}' を作成しました。文書を追加してください。")
        return documents

    for filename in os.listdir(doc_directory):
        file_path = os.path.join(doc_directory, filename)
        metadata = {'filename': filename}

        try:
            # PDFファイルの処理
            if filename.lower().endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() or ""  # None対策
                    if text:
                        documents.append(Document(text, metadata))
                    else:
                        print(f"警告: PDFファイル '{filename}' からテキストを抽出できませんでした。")


            # テキストファイルの処理
            elif filename.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    if text:
                        documents.append(Document(text, metadata))
                    else:
                        print(f"警告: テキストファイル '{filename}' は空です。")

        except Exception as e:
            print(f"エラー: ファイル '{filename}' の読み込み中に問題が発生しました: {e}")


    print(f"ロードされたドキュメント数: {len(documents)}")
    return documents


def create_vector_db(
        documents: List[Document],
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
        db_type: str = config.DEFAULT_DB_TYPE
        ) -> Optional[VectorStore]:
    """
    ドキュメントリストからベクトルデータベースを作成する

    Parameters:
    -----------
    documents : List[Document]
        ベクトル化するドキュメントのリスト
    chunk_size : int
        文書分割サイズ
    chunk_overlap : int
        文書分割時のオーバーラップサイズ
    db_type : str
        使用するベクトルDBの種類 ("chroma" または "faiss")

    Returns:
    --------
    Optional[VectorStore]
        作成されたベクトルデータベースオブジェクト。ドキュメントがない場合はNone。
    """
    if not documents:
        print("ドキュメントが存在しません。ベクトルデータベースを構築できません。")
        return None

    try:
        # 文書の分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        # Document オブジェクトを直接渡す
        texts = text_splitter.split_documents(documents)

        if not texts:
            print("警告: ドキュメントの分割後、有効なテキストチャンクがありません。")
            return None

        # ベクトルDBの作成
        embeddings = OpenAIEmbeddings()

        if db_type.lower() == "faiss":
            vector_db = FAISS.from_documents(texts, embeddings)
            print("FAISS ベクトルデータベースを作成しました。")
        else:
            # デフォルトはChroma
            vector_db = Chroma.from_documents(texts, embeddings)
            print("Chroma ベクトルデータベースを作成しました。")

        return vector_db
    except Exception as e:
        print(f"エラー: ベクトルデータベースの作成中に問題が発生しました: {e}")
        return None


def initialize_sample_docs(doc_directory: str):
    """
    サンプル文書をナレッジベースディレクトリに追加する

    Parameters:
    -----------
    doc_directory : str
        サンプル文書を追加するディレクトリのパス
    """
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(doc_directory):
        os.makedirs(doc_directory)

    # サンプル文書内容 (元のコードからコピー)
    sample_docs = {
            "RAG_guide.txt": """# RAGの基礎ガイド

## 1. はじめに
Retrieval-Augmented Generation (RAG) は、生成AIモデルの回答品質を向上させるための重要な技術です。

## 2. RAGの仕組み
RAGは外部知識源から関連情報を検索し、その情報をLLMの入力に追加することで、より正確で最新の回答を生成します。

## 3. 主なメリット
- 最新性の向上: モデルの訓練データ以降の情報も利用可能
- 事実性の向上: 参照元を明示することでハルシネーションを減少
- ドメイン適応: 特定分野の文書を追加するだけで専門知識を拡張可能

## 4. 典型的な実装
1. ユーザークエリの受け取り
2. 関連文書の検索（ベクトル検索など）
3. 検索結果をプロンプトに組み込み
4. LLMによる回答生成
5. 引用・出典表示

## 5. 応用例
- 企業内部ナレッジベースを活用したチャットボット
- 最新研究論文を参照できる学術Q&Aシステム
- 製品マニュアルに基づくカスタマーサポート
""",
            "XAI_guide.txt": """# XAIの基礎ガイド

## 1. はじめに
Explainable AI (XAI) は、AIシステムの意思決定プロセスを人間が理解できるようにするための技術です。

## 2. XAIの重要性
- 信頼性向上: 結果の妥当性を検証可能に
- 法規制対応: GDPR等の説明責任要件に対応
- モデル改善: 問題点の特定と修正が容易に

## 3. 主なアプローチ
- ポストホック手法: 訓練済みモデルの出力を説明 (SHAP, LIME)
- 透明性重視手法: 解釈可能なモデル構造の採用 (決定木、線形モデル)
- プロセス可視化: 重要特徴や判断根拠の提示

## 4. AIモデル別の説明手法
- ディープラーニング: アテンション可視化、特徴重要度マップ
- 自然言語処理: トークン寄与度、関連テキスト強調
- 強化学習: 状態価値の可視化、方策説明

## 5. 実装のヒント
- ユーザー層に合わせた説明の調整
- 根拠データの透明な提示
- 説明と精度のバランス考慮
""",
            "AIME_concept.txt": """# AIME (Artificial Intelligence Model Explanation)

## 概要
AIMEは、AI・機械学習モデルが特定の出力や判断を行った理由を説明するための包括的フレームワークです。

## 主要コンポーネント
1. **根拠特定**: モデル出力に影響を与えた要素の特定
2. **判断プロセス再現**: モデルが情報をどう処理したかの説明
3. **確信度表示**: 出力の信頼性と不確実性の提示
4. **代替案提示**: 他の可能性や判断境界の説明

## 説明レベル
- **技術レベル**: モデル構造や数学的根拠に基づく詳細説明
- **概念レベル**: モデルロジックを簡略化した概念的説明
- **事例レベル**: 類似事例や比較による説明

## ユースケース
- 医療診断支援システムの判断根拠説明
- 金融審査の承認・拒否理由の透明化
- コンテンツレコメンデーションの推薦理由提示
- 自動運転車の判断プロセス説明

## 実装アプローチ
- 特徴重要度分析
- アテンションメカニズム可視化
- RAGと組み合わせた参照情報の透明化
- マルチモーダル説明（テキスト・ビジュアル併用）
""",
            "attribute_enhancement.txt": """# アトリビュート強化手法

## 概要
アトリビュート強化手法は、AIシステムの回答の根拠を強化し、信頼性を高めるための手法です。

## 主要な強化手法

### 1. 自己検証機能
AIが自身の回答を批判的に検証するステップを追加し、事実性と整合性を確認します。

#### 実装ポイント
- 回答生成後の検証ステップ追加
- 事実と推論の明確な区別
- 自己矛盾や論理的飛躍のチェック
- 確信度の定量的評価

### 2. 逐次検索
初期検索結果を分析し、必要に応じてクエリを改善して再検索を行う反復的プロセス。

#### 実装ポイント
- 初期回答の不足点に基づくクエリ修正
- 複数の検索戦略の並行実行
- 検索結果の多様性確保
- 反復的な情報収集と統合

### 3. 複数情報源の相互検証
異なる情報源から得られた情報を比較し、一致点と相違点を分析。

#### 実装ポイント
- 複数ソースからの情報収集
- 情報の一致度スコアリング
- 矛盾する情報の明示
- 情報源の信頼性評価

### 4. 確信度調整機能
回答の各部分に対する確信度を評価し、明示的に表示。

#### 実装ポイント
- 情報源の質と関連性に基づく確信度計算
- 推論の深さに応じた確信度調整
- 不確実性の明示的な伝達
- 確信度マップの可視化

## 応用事例
- 医療情報システムでの診断根拠提示
- 法律アドバイスシステムでの判例引用
- 学術Q&Aでの文献根拠付与
- ニュース要約での情報源検証
""",
            "evaluation_metrics.txt": """# AI説明の評価指標

## 概要
AI説明システムの品質を継続的に評価・改善するための指標と方法論。

## 主要評価指標

### 1. 説明の一貫性 (Coherence)
説明内容が論理的に一貫しているか、矛盾がないかを評価。

#### 測定方法
- 文間の論理的つながりの分析
- 主張と根拠の関係性評価
- 矛盾検出アルゴリズムの適用
- 専門家による質的評価

### 2. 関連性 (Relevance)
説明がユーザーの質問や要求に適切に対応しているかを評価。

#### 測定方法
- 質問-回答の意味的関連度計算
- キーワード一致率分析
- ユーザーフィードバックの収集
- コンテキスト理解度テスト

### 3. 完全性 (Completeness)
説明が必要な情報を漏れなく含んでいるかを評価。

#### 測定方法
- 情報網羅性スコアの計算
- 欠落情報の自動検出
- 説明の詳細度評価
- ユーザー満足度調査

### 4. 事実的正確性 (Factual Accuracy)
説明内容が事実と一致しているかを評価。

#### 測定方法
- 知識ベースとの整合性確認
- 事実検証アルゴリズムの適用
- 情報源との一致度測定
- 専門家による検証

### 5. ユーザー理解度 (User Comprehension)
ユーザーが説明を理解できるかを評価。

#### 測定方法
- 理解度確認クイズの実施
- 読みやすさ指標の計算
- ユーザーからのフィードバック収集
- 専門用語使用率の分析

## 継続的改善のフレームワーク
1. ベースライン測定の実施
2. 定期的な評価サイクルの確立
3. A/Bテストによる手法比較
4. 定量・定性的フィードバックの統合
5. 改善履歴の記録と分析

## 実装例
- 自動評価パイプラインの構築
- ユーザーフィードバックUI
- 専門家レビューセッション
- 性能ダッシュボードの開発
"""
        }

    # サンプル文書をファイルに書き込む
    added_count = 0
    for filename, content in sample_docs.items():
        file_path = os.path.join(doc_directory, filename)
        if not os.path.exists(file_path):
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"サンプル文書 '{filename}' を追加しました。")
                added_count += 1
            except Exception as e:
                print(f"エラー: サンプル文書 '{filename}' の書き込み中に問題が発生しました: {e}")

    if added_count == 0:
        print("すべてのサンプル文書は既に存在します。")
