"""
AIME-based Explanation System for ChatGPT Outputs
-------------------------------------------------
このシステムは2つの主要機能を提供します：
1. ユーザーの質問に対するChatGPTの回答の根拠を示す
2. ChatGPTが特定の回答をした理由を説明する
"""

import os
from dotenv import load_dotenv
import openai
# 非推奨インポートを修正
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import Dict, List, Any, Optional
import gradio as gr
import PyPDF2
import json
import shutil

# 環境変数の読み込み
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class Document:
    """ドキュメントを表すクラス"""
    def __init__(self, text, metadata):
        self.page_content = text
        self.metadata = metadata


class AIExplanationSystem:
    """ChatGPTの出力説明システムのメインクラス"""

    def __init__(self, doc_directory="knowledge_base", model_name="gpt-4", chunk_size=1000, chunk_overlap=100):
        """
        初期化メソッド
    
        Parameters:
        -----------
        doc_directory : str
            ナレッジベースとなる文書が保存されているディレクトリパス
        model_name : str
            使用するOpenAIのモデル名
        chunk_size : int
            文書分割サイズ
        chunk_overlap : int
            文書分割時のオーバーラップサイズ
        """
        self.doc_directory = doc_directory
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # サンプル文書を追加
        self._initialize_sample_docs()

        # ナレッジベースの構築
        self.documents = self._load_documents()
        self.vector_db = self._create_vector_db()

        # QAチェーンの構築
        self.qa_chain = self._create_qa_chain()

        # 説明システム用のプロンプトテンプレート
        self.explanation_prompt = """
        以下の質問と回答を分析し、ChatGPTがこの回答を生成した理由や根拠を説明してください。
        回答と相関性の高い知識ベースの情報も引用してください。

        ユーザーの質問: {question}
        ChatGPTの回答: {answer}

        説明:
        """

    def _initialize_sample_docs(self):
        """サンプル文書を追加する"""
        # ディレクトリが存在しない場合は作成
        if not os.path.exists(self.doc_directory):
            os.makedirs(self.doc_directory)
    
        # サンプル文書内容
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
"""
        }

        # サンプル文書をファイルに書き込む
        for filename, content in sample_docs.items():
            file_path = os.path.join(self.doc_directory, filename)
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"サンプル文書 '{filename}' を追加しました。")

    def _load_documents(self) -> List[Document]:
        """ドキュメントの読み込み"""
        documents = []

        # ディレクトリが存在しない場合は作成
        if not os.path.exists(self.doc_directory):
            os.makedirs(self.doc_directory)
            print(f"ディレクトリ '{self.doc_directory}' を作成しました。文書を追加してください。")
            return documents

        for filename in os.listdir(self.doc_directory):
            file_path = os.path.join(self.doc_directory, filename)
            metadata = {'filename': filename}

            # PDFファイルの処理
            if filename.endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ''
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text()
                    documents.append(Document(text, metadata))

            # テキストファイルの処理
            elif filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    documents.append(Document(text, metadata))

        print(f"ロードされたドキュメント数: {len(documents)}")
        return documents

    def _create_vector_db(self):
        """ベクトルデータベースの作成"""
        if not self.documents:
            print("ドキュメントが存在しません。知識ベースを構築できません。")
            return None

        # 文書の分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(self.documents)

        # ベクトルDBの作成
        embeddings = OpenAIEmbeddings()
        vector_db = Chroma.from_documents(texts, embeddings)

        return vector_db

    def _create_qa_chain(self):
        """質問応答チェーンの作成"""
        if not self.vector_db:
            print("ベクトルDBが作成されていないため、QAチェーンを作成できません。")
            return None
            
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name=self.model_name),
            chain_type="stuff",
            retriever=self.vector_db.as_retriever()
        )
        
        return qa_chain
    
    def get_answer(self, question: str) -> Dict[str, Any]:
        """質問に対する回答を取得"""
        if not self.qa_chain:
            return {"answer": "知識ベースが構築されていないため、回答できません。", "sources": []}
        
        # 回答の取得
        result = self.qa_chain.invoke({"query": question})
        
        # 回答と関連ドキュメントを抽出
        answer = result["result"]
        
        # 関連ドキュメントの取得を試みる
        try:
            # 関連ドキュメントを取得するための検索
            retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
            sources = retriever.get_relevant_documents(question)
            source_list = [
                {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                for doc in sources
            ]
        except Exception as e:
            print(f"関連ドキュメントの取得中にエラーが発生しました: {e}")
            source_list = []
        
        return {"answer": answer, "sources": source_list}
    
    def explain_answer(self, question: str, answer: str) -> str:
        """回答の説明を生成"""
        if not self.vector_db:
            return "知識ベースが構築されていないため、説明を生成できません。"
        
        # 説明を生成するためのプロンプトを作成
        prompt = self.explanation_prompt.format(
            question=question,
            answer=answer
        )
        
        # 回答ベースの検索と関連ドキュメントの取得
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(question + " " + answer)
        
        # 関連ドキュメントからコンテキストを抽出
        context = "\n\n関連情報:\n"
        for i, doc in enumerate(relevant_docs):
            context += f"[出典 {i+1}] {doc.metadata.get('filename', 'Unknown')}:\n"
            context += f"{doc.page_content}\n\n"
        
        # OpenAI APIを使って説明を生成
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "あなたはAI回答の根拠を説明する専門家です。"},
                {"role": "user", "content": prompt + context}
            ]
        )
        
        explanation = response.choices[0].message.content
        
        return explanation
    
    def chat_and_explain(self, question: str) -> Dict[str, Any]:
        """チャットと説明を同時に行う"""
        # 回答を取得
        result = self.get_answer(question)
        
        # 説明を生成
        explanation = self.explain_answer(question, result["answer"])
        
        return {
            "answer": result["answer"],
            "explanation": explanation,
            "sources": result["sources"]
        }
    
    def add_document_from_text(self, filename: str, content: str) -> bool:
        """テキストからドキュメントを追加"""
        try:
            file_path = os.path.join(self.doc_directory, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # ドキュメントリストに追加
            self.documents.append(Document(content, {'filename': filename}))
            
            # ベクトルDBを再構築
            self.vector_db = self._create_vector_db()
            
            # QAチェーンを再構築
            self.qa_chain = self._create_qa_chain()
            
            return True
        except Exception as e:
            print(f"文書の追加中にエラーが発生しました: {e}")
            return False

def create_gradio_interface():
    """Gradioインターフェースの作成"""
    system = AIExplanationSystem()
    
    with gr.Blocks(title="AIME-based ChatGPT Explanation System") as demo:
        gr.Markdown("# AIME-based ChatGPT Explanation System")
        gr.Markdown("このシステムは、ChatGPTの回答とその根拠を説明します。")
        
        with gr.Tab("チャットと説明"):
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="質問", 
                        placeholder="ここに質問を入力してください..."
                    )
                    submit_btn = gr.Button("送信")
                
            with gr.Row():
                with gr.Column():
                    answer_output = gr.Textbox(label="回答")
                with gr.Column():
                    explanation_output = gr.Textbox(label="回答の説明")
            
            sources_output = gr.JSON(label="参照ソース")
            
            def handle_chat_and_explain(question):
                result = system.chat_and_explain(question)
                return result["answer"], result["explanation"], result["sources"]
            
            submit_btn.click(
                handle_chat_and_explain,
                inputs=[question_input],
                outputs=[answer_output, explanation_output, sources_output]
            )
        
        with gr.Tab("既存の回答を説明"):
            with gr.Row():
                with gr.Column():
                    existing_question = gr.Textbox(
                        label="質問", 
                        placeholder="ここに質問を入力してください..."
                    )
                    existing_answer = gr.Textbox(
                        label="ChatGPTの回答", 
                        placeholder="ここにChatGPTの回答を入力してください..."
                    )
                    explain_btn = gr.Button("説明を生成")
            
            with gr.Row():
                explanation_only_output = gr.Textbox(label="説明")
            
            def handle_explain_only(question, answer):
                explanation = system.explain_answer(question, answer)
                return explanation
            
            explain_btn.click(
                handle_explain_only,
                inputs=[existing_question, existing_answer],
                outputs=[explanation_only_output]
            )
            
        with gr.Tab("文書管理"):
            with gr.Row():
                with gr.Column():
                    document_name = gr.Textbox(
                        label="文書名 (例: example.txt)", 
                        placeholder="ファイル名を入力してください..."
                    )
                    document_content = gr.Textbox(
                        label="文書内容", 
                        placeholder="ここに文書の内容を入力してください...",
                        lines=10
                    )
                    add_doc_btn = gr.Button("文書を追加")
            
            add_result = gr.Textbox(label="追加結果")
            
            def handle_add_document(name, content):
                if not name.strip():
                    return "エラー: ファイル名を入力してください。"
                
                if not content.strip():
                    return "エラー: 文書内容を入力してください。"
                
                # 拡張子の確認と追加
                if not name.endswith('.txt'):
                    name = name + '.txt'
                
                success = system.add_document_from_text(name, content)
                if success:
                    return f"文書 '{name}' が正常に追加されました。"
                else:
                    return f"文書 '{name}' の追加中にエラーが発生しました。"
            
            add_doc_btn.click(
                handle_add_document,
                inputs=[document_name, document_content],
                outputs=[add_result]
            )

    return demo

def main():
    """メイン関数"""
    print("AIME-based Explanation System for ChatGPT Outputs を起動中...")
    demo = create_gradio_interface()
    demo.launch(share=True)

if __name__ == "__main__":
    main()