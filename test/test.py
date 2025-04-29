"""
AIME-based Explanation System for ChatGPT Outputs
-------------------------------------------------
このシステムは2つの主要機能を提供します
1. ユーザーの質問に対するChatGPTの回答の根拠を示す
2. ChatGPTが特定の回答をした理由を説明する
"""

import os
from dotenv import load_dotenv
import openai
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import Dict, List, Any, Optional
import gradio as gr
import PyPDF2
import json

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
    
    def __init__(self, doc_directory="knowledge_base", 
                 model_name="gpt-4", chunk_size=1000, chunk_overlap=100):
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

    return demo

def main():
    """メイン関数"""
    print("AIME-based Explanation System for ChatGPT Outputs を起動中...")
    demo = create_gradio_interface()
    demo.launch(share=True)

if __name__ == "__main__":
    main()