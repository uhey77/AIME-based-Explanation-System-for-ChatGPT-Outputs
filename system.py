# type: ignore
import os
import time
import re
import pandas as pd
import json
import shutil
import csv
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from graph_visualization import ExplanationGraphVisualizer
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores import VectorStore

# 必要なモジュールをインポート
import config
from models import Document
from knowledge_base import load_documents, create_vector_db, initialize_sample_docs
from evaluation import EvaluationMetrics
from xai import XAIMethodSelector
from retriever import SequentialSearchRetriever
from verification import SelfVerificationSystem
from langchain.prompts import ChatPromptTemplate


class EnhancedAIExplanationSystem:
    """強化されたAI説明システムのメインクラス"""

    def __init__(
            self,
            doc_directory: str = config.DEFAULT_DOC_DIRECTORY,
            model_name: str = config.DEFAULT_MODEL_NAME,
            chunk_size: int = config.DEFAULT_CHUNK_SIZE,
            chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
            db_type: str = config.DEFAULT_DB_TYPE
            ):
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
        db_type : str
            使用するベクトルDBの種類 ("chroma" または "faiss")
        """
        print("システム初期化開始...")
        self.doc_directory = doc_directory
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_type = db_type

        # サンプル文書を初期化（必要に応じて）
        initialize_sample_docs(self.doc_directory)

        # ナレッジベースの構築
        self.documents: List[Document] = load_documents(self.doc_directory)
        self.vector_db: Optional[VectorStore] = create_vector_db(
            self.documents, self.chunk_size, self.chunk_overlap, self.db_type
        )

        # QAチェーンの構築
        self.llm = ChatOpenAI(model_name=self.model_name)
        self.qa_chain: Optional[RetrievalQA] = self._create_qa_chain()

        # XAI手法セレクタの初期化
        self.xai_selector = XAIMethodSelector(model_name=model_name)

        # 逐次検索リトリーバーの初期化
        # vector_dbが存在する場合のみ初期化
        self.sequential_retriever: Optional[SequentialSearchRetriever] = None
        if self.vector_db:
            self.sequential_retriever = SequentialSearchRetriever(
                vector_db=self.vector_db,
                llm=self.llm  # QAと同じLLMを使用
            )
        else:
            print("警告: ベクトルDBが利用できないため、逐次検索リトリーバーは初期化されません。")

        # 自己検証システムの初期化
        # 検証用モデルを指定可能にする (例: より強力なモデル 'gpt-4')
        verification_model = os.getenv("VERIFICATION_MODEL_NAME", "gpt-4")  # 環境変数から取得、なければgpt-4
        self.verification_system = SelfVerificationSystem(model_name=verification_model)

        # 評価指標管理の初期化
        self.evaluation_metrics = EvaluationMetrics()

        print("システム初期化完了。")

    def _create_qa_chain(self) -> Optional[RetrievalQA]:
        """質問応答チェーンの作成"""
        if not self.vector_db:
            print("ベクトルDBが作成されていないため、QAチェーンを作成できません。")
            return None
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # 必要に応じて変更 (e.g., "map_reduce")
                retriever=self.vector_db.as_retriever(),
                return_source_documents=True  # ソースドキュメントも返すように設定
            )
            print("QAチェーンを作成しました。")
            return qa_chain
        except Exception as e:
            print(f"QAチェーンの作成中にエラーが発生しました: {e}")
            return None

    def extract_thinking_and_answer(self, text: str) -> Tuple[str, str]:
        """思考プロセスと最終回答を抽出する

        Parameters:
        -----------
        text : str
            <think>タグを含むテキスト

        Returns:
        --------
        tuple
            (思考プロセス, 最終回答)のタプル
        """
        # <think>タグで囲まれた部分を抽出
        thinking_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        thinking_process = thinking_match.group(1).strip() if thinking_match else ""
        
        # <think>タグを除去して最終回答を得る
        final_answer = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        # "最終回答:"のプレフィックスがあれば除去
        final_answer = re.sub(r'^最終回答:\s*', '', final_answer).strip()
        
        return thinking_process, final_answer

    def get_answer(self, question: str) -> Dict[str, Any]:
        """質問に対する回答を取得"""
        if not self.qa_chain or not self.sequential_retriever:
            error_msg = "QAチェーンまたはリトリーバーが初期化されていないため、回答できません。"
            print(f"エラー: {error_msg}")
            return {"answer": error_msg, "sources": [], "search_info": None, "verification": None, "processing_time": 0}

        print(f"質問を受信: {question}")
        start_time = time.time()

        try:
            # 逐次検索を使用して関連ドキュメントを取得
            # ここで取得するドキュメントは検証や最終的なソース表示に使う
            retrieved_docs, search_info = self.sequential_retriever.retrieve(question, k=5)  # k=5 程度取得しておく

            # QAチェーンを使って回答を生成
            # QAチェーン内部でもリトリーバーが動くが、ここではstuffを使っているので、
            # retrieved_docs を直接コンテキストとして渡す方が効率的かもしれない。
            # しかし、元のコードは RetrievalQA を使っているので、それに従う。
            # RetrievalQAは内部で retriever.get_relevant_documents(question) を呼び出す。
            # そのため、sequential_retriever の結果を直接利用しない。
            # TODO: RetrievalQA を使わずに、retrieved_docs を直接 LLM に渡す方法も検討可能。
            result = self.qa_chain.invoke({"query": question})
            answer = result.get("result", "エラー: 回答を生成できませんでした。")
            # RetrievalQAがソースドキュメントを返す場合
            source_docs_from_qa = result.get("source_documents", [])

            # 検証には QA チェーンが参照したドキュメントを使うのが適切
            # source_docs_from_qa があればそれを、なければ sequential_retriever の結果を使う
            docs_for_verification = source_docs_from_qa if source_docs_from_qa else retrieved_docs
            verification_result = self.verification_system.verify_factual_accuracy(answer, docs_for_verification[:3])  # 上位3件で検証

            # 最終的なソースリストの整形 (QAが返したものを優先)
            final_source_docs = source_docs_from_qa if source_docs_from_qa else retrieved_docs
            source_list = [
                {
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                for doc in final_source_docs[:5]  # 上位5件を表示
            ]

            processing_time = time.time() - start_time
            print(f"回答生成完了。処理時間: {processing_time:.2f}秒")

            return {
                "answer": answer,
                "sources": source_list,
                "search_info": search_info,
                "verification": verification_result,
                "processing_time": processing_time
            }

        except Exception as e:
            print(f"回答取得中にエラーが発生しました: {e}")
            processing_time = time.time() - start_time
            return {
                "answer": f"エラーが発生しました: {e}",
                "sources": [],
                "search_info": None,
                "verification": None,
                "processing_time": processing_time
            }

    def explain_answer(self, question: str, answer: str, domain: str = "一般") -> Dict[str, Any]:
        """回答の説明を生成"""
        if not self.vector_db:
            error_msg = "ベクトルDBが利用できないため、説明を生成できません。"
            print(f"エラー: {error_msg}")
            return {"explanation": error_msg, "method": None, "verification": None, "processing_time": 0}

        print(f"回答の説明を生成開始: 質問='{question[:50]}...', 回答='{answer[:50]}...'")
        start_time = time.time()

        try:
            # 最適なXAI手法を選択
            method_selection = self.xai_selector.select_methods(question, answer, domain)
            primary_method = method_selection.get("primary_method", "Qwen3風思考プロセス")  # デフォルトをQwen3風に変更
            print(f"選択されたXAI手法: {primary_method}")

            # 説明生成のために、質問と回答の両方に関連するドキュメントを取得
            # ここでは標準的なリトリーバーを使用
            retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})  # 説明用に5件取得
            # 質問と回答を結合して検索クエリとする
            combined_query = f"質問: {question}\n回答: {answer}"
            relevant_docs = retriever.get_relevant_documents(combined_query)
            print(f"説明生成用に {len(relevant_docs)} 件の関連ドキュメントを取得しました。")

            # 選択された手法で説明を生成
            explanation = self.xai_selector.generate_explanation(
                primary_method,
                question,
                answer,
                relevant_docs  # 説明生成にはここで取得したドキュメントを使用
            )

            # 思考プロセスと最終回答を抽出（Qwen3風思考プロセスの場合）
            thinking_process = ""
            final_answer = answer
            if primary_method == "多角的説明（Qwen3風）":
                thinking_process, extracted_final_answer = self.extract_thinking_and_answer(explanation)
                if extracted_final_answer:
                    final_answer = extracted_final_answer

            # 説明の論理的整合性を検証
            verification_result = self.verification_system.verify_logical_coherence(explanation)

            processing_time = time.time() - start_time
            print(f"説明生成完了。処理時間: {processing_time:.2f}秒")

            # 評価指標を記録 (検証結果を基にスコアを割り当て)
            coherence_score = {
                "非常に高い": 1.0, "高い": 0.8, "中程度": 0.6,
                "低い": 0.4, "非常に低い": 0.2
            }.get(verification_result.get("coherence_rating", "検証失敗"), 0.5)  # デフォルト0.5

            metrics = {
                # 検証結果から取得できるものはそれを使う
                "coherence": coherence_score,
                # 他の指標は現時点ではデフォルト値または簡易的な推定値
                "relevance": 0.8,  # これは別途評価が必要
                "completeness": 0.8,  # これも別途評価が必要
                "factual_accuracy": None,  # 説明自体の事実性はここでは評価しない (回答の事実性は get_answer で評価)
                "user_rating": None,  # UIから取得する必要あり
                "confidence_score": None,  # 説明の信頼度スコア (別途計算ロジックが必要)
                "processing_time": processing_time
            }

            # None の値を除外して記録
            metrics_to_record = {k: v for k, v in metrics.items() if v is not None}

            self.evaluation_metrics.record_metrics(
                question,
                answer,
                explanation,
                primary_method,
                metrics_to_record
            )

            return {
                "explanation": explanation,
                "method": primary_method,
                "method_selection": method_selection,  # 選択理由なども含める
                "verification": verification_result,
                "processing_time": processing_time,
                "thinking_process": thinking_process,  # 思考プロセスを追加
                "final_answer": final_answer  # 最終回答を追加
            }
        except Exception as e:
            print(f"説明生成中にエラーが発生しました: {e}")
            processing_time = time.time() - start_time
            return {
                "explanation": f"説明の生成中にエラーが発生しました: {e}",
                "method": None,
                "verification": None,
                "processing_time": processing_time,
                "thinking_process": "",
                "final_answer": ""
            }

    def chat_and_explain(self, question: str, domain: str = "一般") -> Dict[str, Any]:
        """チャットと説明を同時に行う"""
        print("チャットと説明処理を開始...")
        total_start_time = time.time()

        # 回答を取得
        answer_result = self.get_answer(question)
        answer = answer_result.get("answer", "")
        sources_for_explanation = answer_result.get("sources", [])  # get_answer が返す整形済みソースリストを使う

        # 回答がエラーメッセージの場合は説明をスキップ
        if "エラー" in answer[:10]:  # 簡易的なエラーチェック
            print("回答生成中にエラーが発生したため、説明生成をスキップします。")
            explanation_result = {
                "explanation": "回答が生成されなかったため、説明できません。",
                "method": None,
                "verification": None,
                "processing_time": 0,
                "thinking_process": "",
                "final_answer": ""
            }
            verification_result = {
                "verification_status": "エラー",
                "confidence": "不明",
                "overall_assessment": "回答生成エラーのため評価不可"
            }
        else:
            # 説明を生成
            explanation_result = self.explain_answer(question, answer, domain)
            explanation = explanation_result.get("explanation", "")
            thinking_process = explanation_result.get("thinking_process", "")
            final_answer = explanation_result.get("final_answer", answer)

            # 最終的な検証を実施 (回答と説明の両方に対して)
            # get_answer で取得したソースドキュメントの実体が必要
            # get_answer が source_documents を返すように修正が必要
            # もし get_answer が整形済みリストしか返さない場合は、再度ドキュメントを取得する必要がある
            # ここでは answer_result["sources"] が整形済みリストを指していると仮定し、
            # 検証に必要な Document オブジェクトがないため、再度検索するか、
            # get_answer が Document オブジェクトを返すようにする必要がある。
            # 現状では検証に必要なソースがないため、簡易的な検証結果とする。
            # TODO: get_answer が Document オブジェクト (またはその内容) を返すように改修する
            # 一旦、answer_result の検証結果と explanation_result の検証結果を統合する形にする
            factual_verification = answer_result.get("verification", {})
            logical_verification = explanation_result.get("verification", {})
            verification_result = {
                "verification_status": "統合検証 (簡易)",
                "confidence": "中",  # 仮
                "factual_verification": factual_verification,
                "logical_verification": logical_verification,
                "overall_assessment": f"事実性(回答): {factual_verification.get('accuracy_rating', '不明')}, "
                                    f"論理性(説明): {logical_verification.get('coherence_rating', '不明')}",
                "timestamp": datetime.now().isoformat()
            }
            # 本来の complete_verification を呼ぶには、 sources (List[Document]) が必要
            # verification_result = self.verification_system.complete_verification(answer, explanation, source_documents)

        total_processing_time = time.time() - total_start_time
        print(f"チャットと説明処理完了。総処理時間: {total_processing_time:.2f}秒")

        return {
            "answer": answer,
            "explanation": explanation_result.get("explanation", ""),
            "sources": answer_result.get("sources", []),  # 整形済みソースリスト
            "method": explanation_result.get("method", ""),
            "verification": verification_result,  # 統合された検証結果
            "search_info": answer_result.get("search_info", {}),
            "thinking_process": explanation_result.get("thinking_process", ""),  # 思考プロセスを追加
            "final_answer": explanation_result.get("final_answer", answer),  # 最終回答を追加
            "processing_time": {
                "answer": answer_result.get("processing_time", 0),
                "explanation": explanation_result.get("processing_time", 0),
                "total": total_processing_time
            }
        }

    def add_document_from_text(self, filename: str, content: str) -> bool:
        """テキストからドキュメントを追加し、ナレッジベースを更新"""
        print(f"文書 '{filename}' の追加処理を開始...")
        if not filename or not content:
            print("エラー: ファイル名と内容の両方が必要です。")
            return False

        try:
            # ファイルパスを作成
            file_path = os.path.join(self.doc_directory, filename)
            # ディレクトリが存在しない場合は作成
            os.makedirs(self.doc_directory, exist_ok=True)

            # ファイルに書き込み
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"文書 '{filename}' をファイルに保存しました。")

            # 新しいドキュメントオブジェクトを作成
            new_doc = Document(content, {'filename': filename})

            # ドキュメントリストを更新 (既存のリストに追記)
            self.documents.append(new_doc)
            print("メモリ内のドキュメントリストを更新しました。")

            # ベクトルDBを再構築または更新
            # ChromaやFAISSは追記が難しい場合があるので、再構築が確実
            # 大規模な場合は、差分更新の方法を検討する
            print("ベクトルデータベースの再構築を開始...")
            self.vector_db = create_vector_db(
                self.documents, self.chunk_size, self.chunk_overlap, self.db_type
            )
            if not self.vector_db:
                print("エラー: ベクトルデータベースの再構築に失敗しました。")
                # 追加したドキュメントをリストから削除するなどのロールバック処理が必要な場合がある
                self.documents.pop()  # 直前に追加したものを削除
                return False

            # QAチェーンを再構築
            print("QAチェーンの再構築を開始...")
            self.qa_chain = self._create_qa_chain()
            if not self.qa_chain:
                print("エラー: QAチェーンの再構築に失敗しました。")
                return False

            # 逐次検索リトリーバーを更新 (新しい vector_db で再初期化)
            if self.vector_db:
                print("逐次検索リトリーバーの更新を開始...")
                self.sequential_retriever = SequentialSearchRetriever(
                    vector_db=self.vector_db,
                    llm=self.llm
                )
            else:
                self.sequential_retriever = None  # DBがない場合はNoneに戻す

            print(f"文書 '{filename}' の追加とシステムの更新が完了しました。")
            return True

        except Exception as e:
            print(f"文書の追加中にエラーが発生しました: {e}")
            # エラー発生時のクリーンアップ処理（例：作成したファイルの削除）
            if 'file_path' in locals() and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"エラー発生のため、ファイル '{filename}' を削除しました。")
                except OSError as rm_err:
                    print(f"エラーファイルの削除中に問題が発生しました: {rm_err}")
            return False

    def evaluate_system_performance(self, days: int = 30) -> Dict[str, Any]:
        """システムのパフォーマンス評価指標の要約を取得"""
        print(f"過去 {days} 日間のシステムパフォーマンス評価を開始...")
        summary = self.evaluation_metrics.get_metrics_summary(days=days)
        print("パフォーマンス評価完了。")
        return summary

    def generate_evaluation_report(self) -> Dict[str, Any]:
        """評価レポートを生成"""
        print("評価レポートの生成を開始...")
        report = self.evaluation_metrics.generate_evaluation_report()
        print("評価レポート生成完了。")
        return report

    def record_improvement(self, description: str, baseline_score: float, new_score: float):
        """システム改善を記録"""
        print(f"改善履歴の記録を開始: {description}")
        self.evaluation_metrics.record_improvement(description, baseline_score, new_score)
        # record_improvement内でprintされるのでここでは不要かも
        # print("改善履歴の記録完了。")
        
    def update_model_settings(self, new_model_name: Optional[str] = None, new_chunk_size: Optional[int] = None, 
                              new_chunk_overlap: Optional[int] = None, new_db_type: Optional[str] = None) -> bool:

        print("モデル設定の更新を開始...")
        changes_made = False
        rebuild_db = False
        rebuild_chains = False
        
        # 変更が必要なパラメータをチェック
        if new_model_name and new_model_name != self.model_name:
            self.model_name = new_model_name
            changes_made = True
            rebuild_chains = True
            print(f"モデル名を '{new_model_name}' に更新しました。")
            
        if new_chunk_size and new_chunk_size != self.chunk_size:
            self.chunk_size = new_chunk_size
            changes_made = True
            rebuild_db = True
            print(f"チャンクサイズを {new_chunk_size} に更新しました。")
            
        if new_chunk_overlap and new_chunk_overlap != self.chunk_overlap:
            self.chunk_overlap = new_chunk_overlap
            changes_made = True
            rebuild_db = True
            print(f"チャンクオーバーラップを {new_chunk_overlap} に更新しました。")
            
        if new_db_type and new_db_type != self.db_type:
            self.db_type = new_db_type
            changes_made = True
            rebuild_db = True
            print(f"ベクトルDBタイプを '{new_db_type}' に更新しました。")
            
        # 変更がない場合は早期リターン
        if not changes_made:
            print("変更はありませんでした。")
            return True
            
        try:
            # モデルの更新
            if rebuild_chains or new_model_name:
                self.llm = ChatOpenAI(model_name=self.model_name)
                self.xai_selector = XAIMethodSelector(model_name=self.model_name)
                print("LLMとXAIセレクタを更新しました。")
                
            # ベクトルDBの再構築が必要な場合
            if rebuild_db:
                print("ベクトルデータベースの再構築を開始...")
                self.vector_db = create_vector_db(
                    self.documents, self.chunk_size, self.chunk_overlap, self.db_type
                )
                if not self.vector_db:
                    print("エラー: ベクトルデータベースの再構築に失敗しました。")
                    return False
                print("ベクトルデータベースを再構築しました。")
                
            # チェーンとリトリーバーの再構築
            if rebuild_db or rebuild_chains:
                # QAチェーンの再構築
                self.qa_chain = self._create_qa_chain()
                if not self.qa_chain and self.vector_db:  # vector_dbがあるのにqa_chainが作れない場合はエラー
                    print("エラー: QAチェーンの再構築に失敗しました。")
                    return False
                    
                # リトリーバーの再構築
                if self.vector_db:
                    self.sequential_retriever = SequentialSearchRetriever(
                        vector_db=self.vector_db,
                        llm=self.llm
                    )
                    print("逐次検索リトリーバーを更新しました。")
                    
            print("モデル設定の更新が完了しました。")
            return True
            
        except Exception as e:
            print(f"モデル設定の更新中にエラーが発生しました: {e}")
            return False
            
    def reset_knowledge_base(self, preserve_documents: bool = False) -> bool:
        """ナレッジベースをリセットする
        
        Parameters:
        -----------
        preserve_documents : bool
            True の場合、元のドキュメントファイルは保持し、ベクトルDBのみ再構築する
            False の場合、ドキュメントファイルも含めて完全にリセットする
            
        Returns:
        --------
        bool
            リセットが成功したかどうか
        """
        print(f"ナレッジベースのリセットを開始... (ドキュメント保持: {preserve_documents})")
        
        try:
            if not preserve_documents:
                
                # 現在のディレクトリを一時的に保存
                temp_dir = f"{self.doc_directory}_temp"
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)  # 既存の一時ディレクトリがあれば削除
                    
                # サンプルファイルを再初期化するために、ディレクトリを一度削除して再作成
                if os.path.exists(self.doc_directory):
                    shutil.rmtree(self.doc_directory)
                    
                # ディレクトリを再作成
                os.makedirs(self.doc_directory, exist_ok=True)
                
                # サンプルドキュメントを初期化
                initialize_sample_docs(self.doc_directory)
                
                # ドキュメントリストをリロード
                self.documents = load_documents(self.doc_directory)
                print("ナレッジベースのドキュメントをリセットしました。")
            
            # ベクトルDBを再構築
            print("ベクトルデータベースの再構築を開始...")
            self.vector_db = create_vector_db(
                self.documents, self.chunk_size, self.chunk_overlap, self.db_type
            )
            if not self.vector_db:
                print("エラー: ベクトルデータベースの再構築に失敗しました。")
                return False
                
            # QAチェーンの再構築
            self.qa_chain = self._create_qa_chain()
            if not self.qa_chain:
                print("エラー: QAチェーンの再構築に失敗しました。")
                return False
                
            # リトリーバーの再構築
            if self.vector_db:
                self.sequential_retriever = SequentialSearchRetriever(
                    vector_db=self.vector_db,
                    llm=self.llm
                )
                print("逐次検索リトリーバーを更新しました。")
                
            print("ナレッジベースのリセットが完了しました。")
            return True
            
        except Exception as e:
            print(f"ナレッジベースのリセット中にエラーが発生しました: {e}")
            return False
            
    def export_explanation_records(self, format: str = "json", days: int = 30, 
                                  output_path: Optional[str] = None) -> Optional[str]:
        """説明記録をエクスポートする
        
        Parameters:
        -----------
        format : str
            エクスポート形式 ("json", "csv", "excel")
        days : int
            過去何日分のデータをエクスポートするか
        output_path : Optional[str]
            出力先ファイルパス。指定しない場合はデフォルト名で保存
            
        Returns:
        --------
        Optional[str]
            成功時はファイルパス、失敗時はNone
        """
        print(f"説明記録のエクスポートを開始... 形式: {format}, 期間: 過去{days}日間")
        
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"explanation_records_{timestamp}.{format}"
                output_path = os.path.join(os.getcwd(), filename)
                
            # 評価指標クラスからデータを取得
            records = self.evaluation_metrics.get_records(days=days)
            if not records:
                print("エクスポートするレコードがありません。")
                return None
                
            # 形式に応じてエクスポート
            if format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(records, f, ensure_ascii=False, indent=2)
                    
            elif format.lower() == "csv":
                flattened_records = []
                for record in records:
                    flat_record = {
                        "id": record.get("id"),
                        "timestamp": record.get("timestamp"),
                        "question": record.get("question"),
                        "answer": record.get("answer"),
                        "explanation": record.get("explanation"),
                        "method": record.get("method")
                    }
                    metrics = record.get("metrics", {})
                    for k, v in metrics.items():
                        flat_record[f"metric_{k}"] = v
                        
                    flattened_records.append(flat_record)
 
                # CSVに書き込み
                if flattened_records:
                    keys = flattened_records[0].keys()
                    with open(output_path, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=keys)
                        writer.writeheader()
                        writer.writerows(flattened_records)

            elif format.lower() == "excel":
                try:
                    # DataFrameに変換
                    df_records = pd.DataFrame(records)
                    # メトリクスカラムを展開
                    if 'metrics' in df_records.columns:
                        metrics_df = pd.json_normalize(df_records['metrics'])
                        # メトリクスカラムの接頭辞を追加
                        metrics_df = metrics_df.add_prefix('metric_')
                        # 元のDataFrameから'metrics'カラムを削除
                        df_records = df_records.drop('metrics', axis=1)
                        # メトリクスを結合
                        df_records = pd.concat([df_records, metrics_df], axis=1)
                        
                    # Excelに保存
                    df_records.to_excel(output_path, index=False)
                except ImportError:
                    print("pandasモジュールがインストールされていません。Excelエクスポートにはpandasとopenpyxlが必要です。")
                    return None
            else:
                print(f"未対応の形式です: {format}。'json', 'csv', 'excel'のいずれかを指定してください。")
                return None
                
            print(f"説明記録を正常にエクスポートしました: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"説明記録のエクスポート中にエラーが発生しました: {e}")
            return None
            
    def interactive_explanation(self, question: str, answer: str, followup_question: str, 
                               domain: str = "一般") -> Dict[str, Any]:
        """対話的な説明機能: フォローアップ質問に基づいて説明を深める
        
        Parameters:
        -----------
        question : str
            元の質問
        answer : str
            元の回答
        followup_question : str
            説明についてのフォローアップ質問
        domain : str
            ドメイン/分野名
            
        Returns:
        --------
        Dict[str, Any]
            追加説明を含む結果辞書
        """
        print(f"対話的説明を開始... フォローアップ質問: '{followup_question[:50]}...'")
        start_time = time.time()
        
        if not self.vector_db or not self.llm:
            error_msg = "ベクトルDBまたはLLMが初期化されていないため、対話的説明を生成できません。"
            print(f"エラー: {error_msg}")
            return {"additional_explanation": error_msg, "processing_time": 0}
            
        try:
            # まず基本的な説明結果を取得 (既に生成されている場合は再利用可能)
            # もし既に説明が生成されていて、それを再利用できるならば、ここでは新たに生成せずに
            # 既存の説明情報を利用することも考えられる
            base_explanation_result = self.explain_answer(question, answer, domain)
            base_explanation = base_explanation_result.get("explanation", "")
            method = base_explanation_result.get("method", "")
            thinking_process = base_explanation_result.get("thinking_process", "")
            
            # フォローアップ質問に関連するドキュメントを取得
            # 元の質問、回答、説明、フォローアップ質問を組み合わせて検索
            combined_query = f"元の質問: {question}\n回答: {answer}\n説明: {base_explanation[:200]}...\nフォローアップ質問: {followup_question}"
            
            # 標準的なリトリーバーで検索
            retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
            relevant_docs = retriever.get_relevant_documents(combined_query)
            print(f"フォローアップ説明用に {len(relevant_docs)} 件の関連ドキュメントを取得しました。")
            
            # フォローアップ説明用のプロンプトテンプレート
            followup_prompt = ChatPromptTemplate.from_template(
                """あなたはAI説明システムです。ユーザーからの質問に対するAIの回答について説明を提供しています。
                元の質問、AI回答、基本的な説明を読んだ後、ユーザーが追加の質問をしました。
                以下の情報と参考資料を使って、フォローアップ質問に詳細に答えてください。
                
                #元の質問: {question}
                
                #AI回答: {answer}
                
                #基本的な説明: {base_explanation}
                
                #ユーザーのフォローアップ質問: {followup_question}
                
                #参考資料:
                {relevant_docs}
                
                フォローアップ質問に対する説明を提供してください。元の回答と矛盾しない範囲で、
                できるだけ詳細に、正確に、参考資料の情報を活用して回答してください。
                必要に応じて、{thinking_tag_start}...{thinking_tag_end}タグを使って思考プロセスを示すこともできます。
                """
            )
            
            prompt_args = {
                "question": question,
                "answer": answer,
                "base_explanation": base_explanation,
                "followup_question": followup_question,
                "relevant_docs": "\n\n".join([f"文書{i+1}: {doc.page_content}" for i, doc in enumerate(relevant_docs)]),
                "thinking_tag_start": config.THINKING_START_TAG,
                "thinking_tag_end": config.THINKING_END_TAG
            }
            
            # LLMでフォローアップ説明を生成
            chain = followup_prompt | self.llm
            additional_explanation = chain.invoke(prompt_args).content
            
            # 必要に応じて思考プロセスと最終回答を抽出
            additional_thinking = ""
            final_additional_explanation = additional_explanation
            thinking_match = re.search(f'{config.THINKING_START_TAG}(.*?){config.THINKING_END_TAG}', 
                                      additional_explanation, re.DOTALL)
            if thinking_match:
                additional_thinking = thinking_match.group(1).strip()
                final_additional_explanation = re.sub(
                    f'{config.THINKING_START_TAG}.*?{config.THINKING_END_TAG}', 
                    '', additional_explanation, flags=re.DOTALL
                ).strip()
            
            # 論理的整合性の検証
            verification_result = self.verification_system.verify_logical_coherence(final_additional_explanation)
            
            processing_time = time.time() - start_time
            print(f"対話的説明生成完了。処理時間: {processing_time:.2f}秒")
            
            # 返却する結果辞書
            return {
                "base_explanation": base_explanation,
                "additional_explanation": additional_explanation,
                "method": method,
                "thinking_process": additional_thinking,
                "final_explanation": final_additional_explanation,
                "verification": verification_result,
                "processing_time": processing_time
            }
            
        except Exception as e:
            print(f"対話的説明の生成中にエラーが発生しました: {e}")
            processing_time = time.time() - start_time
            return {
                "additional_explanation": f"説明の生成中にエラーが発生しました: {e}",
                "processing_time": processing_time
            }
    
    def batch_process(self, questions: List[str], domain: str = "一般",
                    output_format: str = "dict") -> Any:
        """複数の質問をバッチ処理する
        
        Parameters:
        -----------
        questions : List[str]
            処理する質問のリスト
        domain : str
            すべての質問に適用するドメイン/分野
        output_format : str
            出力形式 ("dict", "json", "pandas")
            
        Returns:
        --------
        Any
            指定された形式での処理結果
        """
        print(f"バッチ処理を開始... 質問数: {len(questions)}")
        batch_start_time = time.time()
        
        results = []
        for i, question in enumerate(questions):
            print(f"質問 {i+1}/{len(questions)} を処理中: '{question[:50]}...'")
            result = self.chat_and_explain(question, domain)
            result["question_index"] = i + 1 # 結果に質問インデックスと質問テキストを追加
            result["question_text"] = question
            results.append(result)
            
        batch_processing_time = time.time() - batch_start_time
        print(f"バッチ処理完了。総処理時間: {batch_processing_time:.2f}秒")
        
        if output_format.lower() == "dict":
            return {
                "results": results,
                "total_questions": len(questions),
                "total_processing_time": batch_processing_time,
                "average_processing_time": batch_processing_time / len(questions) if questions else 0,
                "timestamp": datetime.now().isoformat()
            }
            
        elif output_format.lower() == "json":
            results_dict = {
                "results": results,
                "total_questions": len(questions),
                "total_processing_time": batch_processing_time,
                "average_processing_time": batch_processing_time / len(questions) if questions else 0,
                "timestamp": datetime.now().isoformat()
            }
            return json.dumps(results_dict, ensure_ascii=False, indent=2)
            
        elif output_format.lower() == "pandas":
            try:
                flat_results = []
                for res in results:
                    flat_res = {
                        "question_index": res.get("question_index"),
                        "question": res.get("question_text"),
                        "answer": res.get("answer"),
                        "explanation": res.get("explanation"),
                        "method": res.get("method"),
                        "thinking_process": res.get("thinking_process"),
                        "final_answer": res.get("final_answer"),
                        "processing_time_total": res.get("processing_time", {}).get("total"),
                        "processing_time_answer": res.get("processing_time", {}).get("answer"),
                        "processing_time_explanation": res.get("processing_time", {}).get("explanation"),
                        "verification_status": res.get("verification", {}).get("verification_status"),
                        "verification_confidence": res.get("verification", {}).get("confidence"),
                        "verification_assessment": res.get("verification", {}).get("overall_assessment")
                    }
                    flat_results.append(flat_res)
                    
                return pd.DataFrame(flat_results)
                
            except ImportError:
                print("pandasモジュールがインストールされていません。'pandas'形式を使用するにはpandasが必要です。")
                # 代替としてdict形式を返す
                return {
                    "results": results,
                    "total_questions": len(questions),
                    "total_processing_time": batch_processing_time,
                    "average_processing_time": batch_processing_time / len(questions) if questions else 0,
                    "timestamp": datetime.now().isoformat()
                }
        else:
            print(f"未対応の出力形式です: {output_format}。'dict', 'json', 'pandas'のいずれかを指定してください。")
            # デフォルトのdict形式で返す
            return {
                "results": results,
                "total_questions": len(questions),
                "total_processing_time": batch_processing_time,
                "average_processing_time": batch_processing_time / len(questions) if questions else 0,
                "timestamp": datetime.now().isoformat()
            }
    
    def visualize_explanation_as_graph(self, explanation: str, question: str = None, answer: str = None, 
                                save_path: str = None, return_base64: bool = True) -> Dict[str, Any]:
        """
        説明をグラフとして可視化する

        Parameters:
        -----------
        explanation : str
            説明テキスト（思考プロセスを含む）
        question : str, optional
            元の質問
        answer : str, optional
            最終的な回答
        save_path : str, optional
            グラフを保存するファイルパス
        return_base64 : bool
            True の場合、画像のbase64エンコードを返す

        Returns:
        --------
        Dict[str, Any]
            グラフの情報と画像データを含む辞書
        """
        print(f"説明のグラフ可視化を開始...")
        
        try:
            # 可視化クラスのインスタンスを作成
            visualizer = ExplanationGraphVisualizer()
            
            # ここから修正部分 -------------------------------------
            # 思考プロセスがあればそれを使用、なければ説明全体を使用
            thinking_process = ""
            if "<think>" in explanation:
                thinking_match = re.search(r'<think>(.*?)</think>', explanation, re.DOTALL)
                if thinking_match:
                    thinking_process = thinking_match.group(1).strip()
            
            text_for_graph = thinking_process if thinking_process and len(thinking_process) > 100 else explanation
            
            # 日本語テキストを適切に処理するための前処理
            # 半角スペースの多重スペースを1つに
            text_for_graph = re.sub(r' +', ' ', text_for_graph)
            # 改行の正規化
            text_for_graph = re.sub(r'\r\n', '\n', text_for_graph)
            # 複数の改行を1つに
            text_for_graph = re.sub(r'\n+', '\n', text_for_graph)
            # ここまで修正部分 -------------------------------------
            
            # 説明からグラフを生成（修正: text_for_graphを使用）
            graph = visualizer.create_explanation_graph(text_for_graph, question, answer)
            
            # グラフの指標を計算
            metrics = visualizer.generate_graph_metrics(graph)
            
            # 結果を格納する辞書
            result = {
                'metrics': metrics,
                'node_count': metrics['node_count'],
                'edge_count': metrics['edge_count'],
                'central_concepts': []
            }
            
            # 中心的な概念を追加
            if metrics.get('central_node'):
                result['central_concepts'].append({
                    'text': metrics['central_node'],
                    'type': metrics['central_node_type']
                })
            
            # save_pathが指定されている場合、画像を保存
            if save_path:
                # ディレクトリが存在しない場合は作成
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                
                # グラフを画像として保存
                title = f"Explanation Graph: {question[:30] + '...' if question and len(question) > 30 else question}"
                visualizer.save_graph_as_image(graph, save_path, title)
                result['image_path'] = save_path
            
            # base64エンコードされた画像を返すオプション
            if return_base64:
                title = f"Explanation Graph"
                if question:
                    title += f" for: {question[:30] + '...' if len(question) > 30 else question}"
                
                img_base64 = visualizer.render_graph_as_base64(graph, title)
                result['image_base64'] = img_base64
            
            print(f"説明のグラフ可視化が完了しました。ノード数: {metrics['node_count']}, エッジ数: {metrics['edge_count']}")
            return result
            
        except Exception as e:
            print(f"説明のグラフ可視化中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'metrics': {},
                'node_count': 0,
                'edge_count': 0,
                'central_concepts': []
            }

    def batch_visualization(self, explanation_records: List[Dict[str, Any]], output_dir: str = "explanation_graphs") -> List[Dict[str, Any]]:
        """
        複数の説明を一括でグラフ可視化する

        Parameters:
        -----------
        explanation_records : List[Dict[str, Any]]
            説明レコードのリスト（各レコードには 'question', 'answer', 'explanation' が含まれる）
        output_dir : str
            出力ディレクトリ

        Returns:
        --------
        List[Dict[str, Any]]
            各説明の可視化結果リスト
        """
        print(f"一括グラフ可視化を開始... レコード数: {len(explanation_records)}")
        
        results = []
        os.makedirs(output_dir, exist_ok=True)
        
        for i, record in enumerate(explanation_records):
            question = record.get('question', f"Question {i+1}")
            answer = record.get('answer', '')
            explanation = record.get('explanation', '')
            
            # ファイル名を生成（質問の先頭部分を使用）
            question_prefix = question[:20].replace(" ", "_").replace("/", "_").replace("\\", "_")
            filename = f"graph_{i+1}_{question_prefix}.png"
            filepath = os.path.join(output_dir, filename)
            
            # グラフを生成して保存
            result = self.visualize_explanation_as_graph(
                explanation=explanation,
                question=question,
                answer=answer,
                save_path=filepath,
                return_base64=False  # ファイルとして保存するので、base64は不要
            )
            
            # 結果にレコード情報を追加
            result['question'] = question
            result['answer'] = answer
            result['record_id'] = record.get('id', i+1)
            result['filename'] = filename
            
            results.append(result)
            print(f"レコード {i+1}/{len(explanation_records)} を処理しました: {filepath}")
        
        print(f"一括グラフ可視化が完了しました。出力ディレクトリ: {output_dir}")
        return results