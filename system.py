# system.py
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
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
            primary_method = method_selection.get("primary_method", "多角的説明")  # デフォルトを設定
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
                "processing_time": processing_time
            }
        except Exception as e:
            print(f"説明生成中にエラーが発生しました: {e}")
            processing_time = time.time() - start_time
            return {
                "explanation": f"説明の生成中にエラーが発生しました: {e}",
                "method": None,
                "verification": None,
                "processing_time": processing_time
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
                "processing_time": 0
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
            "explanation": explanation_result.get("explanation"),
            "sources": answer_result.get("sources"),  # 整形済みソースリスト
            "method": explanation_result.get("method"),
            "verification": verification_result,  # 統合された検証結果
            "search_info": answer_result.get("search_info"),
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
