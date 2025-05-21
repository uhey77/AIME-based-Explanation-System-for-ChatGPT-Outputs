# retriever.py
import hashlib
from typing import List, Dict, Any, Tuple
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# modelsモジュールからDocumentクラスをインポート
from models import Document


class SequentialSearchRetriever:
    """逐次検索リトリーバークラス"""

    def __init__(self, vector_db: VectorStore, llm: BaseLanguageModel, max_iterations: int = 3):
        """
        初期化メソッド

        Parameters:
        -----------
        vector_db : VectorStore
            ベクトルデータベース
        llm : BaseLanguageModel
            言語モデル
        max_iterations : int
            最大検索反復回数
        """
        if not isinstance(vector_db, VectorStore):
            raise TypeError("vector_dbはVectorStoreのインスタンスである必要があります。")
        self.vector_db = vector_db
        self.llm = llm
        self.max_iterations = max(1, max_iterations)  # 少なくとも1回は実行

        self.query_refinement_prompt = PromptTemplate(
            input_variables=["question", "previous_results", "iteration", "max_iterations"],
            template="""
            あなたは情報検索の専門家です。
            以下の質問に対する回答を見つけるために、検索クエリを最適化してください。

            元の質問: {question}

            これまでの検索結果の要約:
            {previous_results}

            現在の反復回数: {iteration}/{max_iterations}

            次の検索クエリをより効果的にするための修正案を提供してください。
            以下の点を考慮してください:
            1. 前回の検索結果から不足している情報は何か
            2. より具体的な用語や同義語を使うべきか
            3. 検索範囲を広げるべきか、絞るべきか

            修正されたクエリのみを出力してください。例: "RAGの具体的な実装ステップ"
            修正されたクエリ:
            """
        )

    def retrieve(self, question: str, k: int = 3) -> Tuple[List[Document], Dict[str, Any]]:
        """
        逐次的に検索を改善しながら関連ドキュメントを取得

        Parameters:
        -----------
        question : str
            ユーザーの質問
        k : int, optional
            各反復で取得するドキュメント数 (1以上)

        Returns:
        --------
        Tuple[List[Document], Dict[str, Any]]
            関連ドキュメントのリストと検索プロセスの詳細情報
        """
        k = max(1, k)  # kは少なくとも1
        all_docs: List[Document] = []
        search_history: List[Dict[str, Any]] = []
        current_query: str = question
        retrieved_doc_ids = set()  # 重複防止用IDセット (metadataにIDがあれば使う、なければpage_contentで代替)

        try:
            chain = LLMChain(llm=self.llm, prompt=self.query_refinement_prompt)

            for i in range(self.max_iterations):
                print(f"検索反復 {i+1}/{self.max_iterations}, クエリ: '{current_query}'")

                # 現在のクエリで検索
                retriever = self.vector_db.as_retriever(search_kwargs={"k": k})
                docs = retriever.get_relevant_documents(current_query)

                # 重複を除外して結果を追加
                new_docs_count = 0
                current_iteration_docs = []
                for doc in docs:
                    # ドキュメントの一意な識別子を取得 (例: metadataの'id'や'source')
                    # なければ page_content のハッシュ値などを使う
                    doc_id = doc.metadata.get('id', doc.metadata.get('filename', hashlib.md5(doc.page_content.encode()).hexdigest()))

                    if doc_id not in retrieved_doc_ids:
                        retrieved_doc_ids.add(doc_id)
                        all_docs.append(doc)
                        current_iteration_docs.append(doc)
                        new_docs_count += 1

                # 検索履歴を記録
                search_history.append({
                    "iteration": i + 1,
                    "query": current_query,
                    "retrieved_docs_count": len(docs),
                    "new_docs_added": new_docs_count,
                    # 詳細情報が必要な場合は追加
                    # "retrieved_docs_preview": [{"content": doc.page_content[:100] + "...", "metadata": doc.metadata} for doc in docs]
                })

                # 十分な文書が見つかったか、最終反復に達した場合は終了
                # 条件を見直し: 新しいドキュメントが見つからなくなったら終了するなど
                if new_docs_count == 0 and i > 0:  # 最初の反復以外で新しいドキュメントがなければ終了
                    print("新しい関連ドキュメントが見つからなかったため、検索を終了します。")
                    break
                if len(all_docs) >= k * 2 or i == self.max_iterations - 1:  # 元の条件も残す
                    if i == self.max_iterations - 1:
                        print(f"最大反復回数 {self.max_iterations} に達しました。")
                    else:
                        print(f"十分なドキュメント数 ({len(all_docs)}) に達しました。")
                    break

                # 前回の結果の要約を作成 (新規追加分だけでなく、その回の検索結果全体で良いか？)
                previous_results_summary = ""
                # limit the number of docs in summary to avoid large context
                docs_for_summary = docs[:min(len(docs), 5)]  # Show max 5 docs summary
                for j, doc in enumerate(docs_for_summary):
                    previous_results_summary += f"ドキュメント {j+1}: {doc.page_content[:150]}...\n"  # 少し短めに

                if not previous_results_summary:
                    previous_results_summary = "関連するドキュメントは見つかりませんでした。"

                # クエリを改善
                response = chain.invoke({
                    "question": question,
                    "previous_results": previous_results_summary,
                    "iteration": i + 1,
                    "max_iterations": self.max_iterations
                })

                refined_query = response["text"].strip()
                # クエリが変わった場合のみ更新（同じクエリでループするのを防ぐ）
                if refined_query and refined_query != current_query:
                    current_query = refined_query
                else:
                    # クエリが変わらない場合はループを抜ける
                    print("クエリが改善されなかったため、検索を終了します。")
                    break

        except Exception as e:
            print(f"逐次検索中にエラーが発生しました: {e}")
            # エラーが発生した場合でも、それまでに見つかったドキュメントと履歴を返す
            search_info = {
                "initial_query": question,
                "final_query": current_query,
                "search_history": search_history,
                "total_docs_retrieved": len(all_docs),
                "error": str(e)
            }
            return all_docs, search_info

        search_info = {
            "initial_query": question,
            "final_query": current_query,
            "search_history": search_history,
            "total_docs_retrieved": len(all_docs)
        }
        print(f"検索完了。合計 {len(all_docs)} 件のユニークなドキュメントを取得。")
        return all_docs, search_info
