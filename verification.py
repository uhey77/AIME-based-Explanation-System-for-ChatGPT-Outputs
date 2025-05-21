# verification.py
import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime

# modelsモジュールからDocumentクラスをインポート
from models import Document
# configモジュールから設定値をインポート
import config


class SelfVerificationSystem:
    """自己検証システムクラス"""

    def __init__(self, model_name: str = config.DEFAULT_MODEL_NAME):
        """
        初期化メソッド

        Parameters:
        -----------
        model_name : str
            検証に使用するLLMのモデル名 (デフォルトは config から取得)
        """
        self.model_name = model_name
        try:
            # 検証には強力なモデルを使うことを推奨する場合があるため、
            # model_name を引数で受け取れるようにしておくのは良い設計
            self.llm = ChatOpenAI(model_name=model_name)
        except Exception as e:
            print(f"エラー: 検証用LLM '{model_name}' の初期化に失敗しました: {e}")
            raise

        # 事実検証プロンプト
        self.factual_verification_prompt = PromptTemplate(
            input_variables=["statement", "sources"],
            template="""
            あなたは事実検証の専門家です。
            以下の文章の事実性を、提供された情報源に基づいて検証してください。

            評価対象の文章:
            "{statement}"

            情報源:
            {sources}

            以下の点について分析してください:
            1. 文章は情報源と一致しているか？
            2. 省略や誇張はないか？
            3. 誤解を招く表現はないか？
            4. 情報源から直接サポートされない推論はないか？

            各問いに対して「はい」または「いいえ」で回答し、理由を説明してください。
            最後に、文章全体の事実性を以下のスケールで評価してください:
            - 完全に正確 (情報源と完全に一致)
            - ほぼ正確 (軽微な不一致あり)
            - 部分的に正確 (重要な情報の欠落または不正確さあり)
            - 不正確 (情報源と大きく矛盾)
            - 検証不能 (情報源に関連情報なし)

            回答形式:
            {{
              "matches_sources": true/false,
              "has_omissions": true/false,
              "is_misleading": true/false,
              "has_unsupported_inferences": true/false,
              "accuracy_rating": "評価カテゴリ",
              "explanation": "詳細な説明",
              "corrected_statement": "必要に応じて修正した文章"
            }}

            JSONフォーマットで回答してください。
            """
        )

        # 論理整合性検証プロンプト
        self.logical_verification_prompt = PromptTemplate(
            input_variables=["explanation"],
            template="""
            あなたは論理的整合性の検証専門家です。
            以下のAI説明の論理構造を分析し、整合性を評価してください。

            AI説明:
            "{explanation}"

            以下の点について分析してください:
            1. 前提と結論の間に論理的なつながりがあるか
            2. 論理的飛躍はないか
            3. 矛盾する主張はないか
            4. 循環論法になっていないか
            5. 因果関係と相関関係が適切に区別されているか

            各問いに対して「はい」または「いいえ」で回答し、理由を説明してください。
            最後に、説明全体の論理的整合性を以下のスケールで評価してください:
            - 非常に高い (完全に論理的)
            - 高い (軽微な論理的問題あり)
            - 中程度 (いくつかの論理的問題あり)
            - 低い (重大な論理的問題あり)
            - 非常に低い (論理的に破綻)

            回答形式:
            {{
              "logical_connection": true/false,
              "logical_leaps": true/false,
              "contradictions": true/false,
              "circular_reasoning": true/false,
              "causation_correlation_distinction": true/false,
              "coherence_rating": "評価カテゴリ",
              "explanation": "詳細な説明",
              "improvement_suggestions": "改善提案"
            }}

            JSONフォーマットで回答してください。
            """
        )

    def _format_sources_for_verification(self, sources: List[Document]) -> str:
        """検証用にソースを整形するヘルパー関数"""
        sources_text = ""
        if not sources:
            return "利用可能な情報源はありません。\n\n"

        for i, source in enumerate(sources):
            filename = source.metadata.get('filename', f'Unknown Source {i+1}')
            content = source.page_content
            # 検証のためには、ある程度の長さが必要かもしれないが、長すぎるとコンテキストウィンドウを超える
            truncated_content = (content[:1000] + '...') if len(content) > 1000 else content
            sources_text += f"[情報源 {i+1}] ファイル名: {filename}\n内容:\n{truncated_content}\n\n"
        return sources_text

    def _parse_verification_response(self, response_text: str, default_rating: str, error_context: str) -> Dict[str, Any]:
        """LLMからのJSON応答をパースするヘルパー関数"""
        try:
            # 文字列からJSONを抽出
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
                result = json.loads(json_str)
                return result
            else:
                raise json.JSONDecodeError("応答に有効なJSONオブジェクトが見つかりません。", response_text, 0)

        except json.JSONDecodeError as e:
            print(f"{error_context} 結果のJSONパース中にエラーが発生しました: {e}")
            print(f"元のレスポンス: {response_text}")
            # フォールバック結果
            rating_key = "accuracy_rating" if "事実" in error_context else "coherence_rating"
            return {
                rating_key: default_rating,
                "explanation": f"{error_context} 結果のパース中にエラーが発生しました。",
                "error": str(e)
            }
        except Exception as e:
            print(f"{error_context} 結果の処理中に予期せぬエラーが発生しました: {e}")
            print(f"元のレスポンス: {response_text}")
            # フォールバック結果
            rating_key = "accuracy_rating" if "事実" in error_context else "coherence_rating"
            return {
                rating_key: default_rating,
                "explanation": f"{error_context} 結果の処理中に予期せぬエラーが発生しました。",
                "error": str(e)
            }

    def verify_factual_accuracy(self, statement: str, sources: List[Document]) -> Dict[str, Any]:
        """
        文章の事実性を検証

        Parameters:
        -----------
        statement : str
            検証対象の文章
        sources : List[Document]
            情報源のリスト

        Returns:
        --------
        dict
            検証結果
        """
        if not statement:
            return {"accuracy_rating": "検証不能", "explanation": "検証対象の文章が空です。"}

        sources_text = self._format_sources_for_verification(sources)
        default_rating = "検証失敗"
        error_context = "事実検証"

        try:
            chain = LLMChain(llm=self.llm, prompt=self.factual_verification_prompt)
            response = chain.invoke({
                "statement": statement,
                "sources": sources_text
            })
            return self._parse_verification_response(response["text"], default_rating, error_context)

        except Exception as e:
            print(f"{error_context}中にエラーが発生しました: {e}")
            return {
                "accuracy_rating": default_rating,
                "explanation": f"{error_context}中にエラーが発生しました: {e}",
                "error": str(e)
            }

    def verify_logical_coherence(self, explanation: str) -> Dict[str, Any]:
        """
        説明の論理的整合性を検証

        Parameters:
        -----------
        explanation : str
            検証対象の説明

        Returns:
        --------
        dict
            検証結果
        """
        if not explanation:
            return {"coherence_rating": "検証不能", "explanation": "検証対象の説明が空です。"}

        default_rating = "検証失敗"
        error_context = "論理検証"

        try:
            chain = LLMChain(llm=self.llm, prompt=self.logical_verification_prompt)
            response = chain.invoke({
                "explanation": explanation
            })
            return self._parse_verification_response(response["text"], default_rating, error_context)
        except Exception as e:
            print(f"{error_context}中にエラーが発生しました: {e}")
            return {
                "coherence_rating": default_rating,
                "explanation": f"{error_context}中にエラーが発生しました: {e}",
                "error": str(e)
            }

    def complete_verification(self, statement: str, explanation: str, sources: List[Document]) -> Dict[str, Any]:
        """
        総合的な検証を実行

        Parameters:
        -----------
        statement : str
            検証対象の文章
        explanation : str
            検証対象の説明
        sources : List[Document]
            情報源のリスト

        Returns:
        --------
        dict
            総合的な検証結果
        """
        factual_result = self.verify_factual_accuracy(statement, sources)
        coherence_result = self.verify_logical_coherence(explanation)

        # 総合評価ロジック
        factual_rating = factual_result.get("accuracy_rating", "検証失敗")
        coherence_rating = coherence_result.get("coherence_rating", "検証失敗")

        # より詳細なステータスと信頼度を決定
        if factual_rating in ["完全に正確", "ほぼ正確"] and coherence_rating in ["非常に高い", "高い"]:
            verification_status = "検証済み (高信頼)"
            confidence = "高"
        elif factual_rating == "不正確" or coherence_rating == "非常に低い":
            verification_status = "問題あり"
            confidence = "低 (要修正)"
        elif factual_rating == "検証失敗" or coherence_rating == "検証失敗":
            verification_status = "検証エラー"
            confidence = "不明"
        else:
            # 部分的に正確、中程度の整合性、検証不能などが含まれる場合
            verification_status = "部分的に検証済み"
            confidence = "中"

        return {
            "verification_status": verification_status,
            "confidence": confidence,
            "factual_verification": factual_result,
            "logical_verification": coherence_result,
            "overall_assessment": f"事実性評価: {factual_rating}, 論理整合性評価: {coherence_rating}",
            "timestamp": datetime.now().isoformat()
        }
