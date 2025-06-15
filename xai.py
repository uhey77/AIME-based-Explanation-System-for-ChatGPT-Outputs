# type: ignore
import json
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from enum import Enum
from dataclasses import dataclass, field
import logging
from functools import lru_cache
import hashlib
from datetime import datetime
import numpy as np

from models import Document
import config

# ロギング設定
logger = logging.getLogger(__name__)


class XAIMethod(str, Enum):
    """XAI手法の列挙型"""
    FEATURE_IMPORTANCE = "特徴重要度分析"
    COUNTERFACTUAL = "反事実説明"
    RULE_BASED = "ルールベース説明"
    CASE_BASED = "事例ベース説明"
    PROCESS_TRACKING = "プロセス追跡"
    ATTENTION_VISUALIZATION = "アテンション可視化"
    LOCAL_APPROXIMATION = "局所的近似"
    MULTI_FACETED = "多角的説明"
    QWEN3_THINKING = "Qwen3風思考プロセス"


@dataclass
class MethodSelection:
    """手法選択の結果を格納するデータクラス"""
    primary_method: str
    secondary_method: Optional[str] = None
    reasoning: str = ""
    approach: str = ""
    confidence: float = 0.5


class XAIMethodSelector:
    """XAI手法選択と説明生成クラス（改善版）"""

    def __init__(self, model_name: str = config.DEFAULT_MODEL_NAME):
        """
        初期化メソッド

        Parameters:
        -----------
        model_name : str
            XAI手法選択に使用するLLMのモデル名
        """
        self.model_name = model_name
        try:
            self.llm = ChatOpenAI(model_name=model_name)
        except Exception as e:
            logger.error(f"ChatOpenAI モデル '{model_name}' の初期化に失敗しました: {e}")
            raise

        # XAI手法選択プロンプト（既存のものを保持）
        # selection_promptを以下のように更新
        self.selection_prompt = PromptTemplate(
            input_variables=["question", "answer", "domain"],
            template="""
            あなたは最適なXAI(Explainable AI)手法を選択する専門家です。
            以下のAI応答に対して最も適切な説明手法を選択してください。

            ユーザーの質問: {question}
            AIの回答: {answer}
            ドメイン: {domain}

            以下の説明手法から、この状況に最適な手法とその理由を選択してください:
            
            【基本的な説明手法】
            1. 特徴重要度分析 - 回答に影響を与えた入力の重要度を可視化
            2. 反事実説明 - 「もし〜が異なっていたら」という仮定での説明
            3. ルールベース説明 - if-thenルールとして表現
            4. 事例ベース説明 - 類似事例と比較して説明
            5. プロセス追跡 - モデルの推論過程を段階的に説明
            
            【高度な説明手法】
            6. アテンション可視化 - モデルが注目した情報を強調
            7. 局所的近似 - 複雑なモデルを局所的に単純化
            8. 多角的説明 - 複数の説明手法を組み合わせ
            9. Qwen3風思考プロセス - 自然な思考の流れを表現
            
            【新しい説明手法】
            10. 思考の連鎖 - 人間が理解しやすい段階的思考プロセス
            11. 比較分析 - 複数の観点から代替案と比較
            12. 不確実性定量化 - 確信度を定量的に評価し可視化

            選択の際は以下を考慮してください：
            - 質問の複雑さと種類
            - 求められる説明の詳細度
            - ドメインの特性
            - ユーザーの技術的背景（推定）

            回答形式:
            {{
            "primary_method": "選択した主要手法",
            "secondary_method": "補助的手法（必要な場合）",
            "reasoning": "選択理由",
            "approach": "具体的な実装アプローチ",
            "confidence": 0.0-1.0の確信度
            }}

            JSONフォーマットで回答してください。
            """
        )

        # 既存の実装を保持
        self.method_implementations = {
            "特徴重要度分析": self._generate_feature_importance_explanation,
            "反事実説明": self._generate_counterfactual_explanation,
            "ルールベース説明": self._generate_rule_based_explanation,
            "事例ベース説明": self._generate_case_based_explanation,
            "プロセス追跡": self._generate_process_tracking_explanation,
            "アテンション可視化": self._generate_attention_visualization_explanation,
            "局所的近似": self._generate_local_approximation_explanation,
            "多角的説明": self._generate_multi_faceted_explanation,
            "Qwen3風思考プロセス": self._generate_qwen3_thinking_process_explanation,
            "思考の連鎖": self._generate_chain_of_thought_explanation,
            "比較分析": self._generate_comparative_analysis_explanation,
            "不確実性定量化": self._generate_uncertainty_quantification_explanation,
        }
        
        self._explanation_cache = {}
        self._cache_max_size = 100  # 最大キャッシュサイズ
        
    @lru_cache(maxsize=32)
    def _get_cache_key(self, question: str, answer: str, method: str) -> str:
        """キャッシュキーを生成"""
        content = f"{method}:{question}:{answer}"
        return hashlib.md5(content.encode()).hexdigest()

    def generate_explanation(self, method: str, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """選択された手法に基づいて説明を生成（キャッシング機能付き）"""
        
        # キャッシュチェック（ソースは含めない - 頻繁に変わるため）
        use_cache = kwargs.get('use_cache', True)
        if use_cache:
            cache_key = self._get_cache_key(question, answer, method)
            if cache_key in self._explanation_cache:
                logger.info(f"キャッシュから説明を取得: {method}")
                return self._explanation_cache[cache_key]
        
        try:
            # 説明を生成
            if method in self.method_implementations:
                explanation = self.method_implementations[method](question, answer, sources, **kwargs)
            else:
                logger.warning(f"不明なXAI手法 '{method}' が指定されました。")
                explanation = self._generate_qwen3_thinking_process_explanation(question, answer, sources, **kwargs)
            
            # キャッシュに保存
            if use_cache and len(self._explanation_cache) < self._cache_max_size:
                self._explanation_cache[cache_key] = explanation
                logger.info(f"説明をキャッシュに保存: {method}")
            elif use_cache and len(self._explanation_cache) >= self._cache_max_size:
                # 最も古いエントリを削除（簡易的なLRU）
                oldest_key = next(iter(self._explanation_cache))
                del self._explanation_cache[oldest_key]
                self._explanation_cache[cache_key] = explanation
            
            return explanation
            
        except Exception as e:
            logger.error(f"'{method}' 説明生成中にエラーが発生しました: {e}")
            return f"説明生成中にエラーが発生しました: {e}"

    def clear_cache(self):
        """キャッシュをクリア"""
        self._explanation_cache.clear()
        self._get_cache_key.cache_clear()
        logger.info("説明キャッシュをクリアしました")


    def _generate_chain_of_thought_explanation(self, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """思考の連鎖による説明生成（新機能）"""
        prompt = PromptTemplate(
            input_variables=["question", "answer", "sources"],
            template="""
            あなたは段階的思考プロセスの専門家です。
            以下のAI回答について、人間が理解しやすい段階的な思考の連鎖を示してください。

            ユーザーの質問: {question}
            AIの回答: {answer}

            参照情報:
            {sources}

            ## 思考の連鎖 (Chain of Thought)

            ### 🎯 ステップ 1: 問題の理解と分解
            **質問の核心**: 
            - 主要な問いかけ: ...
            - 暗黙の前提: ...
            - 期待される回答の種類: ...

            ### 📊 ステップ 2: 利用可能な情報の整理
            **情報源の分類**:
            - 直接関連する情報: ...
            - 補助的な情報: ...
            - 背景知識: ...

            ### 🔍 ステップ 3: 段階的推論
            **推論の流れ**:
            1. まず、... だから ...
            2. 次に、... したがって ...
            3. さらに、... ゆえに ...
            4. 最後に、... よって ...

            ### ✅ ステップ 4: 結論の導出
            **最終的な答え**:
            - 主要な結論: ...
            - 補足情報: ...
            - 注意事項: ...

            ### 🔄 ステップ 5: 自己検証
            **論理的整合性のチェック**:
            - ✓ 前提と結論の整合性: ...
            - ✓ 推論の妥当性: ...
            - ✓ 情報源との一致: ...
            """
        )
        sources_text = self._format_sources_text(sources)
        inputs = {"question": question, "answer": answer, "sources": sources_text}
        return self._run_llm_chain(prompt, inputs)

    def _generate_comparative_analysis_explanation(self, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """比較分析による説明生成（新機能）"""
        prompt = PromptTemplate(
            input_variables=["question", "answer", "sources"],
            template="""
            あなたは比較分析の専門家です。
            以下のAI回答について、複数の観点から比較分析を行い、なぜこの回答が選ばれたかを説明してください。

            ユーザーの質問: {question}
            AIの回答: {answer}

            参照情報:
            {sources}

            ## 📊 比較分析レポート

            ### 1. 回答アプローチの比較
            
            | アプローチ | 内容 | 長所 | 短所 | 採用理由 |
            |-----------|------|------|------|----------|
            | **採用案** | {answer} | ... | ... | ✅ 選択 |
            | 代替案1 | ... | ... | ... | ❌ 不採用 |
            | 代替案2 | ... | ... | ... | ❌ 不採用 |

            ### 2. 情報源の比較評価
            
            **情報源の信頼性マトリックス**:
            - 情報源A: 信頼度 [■■■■□] 80% - 理由: ...
            - 情報源B: 信頼度 [■■■□□] 60% - 理由: ...
            - 情報源C: 信頼度 [■■□□□] 40% - 理由: ...

            ### 3. 視点別の比較
            
            **技術的視点**: 
            - 現在の回答: ...
            - 別の視点: ...
            
            **実用的視点**:
            - 現在の回答: ...
            - 別の視点: ...
            
            **理論的視点**:
            - 現在の回答: ...
            - 別の視点: ...

            ### 4. トレードオフ分析
            
            **精度 vs 理解しやすさ**:
            - 現在の選択: バランス型（精度70%, 理解度80%）
            - 代替選択A: 精度重視（精度90%, 理解度50%）
            - 代替選択B: 理解重視（精度50%, 理解度95%）

            ### 5. 最終評価
            **なぜこの回答が最適か**:
            - 主要な強み: ...
            - 受け入れられるリスク: ...
            - 総合評価: ⭐⭐⭐⭐☆
            """
        )
        sources_text = self._format_sources_text(sources)
        inputs = {"question": question, "answer": answer, "sources": sources_text}
        return self._run_llm_chain(prompt, inputs)

    def _generate_uncertainty_quantification_explanation(self, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """不確実性定量化による説明生成（新機能）"""
        prompt = PromptTemplate(
            input_variables=["question", "answer", "sources"],
            template="""
            あなたは不確実性分析の専門家です。
            以下のAI回答について、各要素の確実性・不確実性を定量的に評価してください。

            ユーザーの質問: {question}
            AIの回答: {answer}

            参照情報:
            {sources}

            ## 🎯 不確実性定量化分析

            ### 1. 知識の不確実性（エピステミック不確実性）
            
            **情報の完全性評価**:
            ```
            完全性スコア: [████████░░] 82/100
            - ✅ カバーされている領域: ...
            - ⚠️ 不足している情報: ...
            - ❌ 未知の領域: ...
            ```

            **情報源の信頼性**:
            ```
            総合信頼度: [███████░░░] 75/100
            - 一次情報源: 90% 信頼度
            - 二次情報源: 70% 信頼度  
            - 推論部分: 60% 信頼度
            ```

            ### 2. 偶然の不確実性（アレアトリック不確実性）
            
            **変動要因の分析**:
            - 文脈依存性: 中程度（状況により解釈が変わる可能性）
            - 時間的変動: 低（情報の時間的安定性は高い）
            - 個人差: 高（読み手により理解が異なる可能性）

            ### 3. 確信度ヒートマップ
            
            回答の各部分の確信度を視覚化：
            
            ```
            [回答の第1文] ████████████ 95% - 情報源から直接確認
            [回答の第2文] ██████████░░ 85% - 複数源から推論
            [回答の第3文] ███████░░░░░ 65% - 単一源からの推論
            [回答の第4文] █████░░░░░░░ 45% - 一般的知識からの推論
            ```

            ### 4. リスク評価マトリックス
            
            | リスク要因 | 発生確率 | 影響度 | リスクレベル |
            |-----------|---------|--------|-------------|
            | 情報の誤解釈 | 低 (20%) | 中 | 🟡 低リスク |
            | 文脈の見落とし | 中 (40%) | 高 | 🟠 中リスク |
            | 推論の誤り | 低 (15%) | 高 | 🟡 低リスク |
            | 情報の陳腐化 | 低 (10%) | 低 | 🟢 極低リスク |

            ### 5. 信頼区間と推奨アクション
            
            **回答の信頼区間**: 70% - 90%
            - 最も可能性の高い解釈: {answer}
            - 下限の解釈: より保守的な見解...
            - 上限の解釈: より積極的な見解...

            **推奨される追加確認**:
            1. 🔍 追加で確認すべき情報源: ...
            2. 💡 専門家への相談が推奨される部分: ...
            3. ⏰ 定期的な見直しが必要な要素: ...

            ### 6. 総合的な確実性評価

            **全体的な確信度**: [███████░░░] 72%

            この評価は以下の要因に基づいています：
            - 情報源の質と量
            - 推論の論理的妥当性
            - 既知の制限事項
            - 潜在的なバイアス
            """
        )
        sources_text = self._format_sources_text(sources)
        inputs = {"question": question, "answer": answer, "sources": sources_text}
        return self._run_llm_chain(prompt, inputs)


    def select_methods(self, question: str, answer: str, domain: str = "一般") -> Dict[str, Any]:
        """
        最適なXAI手法を選択

        Parameters:
        -----------
        question : str
            ユーザーの質問
        answer : str
            AIの回答
        domain : str, optional
            ドメイン情報

        Returns:
        --------
        dict
            選択された手法と理由
        """
        try:
            # RunnableSequenceを使用（LangChain v0.3+推奨）
            chain = self.selection_prompt | self.llm

            response = chain.invoke({
                "question": question,
                "answer": answer,
                "domain": domain
            })

            # responseがAIMessageの場合、contentを取得
            response_text = response.content if hasattr(response, 'content') else str(response)

            # 文字列からJSONを抽出
            json_str = response_text
            if "{" in json_str and "}" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]

            result = json.loads(json_str)

            # 結果の妥当性をチェック
            if "primary_method" not in result:
                raise ValueError("primary_methodが結果に含まれていません")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"XAI手法選択結果のJSONパース中にエラーが発生しました: {e}")
            logger.error(f"元のレスポンス: {response_text if 'response_text' in locals() else 'N/A'}")
            return self._default_method_selection("JSONパースエラー")
        except Exception as e:
            logger.error(f"XAI手法選択中に予期せぬエラーが発生しました: {e}")
            return self._default_method_selection(f"予期せぬエラー: {e}")

    def _default_method_selection(self, reason: str) -> Dict[str, Any]:
        """デフォルトのXAI手法選択結果を返す"""
        return {
            "primary_method": "Qwen3風思考プロセス",
            "secondary_method": "プロセス追跡",
            "reasoning": f"エラーのため、デフォルトの手法を選択しました ({reason})。",
            "approach": "基本的な説明アプローチを適用します。"
        }

    def generate_explanation(self, method: str, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """
        選択された手法に基づいて説明を生成

        Parameters:
        -----------
        method : str
            使用するXAI手法
        question : str
            ユーザーの質問
        answer : str
            AIの回答
        sources : list[Document]
            関連ドキュメントのリスト
        **kwargs : dict
            その他の引数

        Returns:
        --------
        str
            生成された説明
        """
        try:
            if method in self.method_implementations:
                return self.method_implementations[method](question, answer, sources, **kwargs)
            else:
                logger.warning(f"不明なXAI手法 '{method}' が指定されました。デフォルトのQwen3風思考プロセスを使用します。")
                return self._generate_qwen3_thinking_process_explanation(question, answer, sources, **kwargs)
        except Exception as e:
            logger.error(f"'{method}' 説明生成中にエラーが発生しました: {e}")
            return f"説明生成中にエラーが発生しました: {e}. デフォルトの説明を提供します。\n\n" + \
                self._generate_qwen3_thinking_process_explanation(question, answer, sources, **kwargs)

    def _format_sources_text(self, sources: List[Document]) -> str:
        """ソースリストを整形するヘルパー関数"""
        sources_text = ""
        if not sources:
            return "利用可能な情報源はありません。\n\n"

        for i, source in enumerate(sources):
            filename = source.metadata.get('filename', 'Unknown')
            content = source.page_content
            truncated_content = (content[:500] + '...') if len(content) > 500 else content
            sources_text += f"[情報源 {i+1}] {filename}:\n"
            sources_text += f"{truncated_content}\n\n"
        return sources_text

    def _run_llm_chain(self, prompt: PromptTemplate, inputs: Dict[str, Any]) -> str:
        """LLMチェーンを実行し、結果のテキストを返すヘルパー関数"""
        try:
            # RunnableSequenceを使用
            chain = prompt | self.llm
            response = chain.invoke(inputs)
            
            # responseがAIMessageの場合、contentを取得
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"LLMチェーン実行中にエラーが発生しました: {e}")
            return f"エラー: 説明生成中に問題が発生しました ({e})。"

    # === 以下、既存のメソッドをそのまま保持 ===
    
    def _generate_qwen3_thinking_process_explanation(self, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """Qwen3風の思考プロセスによる説明を生成"""
        prompt = PromptTemplate(
            input_variables=["question", "answer", "sources"],
            template="""
            あなたは思考プロセスの可視化を得意とするAIアシスタントです。
            以下のAI回答について、モデルが考えた思考プロセスを<think>タグで囲み、最後に最終回答を示してください。

            ユーザーの質問: {question}
            AIの回答: {answer}

            参照情報:
            {sources}

            Qwen3風の思考プロセスとして以下のフォーマットで説明してください:

            <think>
            ここでは、回答に至るまでの思考過程を詳細に記述します。以下の点を自然な思考の流れで説明してください:
            1. 質問の意図や要件の理解
            2. 関連する情報源からの知識の抽出
            3. 複数の視点からの検討
            4. 論理的推論のステップ
            5. 回答の妥当性評価
            </think>

            最終回答: （ここには元の回答をそのまま、または必要に応じて改善して記載）
            """
        )
        sources_text = self._format_sources_text(sources)
        inputs = {"question": question, "answer": answer, "sources": sources_text}
        return self._run_llm_chain(prompt, inputs)

    def _generate_feature_importance_explanation(self, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """特徴重要度分析による説明を生成"""
        prompt = PromptTemplate(
            input_variables=["question", "answer", "sources"],
            template="""
            あなたは特徴重要度分析の専門家です。
            以下のAI回答について、どの入力情報（キーワードや文脈）が回答生成に最も影響を与えたかを分析し、説明してください。

            ユーザーの質問: {question}
            AIの回答: {answer}

            参照情報:
            {sources}

            ステップ1: 質問内の重要キーワードを特定し、それらが回答にどう影響したかを説明
            ステップ2: 外部知識ソースの重要部分を特定し、重要度を5段階で評価
            ステップ3: 回答の各パートがどの情報ソースに依存しているかをマッピング
            """
        )
        sources_text = self._format_sources_text(sources)
        inputs = {"question": question, "answer": answer, "sources": sources_text}
        return self._run_llm_chain(prompt, inputs)

    def _generate_counterfactual_explanation(self, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """反事実説明を生成"""
        prompt = PromptTemplate(
            input_variables=["question", "answer", "sources"],
            template="""
            あなたは反事実説明の専門家です。
            以下のAI回答について、入力が異なっていた場合に回答がどう変わっていたかを説明してください。

            ユーザーの質問: {question}
            AIの回答: {answer}

            参照情報:
            {sources}

            ステップ1: 質問の重要な要素を特定し、それらが変わった場合の代替回答を示す
            ステップ2: 「もし〜が異なっていたら、回答は〜になっていた」という形式で3つの反事実シナリオを提示
            ステップ3: なぜモデルがこの回答を選択し、他の可能性を除外したのかを説明
            """
        )
        sources_text = self._format_sources_text(sources)
        inputs = {"question": question, "answer": answer, "sources": sources_text}
        return self._run_llm_chain(prompt, inputs)

    def _generate_rule_based_explanation(self, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """ルールベース説明を生成"""
        prompt = PromptTemplate(
            input_variables=["question", "answer", "sources"],
            template="""
            あなたはルールベース説明の専門家です。
            以下のAI回答について、モデルの決定プロセスをif-thenルールの形式で説明してください。

            ユーザーの質問: {question}
            AIの回答: {answer}

            参照情報:
            {sources}

            ステップ1: モデルが使用したと思われる決定ルールを抽出
            ステップ2: 「もし〜ならば、〜と判断する」という形式で5-7個のルールを提示
            ステップ3: これらのルールが実際の回答にどのように適用されたかを説明
            """
        )
        sources_text = self._format_sources_text(sources)
        inputs = {"question": question, "answer": answer, "sources": sources_text}
        return self._run_llm_chain(prompt, inputs)

    def _generate_case_based_explanation(self, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """事例ベース説明を生成"""
        prompt = PromptTemplate(
            input_variables=["question", "answer", "sources"],
            template="""
            あなたは事例ベース説明の専門家です。
            以下のAI回答について、類似する既知の事例と比較することで説明してください。

            ユーザーの質問: {question}
            AIの回答: {answer}

            参照情報:
            {sources}

            ステップ1: この質問/回答に類似する事例を参照情報から特定
            ステップ2: 参照情報の事例とこの回答の類似点と相違点を分析
            ステップ3: なぜモデルがこの事例から学習し、適用したのかを説明
            """
        )
        sources_text = self._format_sources_text(sources)
        inputs = {"question": question, "answer": answer, "sources": sources_text}
        return self._run_llm_chain(prompt, inputs)

    def _generate_process_tracking_explanation(self, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """プロセス追跡による説明を生成"""
        prompt = PromptTemplate(
            input_variables=["question", "answer", "sources"],
            template="""
            あなたはAIの思考プロセスを説明する専門家です。
            以下のAI回答について、モデルの推論過程を段階的に再現してください。

            ユーザーの質問: {question}
            AIの回答: {answer}

            参照情報:
            {sources}

            以下のフォーマットで説明してください:

            ## 思考プロセスの追跡
            1. **最初の理解**: モデルはまず質問をどう解釈したか
            2. **情報収集**: どの知識ソースにアクセスし、何を抽出したか
            3. **推論ステップ**: どのような中間的な結論を導いたか
            4. **検証プロセス**: どのように結論を確認したか
            5. **最終判断**: なぜこの回答が最適と判断したか

            ## 確信度分析
            各ステップの確信度と不確実性

            ## 代替解釈
            考慮されたが採用されなかった代替案
            """
        )
        sources_text = self._format_sources_text(sources)
        inputs = {"question": question, "answer": answer, "sources": sources_text}
        return self._run_llm_chain(prompt, inputs)

    def _generate_attention_visualization_explanation(self, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """アテンション可視化による説明を生成"""
        prompt = PromptTemplate(
            input_variables=["question", "answer", "sources"],
            template="""
            あなたはアテンション可視化の専門家です。
            以下のAI回答について、モデルがどの情報に注目したかをテキストで可視化してください。

            ユーザーの質問: {question}
            AIの回答: {answer}

            参照情報:
            {sources}

            ステップ1: 質問内の重要な部分をハイライトし、重要度を説明
            ステップ2: 回答生成に特に影響を与えた情報源の部分をハイライトし、重要度を説明
            ステップ3: 質問と回答の間の関連性をアテンションの流れとして説明

            重要度の表現には以下の記号を使用してください:
            [重要度★★★] 非常に重要な情報
            [重要度★★] 中程度に重要な情報
            [重要度★] 関連性のある情報
            """
        )
        sources_text = self._format_sources_text(sources)
        inputs = {"question": question, "answer": answer, "sources": sources_text}
        return self._run_llm_chain(prompt, inputs)

    def _generate_local_approximation_explanation(self, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """局所的近似による説明を生成"""
        prompt = PromptTemplate(
            input_variables=["question", "answer", "sources"],
            template="""
            あなたは局所的近似説明の専門家です。
            以下のAI回答について、複雑なモデルの決定を単純なモデルで近似して説明してください。

            ユーザーの質問: {question}
            AIの回答: {answer}

            参照情報:
            {sources}

            ステップ1: モデルの決定を3-5個の主要因子に単純化して説明
            ステップ2: 各因子が最終決定にどの程度影響したかの割合を提示
            ステップ3: この単純化されたモデルが元の複雑なモデルの決定をどの程度正確に近似しているかを説明
            """
        )
        sources_text = self._format_sources_text(sources)
        inputs = {"question": question, "answer": answer, "sources": sources_text}
        return self._run_llm_chain(prompt, inputs)

    def _generate_multi_faceted_explanation(self, question: str, answer: str, sources: List[Document], **kwargs) -> str:
        """多角的説明を生成"""
        primary_explanation = ""
        secondary_explanation = ""
        try:
            primary_explanation = self._generate_process_tracking_explanation(question, answer, sources, **kwargs)
        except Exception as e:
            logger.error(f"多角的説明のプライマリ生成（プロセス追跡）でエラー: {e}")
            primary_explanation = "プロセス追跡による説明の生成に失敗しました。"

        try:
            secondary_explanation = self._generate_feature_importance_explanation(question, answer, sources, **kwargs)
        except Exception as e:
            logger.error(f"多角的説明のセカンダリ生成（特徴重要度）でエラー: {e}")
            secondary_explanation = "特徴重要度による説明の生成に失敗しました。"

        prompt = PromptTemplate(
            input_variables=["primary", "secondary", "question", "answer"],
            template="""
            あなたは多角的AI説明の専門家です。
            以下の2つの異なるアプローチによる説明を統合して、より包括的な説明を生成してください。

            ユーザーの質問: {question}
            AIの回答: {answer}

            説明アプローチ1 (プロセス追跡):
            {primary}

            説明アプローチ2 (特徴重要度):
            {secondary}

            これらの説明を統合し、以下の構造で包括的な説明を提供してください:

            ## 1. 概要
            回答の主要ポイントと根拠の簡潔な要約

            ## 2. 推論プロセス
            モデルがどのように考え、結論に至ったか (アプローチ1に基づく)

            ## 3. 重要な影響要因
            回答形成に最も影響を与えた情報と要素 (アプローチ2に基づく)

            ## 4. 代替可能性
            考慮された他の可能性と、それらが選択されなかった理由 (可能であれば言及)

            ## 5. 信頼性評価
            この回答の確実性と制限事項 (可能であれば言及)
            """
        )

        inputs = {
            "primary": primary_explanation,
            "secondary": secondary_explanation,
            "question": question,
            "answer": answer
        }
        return self._run_llm_chain(prompt, inputs)
    
    def evaluate_explanation_quality(self, explanation: str, question: str, answer: str) -> Dict[str, float]:
        """説明の品質を評価"""
    
        metrics = {
            'length': len(explanation),
            'readability_score': self._calculate_readability(explanation),
            'structure_score': self._evaluate_structure(explanation),
            'completeness_score': self._evaluate_completeness(explanation, question, answer),
            'timestamp': datetime.now().isoformat()
        }
        
        # 総合スコアを計算
        metrics['overall_score'] = np.mean([
            metrics['readability_score'],
            metrics['structure_score'],
            metrics['completeness_score']
        ])
        
        return metrics

    def _calculate_readability(self, text: str) -> float:
        """可読性スコアを計算（0-1）"""
        # 簡易的な実装
        avg_sentence_length = np.mean([len(s.split()) for s in text.split('。') if s])
        
        # 理想的な文長は15-20単語
        if 15 <= avg_sentence_length <= 20:
            return 1.0
        elif avg_sentence_length < 10 or avg_sentence_length > 30:
            return 0.5
        else:
            return 0.8

    def _evaluate_structure(self, text: str) -> float:
        """構造の良さを評価（0-1）"""
        structure_elements = ['##', '###', '1.', '2.', '3.', '-', '*', '|']
        found_elements = sum(1 for elem in structure_elements if elem in text)
        
        # 構造要素が多いほど高スコア
        return min(found_elements / 5.0, 1.0)

    def _evaluate_completeness(self, explanation: str, question: str, answer: str) -> float:
        """説明の完全性を評価（0-1）"""
        # 質問と回答の主要な要素が説明に含まれているかチェック
        question_words = set(question.split())
        answer_words = set(answer.split())
        explanation_words = set(explanation.split())
        
        question_coverage = len(question_words & explanation_words) / len(question_words)
        answer_coverage = len(answer_words & explanation_words) / len(answer_words)
        
        return (question_coverage + answer_coverage) / 2.0

