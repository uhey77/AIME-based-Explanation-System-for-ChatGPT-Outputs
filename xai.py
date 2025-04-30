# xai.py
import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# modelsモジュールからDocumentクラスをインポート
from models import Document
# configモジュールから設定値をインポート
import config


class XAIMethodSelector:
    """XAI手法選択と説明生成クラス"""

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
            print(f"エラー: ChatOpenAI モデル '{model_name}' の初期化に失敗しました: {e}")
            # フォールバックやエラー処理をここに追加することも可能
            raise  # エラーを再発生させて、上位で処理できるようにする

        # XAI手法選択プロンプト
        self.selection_prompt = PromptTemplate(
            input_variables=["question", "answer", "domain"],
            template="""
            あなたは最適なXAI(Explainable AI)手法を選択する専門家です。
            以下のAI応答に対して最も適切な説明手法を選択してください。

            ユーザーの質問: {question}
            AIの回答: {answer}
            ドメイン: {domain}

            以下の説明手法から、この状況に最適な手法とその理由を選択してください:
            1. 特徴重要度分析 - 回答に影響を与えた入力の重要度を可視化
            2. 反事実説明 - 「もし〜が異なっていたら、回答はこう変わっていた」という説明
            3. ルールベース説明 - モデルの決定プロセスをif-thenルールとして表現
            4. 事例ベース説明 - 類似事例と比較して説明
            5. プロセス追跡 - モデルの推論過程を段階的に説明
            6. アテンション可視化 - モデルが注目した情報を強調
            7. 局所的近似 - 複雑なモデルの決定を局所的に単純なモデルで近似
            8. 多角的説明 - 複数の説明手法を組み合わせて総合的に説明

            回答形式:
            {{
              "primary_method": "選択した主要手法",
              "secondary_method": "補助的手法（必要な場合）",
              "reasoning": "選択理由",
              "approach": "具体的な実装アプローチ"
            }}

            JSONフォーマットで回答してください。
            """
        )

        self.method_implementations = {
            "特徴重要度分析": self._generate_feature_importance_explanation,
            "反事実説明": self._generate_counterfactual_explanation,
            "ルールベース説明": self._generate_rule_based_explanation,
            "事例ベース説明": self._generate_case_based_explanation,
            "プロセス追跡": self._generate_process_tracking_explanation,
            "アテンション可視化": self._generate_attention_visualization_explanation,
            "局所的近似": self._generate_local_approximation_explanation,
            "多角的説明": self._generate_multi_faceted_explanation
        }

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
            # LLMChain は非推奨ですが、元のコードに合わせています。
            # 本来は RunnableSequence (例: `prompt | llm`) を使うべきです。
            chain = LLMChain(llm=self.llm, prompt=self.selection_prompt)

            response = chain.invoke({
                "question": question,
                "answer": answer,
                "domain": domain
            })

            # 文字列からJSONを抽出
            json_str = response["text"]
            if "{" in json_str and "}" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]

            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            print(f"XAI手法選択結果のJSONパース中にエラーが発生しました: {e}")
            print(f"元のレスポンス: {response}")
            # フォールバック
            return self._default_method_selection("JSONパースエラー")
        except Exception as e:
            print(f"XAI手法選択中に予期せぬエラーが発生しました: {e}")
            # フォールバック
            return self._default_method_selection(f"予期せぬエラー: {e}")

    def _default_method_selection(self, reason: str) -> Dict[str, Any]:
        """デフォルトのXAI手法選択結果を返す"""
        return {
            "primary_method": "多角的説明",
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
                print(f"警告: 不明なXAI手法 '{method}' が指定されました。デフォルトのプロセス追跡を使用します。")
                # デフォルトはプロセス追跡
                return self._generate_process_tracking_explanation(question, answer, sources, **kwargs)
        except Exception as e:
            print(f"'{method}' 説明生成中にエラーが発生しました: {e}")
            # エラー発生時はシンプルな説明を返すなど、フォールバック処理を行う
            return f"説明生成中にエラーが発生しました: {e}. デフォルトの説明を提供します。\n\n" + \
                self._generate_process_tracking_explanation(question, answer, sources, **kwargs)

    def _format_sources_text(self, sources: List[Document]) -> str:
        """ソースリストを整形するヘルパー関数"""
        sources_text = ""
        if not sources:
            return "利用可能な情報源はありません。\n\n"

        for i, source in enumerate(sources):
            # Document オブジェクトから正しく情報を取得
            filename = source.metadata.get('filename', 'Unknown')
            content = source.page_content
            # コンテンツが長すぎる場合に切り詰める（例：最初の500文字）
            truncated_content = (content[:500] + '...') if len(content) > 500 else content
            sources_text += f"[情報源 {i+1}] {filename}:\n"
            sources_text += f"{truncated_content}\n\n"  # 切り詰めたコンテンツを使用
        return sources_text

    def _run_llm_chain(self, prompt: PromptTemplate, inputs: Dict[str, Any]) -> str:
        """LLMチェーンを実行し、結果のテキストを返すヘルパー関数"""
        try:
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.invoke(inputs)
            return response.get("text", "エラー: LLMからの応答がありませんでした。")
        except Exception as e:
            print(f"LLMチェーン実行中にエラーが発生しました: {e}")
            return f"エラー: 説明生成中に問題が発生しました ({e})。"

    # --- 個別の説明生成メソッド ---
    # (元のコードから _generate_***_explanation メソッドをここに移動)
    # 各メソッド内で _format_sources_text と _run_llm_chain を使用するように修正

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
        # 修正されたメソッドを使って説明を生成
        # エラーハンドリングを考慮し、各説明生成が失敗する可能性に対処
        primary_explanation = ""
        secondary_explanation = ""
        try:
            primary_explanation = self._generate_process_tracking_explanation(question, answer, sources, **kwargs)
        except Exception as e:
            print(f"多角的説明のプライマリ生成（プロセス追跡）でエラー: {e}")
            primary_explanation = "プロセス追跡による説明の生成に失敗しました。"

        try:
            secondary_explanation = self._generate_feature_importance_explanation(question, answer, sources, **kwargs)
        except Exception as e:
            print(f"多角的説明のセカンダリ生成（特徴重要度）でエラー: {e}")
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
