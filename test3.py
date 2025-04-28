"""
Enhanced AIME-based Explanation System for ChatGPT Outputs
---------------------------------------------------------
このシステムは以下の機能を提供します：
1. ユーザーの質問に対するChatGPTの回答の根拠を示す
2. ChatGPTが特定の回答をした理由を説明する
3. 自己検証と逐次検索によるアトリビュート強化
4. 複数のXAI手法の組み合わせによる多角的説明
5. 説明品質の評価と継続的改善機能
"""

import os
from dotenv import load_dotenv
import openai
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, SequentialChain, LLMChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import SelfQueryRetriever
from langchain_community.llms import LlamaCpp
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from typing import Dict, List, Any, Optional, Tuple, Union
import gradio as gr
import PyPDF2
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import time
import sqlite3
import pandas as pd
from datetime import datetime
import hashlib
import torch
from transformers import pipeline

# 環境変数の読み込み
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class Document:
    """ドキュメントを表すクラス"""
    def __init__(self, text, metadata):
        self.page_content = text
        self.metadata = metadata


class EvaluationMetrics:
    """説明品質の評価指標を管理するクラス"""
    
    def __init__(self, db_path="explanation_metrics.db"):
        """
        初期化メソッド
        
        Parameters:
        -----------
        db_path : str
            評価指標を保存するSQLiteデータベースのパス
        """
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """必要なテーブルを作成"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 評価データテーブル
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS explanation_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question_hash TEXT,
            question TEXT,
            answer_hash TEXT,
            explanation_hash TEXT,
            method TEXT,
            coherence REAL,
            relevance REAL,
            completeness REAL,
            factual_accuracy REAL,
            user_rating INTEGER,
            confidence_score REAL,
            processing_time REAL
        )
        ''')
        
        # 改善履歴テーブル
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS improvement_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            change_description TEXT,
            baseline_avg_score REAL,
            new_avg_score REAL
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_metrics(self, question, answer, explanation, method, metrics):
        """
        評価指標を記録
        
        Parameters:
        -----------
        question : str
            ユーザーの質問
        answer : str
            システムの回答
        explanation : str
            生成された説明
        method : str
            使用された説明生成方法
        metrics : dict
            評価指標のディクショナリ
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ハッシュ生成
        question_hash = hashlib.md5(question.encode()).hexdigest()
        answer_hash = hashlib.md5(answer.encode()).hexdigest()
        explanation_hash = hashlib.md5(explanation.encode()).hexdigest()
        
        # 現在のタイムスタンプ
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO explanation_metrics (
            timestamp, question_hash, question, answer_hash, explanation_hash, 
            method, coherence, relevance, completeness, factual_accuracy, 
            user_rating, confidence_score, processing_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, question_hash, question, answer_hash, explanation_hash,
            method, metrics.get('coherence', 0), metrics.get('relevance', 0),
            metrics.get('completeness', 0), metrics.get('factual_accuracy', 0),
            metrics.get('user_rating', 0), metrics.get('confidence_score', 0),
            metrics.get('processing_time', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def record_improvement(self, description, baseline_score, new_score):
        """
        システム改善を記録
        
        Parameters:
        -----------
        description : str
            改善内容の説明
        baseline_score : float
            ベースラインの平均スコア
        new_score : float
            新しい平均スコア
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
        INSERT INTO improvement_history (
            timestamp, change_description, baseline_avg_score, new_avg_score
        ) VALUES (?, ?, ?, ?)
        ''', (timestamp, description, baseline_score, new_score))
        
        conn.commit()
        conn.close()
    
    def get_metrics_summary(self, days=30, method=None):
        """
        指定された期間の評価指標の要約を取得
        
        Parameters:
        -----------
        days : int
            遡る日数
        method : str, optional
            特定の説明生成方法でフィルタリング
        
        Returns:
        --------
        dict
            評価指標の要約統計
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        time_limit = (datetime.now() - pd.Timedelta(days=days)).isoformat()
        
        if method:
            cursor.execute('''
            SELECT 
                AVG(coherence), AVG(relevance), AVG(completeness), 
                AVG(factual_accuracy), AVG(user_rating), AVG(confidence_score),
                AVG(processing_time), COUNT(*)
            FROM explanation_metrics
            WHERE timestamp > ? AND method = ?
            ''', (time_limit, method))
        else:
            cursor.execute('''
            SELECT 
                AVG(coherence), AVG(relevance), AVG(completeness), 
                AVG(factual_accuracy), AVG(user_rating), AVG(confidence_score),
                AVG(processing_time), COUNT(*)
            FROM explanation_metrics
            WHERE timestamp > ?
            ''', (time_limit,))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'coherence': result[0] or 0,
            'relevance': result[1] or 0,
            'completeness': result[2] or 0,
            'factual_accuracy': result[3] or 0,
            'user_rating': result[4] or 0,
            'confidence_score': result[5] or 0,
            'processing_time': result[6] or 0,
            'sample_count': result[7] or 0
        }
    
    def get_improvement_history(self):
        """
        改善履歴を取得
        
        Returns:
        --------
        pd.DataFrame
            改善履歴のデータフレーム
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM improvement_history ORDER BY timestamp DESC"
        history = pd.read_sql_query(query, conn)
        
        conn.close()
        return history
    
    def generate_evaluation_report(self):
        """
        評価レポートを生成
        
        Returns:
        --------
        dict
            評価レポート
        """
        conn = sqlite3.connect(self.db_path)
        
        # 全体の傾向
        overall = pd.read_sql_query('''
        SELECT 
            AVG(coherence) as avg_coherence, 
            AVG(relevance) as avg_relevance,
            AVG(completeness) as avg_completeness,
            AVG(factual_accuracy) as avg_factual_accuracy,
            AVG(user_rating) as avg_user_rating,
            AVG(confidence_score) as avg_confidence
        FROM explanation_metrics
        ''', conn)
        
        # 方法別の比較
        methods = pd.read_sql_query('''
        SELECT 
            method,
            AVG(coherence) as avg_coherence, 
            AVG(relevance) as avg_relevance,
            AVG(completeness) as avg_completeness,
            AVG(factual_accuracy) as avg_factual_accuracy,
            AVG(user_rating) as avg_user_rating,
            COUNT(*) as count
        FROM explanation_metrics
        GROUP BY method
        ORDER BY avg_user_rating DESC
        ''', conn)
        
        # 時系列トレンド
        trend = pd.read_sql_query('''
        SELECT 
            DATE(timestamp) as date,
            AVG(coherence) as avg_coherence, 
            AVG(relevance) as avg_relevance,
            AVG(completeness) as avg_completeness,
            AVG(user_rating) as avg_user_rating
        FROM explanation_metrics
        GROUP BY DATE(timestamp)
        ORDER BY date
        ''', conn)
        
        conn.close()
        
        return {
            'overall_metrics': overall.to_dict('records')[0] if not overall.empty else {},
            'method_comparison': methods.to_dict('records'),
            'time_trend': trend.to_dict('records')
        }


# 修正済みの XAIMethodSelector クラス
class XAIMethodSelector:
    """XAI手法選択クラス"""

    def __init__(self, model_name="gpt-4"):
        """
        初期化メソッド

        Parameters:
        -----------
        model_name : str
            XAI手法選択に使用するLLMのモデル名
        """
        self.model_name = model_name
        self.llm = ChatOpenAI(model_name=model_name)

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

    def select_methods(self, question, answer, domain="一般"):
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
        # LLMChain は非推奨ですが、元のコードに合わせています。
        # 本来は RunnableSequence (例: `prompt | llm`) を使うべきです。
        chain = LLMChain(llm=self.llm, prompt=self.selection_prompt)

        response = chain.invoke({
            "question": question,
            "answer": answer,
            "domain": domain
        })

        try:
            # 文字列からJSONを抽出
            json_str = response["text"]
            if "{" in json_str and "}" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]

            result = json.loads(json_str)
            return result
        except Exception as e:
            print(f"XAI手法選択結果のパース中にエラーが発生しました: {e}")
            print(f"元のレスポンス: {response}")
            return {
                "primary_method": "多角的説明",
                "secondary_method": "プロセス追跡",
                "reasoning": "パースエラーのため、デフォルトの手法を選択しました。",
                "approach": "基本的な説明アプローチを適用します。"
            }

    def generate_explanation(self, method, question, answer, sources, **kwargs):
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
        sources : list[Document] # 型ヒントを明確化
            関連ドキュメントのリスト
        **kwargs : dict
            その他の引数

        Returns:
        --------
        str
            生成された説明
        """
        if method in self.method_implementations:
            return self.method_implementations[method](question, answer, sources, **kwargs)
        else:
            # デフォルトはプロセス追跡
            return self._generate_process_tracking_explanation(question, answer, sources, **kwargs)

    def _format_sources_text(self, sources: List[Document]) -> str:
        """ソースリストを整形するヘルパー関数"""
        sources_text = ""
        for i, source in enumerate(sources):
            # Document オブジェクトから正しく情報を取得
            filename = source.metadata.get('filename', 'Unknown')
            content = source.page_content
            sources_text += f"[情報源 {i+1}] {filename}:\n"
            sources_text += f"{content}\n\n"
        return sources_text

    def _generate_feature_importance_explanation(self, question, answer, sources, **kwargs):
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
        # ヘルパー関数を使用してソーステキストを整形
        sources_text = self._format_sources_text(sources)

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.invoke({
            "question": question,
            "answer": answer,
            "sources": sources_text
        })
        return response["text"]

    def _generate_counterfactual_explanation(self, question, answer, sources, **kwargs):
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
        # ヘルパー関数を使用してソーステキストを整形
        sources_text = self._format_sources_text(sources)

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.invoke({
            "question": question,
            "answer": answer,
            "sources": sources_text
        })
        return response["text"]

    def _generate_rule_based_explanation(self, question, answer, sources, **kwargs):
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
        # ヘルパー関数を使用してソーステキストを整形
        sources_text = self._format_sources_text(sources)

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.invoke({
            "question": question,
            "answer": answer,
            "sources": sources_text
        })
        return response["text"]

    def _generate_case_based_explanation(self, question, answer, sources, **kwargs):
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
        # ヘルパー関数を使用してソーステキストを整形
        sources_text = self._format_sources_text(sources)

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.invoke({
            "question": question,
            "answer": answer,
            "sources": sources_text
        })
        return response["text"]

    def _generate_process_tracking_explanation(self, question, answer, sources, **kwargs):
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
        # ヘルパー関数を使用してソーステキストを整形
        sources_text = self._format_sources_text(sources)

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.invoke({
            "question": question,
            "answer": answer,
            "sources": sources_text
        })
        return response["text"]

    def _generate_attention_visualization_explanation(self, question, answer, sources, **kwargs):
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
        # ヘルパー関数を使用してソーステキストを整形
        sources_text = self._format_sources_text(sources)

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.invoke({
            "question": question,
            "answer": answer,
            "sources": sources_text
        })
        return response["text"]

    def _generate_local_approximation_explanation(self, question, answer, sources, **kwargs):
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
        # ヘルパー関数を使用してソーステキストを整形
        sources_text = self._format_sources_text(sources)

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.invoke({
            "question": question,
            "answer": answer,
            "sources": sources_text
        })
        return response["text"]

    def _generate_multi_faceted_explanation(self, question, answer, sources, **kwargs):
        """多角的説明を生成"""
        # 修正されたメソッドを使って説明を生成
        primary = self._generate_process_tracking_explanation(question, answer, sources, **kwargs)
        secondary = self._generate_feature_importance_explanation(question, answer, sources, **kwargs)

        prompt = PromptTemplate(
            input_variables=["primary", "secondary", "question", "answer"],
            template="""
            あなたは多角的AI説明の専門家です。
            以下の2つの異なるアプローチによる説明を統合して、より包括的な説明を生成してください。

            ユーザーの質問: {question}
            AIの回答: {answer}

            説明アプローチ1:
            {primary}

            説明アプローチ2:
            {secondary}

            これらの説明を統合し、以下の構造で包括的な説明を提供してください:

            ## 1. 概要
            回答の主要ポイントと根拠の簡潔な要約

            ## 2. 推論プロセス
            モデルがどのように考え、結論に至ったか

            ## 3. 重要な影響要因
            回答形成に最も影響を与えた情報と要素

            ## 4. 代替可能性
            考慮された他の可能性と、それらが選択されなかった理由

            ## 5. 信頼性評価
            この回答の確実性と制限事項
            """
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.invoke({
            "primary": primary,
            "secondary": secondary,
            "question": question,
            "answer": answer
        })
        return response["text"]


class SequentialSearchRetriever:
    """逐次検索リトリーバークラス"""
    
    def __init__(self, vector_db, llm, max_iterations=3):
        """
        初期化メソッド
        
        Parameters:
        -----------
        vector_db : VectorStore
            ベクトルデータベース
        llm : LLM
            言語モデル
        max_iterations : int
            最大検索反復回数
        """
        self.vector_db = vector_db
        self.llm = llm
        self.max_iterations = max_iterations
        
        self.query_refinement_prompt = PromptTemplate(
            input_variables=["question", "previous_results", "iteration"],
            template="""
            あなたは情報検索の専門家です。
            以下の質問に対する回答を見つけるために、検索クエリを最適化してください。

            元の質問: {question}

            これまでの検索結果:
            {previous_results}

            現在の反復回数: {iteration}/{max_iterations}

            次の検索クエリをより効果的にするための修正案を提供してください。
            以下の点を考慮してください:
            1. 前回の検索結果から不足している情報は何か
            2. より具体的な用語や同義語を使うべきか
            3. 検索範囲を広げるべきか、絞るべきか

            修正されたクエリ:
            """
        )
    
    def retrieve(self, question, k=3):
        """
        逐次的に検索を改善しながら関連ドキュメントを取得
        
        Parameters:
        -----------
        question : str
            ユーザーの質問
        k : int, optional
            各反復で取得するドキュメント数
        
        Returns:
        --------
        list
            関連ドキュメントのリスト
        dict
            検索プロセスの詳細情報
        """
        all_docs = []
        search_history = []
        current_query = question
        
        for i in range(self.max_iterations):
            # 現在のクエリで検索
            retriever = self.vector_db.as_retriever(search_kwargs={"k": k})
            docs = retriever.get_relevant_documents(current_query)
            
            # 重複を除外して結果を追加
            new_docs = []
            existing_contents = [doc.page_content for doc in all_docs]
            
            for doc in docs:
                if doc.page_content not in existing_contents:
                    new_docs.append(doc)
                    all_docs.append(doc)
            
            # 検索履歴を記録
            search_history.append({
                "iteration": i + 1,
                "query": current_query,
                "new_docs_count": len(new_docs),
                "new_docs": [{"content": doc.page_content[:100] + "...", "metadata": doc.metadata} for doc in new_docs]
            })
            
            # 十分な文書が見つかったか、最終反復に達した場合は終了
            if len(all_docs) >= k * 2 or i == self.max_iterations - 1:
                break
            
            # 前回の結果の要約を作成
            previous_results = ""
            for j, doc in enumerate(docs):
                previous_results += f"ドキュメント {j+1}: {doc.page_content[:200]}...\n\n"
            
            # クエリを改善
            chain = LLMChain(llm=self.llm, prompt=self.query_refinement_prompt)
            response = chain.invoke({
                "question": question,
                "previous_results": previous_results,
                "iteration": i + 1,
                "max_iterations": self.max_iterations
            })
            
            current_query = response["text"].strip()
        
        return all_docs, {
            "initial_query": question,
            "final_query": current_query,
            "search_history": search_history,
            "total_docs_retrieved": len(all_docs)
        }


class SelfVerificationSystem:
    """自己検証システムクラス"""
    
    def __init__(self, model_name="gpt-4"):
        """
        初期化メソッド
        
        Parameters:
        -----------
        model_name : str
            検証に使用するLLMのモデル名
        """
        self.model_name = model_name
        self.llm = ChatOpenAI(model_name=model_name)
        
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
    
    def verify_factual_accuracy(self, statement, sources):
        """
        文章の事実性を検証
        
        Parameters:
        -----------
        statement : str
            検証対象の文章
        sources : list
            情報源のリスト
        
        Returns:
        --------
        dict
            検証結果
        """
        sources_text = ""
        for i, source in enumerate(sources):
            if isinstance(source, dict):
                sources_text += f"[情報源 {i+1}] {source.get('filename', 'Unknown')}:\n"
                sources_text += f"{source.get('content', '')}\n\n"
            else:
                # Document オブジェクトの場合
                sources_text += f"[情報源 {i+1}] {source.metadata.get('filename', 'Unknown')}:\n"
                sources_text += f"{source.page_content}\n\n"
        
        chain = LLMChain(llm=self.llm, prompt=self.factual_verification_prompt)
        
        response = chain.invoke({
            "statement": statement,
            "sources": sources_text
        })
        
        try:
            # 文字列からJSONを抽出
            json_str = response["text"]
            # {}で囲まれた部分を抽出
            if "{" in json_str and "}" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]
            
            result = json.loads(json_str)
            return result
        except Exception as e:
            print(f"事実検証結果のパース中にエラーが発生しました: {e}")
            print(f"元のレスポンス: {response}")
            # フォールバック結果
            return {
                "matches_sources": None,
                "has_omissions": None,
                "is_misleading": None,
                "has_unsupported_inferences": None,
                "accuracy_rating": "検証失敗",
                "explanation": f"検証結果のパース中にエラーが発生しました: {e}",
                "corrected_statement": statement
            }
    
    def verify_logical_coherence(self, explanation):
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
        chain = LLMChain(llm=self.llm, prompt=self.logical_verification_prompt)
        
        response = chain.invoke({
            "explanation": explanation
        })
        
        try:
            # 文字列からJSONを抽出
            json_str = response["text"]
            # {}で囲まれた部分を抽出
            if "{" in json_str and "}" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]
            
            result = json.loads(json_str)
            return result
        except Exception as e:
            print(f"論理検証結果のパース中にエラーが発生しました: {e}")
            print(f"元のレスポンス: {response}")
            # フォールバック結果
            return {
                "logical_connection": None,
                "logical_leaps": None,
                "contradictions": None,
                "circular_reasoning": None,
                "causation_correlation_distinction": None,
                "coherence_rating": "検証失敗",
                "explanation": f"検証結果のパース中にエラーが発生しました: {e}",
                "improvement_suggestions": "検証できませんでした。"
            }
    
    def complete_verification(self, statement, explanation, sources):
        """
        総合的な検証を実行
        
        Parameters:
        -----------
        statement : str
            検証対象の文章
        explanation : str
            検証対象の説明
        sources : list
            情報源のリスト
        
        Returns:
        --------
        dict
            検証結果
        """
        factual_result = self.verify_factual_accuracy(statement, sources)
        coherence_result = self.verify_logical_coherence(explanation)
        
        # 総合評価
        if factual_result["accuracy_rating"] in ["完全に正確", "ほぼ正確"] and \
           coherence_result["coherence_rating"] in ["非常に高い", "高い"]:
            verification_status = "検証済み"
            confidence = "高"
        elif factual_result["accuracy_rating"] in ["不正確"] or \
             coherence_result["coherence_rating"] in ["非常に低い"]:
            verification_status = "不正確"
            confidence = "高"
        else:
            verification_status = "部分的に検証"
            confidence = "中"
        
        return {
            "verification_status": verification_status,
            "confidence": confidence,
            "factual_verification": factual_result,
            "logical_verification": coherence_result,
            "overall_assessment": f"事実性: {factual_result['accuracy_rating']}, 論理性: {coherence_result['coherence_rating']}",
            "timestamp": datetime.now().isoformat()
        }


class EnhancedAIExplanationSystem:
    """強化されたAI説明システムのメインクラス"""

    def __init__(self, doc_directory="knowledge_base", model_name="gpt-4", chunk_size=1000, chunk_overlap=100, db_type="chroma"):
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
        self.doc_directory = doc_directory
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db_type = db_type

        # サンプル文書を追加
        self._initialize_sample_docs()

        # ナレッジベースの構築
        self.documents = self._load_documents()
        self.vector_db = self._create_vector_db()

        # QAチェーンの構築
        self.qa_chain = self._create_qa_chain()

        # XAI手法セレクタの初期化
        self.xai_selector = XAIMethodSelector(model_name=model_name)
        
        # 逐次検索リトリーバーの初期化
        self.sequential_retriever = SequentialSearchRetriever(
            vector_db=self.vector_db,
            llm=ChatOpenAI(model_name=model_name)
        )
        
        # 自己検証システムの初期化
        self.verification_system = SelfVerificationSystem(model_name=model_name)
        
        # 評価指標管理の初期化
        self.evaluation_metrics = EvaluationMetrics()

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
        
        if self.db_type.lower() == "faiss":
            vector_db = FAISS.from_documents(texts, embeddings)
        else:
            # デフォルトはChroma
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
            return {"answer": "知識ベースが構築されていないため、回答できません。", "sources": [], "search_info": None}
        
        # 開始時間を記録
        start_time = time.time()
        
        # 逐次検索を使用して関連ドキュメントを取得
        docs, search_info = self.sequential_retriever.retrieve(question)
        
        # 回答の生成
        result = self.qa_chain.invoke({"query": question})
        answer = result["result"]
        
        # 処理時間を計算
        processing_time = time.time() - start_time
        
        # 回答の検証
        verification_result = self.verification_system.verify_factual_accuracy(answer, docs[:3])
        
        # 関連ドキュメントの整形
        source_list = [
            {
                "filename": doc.metadata.get("filename", "Unknown"),
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            for doc in docs[:5]  # 最も関連性の高い5つのドキュメントのみを含める
        ]
        
        return {
            "answer": answer,
            "sources": source_list,
            "search_info": search_info,
            "verification": verification_result,
            "processing_time": processing_time
        }
    
    def explain_answer(self, question: str, answer: str, domain: str = "一般") -> Dict[str, Any]:
        """回答の説明を生成"""
        if not self.vector_db:
            return {
                "explanation": "知識ベースが構築されていないため、説明を生成できません。",
                "method": None,
                "verification": None,
                "processing_time": 0
            }
        
        # 開始時間を記録
        start_time = time.time()
        
        # 最適なXAI手法を選択
        method_selection = self.xai_selector.select_methods(question, answer, domain)
        primary_method = method_selection.get("primary_method", "多角的説明")
        
        # 回答ベースの検索と関連ドキュメントの取得
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.get_relevant_documents(question + " " + answer)
        
        # 選択された手法で説明を生成
        explanation = self.xai_selector.generate_explanation(
            primary_method, 
            question, 
            answer, 
            relevant_docs
        )
        
        # 説明の論理的整合性を検証
        verification_result = self.verification_system.verify_logical_coherence(explanation)
        
        # 処理時間を計算
        processing_time = time.time() - start_time
        
        # 評価指標を記録
        metrics = {
            "coherence": 1.0 if verification_result.get("coherence_rating") == "非常に高い" else
                        0.8 if verification_result.get("coherence_rating") == "高い" else
                        0.6 if verification_result.get("coherence_rating") == "中程度" else
                        0.4 if verification_result.get("coherence_rating") == "低い" else
                        0.2 if verification_result.get("coherence_rating") == "非常に低い" else 0.5,
            "relevance": 0.8,  # デフォルト値
            "completeness": 0.8,  # デフォルト値
            "factual_accuracy": 0.8,  # デフォルト値
            "confidence_score": 0.8,  # デフォルト値
            "processing_time": processing_time
        }
        
        self.evaluation_metrics.record_metrics(
            question, 
            answer, 
            explanation, 
            primary_method, 
            metrics
        )
        
        return {
            "explanation": explanation,
            "method": primary_method,
            "method_selection": method_selection,
            "verification": verification_result,
            "processing_time": processing_time
        }
    
    def chat_and_explain(self, question: str, domain: str = "一般") -> Dict[str, Any]:
        """チャットと説明を同時に行う"""
        # 回答を取得
        answer_result = self.get_answer(question)
        
        # 説明を生成
        explanation_result = self.explain_answer(question, answer_result["answer"], domain)
        
        # 最終的な検証を実施
        verification_result = self.verification_system.complete_verification(
            answer_result["answer"],
            explanation_result["explanation"],
            answer_result["sources"]
        )
        
        return {
            "answer": answer_result["answer"],
            "explanation": explanation_result["explanation"],
            "sources": answer_result["sources"],
            "method": explanation_result["method"],
            "verification": verification_result,
            "search_info": answer_result.get("search_info"),
            "processing_time": {
                "answer": answer_result.get("processing_time", 0),
                "explanation": explanation_result.get("processing_time", 0),
                "total": answer_result.get("processing_time", 0) + explanation_result.get("processing_time", 0)
            }
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
            
            # 逐次検索リトリーバーを更新
            self.sequential_retriever = SequentialSearchRetriever(
                vector_db=self.vector_db,
                llm=ChatOpenAI(model_name=self.model_name)
            )
            
            return True
        except Exception as e:
            print(f"文書の追加中にエラーが発生しました: {e}")
            return False
    
    def evaluate_system_performance(self, days=30):
        """システムのパフォーマンスを評価"""
        return self.evaluation_metrics.get_metrics_summary(days=days)
    
    def generate_evaluation_report(self):
        """評価レポートを生成"""
        return self.evaluation_metrics.generate_evaluation_report()
    
    def record_improvement(self, description, baseline_score, new_score):
        """システム改善を記録"""
        return self.evaluation_metrics.record_improvement(description, baseline_score, new_score)


def create_gradio_interface():
    """Gradioインターフェースの作成"""
    system = EnhancedAIExplanationSystem()
    
    with gr.Blocks(title="Enhanced AIME-based ChatGPT Explanation System") as demo:
        gr.Markdown("# Enhanced AIME-based ChatGPT Explanation System")
        gr.Markdown("このシステムは、ChatGPTの回答とその根拠を説明します。アトリビュート強化と多角的XAI手法を適用しています。")
        
        with gr.Tab("チャットと説明"):
            with gr.Row():
                with gr.Column(scale=3):
                    question_input = gr.Textbox(
                        label="質問", 
                        placeholder="ここに質問を入力してください..."
                    )
                    domain_input = gr.Dropdown(
                        label="ドメイン",
                        choices=["一般", "技術", "医療", "法律", "ビジネス", "教育", "科学"],
                        value="一般"
                    )
                    submit_btn = gr.Button("送信")
                
            with gr.Row():
                with gr.Column(scale=2):
                    answer_output = gr.Textbox(label="回答")
                with gr.Column(scale=3):
                    explanation_output = gr.Textbox(label="回答の説明", lines=10)
            
            with gr.Row():
                with gr.Column():
                    sources_output = gr.JSON(label="参照ソース")
                with gr.Column():
                    verification_output = gr.JSON(label="検証結果")
                with gr.Column():
                    method_output = gr.Textbox(label="使用されたXAI手法")
            
            search_info_output = gr.JSON(label="検索情報", visible=False)
            
            def handle_chat_and_explain(question, domain):
                if not question.strip():
                    return "質問を入力してください。", "質問を入力してください。", [], {}, "N/A", {}
                
                result = system.chat_and_explain(question, domain)
                
                return (
                    result["answer"], 
                    result["explanation"], 
                    result["sources"], 
                    result["verification"], 
                    result["method"],
                    result.get("search_info", {})
                )
            
            submit_btn.click(
                handle_chat_and_explain,
                inputs=[question_input, domain_input],
                outputs=[answer_output, explanation_output, sources_output, verification_output, method_output, search_info_output]
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
                    existing_domain = gr.Dropdown(
                        label="ドメイン",
                        choices=["一般", "技術", "医療", "法律", "ビジネス", "教育", "科学"],
                        value="一般"
                    )
                    explain_btn = gr.Button("説明を生成")
            
            with gr.Row():
                with gr.Column():
                    explanation_only_output = gr.Textbox(label="説明", lines=10)
                with gr.Column():
                    method_only_output = gr.Textbox(label="使用されたXAI手法")
                    verification_only_output = gr.JSON(label="検証結果")
            
            def handle_explain_only(question, answer, domain):
                if not question.strip() or not answer.strip():
                    return "質問と回答を入力してください。", "N/A", {}
                
                result = system.explain_answer(question, answer, domain)
                
                return (
                    result["explanation"],
                    result["method"],
                    result["verification"]
                )
            
            explain_btn.click(
                handle_explain_only,
                inputs=[existing_question, existing_answer, existing_domain],
                outputs=[explanation_only_output, method_only_output, verification_only_output]
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
            
        with gr.Tab("パフォーマンス評価"):
            with gr.Row():
                with gr.Column():
                    time_period = gr.Dropdown(
                        label="評価期間",
                        choices=["7日間", "30日間", "90日間", "全期間"],
                        value="30日間"
                    )
                    evaluate_btn = gr.Button("評価を実行")
            
            with gr.Row():
                with gr.Column():
                    metrics_output = gr.JSON(label="性能指標")
                with gr.Column():
                    report_output = gr.JSON(label="詳細レポート")
            
            def handle_evaluation(period):
                days = 7 if period == "7日間" else 30 if period == "30日間" else 90 if period == "90日間" else 365
                
                metrics = system.evaluate_system_performance(days=days)
                report = system.generate_evaluation_report()
                
                return metrics, report
            
            evaluate_btn.click(
                handle_evaluation,
                inputs=[time_period],
                outputs=[metrics_output, report_output]
            )
            
        with gr.Tab("改善履歴"):
            with gr.Row():
                with gr.Column():
                    improvement_desc = gr.Textbox(
                        label="改善内容の説明", 
                        placeholder="システムに適用した改善内容を説明してください..."
                    )
                    baseline_score = gr.Number(label="ベースラインスコア", value=0.75)
                    new_score = gr.Number(label="新スコア", value=0.85)
                    record_btn = gr.Button("改善を記録")
            
            with gr.Row():
                record_result = gr.Textbox(label="記録結果")
                
            def handle_record_improvement(desc, baseline, new_sc):
                if not desc.strip():
                    return "エラー: 改善内容を入力してください。"
                
                system.record_improvement(desc, baseline, new_sc)
                return f"改善 '{desc}' が正常に記録されました。"
            
            record_btn.click(
                handle_record_improvement,
                inputs=[improvement_desc, baseline_score, new_score],
                outputs=[record_result]
            )
            
            with gr.Row():
                refresh_history_btn = gr.Button("履歴を更新")
                history_output = gr.DataFrame(label="改善履歴")
            
            def get_improvement_history():
                return system.evaluation_metrics.get_improvement_history()
            
            refresh_history_btn.click(
                get_improvement_history,
                inputs=[],
                outputs=[history_output]
            )

    return demo

def main():
    """メイン関数"""
    print("Enhanced AIME-based Explanation System for ChatGPT Outputs を起動中...")
    demo = create_gradio_interface()
    demo.launch(share=True)

if __name__ == "__main__":
    main()