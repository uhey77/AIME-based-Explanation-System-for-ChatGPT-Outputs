# evaluation.py
import sqlite3
import pandas as pd
from datetime import datetime
import hashlib
from typing import Dict, Any

# configモジュールから設定値をインポート
import config


class EvaluationMetrics:
    """説明品質の評価指標を管理するクラス"""

    def __init__(self, db_path: str = config.EVALUATION_DB_PATH):
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
        try:
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
        except sqlite3.Error as e:
            print(f"データベーステーブル作成エラー: {e}")
        finally:
            if conn:
                conn.close()

    def record_metrics(self, question: str, answer: str, explanation: str, method: str, metrics: Dict[str, Any]):
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
        conn = None
        try:
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
                method, metrics.get('coherence'), metrics.get('relevance'),
                metrics.get('completeness'), metrics.get('factual_accuracy'),
                metrics.get('user_rating'), metrics.get('confidence_score'),
                metrics.get('processing_time')
            ))

            conn.commit()
        except sqlite3.Error as e:
            print(f"メトリクス記録エラー: {e}")
        finally:
            if conn:
                conn.close()

    def record_improvement(self, description: str, baseline_score: float, new_score: float):
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
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            timestamp = datetime.now().isoformat()

            cursor.execute('''
            INSERT INTO improvement_history (
                timestamp, change_description, baseline_avg_score, new_avg_score
            ) VALUES (?, ?, ?, ?)
            ''', (timestamp, description, baseline_score, new_score))

            conn.commit()
            print(f"改善履歴を記録しました: {description}")
        except sqlite3.Error as e:
            print(f"改善履歴記録エラー: {e}")
        finally:
            if conn:
                conn.close()

    def get_metrics_summary(self, days: int = 30, method: str = None) -> Dict[str, Any]:
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
        conn = None
        summary = {
            'coherence': 0, 'relevance': 0, 'completeness': 0,
            'factual_accuracy': 0, 'user_rating': 0, 'confidence_score': 0,
            'processing_time': 0, 'sample_count': 0
        }
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            time_limit = (datetime.now() - pd.Timedelta(days=days)).isoformat()

            base_query = '''
                SELECT
                    AVG(coherence), AVG(relevance), AVG(completeness),
                    AVG(factual_accuracy), AVG(user_rating), AVG(confidence_score),
                    AVG(processing_time), COUNT(*)
                FROM explanation_metrics
                WHERE timestamp > ?
            '''
            params = [time_limit]

            if method:
                base_query += " AND method = ?"
                params.append(method)

            cursor.execute(base_query, tuple(params))
            result = cursor.fetchone()

            if result and result[7] > 0:  # COUNT(*) > 0
                summary = {
                    'coherence': result[0] or 0,
                    'relevance': result[1] or 0,
                    'completeness': result[2] or 0,
                    'factual_accuracy': result[3] or 0,
                    'user_rating': result[4] or 0,
                    'confidence_score': result[5] or 0,
                    'processing_time': result[6] or 0,
                    'sample_count': result[7] or 0
                }
        except sqlite3.Error as e:
            print(f"メトリクス要約取得エラー: {e}")
        finally:
            if conn:
                conn.close()
        return summary

    def get_improvement_history(self) -> pd.DataFrame:
        """
        改善履歴を取得

        Returns:
        --------
        pd.DataFrame
            改善履歴のデータフレーム
        """
        conn = None
        history = pd.DataFrame()  # 空のDataFrameをデフォルト値とする
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM improvement_history ORDER BY timestamp DESC"
            history = pd.read_sql_query(query, conn)
        except sqlite3.Error as e:
            print(f"改善履歴取得エラー: {e}")
        except Exception as e:  # pd.read_sql_queryが他のエラーを出す可能性
            print(f"改善履歴の読み込み中に予期せぬエラーが発生しました: {e}")
        finally:
            if conn:
                conn.close()
        return history

    def generate_evaluation_report(self) -> Dict[str, Any]:
        """
        評価レポートを生成

        Returns:
        --------
        dict
            評価レポート
        """
        conn = None
        report = {
            'overall_metrics': {},
            'method_comparison': [],
            'time_trend': []
        }
        try:
            conn = sqlite3.connect(self.db_path)

            # 全体の傾向
            overall_query = '''
            SELECT
                AVG(coherence) as avg_coherence,
                AVG(relevance) as avg_relevance,
                AVG(completeness) as avg_completeness,
                AVG(factual_accuracy) as avg_factual_accuracy,
                AVG(user_rating) as avg_user_rating,
                AVG(confidence_score) as avg_confidence
            FROM explanation_metrics
            '''
            overall = pd.read_sql_query(overall_query, conn)
            if not overall.empty and overall.notna().all().all():  # データがあり、NaNでないことを確認
                report['overall_metrics'] = overall.to_dict('records')[0]

            # 方法別の比較
            methods_query = '''
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
            '''
            methods = pd.read_sql_query(methods_query, conn)
            report['method_comparison'] = methods.to_dict('records')

            # 時系列トレンド
            trend_query = '''
            SELECT
                DATE(timestamp) as date,
                AVG(coherence) as avg_coherence,
                AVG(relevance) as avg_relevance,
                AVG(completeness) as avg_completeness,
                AVG(user_rating) as avg_user_rating
            FROM explanation_metrics
            GROUP BY DATE(timestamp)
            ORDER BY date
            '''
            trend = pd.read_sql_query(trend_query, conn)
            report['time_trend'] = trend.to_dict('records')

        except sqlite3.Error as e:
            print(f"評価レポート生成エラー (SQLite): {e}")
        except Exception as e:
            print(f"評価レポート生成エラー: {e}")
        finally:
            if conn:
                conn.close()

        return report
