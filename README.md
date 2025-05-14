# 概要
AIME-based Explanation Systemは、ChatGPTなどの大規模言語モデル(LLM)の出力に対して、その根拠や判断プロセスを説明するための包括的なフレームワークです。最新のXAI（説明可能なAI）手法と信頼性強化メカニズムを組み合わせ、AIの判断に対する透明性と理解性を高めます。


# 主な特徴

1. アトリビュート強化機能
- 自己検証システムによる回答の事実性と論理整合性の評価
- 逐次検索による情報収集の質と幅の向上
- 複数情報源の相互検証と確信度の明示的表示


2. 多角的XAI手法
- 質問と回答の特性に応じた最適なXAI手法の自動選択
- 8種類の説明手法（特徴重要度分析、反事実説明、プロセス追跡など）
- 複数手法の組み合わせによる包括的な説明生成


3. 拡張性と性能
- 複数のベクトルデータベース（ChromaとFAISS）のサポート
- モジュール化された設計で新しいXAI手法や検証機能を容易に追加可能
- 堅牢なエラーハンドリングと代替処理パス


4. 説明品質の評価と改善
- 一貫性、関連性、完全性、事実的正確性などの指標による評価
- SQLiteデータベースを使用した評価データの記録と分析
- 時系列トレンド分析による継続的改善の支援

# 機能詳細
1. ナレッジベース管理
システムは指定されたディレクトリからPDFやテキストファイルを読み込み、ナレッジベースを構築します。新しい文書はGradioインターフェースまたはAPIを通じて追加できます。

```python
# 新しい文書を追加
system.add_document_from_text("example.txt", "文書の内容をここに記述")
```

2. XAI手法の選択
システムは質問と回答の性質に基づいて最適なXAI手法を自動選択します。特定の手法を指定することも可能です。

```python
# 特定のXAI手法を使用して説明を生成
explanation = system.xai_selector.generate_explanation(
    "特徴重要度分析", 
    "質問", 
    "回答", 
    sources
)
```

3. 評価と改善
システムは説明の品質を継続的に評価し、改善の機会を特定します。

```python
# パフォーマンス評価を実行
metrics = system.evaluate_system_performance(days=30)

# 詳細なレポートを生成
report = system.generate_evaluation_report()

# 改善を記録
system.record_improvement(
    "XAI手法の選択ロジックを改善",
    baseline_score=0.75,
    new_score=0.85
)
```

4. アーキテクチャ
システムは以下の主要コンポーネントで構成されています：

- **EnhancedAIExplanationSystem**: メインシステムクラス
- **XAIMethodSelector**: 最適なXAI手法を選択するクラス
- **SequentialSearchRetriever**: 逐次検索を行うクラス
- **SelfVerificationSystem**: 回答と説明を検証するクラス
- **EvaluationMetrics**: 説明品質を評価・記録するクラス




