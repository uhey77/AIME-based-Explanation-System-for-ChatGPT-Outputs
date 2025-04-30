# ui.py
import gradio as gr
from typing import Dict, Any, Tuple, List  # 型ヒントのために追加

# systemモジュールからメインシステムクラスをインポート
from system import EnhancedAIExplanationSystem


def create_gradio_interface(system: EnhancedAIExplanationSystem):
    """
    Gradioインターフェースを作成する

    Parameters:
    -----------
    system : EnhancedAIExplanationSystem
        使用する説明システムのインスタンス
    """

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
                    submit_btn = gr.Button("送信", variant="primary")  # ボタンを強調

            with gr.Accordion("詳細結果", open=False):  # 結果をアコーディオンに格納
                with gr.Row():
                    with gr.Column(scale=2):
                        answer_output = gr.Textbox(label="回答", lines=5)  # 回答欄を少し広げる
                    with gr.Column(scale=3):
                        explanation_output = gr.Textbox(label="回答の説明", lines=10)

                with gr.Row():
                    with gr.Column(scale=1):
                        method_output = gr.Textbox(label="使用されたXAI手法")
                    with gr.Column(scale=2):
                        sources_output = gr.JSON(label="参照ソース")
                    with gr.Column(scale=2):
                        verification_output = gr.JSON(label="検証結果")

                search_info_output = gr.JSON(label="検索プロセス情報")  # 検索情報も表示

            # 処理中を示すインジケータ用
            status_output = gr.Textbox(label="ステータス", interactive=False, visible=False)

            # handle_chat_and_explain が返すタプルの型ヒントを修正
            def handle_chat_and_explain(question: str, domain: str) -> Tuple[str, str, str, List[Dict[str, str]], Dict[str, Any], Dict[str, Any], Any]: # 最後のAnyはgr.update用
                if not question.strip():
                    # 7つの値を返すように修正
                    return "質問を入力してください。", "", "N/A", [], {}, {}, gr.update(visible=True, value="質問が空です。入力してください。")

                # 処理開始を表示 (7つの値をyield)
                yield "処理中...", "", "処理中...", [], {}, {}, gr.update(visible=True, value="回答と説明を生成しています...")

                try:
                    result = system.chat_and_explain(question, domain)
                    answer = result.get("answer", "エラー: 回答なし")
                    explanation = result.get("explanation", "エラー: 説明なし")
                    method = result.get("method", "N/A")
                    sources = result.get("sources", [])
                    verification = result.get("verification", {})
                    search_info = result.get("search_info", {})

                    # 処理完了を表示 (7つの値をyield)
                    yield answer, explanation, method, sources, verification, search_info, gr.update(visible=True, value="処理完了")

                except Exception as e:
                    error_message = f"エラーが発生しました: {e}"
                    print(error_message)
                    # エラー発生時の表示 (7つの値をyield)
                    yield error_message, error_message, "N/A", [], {}, {}, gr.update(visible=True, value=f"エラー: {e}")

            submit_btn.click(
                handle_chat_and_explain,
                inputs=[question_input, domain_input],
                # outputsの数（7つ）とyieldの数（7つ）を合わせる
                outputs=[answer_output, explanation_output, method_output, sources_output, verification_output, search_info_output, status_output]
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
                        placeholder="ここにChatGPTの回答を入力してください...",
                        lines=5  # 入力欄を少し広げる
                    )
                    existing_domain = gr.Dropdown(
                        label="ドメイン",
                        choices=["一般", "技術", "医療", "法律", "ビジネス", "教育", "科学"],
                        value="一般"
                    )
                    explain_btn = gr.Button("説明を生成", variant="primary")

            with gr.Accordion("説明結果", open=False):
                with gr.Row():
                    explanation_only_output = gr.Textbox(label="説明", lines=10)
                with gr.Row():
                    method_only_output = gr.Textbox(label="使用されたXAI手法")
                    verification_only_output = gr.JSON(label="検証結果 (論理整合性)")

            explain_status_output = gr.Textbox(label="ステータス", interactive=False, visible=False)

            # handle_explain_only が返すタプルの型ヒントを修正
            def handle_explain_only(question: str, answer: str, domain: str) -> Tuple[str, str, Dict[str, Any], Any]:  # 最後のAnyはgr.update用
                if not question.strip() or not answer.strip():
                    # 4つの値を返すように修正
                    return "質問と回答を入力してください。", "N/A", {}, gr.update(visible=True, value="質問と回答の両方を入力してください。")

                # 4つの値をyield
                yield "処理中...", "処理中...", {}, gr.update(visible=True, value="説明を生成しています...")

                try:
                    result = system.explain_answer(question, answer, domain)
                    explanation = result.get("explanation", "エラー: 説明なし")
                    method = result.get("method", "N/A")
                    # explain_answerは論理整合性の検証結果を返す
                    verification = result.get("verification", {})
                    # 4つの値をyield
                    yield explanation, method, verification, gr.update(visible=True, value="処理完了")
                except Exception as e:
                    error_message = f"エラーが発生しました: {e}"
                    print(error_message)
                    # 4つの値をyield
                    yield error_message, "N/A", {}, gr.update(visible=True, value=f"エラー: {e}")

            explain_btn.click(
                handle_explain_only,
                inputs=[existing_question, existing_answer, existing_domain],
                # outputsの数（4つ）とyieldの数（4つ）を合わせる
                outputs=[explanation_only_output, method_only_output, verification_only_output, explain_status_output]
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
                    add_doc_btn = gr.Button("文書を追加", variant="primary")

            add_result = gr.Textbox(label="追加結果", interactive=False)

            def handle_add_document(name: str, content: str) -> str:
                if not name.strip():
                    return "エラー: ファイル名を入力してください。"

                if not content.strip():
                    return "エラー: 文書内容を入力してください。"

                # 拡張子の確認と追加 (.txt以外も許可する可能性も考慮)
                if '.' not in name:
                    name = name + '.txt'  # 拡張子がない場合 .txt を付与

                # 処理中メッセージ（Gradioはyield非対応なので同期的に実行される）
                # add_result.update("文書を追加しています...") # クリックハンドラ内での .update は推奨されない場合がある

                try:
                    # ボタンクリックで直接結果を返す
                    add_result_value = "文書を追加しています..."  # 一時的な表示
                    # 非同期処理ではないため、すぐに system の関数が呼ばれる
                    success = system.add_document_from_text(name, content)
                    if success:
                        add_result_value = f"文書 '{name}' が正常に追加され、システムが更新されました。"
                    else:
                        # system側のログで詳細が出力されるはず
                        add_result_value = f"文書 '{name}' の追加またはシステム更新中にエラーが発生しました。詳細はログを確認してください。"
                except Exception as e:
                    add_result_value = f"文書追加処理中に予期せぬエラーが発生しました: {e}"
                return add_result_value  # 最終的な結果を返す

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
                    evaluate_btn = gr.Button("評価を実行", variant="primary")

            with gr.Row():
                with gr.Column(scale=1):
                    metrics_output = gr.JSON(label="性能指標 (要約)")
                with gr.Column(scale=2):
                    report_output = gr.JSON(label="詳細レポート")

            eval_status_output = gr.Textbox(label="ステータス", interactive=False, visible=False)

            # handle_evaluation が返すタプルの型ヒントを修正
            def handle_evaluation(period: str) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:  # 最後のAnyはgr.update用
                # 3つの値をyield
                yield {}, {}, gr.update(visible=True, value="評価を実行中...")
                try:
                    days_map = {"7日間": 7, "30日間": 30, "90日間": 90}
                    # "全期間" は非常に大きな値やNoneなどで扱う (get_metrics_summaryが対応する場合)
                    # ここでは大きな日数（例：10年）を指定しておく
                    days = days_map.get(period, 3650)

                    metrics = system.evaluate_system_performance(days=days)
                    report = system.generate_evaluation_report()
                    # 3つの値をyield
                    yield metrics, report, gr.update(visible=True, value="評価完了")
                except Exception as e:
                    error_message = f"評価中にエラーが発生しました: {e}"
                    print(error_message)
                    # 3つの値をyield
                    yield {}, {}, gr.update(visible=True, value=f"エラー: {e}")

            evaluate_btn.click(
                handle_evaluation,
                inputs=[time_period],
                # outputsの数（3つ）とyieldの数（3つ）を合わせる
                outputs=[metrics_output, report_output, eval_status_output]
            )

        with gr.Tab("改善履歴"):
            with gr.Row():
                with gr.Column(scale=2):
                    improvement_desc = gr.Textbox(
                        label="改善内容の説明",
                        placeholder="システムに適用した改善内容を説明してください..."
                    )
                with gr.Column(scale=1):
                    baseline_score = gr.Number(label="ベースラインスコア", value=0.75, step=0.01)  # step追加
                    new_score = gr.Number(label="新スコア", value=0.85, step=0.01)  # step追加
                with gr.Column(scale=1):
                    record_btn = gr.Button("改善を記録", variant="primary")

            record_result = gr.Textbox(label="記録結果", interactive=False)

            def handle_record_improvement(desc: str, baseline: float, new_sc: float) -> str:
                if not desc.strip():
                    return "エラー: 改善内容を入力してください。"
                if baseline is None or new_sc is None:
                    return "エラー: ベースラインスコアと新スコアを入力してください。"

                try:
                    # system.record_improvement は内部でprintするので、ここでは完了メッセージのみ
                    system.record_improvement(desc, baseline, new_sc)
                    return f"改善 '{desc}' が正常に記録されました。"
                except Exception as e:
                    return f"改善記録中にエラーが発生しました: {e}"

            record_btn.click(
                handle_record_improvement,
                inputs=[improvement_desc, baseline_score, new_score],
                outputs=[record_result]
            )

            gr.Markdown("---")  # 区切り線

            with gr.Row():
                refresh_history_btn = gr.Button("改善履歴を表示/更新")
                history_output = gr.DataFrame(label="改善履歴", interactive=False)  # DataFrameを表示

            def get_improvement_history_df():
                # エラーハンドリングを追加
                try:
                    history_df = system.evaluation_metrics.get_improvement_history()
                    # DataFrameが空の場合も考慮
                    if history_df.empty:
                        # gr.Info は Blocks 内の関数でのみ利用可能。ここでは print する。
                        print("改善履歴はまだありません。")
                    return history_df
                except Exception as e:
                    print(f"改善履歴の取得中にエラー: {e}")
                    # gr.Error も Blocks 内関数でのみ利用可能
                    print(f"エラー: 改善履歴の取得中にエラーが発生しました: {e}")
                    # 空のDataFrameを返すか、エラーを示すDataFrameを返す
                    import pandas as pd
                    return pd.DataFrame({"エラー": [str(e)]})

            refresh_history_btn.click(
                get_improvement_history_df,
                inputs=[],
                outputs=[history_output]
            )
            # 初期表示のためにも呼び出す (Blocksのloadイベント)
            demo.load(get_improvement_history_df, None, history_output)

    return demo
