# type: ignore
import re
import os
import json
import time
import traceback
import gradio as gr
from typing import Dict, Any, Optional, Tuple, List, Generator

# 自作モジュールの追加
from system import EnhancedAIExplanationSystem


def create_gradio_interface(system_instance: EnhancedAIExplanationSystem) -> gr.Blocks:
    """Gradioインターフェースを作成する"""

    # 思考プロセスを抽出する関数
    def extract_thinking_process(explanation: str) -> tuple:
        """思考プロセスと最終回答を抽出 - マークダウン見出しにも対応"""
        # まず<think>タグがある場合はそこから抽出
        thinking_match = re.search(r'<think>(.*?)</think>', explanation, re.DOTALL)
        
        if thinking_match:
            thinking_process = thinking_match.group(1).strip()
            print(f"<think>タグから思考プロセスを抽出しました: {thinking_process[:50]}...")
            
            # <think>タグを除去して最終回答を得る
            final_answer = re.sub(r'<think>.*?</think>', '', explanation, flags=re.DOTALL).strip()
            # "最終回答:"のプレフィックスがあれば除去
            final_answer = re.sub(r'^最終回答:\s*', '', final_answer).strip()
            
            return thinking_process, final_answer
        
        # タグがない場合はマークダウン見出しから抽出を試みる
        # "## 思考プロセスの追跡" や "# 思考プロセスの詳細" などのパターンを検出
        markdown_section_match = re.search(r'(#+\s*思考プロセス[^#]*?)(?=#+|$)', explanation, re.DOTALL | re.IGNORECASE)
        
        if markdown_section_match:
            thinking_process = markdown_section_match.group(1).strip()
            print(f"マークダウン見出しから思考プロセスを抽出しました: {thinking_process[:50]}...")
            
            # 思考プロセス部分を除いた残りを最終回答とする
            final_answer = re.sub(re.escape(thinking_process), '', explanation, flags=re.DOTALL).strip()
            return thinking_process, final_answer
        
        # 両方の方法で見つからない場合は、元の説明全体を思考プロセスとして扱う
        print("思考プロセスのタグもマークダウン見出しも見つかりませんでした。元の説明全体を使用します。")
        return explanation, explanation

    # チャットと説明を処理する関数
    def process_chat_explain(question: str, domain: str = "一般") -> Tuple[str, str, str, str, str, str, str]:
        """チャットと説明を処理する関数"""
        if not question:
            return ("質問を入力してください。", "", "N/A", "[]", "{}", "{}", "質問を入力してください。")

        try:
            # チャットと説明を実行
            result = system_instance.chat_and_explain(question, domain)
            
            # 結果から各種情報を取得
            answer = result.get("answer", "エラー: 回答なし")
            explanation = result.get("explanation", "エラー: 説明なし")
            method = result.get("method", "N/A")
            
            # JSON形式のデータを文字列に変換
            sources = json.dumps(result.get("sources", []), ensure_ascii=False, indent=2)
            verification = json.dumps(result.get("verification", {}), ensure_ascii=False, indent=2)
            search_info = json.dumps(result.get("search_info", {}), ensure_ascii=False, indent=2)
            
            # 処理完了メッセージ
            status_message = "回答と説明の生成が完了しました。"
            
            # 処理時間があれば追加
            processing_time = result.get("processing_time", {})
            if processing_time:
                total_time = processing_time.get("total", 0)
                if total_time > 0:
                    status_message += f" 処理時間: {total_time:.2f}秒"
            
            # デバッグ出力
            print(f"回答: {answer[:100]}...")
            print(f"説明: {explanation[:100]}...")
            print(f"手法: {method}")
            print(f"ソース長さ: {len(sources)}")
            print(f"検証結果長さ: {len(verification)}")
            print(f"ステータス: {status_message}")
            
            # 7つの値をタプルとして返す
            return (
                answer,           # answer_output
                explanation,      # explanation_output
                method,           # method_output
                sources,          # sources_output (文字列化JSON)
                verification,     # verification_output (文字列化JSON)
                search_info,      # search_info_output (文字列化JSON)
                status_message    # status_output
            )
        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = f"エラーが発生しました: {e}\n{tb_str}"
            print(error_message)
            
            # エラー時も7つの値を返す
            return (error_message, error_message, "N/A", "[]", "{}", "{}", f"エラーが発生しました: {e}")

    # 既存の回答に対する説明を生成する関数
    def generate_explanation(question: str, answer: str, domain: str = "一般") -> Generator[Tuple[str, str, str, str], None, None]:
        """既存の回答に対する説明を生成する"""
        if not question or not answer:
            # 4つの値を返す
            return ("質問と回答の両方を入力してください。", "N/A", "{}", "質問と回答の両方を入力してください。")

        try:
            # 説明を生成
            result = system_instance.explain_answer(question, answer, domain)
            explanation = result.get("explanation", "エラー: 説明なし")
            method = result.get("method", "N/A")
            
            # JSON形式のデータを文字列に変換
            verification = json.dumps(result.get("verification", {}), ensure_ascii=False, indent=2)
            
            # 処理完了メッセージ
            status_message = "説明の生成が完了しました。"
            
            # 処理時間があれば追加
            processing_time = result.get("processing_time", 0)
            if processing_time > 0:
                status_message += f" 処理時間: {processing_time:.2f}秒"
            
            # 4つの値をタプルとして返す
            return (
                explanation,      # existing_explanation
                method,           # existing_method
                verification,     # existing_verification (文字列化JSON)
                status_message    # explain_status_output
            )
        except Exception as e:
            error_message = f"エラーが発生しました: {e}"
            print(error_message)
            # エラー時も4つの値を返す
            return (error_message, "N/A", "{}", f"エラーが発生しました: {e}")

    # 文書追加の処理関数
    def add_document(filename: str, content: str) -> Dict[str, str]:
        """新しい文書をナレッジベースに追加する"""
        if not filename or not content:
            return {"result": "ファイル名と内容を両方入力してください。"}
        
        # 拡張子がない場合は.txtを追加
        if not os.path.splitext(filename)[1]:
            filename += ".txt"
        
        success = system_instance.add_document_from_text(filename, content)
        if success:
            return {"result": f"文書 '{filename}' が正常に追加されました。"}
        else:
            return {"result": f"文書 '{filename}' の追加に失敗しました。"}

    # Gradioインターフェースの作成
    with gr.Blocks(title="AIME-based Explanation System for ChatGPT Outputs", 
                theme=gr.themes.Soft()) as demo:
        gr.Markdown("# AIME-based Explanation System for ChatGPT Outputs")
        gr.Markdown("AI回答の根拠と思考プロセスを可視化するシステム")
        
        with gr.Tabs():
            # チャットと説明のタブ
            with gr.TabItem("チャットと説明"):
                with gr.Row():
                    with gr.Column():
                        question_input = gr.Textbox(label="質問", placeholder="ここに質問を入力してください", lines=3)
                        domain_input = gr.Dropdown(
                            choices=["一般", "医療", "法律", "科学", "技術", "ビジネス", "教育"], 
                            label="ドメイン", 
                            value="一般"
                        )
                        submit_btn = gr.Button("送信", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        answer_output = gr.Textbox(label="回答", lines=8, interactive=False)
                    with gr.Column():
                        explanation_output = gr.Textbox(label="回答の説明", lines=8, interactive=False)
                
                method_output = gr.Textbox(label="使用されたXAI手法", interactive=False)
                
                # タブで詳細情報を表示
                with gr.Tabs():
                    with gr.TabItem("参照ソース"):
                        sources_output = gr.Code(language="json", label="", show_label=False)
                    
                    with gr.TabItem("検証結果"):
                        verification_output = gr.Code(language="json", label="", show_label=False)
                    
                    with gr.TabItem("検索プロセス情報"):
                        search_info_output = gr.Code(language="json", label="", show_label=False)
                
                # ステータス表示 - 常に表示
                status_output = gr.Textbox(label="ステータス", interactive=False)
            
            # 既存回答の説明タブ
            with gr.TabItem("既存回答の説明"):
                with gr.Row():
                    with gr.Column():
                        existing_question = gr.Textbox(label="質問", placeholder="ここに質問を入力してください", lines=3)
                        existing_answer = gr.Textbox(label="ChatGPTの回答", placeholder="ChatGPTの回答を入力してください", lines=5)
                        existing_domain = gr.Dropdown(
                            choices=["一般", "医療", "法律", "科学", "技術", "ビジネス", "教育"], 
                            label="ドメイン", 
                            value="一般"
                        )
                        explain_btn = gr.Button("説明を生成", variant="primary")
                
                existing_explanation = gr.Textbox(label="説明", lines=8, interactive=False)
                existing_method = gr.Textbox(label="使用されたXAI手法", interactive=False)
                
                with gr.Tabs():
                    with gr.TabItem("検証結果"):
                        existing_verification = gr.Code(language="json", label="", show_label=False)
                
                # ステータス表示
                explain_status_output = gr.Textbox(label="ステータス", interactive=False)
            
            # ナレッジベース管理タブ
            with gr.TabItem("ナレッジベース管理"):
                with gr.Row():
                    with gr.Column():
                        doc_filename = gr.Textbox(label="ファイル名", placeholder="filename.txt")
                        doc_content = gr.Textbox(label="文書内容", placeholder="ここに文書の内容を入力してください", lines=10)
                        add_doc_btn = gr.Button("文書を追加", variant="primary")
                        add_result = gr.Textbox(label="結果", interactive=False)
        
        # イベントの設定
        submit_btn.click(
            fn=process_chat_explain,
            inputs=[question_input, domain_input],
            outputs=[
                answer_output, explanation_output, method_output,
                sources_output, verification_output, search_info_output,
                status_output
            ],
            api_name="chat_and_explain"
        ).then(
            fn=lambda: None,
            inputs=None,
            outputs=None,
            js="() => { console.log('処理が完了しました'); }"
        )
        
        explain_btn.click(
            fn=generate_explanation,
            inputs=[existing_question, existing_answer, existing_domain],
            outputs=[
                existing_explanation, existing_method, existing_verification,
                explain_status_output
            ]
        )
        
        add_doc_btn.click(
            fn=add_document,
            inputs=[doc_filename, doc_content],
            outputs=[add_result]
        )
        
    return demo
