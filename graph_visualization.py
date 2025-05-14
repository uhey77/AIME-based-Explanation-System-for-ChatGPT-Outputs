import re
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import io
import base64
from PIL import Image
import random
import spacy
import os

# spaCyの英語モデルをロード (日本語の場合は 'ja_core_news_sm' などを使用)
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    try:
        # 日本語モデルを試す
        nlp = spacy.load('ja_core_news_sm')
    except OSError:
        print("spaCyモデルがインストールされていません。'python -m spacy download en_core_web_sm' または 'python -m spacy download ja_core_news_sm' を実行してください。")
        # フォールバックとして、シンプルな分割関数を使用
        nlp = None


class ExplanationGraphVisualizer:
    """AIの説明をグラフとして視覚化するクラス"""

    def __init__(self, color_palette: Optional[Dict[str, str]] = None):
        """
        初期化メソッド

        Parameters:
        -----------
        color_palette : Optional[Dict[str, str]]
            ノードタイプごとの色を指定する辞書
        """
        # デフォルトの色パレット
        self.color_palette = color_palette or {
            'concept': '#3498db',  # 青
            'fact': '#2ecc71',     # 緑
            'inference': '#e74c3c', # 赤
            'source': '#9b59b6',   # 紫
            'question': '#f39c12', # オレンジ
            'answer': '#f1c40f',   # 黄
            'step': '#1abc9c',     # ターコイズ
            'default': '#95a5a6'   # グレー
        }

    # graph_visualization.py の _extract_concepts_from_text メソッドを修正

    def _extract_concepts_from_text(self, text: str) -> List[Tuple[str, str]]:
        """
        テキストから重要な概念やキーフレーズを抽出する（日本語対応強化版）
        
        Parameters:
        -----------
        text : str
            抽出対象のテキスト
            
        Returns:
        --------
        List[Tuple[str, str]]
            (概念テキスト, タイプ) のタプルのリスト
        """
        concepts = []
        
        # テキストが空の場合は早期リターン
        if not text or len(text.strip()) == 0:
            return []
        
        # 日本語用の特殊処理
        # 日本語の文を簡易的に分割（句点や改行で区切る）
        sentences = re.split(r'[。\n]', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 短い文は概念として追加
            if 10 <= len(sentence) <= 100:
                concepts.append((sentence, 'concept'))
        
        # 日本語の特徴的なフレーズやキーワードを抽出
        # キーワードパターン
        keyword_patterns = [
            (r'「(.+?)」', 'fact'),  # 鍵括弧内のテキスト
            (r'『(.+?)』', 'fact'),  # 二重鍵括弧内のテキスト
            (r'（(.+?)）', 'fact'),  # 丸括弧内のテキスト
            (r'\((.+?)\)', 'fact'),  # 半角丸括弧内のテキスト
            (r'【(.+?)】', 'concept'),  # 隅付き括弧内のテキスト
            (r'［(.+?)］', 'concept'),  # 角括弧内のテキスト
        ]
        
        for pattern, concept_type in keyword_patterns:
            for match in re.finditer(pattern, text):
                content = match.group(1).strip()
                if content and 1 <= len(content) <= 100:
                    concepts.append((content, concept_type))
        
        # 段落や見出しのパターン
        for match in re.finditer(r'^(.*?)[:：]', text, re.MULTILINE):
            header = match.group(1).strip()
            if header and 1 <= len(header) <= 50:
                concepts.append((header, 'concept'))
        
        # 思考ステップのパターンを検出
        step_pattern = re.compile(r'(ステップ\s*\d+|Step\s*\d+|手順\s*\d+|^\d+\.|^\d+\s*、)')
        for match in step_pattern.finditer(text):
            step_text = match.group()
            
            # ステップに続く文も含める（行の残り）
            line_start = match.start()
            line_end = text.find('\n', line_start)
            if line_end == -1:
                line_end = len(text)
            
            step_context = text[line_start:line_end].strip()
            
            if step_context and len(step_context) <= 100:
                concepts.append((step_context, 'step'))
        
        # 推論を示すパターンを検出
        inference_patterns = [
            r'(したがって|よって|つまり|すなわち|要するに|結論として|このことから)',
            r'(考えられる|推測される|推論できる|判断できる|結論づけられる)',
            r'(〜から考えると|〜という結論になる)'
        ]
        
        for pattern in inference_patterns:
            for match in re.finditer(pattern, text):
                # マッチした周辺のテキストを取得（前後50文字）
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                concepts.append((context, 'inference'))
        
        # 質問に関連する概念を抽出
        question_patterns = [
            r'(なぜ|どのように|どのような|何が|いつ|どこで|誰が|何を)',
            r'(\?|？)'
        ]
        
        for pattern in question_patterns:
            for match in re.finditer(pattern, text):
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end]
                if len(context) <= 100:
                    concepts.append((context, 'question'))
        
        # 事実や具体例を示すパターン
        fact_patterns = [
            r'(例えば|具体的には|実際に|事実として)',
            r'(データによると|調査によれば|研究結果では)'
        ]
        
        for pattern in fact_patterns:
            for match in re.finditer(pattern, text):
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 60)  # 事実は後ろのほうが重要なことが多い
                context = text[start:end]
                concepts.append((context, 'fact'))
        
        # 重複を削除
        unique_concepts = []
        seen = set()
        for concept, concept_type in concepts:
            # 長すぎる概念や空の概念は除外
            if not concept or len(concept) > 100:
                continue
            if concept not in seen:
                seen.add(concept)
                unique_concepts.append((concept, concept_type))
        
        # 概念が少なすぎる場合は、単純に短めの行も追加
        if len(unique_concepts) < 5:
            for line in text.split('\n'):
                line = line.strip()
                if line and 10 <= len(line) <= 80 and line not in seen:
                    seen.add(line)
                    unique_concepts.append((line, 'concept'))
                    if len(unique_concepts) >= 10:  # 最大10個まで
                        break
        
        return unique_concepts

    def _extract_relations_from_concepts(self, concepts: List[Tuple[str, str]], text: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        抽出された概念間の関係を推測する

        Parameters:
        -----------
        concepts : List[Tuple[str, str]]
            (概念テキスト, タイプ) のタプルのリスト
        text : str
            元のテキスト

        Returns:
        --------
        List[Tuple[str, str, Dict[str, Any]]]
            (ソース概念, ターゲット概念, 属性) のタプルのリスト
        """
        relations = []
        concept_texts = [c[0] for c in concepts]
        
        # 概念のテキスト内での登場順序を記録
        concept_positions = {}
        for concept, _ in concepts:
            # 複数回出現する場合は最初の位置を使用
            pos = text.find(concept)
            if pos >= 0:
                concept_positions[concept] = pos
        
        # ステップ間の関係（順次関係）を作成
        step_concepts = [(c, t) for c, t in concepts if t == 'step']
        step_concepts.sort(key=lambda x: concept_positions.get(x[0], float('inf')))
        for i in range(len(step_concepts) - 1):
            relations.append((
                step_concepts[i][0],
                step_concepts[i+1][0],
                {'type': 'next_step', 'weight': 2.0}
            ))
        
        # 思考パターンを検出するための正規表現パターンを定義
        patterns = [
            # 因果関係
            (r'(.*?)(が原因で|のため|によって|because of|due to|causes)(.*?)', 'causes'),
            # 包含関係
            (r'(.*?)(を含む|含まれる|includes|contains|part of)(.*?)', 'includes'),
            # 対照関係
            (r'(.*?)(とは対照的に|一方|しかし|in contrast|however|but)(.*?)', 'contrasts'),
            # その他の関係も追加可能
        ]
        
        # パターンに基づいて関係を抽出
        for pattern, rel_type in patterns:
            for match in re.finditer(pattern, text):
                # 前後のテキストから関連する概念を探す
                before_text = match.group(1)
                after_text = match.group(3)
                
                # 前後のテキストに最も近い概念を見つける
                before_concept = None
                after_concept = None
                min_before_dist = float('inf')
                min_after_dist = float('inf')
                
                for concept in concept_texts:
                    # 前のテキストに含まれる概念を探す
                    if concept in before_text:
                        dist = len(before_text) - before_text.find(concept) - len(concept)
                        if dist < min_before_dist:
                            min_before_dist = dist
                            before_concept = concept
                    
                    # 後のテキストに含まれる概念を探す
                    if concept in after_text:
                        dist = after_text.find(concept)
                        if dist < min_after_dist:
                            min_after_dist = dist
                            after_concept = concept
                
                # 両方の概念が見つかった場合、関係を追加
                if before_concept and after_concept:
                    relations.append((
                        before_concept,
                        after_concept,
                        {'type': rel_type, 'weight': 1.5}
                    ))
        
        # 近接性に基づく関係の推定
        sorted_concepts = sorted(concepts, key=lambda x: concept_positions.get(x[0], float('inf')))
        for i in range(len(sorted_concepts) - 1):
            current_concept, current_type = sorted_concepts[i]
            next_concept, next_type = sorted_concepts[i + 1]
            
            # 近い位置にある概念同士を関連付ける
            if concept_positions.get(next_concept, 0) - concept_positions.get(current_concept, 0) < 200:
                relations.append((
                    current_concept,
                    next_concept,
                    {'type': 'related', 'weight': 1.0}
                ))
        
        # 推論関係の処理
        inference_concepts = [(c, t) for c, t in concepts if t == 'inference']
        for inference, _ in inference_concepts:
            # 推論の前後に出現する概念を見つける
            infer_pos = text.find(inference)
            if infer_pos >= 0:
                # 推論の前に出現する概念（前提）
                premises = [(c, t) for c, t in concepts if c != inference and 
                           concept_positions.get(c, float('inf')) < infer_pos]
                
                # 推論の後に出現する概念（結論）
                conclusions = [(c, t) for c, t in concepts if c != inference and 
                              concept_positions.get(c, 0) > infer_pos + len(inference)]
                
                # 前提から推論への関係
                for premise, _ in premises[-2:]:  # 直前の2つまで
                    relations.append((
                        premise,
                        inference,
                        {'type': 'premise', 'weight': 1.8}
                    ))
                
                # 推論から結論への関係
                for conclusion, _ in conclusions[:2]:  # 直後の2つまで
                    relations.append((
                        inference,
                        conclusion,
                        {'type': 'conclusion', 'weight': 1.8}
                    ))
        
        return relations

    def extract_graph_from_thinking_process(self, thinking_process: str, question: str = None, answer: str = None, max_nodes: int = 15) -> nx.DiGraph:
      """
      思考プロセスからグラフを抽出する（日本語対応・最適化版）
      """
      # グラフを作成
      G = nx.DiGraph()
      
      # 非ASCII文字（特に日本語）を処理するための前処理
      def sanitize_text(text):
          if not text:
              return ""
          # 制御文字を除去
          text = re.sub(r'[\x00-\x1F\x7F]', '', text)
          # 長すぎるテキストを切り詰め
          if len(text) > 150:
              text = text[:147] + '...'
          return text
      
      # 質問と回答をグラフに追加
      if question:
          sanitized_question = sanitize_text(question)
          G.add_node(sanitized_question, type='question', size=25)
      
      if answer:
          sanitized_answer = sanitize_text(answer)
          G.add_node(sanitized_answer, type='answer', size=25)
          if question:
              G.add_edge(sanitized_question, sanitized_answer, type='leads_to', weight=3.0)
      
      # 思考プロセスから概念を抽出
      concepts = self._extract_concepts_from_text(thinking_process)
      
      # ノード数を制限
      if len(concepts) > max_nodes:
          # より重要な概念を選択（長さなどに基づいて）
          # ここでは単純に文字数でフィルタリング
          filtered_concepts = []
          min_concept_len = 5  # 最小概念長
          
          # 特定タイプ（inference, step）を優先
          priority_types = ['inference', 'step', 'fact']
          for concept_type in priority_types:
              type_concepts = [(c, t) for c, t in concepts if t == concept_type and len(c) >= min_concept_len]
              filtered_concepts.extend(type_concepts[:max_nodes // 3])  # 各タイプで上限を設定
          
          # 残りを埋める
          remaining_concepts = [(c, t) for c, t in concepts if t not in priority_types and len(c) >= min_concept_len]
          filtered_concepts.extend(remaining_concepts[:max_nodes - len(filtered_concepts)])
          
          concepts = filtered_concepts
      
      # 抽出された概念をグラフに追加
      for concept, concept_type in concepts:
          # サニタイズされた概念テキスト
          sanitized_concept = sanitize_text(concept)
          if not sanitized_concept:
              continue
              
          # ノードが存在しない場合のみ追加
          if not G.has_node(sanitized_concept):
              # タイプに基づいてサイズを決定
              if concept_type == 'inference':
                  size = 20
              elif concept_type == 'fact':
                  size = 18
              elif concept_type == 'step':
                  size = 22  # ステップはより大きく
              else:
                  size = 15
              G.add_node(sanitized_concept, type=concept_type, size=size)
      
      # 概念間の関係を抽出してグラフに追加
      relations = self._extract_relations_from_concepts(concepts, thinking_process)
      for source, target, attrs in relations:
          # サニタイズされた概念テキスト
          sanitized_source = sanitize_text(source)
          sanitized_target = sanitize_text(target)
          
          # 両方のノードがグラフに存在する場合のみエッジを追加
          if G.has_node(sanitized_source) and G.has_node(sanitized_target):
              G.add_edge(sanitized_source, sanitized_target, **attrs)
      
      # 質問と回答を主要概念と接続
      if question and concepts:
          sanitized_question = sanitize_text(question)
          # 最初の数個の概念を質問と接続
          for concept, _ in concepts[:min(3, len(concepts))]:
              sanitized_concept = sanitize_text(concept)
              if G.has_node(sanitized_concept):
                  G.add_edge(sanitized_question, sanitized_concept, type='asks_about', weight=2.0)
      
      if answer and concepts:
          sanitized_answer = sanitize_text(answer)
          # 最後の数個の概念を回答と接続
          for concept, _ in concepts[-min(3, len(concepts)):]:
              sanitized_concept = sanitize_text(concept)
              if G.has_node(sanitized_concept):
                  G.add_edge(sanitized_concept, sanitized_answer, type='supports', weight=2.0)
      
      return G

    # graph_visualization.py の create_explanation_graph メソッドに追加

    def create_explanation_graph(self, explanation: str, question: str = None, answer: str = None, max_nodes: int = 15) -> nx.DiGraph:
        """
        説明テキストからグラフを作成する（最適化版）
        
        Parameters:
        -----------
        explanation : str
            説明テキスト
        question : str, optional
            元の質問
        answer : str, optional
            最終的な回答
        max_nodes : int
            グラフに含める最大ノード数
            
        Returns:
        --------
        nx.DiGraph
            説明グラフ
        """
        # <think>タグがある場合は抽出
        thinking_match = re.search(r'<think>(.*?)</think>', explanation, re.DOTALL)
        thinking_process = thinking_match.group(1).strip() if thinking_match else explanation
        
        # まずは通常通りグラフを生成
        G = self.extract_graph_from_thinking_process(thinking_process, question, answer)
        
        # ノード数が多すぎる場合は重要なノードだけに絞る
        if G.number_of_nodes() > max_nodes + 2:  # +2 は質問と回答のノード
            # 重要度計算（中心性などを使用）
            try:
                # 中心性が高いノードほど重要と考える
                centrality = nx.betweenness_centrality(G)
                # 質問と回答は常に含める
                if question:
                    centrality[question] = float('inf')
                if answer:
                    centrality[answer] = float('inf')
                    
                # 中心性の低いノードを削除
                nodes_to_keep = sorted(centrality, key=centrality.get, reverse=True)[:max_nodes]
                nodes_to_remove = [n for n in G.nodes() if n not in nodes_to_keep and n != question and n != answer]
                
                for node in nodes_to_remove:
                    G.remove_node(node)
                    
                print(f"グラフを最適化: {len(nodes_to_remove)} ノードを削除し、{G.number_of_nodes()} ノードを保持")
            except:
                # 中心性計算に失敗した場合は何もしない
                pass
        
        return G

    def render_graph_as_base64(self, G: nx.DiGraph, title: str = "Explanation Graph") -> str:
      """
      グラフをレンダリングし、base64エンコードされた画像として返す（日本語対応強化版）
      """
      import matplotlib
      matplotlib.use('Agg')  # GUIバックエンドを使用しない設定
      import matplotlib.pyplot as plt
      
      # 日本語フォントの設定
      self._setup_japanese_font()
      
      # フィギュアを作成
      plt.figure(figsize=(14, 10), dpi=150)
      plt.title(title, fontsize=16)
      
      # グラフが空の場合は空の画像を返す
      if G.number_of_nodes() == 0:
          plt.text(0.5, 0.5, "グラフデータがありません", ha='center', va='center', fontsize=14)
          plt.axis('off')
          img_buf = io.BytesIO()
          plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
          plt.close()
          img_buf.seek(0)
          return base64.b64encode(img_buf.read()).decode('utf-8')
      
      # ノードの位置計算を改善
      try:
          # kamada_kawai_layout はより良い配置になることが多い
          pos = nx.kamada_kawai_layout(G)
      except:
          # 失敗した場合は spring_layout にフォールバック
          pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42)
      
      # ノードのグループ化
      node_types = {}
      for node in G.nodes():
          node_type = G.nodes[node].get('type', 'default')
          if node_type not in node_types:
              node_types[node_type] = []
          node_types[node_type].append(node)
      
      # 強調表示するエッジの設定
      highlighted_edge_types = {
          'next_step': {'color': 'red', 'width_factor': 2.0, 'style': 'solid'},
          'premise': {'color': 'blue', 'width_factor': 1.8, 'style': 'solid'},
          'conclusion': {'color': 'blue', 'width_factor': 1.8, 'style': 'solid'},
          'causes': {'color': 'green', 'width_factor': 1.5, 'style': 'solid'},
          'contrasts': {'color': 'purple', 'width_factor': 1.5, 'style': 'solid'},
          'asks_about': {'color': 'orange', 'width_factor': 1.5, 'style': 'dashed'},
          'supports': {'color': 'brown', 'width_factor': 1.5, 'style': 'dashed'}
      }
      
      # エッジを描画
      for edge_type, edge_props in highlighted_edge_types.items():
          # 特定タイプのエッジだけを抽出
          edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type', 'default') == edge_type]
          if edges:
              weights = [G[u][v].get('weight', 1.0) * edge_props['width_factor'] for u, v in edges]
              nx.draw_networkx_edges(
                  G, pos, 
                  edgelist=edges,
                  width=weights,
                  edge_color=edge_props['color'],
                  style=edge_props.get('style', 'solid'),
                  arrowsize=15,
                  arrowstyle='->',
                  connectionstyle='arc3,rad=0.1',
                  alpha=0.8
              )
      
      # デフォルトエッジを描画（上記で指定したタイプ以外のすべてのエッジ）
      default_edges = [(u, v) for u, v, d in G.edges(data=True) 
                      if d.get('type', 'default') not in highlighted_edge_types]
      if default_edges:
          weights = [G[u][v].get('weight', 1.0) for u, v in default_edges]
          nx.draw_networkx_edges(
              G, pos, 
              edgelist=default_edges,
              width=weights,
              edge_color='gray',
              style='solid',
              arrowsize=10,
              arrowstyle='->',
              connectionstyle='arc3,rad=0.1',
              alpha=0.5
          )
      
      # タイプごとにノードを描画
      for node_type, nodes in node_types.items():
          node_color = self.color_palette.get(node_type, self.color_palette['default'])
          # サイズを調整
          node_sizes = [G.nodes[n].get('size', 10) * 40 for n in nodes]
          
          nx.draw_networkx_nodes(
              G, pos, 
              nodelist=nodes, 
              node_color=node_color,
              node_size=node_sizes, 
              alpha=0.8
          )
      
      # ラベルを描画（重要: 日本語対応）
      # 1. すべてのノードのラベルを作成
      all_labels = {}
      for node in G.nodes():
          # 長いラベルを短くする
          label = node
          if len(label) > 25:
              label = label[:22] + '...'
          all_labels[node] = label
      
      # 2. ノードラベルを描画（バックグラウンドありでより読みやすく）
      for node, label in all_labels.items():
          x, y = pos[node]
          node_type = G.nodes[node].get('type', 'default')
          bg_color = self.color_palette.get(node_type, self.color_palette['default'])
          
          # 中心性に基づいてフォントサイズを調整
          font_size = 10
          if node_type in ['question', 'answer']:
              font_size = 12  # 質問と回答は大きく
          elif node_type == 'step':
              font_size = 11  # ステップも少し大きく
          
          plt.text(
              x, y, label,
              fontsize=font_size,
              ha='center', va='center',
              bbox=dict(
                  boxstyle="round,pad=0.3",
                  fc=bg_color,
                  ec='black',
                  alpha=0.8
              )
          )
      
      # 凡例を追加
      legend_elements = []
      import matplotlib.patches as mpatches
      
      # 日本語の対応表
      label_map = {
          'concept': '概念',
          'fact': '事実',
          'inference': '推論',
          'source': '情報源',
          'question': '質問',
          'answer': '回答',
          'step': 'プロセス',
          'default': 'その他'
      }
      
      for node_type, color in self.color_palette.items():
          if node_type in node_types:
              label = label_map.get(node_type, node_type.capitalize())
              legend_elements.append(mpatches.Patch(color=color, label=label))
      
      plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
      plt.tight_layout()
      plt.axis('off')
      
      # 画像をメモリ上に保存
      img_buf = io.BytesIO()
      plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
      plt.close()
      
      # base64エンコード
      img_buf.seek(0)
      img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
      
      return img_base64

    def save_graph_as_image(self, G: nx.DiGraph, filepath: str, title: str = "AI Explanation Graph"):
        """
        グラフを画像として保存する

        Parameters:
        -----------
        G : nx.DiGraph
            保存するグラフ
        filepath : str
            保存先のファイルパス
        title : str
            グラフのタイトル
        """
        plt.figure(figsize=(12, 9))
        plt.title(title, fontsize=16)
        
        # ノードの位置を計算
        pos = nx.spring_layout(G, k=0.25, iterations=50, seed=42)
        
        # ノードをタイプごとにグループ化
        node_types = {}
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'default')
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append(node)
        
        # タイプごとにノードを描画
        for node_type, nodes in node_types.items():
            node_color = self.color_palette.get(node_type, self.color_palette['default'])
            node_sizes = [G.nodes[n].get('size', 10) * 20 for n in nodes]
            
            # ノード名が長すぎる場合は省略
            node_labels = {}
            for n in nodes:
                if len(n) > 30:
                    node_labels[n] = n[:27] + '...'
                else:
                    node_labels[n] = n
            
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=node_color, 
                                  node_size=node_sizes, alpha=0.8)
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)
        
        # エッジを描画
        edge_colors = []
        edge_widths = []
        for u, v, data in G.edges(data=True):
            edge_type = data.get('type', 'default')
            if edge_type == 'next_step':
                edge_colors.append('red')
            elif edge_type == 'premise' or edge_type == 'conclusion':
                edge_colors.append('blue')
            elif edge_type == 'causes':
                edge_colors.append('green')
            elif edge_type == 'contrasts':
                edge_colors.append('purple')
            else:
                edge_colors.append('gray')
            
            edge_widths.append(data.get('weight', 1.0))
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, 
                              arrowsize=15, arrowstyle='->', alpha=0.7)
        
        # 凡例を追加
        legend_elements = []
        import matplotlib.patches as mpatches
        for node_type, color in self.color_palette.items():
            if node_type in node_types:
                legend_elements.append(mpatches.Patch(color=color, label=node_type.capitalize()))
        
        plt.legend(handles=legend_elements, loc='upper right')
        plt.tight_layout()
        plt.axis('off')
        
        # 画像を保存
        plt.savefig(filepath, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"グラフを保存しました: {filepath}")

    def generate_graph_metrics(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        グラフの特性に関する指標を生成する

        Parameters:
        -----------
        G : nx.DiGraph
            分析するグラフ

        Returns:
        --------
        Dict[str, Any]
            グラフ指標の辞書
        """
        metrics = {}
        
        # 基本的な指標
        metrics['node_count'] = G.number_of_nodes()
        metrics['edge_count'] = G.number_of_edges()
        metrics['density'] = nx.density(G)
        
        # 次数に関する指標
        if G.number_of_nodes() > 0:
            in_degrees = [d for _, d in G.in_degree()]
            out_degrees = [d for _, d in G.out_degree()]
            
            metrics['avg_in_degree'] = sum(in_degrees) / len(in_degrees) if in_degrees else 0
            metrics['avg_out_degree'] = sum(out_degrees) / len(out_degrees) if out_degrees else 0
            metrics['max_in_degree'] = max(in_degrees) if in_degrees else 0
            metrics['max_out_degree'] = max(out_degrees) if out_degrees else 0
        
        # 中心性指標
        try:
            betweenness = nx.betweenness_centrality(G)
            metrics['avg_betweenness'] = sum(betweenness.values()) / len(betweenness) if betweenness else 0
            
            # 最も中心的なノードを特定
            if betweenness:
                max_betweenness_node = max(betweenness, key=betweenness.get)
                metrics['central_node'] = max_betweenness_node
                metrics['central_node_type'] = G.nodes[max_betweenness_node].get('type', 'unknown')
        except:
            metrics['avg_betweenness'] = None
            metrics['central_node'] = None
            metrics['central_node_type'] = None
        
        # ノードタイプの分布
        node_type_distribution = {}
        for node in G.nodes():
            node_type = G.nodes[node].get('type', 'default')
            node_type_distribution[node_type] = node_type_distribution.get(node_type, 0) + 1
        
        metrics['node_type_distribution'] = node_type_distribution
        
        # エッジタイプの分布
        edge_type_distribution = {}
        for _, _, data in G.edges(data=True):
            edge_type = data.get('type', 'default')
            edge_type_distribution[edge_type] = edge_type_distribution.get(edge_type, 0) + 1
        
        metrics['edge_type_distribution'] = edge_type_distribution
        
        return metrics
    
    # graph_visualization.py に追加するフォント設定関数
    def _setup_japanese_font(self):
        """日本語フォントを設定する"""
        import matplotlib as mpl
        import platform
        
        # システムの種類を判別
        system = platform.system()
        
        # フォントファミリーを設定
        mpl.rcParams['font.family'] = 'sans-serif'
        
        if system == 'Windows':
            # Windowsの場合
            font_list = ['Yu Gothic', 'Meiryo', 'MS Gothic', 'Malgun Gothic']
        elif system == 'Darwin':
            # macOSの場合
            font_list = ['Hiragino Sans', 'Hiragino Kaku Gothic Pro', 'Osaka']
        else:
            # Linux/その他の場合
            font_list = ['IPAGothic', 'IPAPGothic', 'VL Gothic', 'Noto Sans CJK JP']
        
        # 使用可能なフォントのリストからサンス系フォントを検索
        import matplotlib.font_manager as fm
        available_fonts = set(f.name for f in fm.fontManager.ttflist)
        
        # 使用可能なフォントから、リストにあるフォントを選択
        for font in font_list:
            if font in available_fonts:
                mpl.rcParams['font.sans-serif'] = [font] + mpl.rcParams['font.sans-serif']
                print(f"日本語フォント '{font}' を使用します")
                return True
        
        # どのフォントも見つからなかった場合のフォールバック
        print("警告: 日本語フォントが見つかりませんでした。デフォルトフォントを使用します。")
        return False