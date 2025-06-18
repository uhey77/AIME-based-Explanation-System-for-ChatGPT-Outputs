import itertools
from collections import Counter
from typing import List, Dict, Any, Optional

import networkx as nx
import matplotlib.pyplot as plt
import japanize_matplotlib # Matplotlibで日本語を正しく表示するため
from transformers import AutoTokenizer, PreTrainedTokenizer

# --- 定数定義 ---
DEFAULT_MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
# 他のモデル候補:
# DEFAULT_MODEL_NAME = "cl-tohoku/bert-base-japanese-whole-word-masking"
# DEFAULT_MODEL_NAME = "rinna/japanese-roberta-base"
# DEFAULT_MODEL_NAME = "cl-tohoku/bert-base-japanese-v2"

DEFAULT_SAMPLE_TEXT = "AIで単語のつながりを分析し、NetworkXでネットワークを可視化したい。"
DEFAULT_WINDOW_SIZE = 3

# --- Tokenizer関連 ---

def load_tokenizer(model_name: str) -> Optional[PreTrainedTokenizer]:
    """
    指定されたモデル名のTokenizerをロードします。

    Args:
        model_name: ロードするHugging Faceモデル名。

    Returns:
        成功時はTokenizerインスタンス、失敗時はNone。
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Tokenizer '{model_name}' のロードに成功しました。")
        return tokenizer
    except Exception as e:
        print(f"Tokenizer '{model_name}' のロードに失敗しました: {e}")
        print("モデル名が正しいか、インターネット接続があるか確認してください。")
        return None

def get_simple_tokens(text: str, tokenizer_instance: PreTrainedTokenizer) -> List[str]:
    """
    テキストをサブワードトークンのリスト（文字列）に単純に変換します。
    特殊トークン（[CLS], [SEP]など）は含めません。

    Args:
        text: トークン化するテキスト。
        tokenizer_instance: 使用するTokenizerインスタンス。

    Returns:
        トークン文字列のリスト。
    """
    return tokenizer_instance.tokenize(text)

def tokenize_for_model_input(
    text: str,
    tokenizer_instance: PreTrainedTokenizer,
    add_special_tokens: bool = True,
    return_tensors: str = 'pt'
) -> Dict[str, Any]:
    """
    モデル入力用にテキストをエンコードし、ID、アテンションマスクなどを取得します。

    Args:
        text: エンコードするテキスト。
        tokenizer_instance: 使用するTokenizerインスタンス。
        add_special_tokens: 特殊トークンを付加するかどうか。
        return_tensors: 返すテンソルの種類（'pt' for PyTorch, 'tf' for TensorFlow）。

    Returns:
        エンコードされた入力データ（input_ids, attention_maskなどを含む辞書）。
    """
    encoded_input = tokenizer_instance(
        text,
        add_special_tokens=add_special_tokens,
        return_attention_mask=True,
        return_tensors=return_tensors
        # 必要に応じて以下のパラメータも設定可能
        # max_length=512,
        # truncation=True,
        # padding='max_length'
    )
    return encoded_input

# --- 共起計算関連 ---

def calculate_cooccurrence(
    tokens: List[str],
    window_size: int = 2
) -> Counter:
    """
    トークンリストから共起ペアとその頻度を計算します。

    Args:
        tokens: トークン文字列のリスト。
        window_size: 共起とみなすトークン間の最大距離（ウィンドウサイズ）。
     
    Returns:
        共起ペアをキー、頻度を値とするCounterオブジェクト。
        ペアは (token1, token2) のタプルで、token1 <= token2 となるようソートされます。
    """
    if not tokens or len(tokens) < 2:
        return Counter()

    pairs = []
    # ウィンドウサイズがリストの長さより大きい場合は、リストの長さに合わせる
    effective_window_size = min(window_size, len(tokens))

    for i in range(len(tokens) - effective_window_size + 1):
        current_window = tokens[i : i + effective_window_size]
        # ウィンドウ内の要素数が2未満の場合は combinations でエラーになるためスキップ
        if len(current_window) < 2:
            continue
        for token1, token2 in itertools.combinations(current_window, 2):
            # 同じトークン同士のペアは含めない (例: ('単語', '単語'))
            # itertools.combinations は異なる位置にあれば同じ値でもペアを生成する
            # 例: ['A', 'B', 'A'] -> ('A', 'A') は最初のAと3番目のA
            if token1 != token2:
                # ペアの順序を正規化するためソートしてタプルにする
                pairs.append(tuple(sorted((token1, token2))))

    return Counter(pairs)

# --- グラフ構築・可視化関連 ---

def create_cooccurrence_graph(
    cooccurrence_counts: Counter,
    initial_nodes: Optional[List[str]] = None
) -> nx.Graph:
    """
    共起カウントデータからNetworkXのグラフオブジェクトを構築します。

    Args:
        cooccurrence_counts: calculate_cooccurrence関数で得られた共起ペアと頻度のCounter。
        initial_nodes: グラフに含める初期ノードのリスト。指定した場合、共起がなくてもノードとして存在します。

    Returns:
        構築されたNetworkXグラフオブジェクト。
    """
    graph = nx.Graph()
    if initial_nodes:
        graph.add_nodes_from(list(set(initial_nodes))) # 重複を除いてノード追加

    for (token1, token2), weight in cooccurrence_counts.items():
        if weight > 0: # 頻度が1回以上の場合のみエッジを追加
            graph.add_edge(token1, token2, weight=weight)
    return graph

def visualize_network(graph: nx.Graph, title: str = "Word Network"):
    """
    NetworkXグラフをMatplotlibで可視化します。

    Args:
        graph: 可視化するNetworkXグラフオブジェクト。
        title: グラフのタイトル。
    """
    if not graph.nodes():
        print("グラフにノードがありません。可視化をスキップします。")
        return

    plt.figure(figsize=(14, 12)) # サイズを少し大きく

    # ノードの配置アルゴリズム (kでノード間隔、iterationsで安定度を調整)
    pos = nx.spring_layout(graph, k=0.7, iterations=50, seed=42)

    # ノードのサイズを次数（接続エッジ数）に応じて変更
    node_degrees = [graph.degree(n) for n in graph.nodes()]
    min_size, max_size = 200, 2000 # ノードサイズの最小・最大値
    node_sizes = [min_size + (max_size - min_size) * (d / max(node_degrees, default=1)) if node_degrees else min_size for d in node_degrees]
    if not any(node_degrees): # 全ノードが孤立している場合
        node_sizes = [min_size for _ in graph.nodes()]


    # エッジの太さを重み (weight) に応じて変更
    # 重みが存在しないエッジはデフォルトの太さ 1 とする
    edge_weights = [graph.edges[edge].get('weight', 1) * 0.5 for edge in graph.edges()]

    # 描画
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color='skyblue', alpha=0.9, linewidths=0.5, edgecolors='black')
    nx.draw_networkx_edges(graph, pos, width=edge_weights, edge_color='grey', alpha=0.7)
    nx.draw_networkx_labels(graph, pos, font_family='IPAexGothic', font_size=10, font_weight='bold')

    plt.title(title, fontsize=18, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# --- メイン処理 ---
def main():
    """
    メイン処理を実行します。
    1. Tokenizerのロード
    2. テキストのトークン化
    3. 共起頻度の計算
    4. 共起ネットワークグラフの構築
    5. ネットワークの可視化
    """
    # --- 設定 ---
    model_name_to_use = DEFAULT_MODEL_NAME
    sample_text_to_process = DEFAULT_SAMPLE_TEXT
    window_size_for_cooccurrence = DEFAULT_WINDOW_SIZE

    print(f"使用モデル: {model_name_to_use}")
    print(f"処理テキスト: 「{sample_text_to_process}」")
    print(f"共起ウィンドウサイズ: {window_size_for_cooccurrence}")

    # --- 1. Tokenizerのロード ---
    tokenizer = load_tokenizer(model_name_to_use)

    if not tokenizer:
        print("Tokenizerのロードに失敗したため、処理を中断します。")
        return

    # --- 2. テキストのトークン化 (NetworkX用) ---
    # 今回はシンプルな `tokenize` メソッドを使用します。
    # 特殊トークンを除いたリストを得る他の方法としては、
    # `tokenize_for_model_input` を呼び出し、IDから変換する方法もあります:
    # encoded = tokenize_for_model_input(sample_text_to_process, tokenizer)
    # tokens_for_network = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0].tolist(), skip_special_tokens=True)
    tokens_for_network = get_simple_tokens(sample_text_to_process, tokenizer)

    if not tokens_for_network:
        print("トークン化結果が空です。処理を中断します。")
        return
    print(f"\nサブワードトークン (NetworkX用): {tokens_for_network}")

    # (参考) モデル入力用のエンコード結果
    # encoded_output = tokenize_for_model_input(sample_text_to_process, tokenizer)
    # print(f"\nエンコードされた入力ID (参考): {encoded_output['input_ids']}")
    # tokens_from_ids = tokenizer.convert_ids_to_tokens(encoded_output['input_ids'][0].tolist())
    # print(f"IDから変換したトークン (参考, 特殊トークン含む): {tokens_from_ids}")


    # --- 3. 共起頻度の計算 ---
    cooc_counts = calculate_cooccurrence(tokens_for_network, window_size=window_size_for_cooccurrence)

    if not cooc_counts:
        print("共起ペアが見つかりませんでした。処理を中断します。")
        return
    print(f"\n共起ペアと頻度 (上位10件): {cooc_counts.most_common(10)}")


    # --- 4. 共起ネットワークグラフの構築 ---
    # オプション: 元のテキストに含まれる全トークンをノードとして含める場合
    # cooccurrence_graph = create_cooccurrence_graph(cooc_counts, initial_nodes=tokens_for_network)
    # ここでは共起があったトークンのみでグラフを構成します
    cooccurrence_graph = create_cooccurrence_graph(cooc_counts)

    if not cooccurrence_graph.nodes():
        print("グラフ構築の結果、ノードが存在しません。処理を中断します。")
        return
    print(f"\nグラフのノード数: {cooccurrence_graph.number_of_nodes()}, エッジ数: {cooccurrence_graph.number_of_edges()}")


    # --- 5. ネットワークの可視化 ---
    graph_title = f"「{model_name_to_use}」による単語共起ネットワーク\n(テキスト: 「{sample_text_to_process[:20]}...」)"
    visualize_network(cooccurrence_graph, title=graph_title)

if __name__ == "__main__":
    main()