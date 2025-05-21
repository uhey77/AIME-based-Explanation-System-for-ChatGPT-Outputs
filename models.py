# models.py
from typing import Dict, Any


class Document:
    """ドキュメントを表すクラス"""
    def __init__(self, text: str, metadata: Dict[str, Any]):
        """
        初期化メソッド

        Parameters:
        -----------
        text : str
            ドキュメントのテキスト内容
        metadata : dict
            ドキュメントのメタデータ (例: {'filename': 'example.txt'})
        """
        self.page_content = text
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

# 必要に応じて他のデータモデルもここに追加
