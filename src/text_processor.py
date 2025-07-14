"""
Text Processor for Medical RAG System
医療RAGシステム用テキスト前処理・チャンク分割モジュール
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """テキストチャンクのデータクラス"""
    id: str
    content: str
    metadata: Dict
    source: str = "pubmed"
    chunk_index: int = 0
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'source': self.source,
            'chunk_index': self.chunk_index
        }


class MedicalTextProcessor:
    """医療文献テキスト前処理クラス"""
    
    def __init__(self, 
                 chunk_size: int = None, 
                 chunk_overlap: int = None):
        """
        Args:
            chunk_size: チャンクサイズ（文字数）
            chunk_overlap: チャンク間オーバーラップ（文字数）
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        # 医学用語正規化辞書
        self.medical_abbreviations = {
            'MI': 'myocardial infarction',
            'HTN': 'hypertension',
            'DM': 'diabetes mellitus',
            'CAD': 'coronary artery disease',
            'COPD': 'chronic obstructive pulmonary disease',
            'CHF': 'congestive heart failure',
            'CVA': 'cerebrovascular accident',
            'ICU': 'intensive care unit',
            'ER': 'emergency room',
            'OR': 'operating room',
            'CT': 'computed tomography',
            'MRI': 'magnetic resonance imaging',
            'ECG': 'electrocardiogram',
            'EKG': 'electrocardiogram',
            'CBC': 'complete blood count',
            'BUN': 'blood urea nitrogen',
            'HIV': 'human immunodeficiency virus',
            'AIDS': 'acquired immunodeficiency syndrome',
            'COVID': 'coronavirus disease',
            'SARS': 'severe acute respiratory syndrome',
            'MERS': 'Middle East respiratory syndrome'
        }
        
        logger.info(f"TextProcessor initialized: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def clean_text(self, text: str) -> str:
        """
        テキストクリーニング
        
        Args:
            text: 原文テキスト
            
        Returns:
            クリーニング済みテキスト
        """
        if not text:
            return ""
        
        # 改行・タブの正規化
        text = re.sub(r'\r\n|\r|\n', ' ', text)
        text = re.sub(r'\t', ' ', text)
        
        # 複数スペースの削除
        text = re.sub(r'\s+', ' ', text)
        
        # HTMLタグの除去
        text = re.sub(r'<[^>]+>', '', text)
        
        # 特殊文字の正規化
        text = re.sub(r'[""''‚„]', '"', text)
        text = re.sub(r'[–—]', '-', text)
        text = re.sub(r'[…]', '...', text)
        
        # 文字化け除去
        text = re.sub(r'[^\x00-\x7F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', '', text)
        
        # 前後空白削除
        text = text.strip()
        
        return text
    
    def normalize_medical_terms(self, text: str) -> str:
        """
        医学用語の正規化
        
        Args:
            text: 入力テキスト
            
        Returns:
            正規化済みテキスト
        """
        normalized_text = text
        
        # 略語の展開
        for abbrev, full_form in self.medical_abbreviations.items():
            # 単語境界を考慮した置換
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            replacement = f"{abbrev} ({full_form})"
            normalized_text = re.sub(pattern, replacement, normalized_text, flags=re.IGNORECASE)
        
        # 単位の正規化
        normalized_text = re.sub(r'\bmg/dl\b', 'mg/dL', normalized_text, flags=re.IGNORECASE)
        normalized_text = re.sub(r'\bmmhg\b', 'mmHg', normalized_text, flags=re.IGNORECASE)
        normalized_text = re.sub(r'\bkg/m2\b', 'kg/m²', normalized_text, flags=re.IGNORECASE)
        
        return normalized_text
    
    def split_into_chunks(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """
        テキストをチャンクに分割
        
        Args:
            text: 分割対象テキスト
            metadata: メタデータ
            
        Returns:
            テキストチャンクのリスト
        """
        if not text:
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # 文単位での分割を優先
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # チャンクサイズを超える場合
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # 現在のチャンクを保存
                chunk_id = f"{metadata.get('pmid', 'unknown')}_{chunk_index}"
                chunk = TextChunk(
                    id=chunk_id,
                    content=current_chunk.strip(),
                    metadata={**metadata, 'chunk_type': 'sentence_based'},
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                
                # 新しいチャンク開始（オーバーラップ考慮）
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + " " + sentence
                chunk_index += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # 最後のチャンク
        if current_chunk.strip():
            chunk_id = f"{metadata.get('pmid', 'unknown')}_{chunk_index}"
            chunk = TextChunk(
                id=chunk_id,
                content=current_chunk.strip(),
                metadata={**metadata, 'chunk_type': 'sentence_based'},
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        logger.info(f"Split text into {len(chunks)} chunks (avg size: {sum(len(c.content) for c in chunks) / len(chunks):.0f} chars)")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """文単位での分割"""
        # 医学文献特有の略語を考慮した文分割
        sentence_endings = r'[.!?]\s+'
        sentences = re.split(sentence_endings, text)
        
        # 短すぎる文や不完全な文を結合
        processed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 and processed_sentences:
                # 前の文と結合
                processed_sentences[-1] += ". " + sentence
            elif sentence:
                processed_sentences.append(sentence)
        
        return processed_sentences
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """オーバーラップテキストの取得"""
        if len(text) <= overlap_size:
            return text
        
        # 単語境界を考慮したオーバーラップ
        overlap_text = text[-overlap_size:]
        space_index = overlap_text.find(' ')
        if space_index > 0:
            overlap_text = overlap_text[space_index + 1:]
        
        return overlap_text
    
    def create_searchable_content(self, article: Dict) -> str:
        """
        検索用コンテンツの作成（複数フィールドの結合）
        
        Args:
            article: PubMed文献データ
            
        Returns:
            検索用統合テキスト
        """
        content_parts = []
        
        # 優先度A: 主要フィールド
        if article.get('title'):
            content_parts.append(f"Title: {article['title']}")
        
        if article.get('abstract'):
            content_parts.append(f"Abstract: {article['abstract']}")
        
        if article.get('mesh_terms'):
            mesh_text = ", ".join(article['mesh_terms'])
            content_parts.append(f"MeSH Terms: {mesh_text}")
        
        # 優先度B: 補助フィールド
        if article.get('keywords'):
            keywords_text = ", ".join(article['keywords'])
            content_parts.append(f"Keywords: {keywords_text}")
        
        return "\n\n".join(content_parts)
    
    def process_medical_data(self, data_file: str) -> List[TextChunk]:
        """
        医療データファイルの処理
        
        Args:
            data_file: JSONデータファイルパス
            
        Returns:
            処理済みテキストチャンクのリスト
        """
        logger.info(f"Processing medical data from: {data_file}")
        
        # データ読み込み
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = data.get('articles', [])
        all_chunks = []
        
        for article in articles:
            try:
                # 検索用コンテンツ作成
                searchable_content = self.create_searchable_content(article)
                
                # テキストクリーニング
                cleaned_content = self.clean_text(searchable_content)
                
                # 医学用語正規化
                normalized_content = self.normalize_medical_terms(cleaned_content)
                
                # メタデータ準備
                metadata = {
                    'pmid': article.get('pmid'),
                    'title': article.get('title'),
                    'authors': article.get('authors', []),
                    'journal': article.get('journal'),
                    'publication_date': article.get('publication_date'),
                    'doi': article.get('doi'),
                    'mesh_terms': article.get('mesh_terms', []),
                    'keywords': article.get('keywords', []),
                    'publication_types': article.get('publication_types', [])
                }
                
                # チャンク分割
                chunks = self.split_into_chunks(normalized_content, metadata)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Failed to process article {article.get('pmid', 'unknown')}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(articles)} articles into {len(all_chunks)} chunks")
        return all_chunks
    
    def save_processed_chunks(self, chunks: List[TextChunk], output_file: str):
        """
        処理済みチャンクの保存
        
        Args:
            chunks: テキストチャンクリスト
            output_file: 出力ファイルパス
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        chunk_data = {
            'metadata': {
                'total_chunks': len(chunks),
                'processing_settings': {
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap
                },
                'processed_at': logger.handlers[0].format(logging.LogRecord(
                    name='', level=0, pathname='', lineno=0, msg='', args=(), exc_info=None
                )).split(' - ')[0] if logger.handlers else 'unknown'
            },
            'chunks': [chunk.to_dict() for chunk in chunks]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_file}")


def main():
    """テスト実行"""
    processor = MedicalTextProcessor()
    
    # テストデータの処理
    input_file = "data/sample_data.json"
    output_file = "data/processed_chunks.json"
    
    if Path(input_file).exists():
        chunks = processor.process_medical_data(input_file)
        processor.save_processed_chunks(chunks, output_file)
        
        print(f"\n=== Processing Results ===")
        print(f"Total chunks: {len(chunks)}")
        print(f"Average chunk size: {sum(len(c.content) for c in chunks) / len(chunks):.0f} characters")
        print(f"Output saved to: {output_file}")
        
        # サンプル表示
        if chunks:
            print(f"\n=== Sample Chunk ===")
            sample = chunks[0]
            print(f"ID: {sample.id}")
            print(f"Content preview: {sample.content[:200]}...")
            print(f"Metadata: {sample.metadata}")
    else:
        print(f"Input file not found: {input_file}")


if __name__ == "__main__":
    main()