"""
ChromaDB Vector Store for Medical RAG System
医療RAGシステム用ChromaDBベクトルストアモジュール
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from config.settings import settings

logger = logging.getLogger(__name__)


class MedicalVectorStore:
    """医療文献用ベクトルストアクラス"""
    
    def __init__(self, 
                 chroma_db_path: str = None,
                 collection_name: str = None,
                 embedding_model: str = None):
        """
        Args:
            chroma_db_path: ChromaDBのパス
            collection_name: コレクション名
            embedding_model: Embeddingモデル名
        """
        self.chroma_db_path = chroma_db_path or settings.CHROMA_DB_PATH
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL
        
        # ChromaDBクライアント初期化
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        self._initialize_chromadb()
        self._initialize_embedding_model()
        
        logger.info(f"VectorStore initialized: {self.chroma_db_path}/{self.collection_name}")
    
    def _initialize_chromadb(self):
        """ChromaDBクライアントとコレクションの初期化"""
        try:
            # ChromaDBクライアント作成
            self.client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # コレクション取得または作成
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except (ValueError, chromadb.errors.NotFoundError):
                # コレクションが存在しない場合は新規作成
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Medical literature chunks for RAG system"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _initialize_embedding_model(self):
        """Embeddingモデルの初期化"""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        テキストリストからEmbeddingを生成
        
        Args:
            texts: テキストリスト
            
        Returns:
            Embeddingのリスト
        """
        if not texts:
            return []
        
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts")
            
            # バッチ処理でEmbedding生成
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # ChromaDB用にリスト形式に変換
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            logger.info(f"Successfully created {len(embeddings_list)} embeddings")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise
    
    def add_documents(self, 
                     chunk_data: Dict,
                     batch_size: int = 100) -> int:
        """
        処理済みチャンクデータをベクトルストアに追加
        
        Args:
            chunk_data: 処理済みチャンクデータ（JSON形式）
            batch_size: バッチサイズ
            
        Returns:
            追加された文書数
        """
        chunks = chunk_data.get('chunks', [])
        if not chunks:
            logger.warning("No chunks found in data")
            return 0
        
        logger.info(f"Adding {len(chunks)} documents to vector store")
        
        total_added = 0
        
        # バッチ処理で追加
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            
            try:
                # バッチデータ準備
                ids = []
                documents = []
                metadatas = []
                
                for chunk in batch_chunks:
                    ids.append(chunk['id'])
                    documents.append(chunk['content'])
                    
                    # メタデータ準備（ChromaDB形式に適合）
                    metadata = self._prepare_metadata(chunk['metadata'])
                    metadatas.append(metadata)
                
                # Embedding生成
                embeddings = self.create_embeddings(documents)
                
                # ChromaDBに追加
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                total_added += len(batch_chunks)
                logger.info(f"Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} "
                           f"({len(batch_chunks)} documents)")
                
            except Exception as e:
                logger.error(f"Failed to add batch {i//batch_size + 1}: {e}")
                continue
        
        logger.info(f"Successfully added {total_added} documents to vector store")
        return total_added
    
    def _prepare_metadata(self, metadata: Dict) -> Dict:
        """
        ChromaDB用メタデータの準備
        
        Args:
            metadata: 元のメタデータ
            
        Returns:
            ChromaDB互換メタデータ
        """
        # ChromaDBはネストした構造やリストを直接サポートしないため、文字列化
        prepared_metadata = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (list, dict)):
                # リストや辞書は文字列化
                prepared_metadata[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (str, int, float, bool)):
                # プリミティブ型はそのまま
                prepared_metadata[key] = value
            else:
                # その他は文字列化
                prepared_metadata[key] = str(value)
        
        return prepared_metadata
    
    def search_similar(self, 
                      query: str, 
                      top_k: int = None,
                      filters: Dict = None) -> List[Dict]:
        """
        類似度検索の実行
        
        Args:
            query: 検索クエリ
            top_k: 取得する文書数
            filters: メタデータフィルタ
            
        Returns:
            検索結果のリスト
        """
        top_k = top_k or settings.SEARCH_TOP_K
        
        try:
            logger.info(f"Searching for: '{query}' (top_k={top_k})")
            
            # クエリのEmbedding生成
            query_embedding = self.create_embeddings([query])[0]
            
            # ChromaDBで検索
            search_params = {
                'query_embeddings': [query_embedding],
                'n_results': top_k
            }
            
            # フィルタ適用
            if filters:
                search_params['where'] = filters
            
            results = self.collection.query(**search_params)
            
            # 結果整形
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': self._restore_metadata(results['metadatas'][0][i]),
                        'similarity_score': 1 - results['distances'][0][i],  # 距離を類似度に変換
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _restore_metadata(self, metadata: Dict) -> Dict:
        """
        メタデータの復元（JSON文字列を元の形式に戻す）
        
        Args:
            metadata: ChromaDBから取得したメタデータ
            
        Returns:
            復元されたメタデータ
        """
        restored_metadata = {}
        
        for key, value in metadata.items():
            if key in ['mesh_terms', 'keywords', 'authors', 'publication_types']:
                # リスト形式のフィールドを復元
                try:
                    restored_metadata[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    restored_metadata[key] = value
            else:
                restored_metadata[key] = value
        
        return restored_metadata
    
    def get_collection_stats(self) -> Dict:
        """
        コレクションの統計情報取得
        
        Returns:
            統計情報
        """
        try:
            count = self.collection.count()
            
            stats = {
                'total_documents': count,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model_name,
                'chroma_db_path': self.chroma_db_path
            }
            
            # サンプル文書を取得して詳細情報
            if count > 0:
                sample = self.collection.peek(limit=1)
                if sample['ids']:
                    sample_metadata = self._restore_metadata(sample['metadatas'][0])
                    stats['sample_metadata_keys'] = list(sample_metadata.keys())
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {'error': str(e)}
    
    def delete_collection(self):
        """コレクションの削除"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def reset_collection(self):
        """コレクションのリセット（削除して再作成）"""
        try:
            # 既存コレクション削除
            try:
                self.delete_collection()
            except:
                pass
            
            # 新規コレクション作成
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Medical literature chunks for RAG system"}
            )
            logger.info(f"Reset collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise


def main():
    """テスト実行"""
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    vector_store = MedicalVectorStore()
    
    # 処理済みデータの読み込み
    processed_data_file = "data/processed_chunks.json"
    
    if not Path(processed_data_file).exists():
        print(f"Processed data file not found: {processed_data_file}")
        print("Run text_processor.py first to generate processed chunks.")
        return
    
    with open(processed_data_file, 'r', encoding='utf-8') as f:
        chunk_data = json.load(f)
    
    print(f"\n=== Loading Medical Literature into Vector Store ===")
    print(f"Total chunks to process: {len(chunk_data.get('chunks', []))}")
    
    # ベクトルストアにデータ追加
    added_count = vector_store.add_documents(chunk_data)
    
    # 統計情報表示
    stats = vector_store.get_collection_stats()
    print(f"\n=== Vector Store Statistics ===")
    print(f"Total documents: {stats.get('total_documents', 0)}")
    print(f"Collection name: {stats.get('collection_name')}")
    print(f"Embedding model: {stats.get('embedding_model')}")
    print(f"Storage path: {stats.get('chroma_db_path')}")
    
    # テスト検索
    print(f"\n=== Test Search ===")
    test_queries = [
        "COVID-19 treatment options",
        "artificial intelligence medical diagnosis",
        "cancer immunotherapy"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.search_similar(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['similarity_score']:.3f}")
            print(f"     Title: {result['metadata'].get('title', 'N/A')}")
            print(f"     Content: {result['content'][:100]}...")
    
    print(f"\n✅ Vector store setup completed successfully!")


if __name__ == "__main__":
    main()