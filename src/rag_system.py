"""
Medical RAG System Core Implementation
医療RAGシステムのコア実装（検索・回答生成機能）
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from datetime import datetime
from pathlib import Path

from src.vector_store import MedicalVectorStore
from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """RAG回答のデータクラス"""
    query: str
    answer: str
    source_documents: List[Dict]
    search_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    metadata: Dict


class MedicalRAGSystem:
    """医療RAGシステムのメインクラス"""
    
    def __init__(self):
        """RAGシステムの初期化"""
        self.vector_store = None
        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self.ollama_model = settings.OLLAMA_MODEL
        
        # プロンプトテンプレート
        self.system_prompt = """あなたは医療文献に基づいて回答する医療AIアシスタントです。

【重要な制約】
1. 提供された医学文献の情報のみを使用して回答してください
2. 医学的診断や治療の助言は行わず、文献情報の要約に留めてください
3. 不確実な情報については明確に「文献では言及されていません」と述べてください
4. 回答の最後に参考文献のPMIDを必ず記載してください

【回答形式】
- 簡潔で分かりやすい日本語で回答
- 根拠となる文献情報を明示
- 医療従事者への相談を推奨する文言を含める

以下の医学文献を参考にして、ユーザーの質問に回答してください：

{context}

質問: {question}

回答:"""
        
        self._initialize_components()
    
    def _initialize_components(self):
        """コンポーネントの初期化"""
        try:
            # ベクトルストアの初期化
            self.vector_store = MedicalVectorStore()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def translate_query_to_english(self, query: str) -> Tuple[str, float]:
        """
        日本語クエリを英語に翻訳
        
        Args:
            query: 日本語の検索クエリ
            
        Returns:
            (英訳クエリ, 翻訳時間(ms))
        """
        start_time = datetime.now()
        
        try:
            # Ollamaで医学専門用語を考慮した翻訳
            translation_prompt = f"""Translate this Japanese medical query to English. Use precise medical terminology. Give only the English translation, no explanations.

Japanese: {query}
English:"""
            
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": translation_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 50,
                    }
                },
                timeout=15
            )
            
            response.raise_for_status()
            result = response.json()
            
            english_query = result.get('response', '').strip()
            
            # 翻訳結果のクリーニング
            english_query = english_query.replace('\n', ' ').strip()
            
            # 冗長な説明を削除（最初の文のみ取得）
            if '.' in english_query:
                english_query = english_query.split('.')[0].strip()
            
            if not english_query:
                english_query = query  # フォールバック
            
            translation_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Translated '{query}' -> '{english_query}' ({translation_time:.2f}ms)")
            
            return english_query, translation_time
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            translation_time = (datetime.now() - start_time).total_seconds() * 1000
            return query, translation_time  # フォールバック

    def search_relevant_documents(self, 
                                query: str, 
                                top_k: int = None,
                                similarity_threshold: float = None) -> Tuple[List[Dict], float, str]:
        """
        関連文書の検索（翻訳対応）
        
        Args:
            query: 検索クエリ
            top_k: 取得する文書数
            similarity_threshold: 類似度閾値
            
        Returns:
            (関連文書リスト, 検索時間(ms), 使用されたクエリ)
        """
        start_time = datetime.now()
        
        try:
            top_k = top_k or settings.SEARCH_TOP_K
            similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
            
            # 日本語クエリを英語に翻訳
            english_query, translation_time = self.translate_query_to_english(query)
            
            # ベクトル検索実行
            search_results = self.vector_store.search_similar(
                query=english_query,
                top_k=top_k * 2  # 閾値フィルタリング用に多めに取得
            )
            
            # 類似度閾値でフィルタリング
            filtered_results = [
                result for result in search_results 
                if result['similarity_score'] >= similarity_threshold
            ]
            
            # 上位k件に制限
            final_results = filtered_results[:top_k]
            
            search_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Found {len(final_results)} relevant documents (total: {search_time:.2f}ms, translation: {translation_time:.2f}ms)")
            return final_results, search_time, english_query
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return [], 0.0, query
    
    def _format_context_from_documents(self, documents: List[Dict]) -> str:
        """
        文書リストからコンテキスト文字列を生成
        
        Args:
            documents: 検索結果文書リスト
            
        Returns:
            フォーマット済みコンテキスト
        """
        if not documents:
            return "関連する医学文献が見つかりませんでした。"
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc['metadata']
            content = doc['content']
            
            # 文書情報のフォーマット
            doc_info = f"【文献{i}】"
            if metadata.get('title'):
                doc_info += f" {metadata['title']}"
            if metadata.get('authors'):
                authors = metadata['authors'][:2] if isinstance(metadata['authors'], list) else []
                if authors:
                    doc_info += f" (著者: {', '.join(authors)})"
            if metadata.get('journal'):
                doc_info += f" - {metadata['journal']}"
            if metadata.get('publication_date'):
                doc_info += f" ({metadata['publication_date']})"
            if metadata.get('pmid'):
                doc_info += f" [PMID: {metadata['pmid']}]"
            
            doc_info += f"\n類似度: {doc['similarity_score']:.3f}\n"
            doc_info += f"内容: {content}\n"
            
            context_parts.append(doc_info)
        
        return "\n" + "="*80 + "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> Tuple[str, float]:
        """
        Ollamaを使用した回答生成
        
        Args:
            query: ユーザーの質問
            context: 検索コンテキスト
            
        Returns:
            (生成された回答, 生成時間(ms))
        """
        start_time = datetime.now()
        
        try:
            # プロンプト構築
            prompt = self.system_prompt.format(
                context=context,
                question=query
            )
            
            # Ollama APIへのリクエスト
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # 一貫性重視
                        "top_p": 0.9,
                        "num_predict": 1000,  # 最大トークン数
                    }
                },
                timeout=settings.GENERATION_TIMEOUT
            )
            
            response.raise_for_status()
            result = response.json()
            
            answer = result.get('response', '回答の生成に失敗しました。')
            generation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            logger.info(f"Answer generated (generation: {generation_time:.2f}ms)")
            return answer, generation_time
            
        except requests.exceptions.Timeout:
            logger.error("Answer generation timed out")
            return "回答生成がタイムアウトしました。より簡潔な質問で再試行してください。", 0.0
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama server")
            return "Ollamaサーバーに接続できません。サーバーが起動しているか確認してください。", 0.0
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"回答生成中にエラーが発生しました: {str(e)}", 0.0
    
    def query(self, 
             user_query: str,
             top_k: int = None,
             similarity_threshold: float = None) -> RAGResponse:
        """
        RAGクエリの実行（検索 + 生成）
        
        Args:
            user_query: ユーザーの質問
            top_k: 検索する文書数
            similarity_threshold: 類似度閾値
            
        Returns:
            RAG回答オブジェクト
        """
        total_start_time = datetime.now()
        
        logger.info(f"Processing RAG query: {user_query}")
        
        # 1. 関連文書検索
        relevant_docs, search_time, english_query = self.search_relevant_documents(
            query=user_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        if not relevant_docs:
            # 関連文書が見つからない場合
            total_time = (datetime.now() - total_start_time).total_seconds() * 1000
            return RAGResponse(
                query=user_query,
                answer="申し訳ございませんが、ご質問に関連する医学文献が見つかりませんでした。異なるキーワードで再度お試しください。",
                source_documents=[],
                search_time_ms=search_time,
                generation_time_ms=0.0,
                total_time_ms=total_time,
                metadata={
                    'similarity_threshold': similarity_threshold or settings.SIMILARITY_THRESHOLD,
                    'requested_top_k': top_k or settings.SEARCH_TOP_K,
                    'found_documents': 0,
                    'english_query': english_query
                }
            )
        
        # 2. コンテキスト構築
        context = self._format_context_from_documents(relevant_docs)
        
        # 3. 回答生成
        answer, generation_time = self.generate_answer(user_query, context)
        
        # 4. 結果構築
        total_time = (datetime.now() - total_start_time).total_seconds() * 1000
        
        return RAGResponse(
            query=user_query,
            answer=answer,
            source_documents=relevant_docs,
            search_time_ms=search_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            metadata={
                'similarity_threshold': similarity_threshold or settings.SIMILARITY_THRESHOLD,
                'requested_top_k': top_k or settings.SEARCH_TOP_K,
                'found_documents': len(relevant_docs),
                'ollama_model': self.ollama_model,
                'english_query': english_query
            }
        )
    
    def get_system_status(self) -> Dict:
        """
        システム状態の確認
        
        Returns:
            システム状態情報
        """
        status = {
            'vector_store': False,
            'ollama_server': False,
            'total_documents': 0,
            'available_models': []
        }
        
        try:
            # ベクトルストア状態確認
            if self.vector_store:
                stats = self.vector_store.get_collection_stats()
                status['vector_store'] = True
                status['total_documents'] = stats.get('total_documents', 0)
        except Exception as e:
            logger.error(f"Vector store check failed: {e}")
        
        try:
            # Ollama サーバー状態確認
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                status['ollama_server'] = True
                models_data = response.json()
                status['available_models'] = [model['name'] for model in models_data.get('models', [])]
        except Exception as e:
            logger.error(f"Ollama server check failed: {e}")
        
        return status


def main():
    """テスト実行"""
    # ログ設定
    logging.basicConfig(level=logging.INFO)
    
    # RAGシステム初期化
    rag_system = MedicalRAGSystem()
    
    # システム状態確認
    status = rag_system.get_system_status()
    print("🔍 Medical RAG System Status")
    print("=" * 40)
    print(f"Vector Store: {'✅' if status['vector_store'] else '❌'}")
    print(f"Ollama Server: {'✅' if status['ollama_server'] else '❌'}")
    print(f"Total Documents: {status['total_documents']}")
    print(f"Available Models: {status['available_models']}")
    
    if not (status['vector_store'] and status['ollama_server']):
        print("\n❌ System not ready. Check vector store and Ollama server.")
        return
    
    # テストクエリ実行
    test_queries = [
        "COVID-19の治療法について教えてください",
        "AIによる医療診断の精度はどの程度ですか？",
        "がん免疫療法の最新の研究成果は？"
    ]
    
    print(f"\n🧪 Running Test Queries")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n【クエリ {i}】{query}")
        print("-" * 50)
        
        response = rag_system.query(query)
        
        print(f"回答: {response.answer[:200]}...")
        print(f"検索時間: {response.search_time_ms:.2f}ms")
        print(f"生成時間: {response.generation_time_ms:.2f}ms")
        print(f"総時間: {response.total_time_ms:.2f}ms")
        print(f"参考文献数: {len(response.source_documents)}")
        
        if response.source_documents:
            print("参考文献:")
            for j, doc in enumerate(response.source_documents[:2], 1):
                metadata = doc['metadata']
                title = metadata.get('title', 'Unknown')[:50]
                pmid = metadata.get('pmid', 'Unknown')
                score = doc['similarity_score']
                print(f"  {j}. {title}... [PMID: {pmid}] (類似度: {score:.3f})")
    
    print(f"\n✅ RAG System test completed!")


if __name__ == "__main__":
    main()