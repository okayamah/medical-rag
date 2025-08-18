"""
Medical RAG System - Streamlit Application
医療RAGシステム統合Streamlitアプリケーション
"""

import streamlit as st
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging
import requests

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_system import MedicalRAGSystem, RAGResponse
from config.settings import settings

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ページ設定
st.set_page_config(
    page_title="Medical RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# セッション状態の初期化
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = ""
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""
if 'rag_response' not in st.session_state:
    st.session_state.rag_response = None
if 'llm_response' not in st.session_state:
    st.session_state.llm_response = None

def initialize_rag_system():
    """RAGシステムの初期化"""
    try:
        with st.spinner("RAGシステムを初期化しています..."):
            rag_system = MedicalRAGSystem()
            status = rag_system.get_system_status()
            
            if status['vector_store'] and status['ollama_server']:
                st.session_state.rag_system = rag_system
                st.session_state.system_initialized = True
                st.success("✅ RAGシステムが正常に初期化されました")
                return status
            else:
                st.error("❌ システムの初期化に失敗しました")
                return status
                
    except Exception as e:
        st.error(f"❌ 初期化エラー: {str(e)}")
        return None

def display_system_status(status: Dict):
    """システム状態の表示"""
    st.sidebar.markdown("### 🔍 システム状態")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric(
            "ベクトルストア",
            "✅" if status['vector_store'] else "❌",
            delta=None
        )
        
    with col2:
        st.metric(
            "Ollamaサーバー", 
            "✅" if status['ollama_server'] else "❌",
            delta=None
        )
    
    st.sidebar.metric("文書数", f"{status['total_documents']:,}件")
    
    if status['available_models']:
        st.sidebar.markdown("**利用可能モデル:**")
        for model in status['available_models']:
            st.sidebar.text(f"• {model}")

def display_search_settings():
    """検索設定の表示"""
    st.sidebar.markdown("### ⚙️ 検索設定")
    
    # 検索パラメータ設定
    top_k = st.sidebar.slider(
        "取得文書数",
        min_value=1,
        max_value=10,
        value=settings.SEARCH_TOP_K,
        help="検索で取得する関連文書の最大数"
    )
    
    similarity_threshold = st.sidebar.slider(
        "類似度閾値",
        min_value=0.0,
        max_value=1.0,
        value=settings.SIMILARITY_THRESHOLD,
        step=0.05,
        help="文書を関連ありと判定する最小類似度"
    )
    
    return top_k, similarity_threshold

def display_query_examples():
    """クエリ例の表示"""
    st.sidebar.markdown("### 💡 質問例")
    
    examples = [
        "COVID-19の治療法について教えてください",
        "AIによる医療診断の精度はどの程度ですか？",
        "がん免疫療法の最新の研究成果は？",
        "遠隔医療の効果と課題について",
        "機械学習を用いた創薬研究の現状"
    ]
    
    for i, example in enumerate(examples):
        if st.sidebar.button(f"📝 {example[:20]}...", key=f"example_{i}"):
            st.session_state.selected_query = example
            st.rerun()

def format_response_display(response: RAGResponse, response_type: str = "RAG"):
    """レスポンスの表示フォーマット"""
    
    # 基本情報
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("検索時間", f"{response.search_time_ms:.0f}ms" if hasattr(response, 'search_time_ms') else "N/A")
    with col2:
        st.metric("生成時間", f"{response.generation_time_ms:.0f}ms" if hasattr(response, 'generation_time_ms') else "N/A")
    with col3:
        if response_type == "RAG" and hasattr(response, 'source_documents'):
            st.metric("参考文献数", len(response.source_documents))
        else:
            st.metric("参考文献数", "N/A")
    
    # 英訳クエリ表示
    if hasattr(response, 'metadata') and response.metadata and 'english_query' in response.metadata:
        st.info(f"🔄 **検索に使用された英訳**: {response.metadata['english_query']}")
    
    # 回答表示
    st.markdown("### 💬 回答")
    if hasattr(response, 'answer'):
        st.markdown(response.answer)
    else:
        st.markdown(response)
    
    # 参考文献表示（RAGのみ）
    if response_type == "RAG" and hasattr(response, 'source_documents') and response.source_documents:
        st.markdown("### 📚 参考文献")
        
        for i, doc in enumerate(response.source_documents, 1):
            with st.expander(f"📄 文献 {i} (類似度: {doc['similarity_score']:.3f})"):
                metadata = doc['metadata']
                
                # メタデータ表示
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**タイトル**: {metadata.get('title', 'N/A')}")
                    if metadata.get('authors'):
                        authors = metadata['authors'][:3] if isinstance(metadata['authors'], list) else []
                        if authors:
                            st.markdown(f"**著者**: {', '.join(authors)}")
                    st.markdown(f"**ジャーナル**: {metadata.get('journal', 'N/A')}")
                
                with col2:
                    st.markdown(f"**発行日**: {metadata.get('publication_date', 'N/A')}")
                    if metadata.get('pmid'):
                        pmid_url = f"https://pubmed.ncbi.nlm.nih.gov/{metadata['pmid']}"
                        st.markdown(f"**PMID**: [{metadata['pmid']}]({pmid_url})")
                    if metadata.get('doi'):
                        doi_url = f"https://doi.org/{metadata['doi']}"
                        st.markdown(f"**DOI**: [{metadata['doi']}]({doi_url})")
                
                # コンテンツ表示
                st.markdown("**内容抜粋**:")
                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                
                # MeSH用語とキーワード
                if metadata.get('mesh_terms'):
                    mesh_terms = metadata['mesh_terms'][:5] if isinstance(metadata['mesh_terms'], list) else []
                    if mesh_terms:
                        st.markdown(f"**MeSH用語**: {', '.join(mesh_terms)}")
                
                if metadata.get('keywords'):
                    keywords = metadata['keywords'][:5] if isinstance(metadata['keywords'], list) else []
                    if keywords:
                        st.markdown(f"**キーワード**: {', '.join(keywords)}")

def display_query_history():
    """クエリ履歴の表示"""
    if st.session_state.query_history:
        st.sidebar.markdown("### 📝 最近の質問")
        
        for i, (query, timestamp) in enumerate(reversed(st.session_state.query_history[-5:])):
            time_str = timestamp.strftime("%H:%M")
            if st.sidebar.button(f"{time_str} {query[:15]}...", key=f"history_{i}"):
                st.session_state.selected_query = query
                st.rerun()

def save_query_to_history(query: str, response=None):
    """クエリ履歴への保存"""
    timestamp = datetime.now()
    st.session_state.query_history.append((query, timestamp))
    
    # 履歴の制限（最大50件）
    if len(st.session_state.query_history) > 50:
        st.session_state.query_history = st.session_state.query_history[-50:]

def display_medical_disclaimer():
    """医療免責事項の表示"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚠️ 重要な注意事項")
    st.sidebar.warning(
        "このシステムは研究・教育目的のものです。\n\n"
        "医学的な診断や治療の助言を行うものではありません。\n\n"
        "医療に関する判断は必ず医療従事者にご相談ください。"
    )

class LLMResponse:
    """直接LLM回答のデータクラス"""
    def __init__(self, query: str, answer: str, generation_time_ms: float):
        self.query = query
        self.answer = answer
        self.generation_time_ms = generation_time_ms
        self.search_time_ms = 0.0  # LLMは検索しないため
        self.source_documents = []  # LLMは参考文献なし
        self.metadata = {}

def get_llm_response(query: str) -> LLMResponse:
    """直接LLM回答を取得"""
    start_time = datetime.now()
    
    try:
        # 医療専用プロンプト
        medical_prompt = f"""あなたは医療知識に特化したAIアシスタントです。

【重要な制約】
1. 医学的診断や治療の助言は行わず、一般的な医療情報の提供に留めてください
2. 不確実な情報については明確に「一般的な情報です」と述べてください
3. 回答の最後に医療従事者への相談を推奨する文言を含めてください

【回答形式】
- 簡潔で分かりやすい日本語で回答
- 一般的な医療知識を基にした情報提供
- 最新の研究結果や文献情報は不明であることを明示

質問: {query}

回答:"""
        
        # Ollama / Langfuse セットアップ
        from ollama import Client
        from langfuse.decorators import langfuse_context, observe
        langfuse_context.configure(
        secret_key=settings.LANGFUSE_SECRET_KEY,
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        host=settings.LANGFUSE_HOST
        )
        # Ollama APIへのリクエスト
        client = Client(host=settings.OLLAMA_BASE_URL)
        @observe(as_type="generation")
        def generate(prompt, model_id=settings.OLLAMA_MODEL, mode=None, system="あなたは優秀な日本人アシスタントです。特に指示が無い場合は、常に日本語で回答してください。"):
            response = client.generate(model=model_id, prompt=prompt, format=mode, system=system)
            return response
        response = generate(prompt=medical_prompt)
        answer = response.get('response', '回答の生成に失敗しました。')
        generation_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"LLM direct response generated ({generation_time:.2f}ms)")
        return LLMResponse(query, answer, generation_time)
        
    except requests.exceptions.Timeout:
        logger.error("LLM response generation timed out")
        return LLMResponse(query, "回答生成がタイムアウトしました。より簡潔な質問で再試行してください。", 0.0)
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Ollama server for LLM response")
        return LLMResponse(query, "Ollamaサーバーに接続できません。サーバーが起動しているか確認してください。", 0.0)
    except Exception as e:
        logger.error(f"LLM response generation failed: {e}")
        return LLMResponse(query, f"回答生成中にエラーが発生しました: {str(e)}", 0.0)

def display_comparison_view(user_query: str, top_k: int, similarity_threshold: float):
    """比較表示ビュー"""
    st.markdown("### 🔍 クエリ実行")
    
    # テキスト入力の有無でボタンの活性化を制御
    has_query = bool(user_query.strip())
    
    # ボタンレイアウト
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        rag_button = st.button("🔍 RAG回答", type="primary", disabled=not has_query)
    
    with col2:
        llm_button = st.button("🤖 LLM回答", type="secondary", disabled=not has_query)
    
    with col3:
        both_button = st.button("🔄 両方同時実行", type="secondary", disabled=not has_query)
    
    # クエリ実行
    if has_query:
        if rag_button or both_button:
            execute_rag_query(user_query, top_k, similarity_threshold)
        
        if llm_button or both_button:
            execute_llm_query(user_query)
    
    # 結果表示
    if st.session_state.rag_response or st.session_state.llm_response:
        st.markdown("---")
        st.markdown(f"### 📋 質問: {user_query}")
        
        # タブ表示
        tab1, tab2, tab3 = st.tabs(["🔍 RAG回答", "🤖 LLM回答", "🔄 比較"])
        
        with tab1:
            if st.session_state.rag_response:
                format_response_display(st.session_state.rag_response, "RAG")
            else:
                st.info("RAG回答を実行してください")
        
        with tab2:
            if st.session_state.llm_response:
                format_response_display(st.session_state.llm_response, "LLM")
            else:
                st.info("LLM回答を実行してください")
        
        with tab3:
            if st.session_state.rag_response and st.session_state.llm_response:
                display_comparison_results()
            else:
                st.info("比較には両方の回答が必要です")

def execute_rag_query(user_query: str, top_k: int, similarity_threshold: float):
    """RAGクエリ実行"""
    with st.spinner("🔍 RAG回答を生成中..."):
        try:
            response = st.session_state.rag_system.query(
                user_query=user_query.strip(),
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            st.session_state.rag_response = response
            st.success(f"✅ RAG回答が完了しました ({response.generation_time_ms:.0f}ms)")
        except Exception as e:
            st.error(f"❌ RAGエラー: {str(e)}")
            logger.error(f"RAG query execution failed: {e}")

def execute_llm_query(user_query: str):
    """直接LLMクエリ実行"""
    with st.spinner("🤖 LLM回答を生成中..."):
        try:
            response = get_llm_response(user_query)
            st.session_state.llm_response = response
            # 成功メッセージに生成時間を表示
            st.success(f"✅ LLM回答が完了しました ({response.generation_time_ms:.0f}ms)")
        except Exception as e:
            st.error(f"❌ LLMエラー: {str(e)}")
            logger.error(f"LLM query execution failed: {e}")

def display_comparison_results():
    """比較結果表示"""
    st.markdown("### 🔄 比較結果")
    
    # パフォーマンス比較
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 RAG回答")
        if hasattr(st.session_state.rag_response, 'generation_time_ms'):
            st.metric("生成時間", f"{st.session_state.rag_response.generation_time_ms:.0f}ms")
        if hasattr(st.session_state.rag_response, 'source_documents'):
            st.metric("参考文献数", len(st.session_state.rag_response.source_documents))
    
        # 回答表示
        st.markdown("### 💬 回答")
        if hasattr(st.session_state.rag_response, 'answer'):
            st.markdown(st.session_state.rag_response.answer)
        else:
            st.markdown(st.session_state.rag_response)
    
    with col2:
        st.markdown("#### 🤖 LLM回答")
        if hasattr(st.session_state.llm_response, 'generation_time_ms'):
            st.metric("生成時間", f"{st.session_state.llm_response.generation_time_ms:.0f}ms")
        else:
            st.metric("生成時間", "N/A")
        st.metric("参考文献数", "0")
    
        # 回答表示
        st.markdown("### 💬 回答")
        if hasattr(st.session_state.llm_response, 'answer'):
            st.markdown(st.session_state.llm_response.answer)
        else:
            st.markdown(st.session_state.llm_response)

def main():
    """メインアプリケーション"""
    
    # ヘッダー
    st.title("🏥 Medical RAG System")
    st.markdown("**医療文献検索・回答生成システム**")
    st.markdown("---")
    
    # システム初期化
    if not st.session_state.system_initialized:
        st.markdown("### 📋 システム初期化")
        
        if st.button("🚀 RAGシステムを開始", type="primary"):
            status = initialize_rag_system()
            if status:
                display_system_status(status)
                st.rerun()
        
        st.info("「RAGシステムを開始」ボタンをクリックしてシステムを初期化してください。")
        return
    
    # システム状態表示
    if st.session_state.rag_system:
        status = st.session_state.rag_system.get_system_status()
        display_system_status(status)
        
        # システムが正常でない場合の警告
        if not (status['vector_store'] and status['ollama_server']):
            st.error("⚠️ システムに問題があります。ChromaDBまたはOllamaサーバーの状態を確認してください。")
            return
    
    # 検索設定
    top_k, similarity_threshold = display_search_settings()
    
    # クエリ履歴表示
    display_query_history()
    
    # メインエリア
    st.markdown("### 🔍 医療文献検索")
    
    user_query = st.text_area(
        "医療に関する質問を日本語で入力してください：",
        height=100,
        placeholder="例：COVID-19の治療法について教えてください",
    )
    
    # システムチェック
    if not st.session_state.rag_system:
        st.error("RAGシステムが初期化されていません。")
        return
    
    # 比較モードで表示（常に表示）
    display_comparison_view(user_query, top_k, similarity_threshold)
    
    # 履歴に保存
    if st.session_state.rag_response or st.session_state.llm_response:
        save_query_to_history(user_query)
    
    # 履歴クリア
    if st.button("🗑️ 履歴をクリア"):
        st.session_state.query_history = []
        st.session_state.rag_response = None
        st.session_state.llm_response = None
        st.success("履歴をクリアしました")
    
    # 免責事項
    display_medical_disclaimer()
    
    # フッター
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <small>
        🔬 Medical RAG System v1.0 | 
        📚 PubMed文献データベース連携 | 
        🤖 Llama3.2-3B + ChromaDB
        </small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()