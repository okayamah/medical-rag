"""
Medical RAG System - Streamlit Application
医療RAGシステム統合Streamlitアプリケーション
"""

import streamlit as st
import time
import json
from datetime import datetime
from typing import Dict, List
import logging

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

def format_response_display(response: RAGResponse):
    """レスポンスの表示フォーマット"""
    
    # 基本情報
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("検索時間", f"{response.search_time_ms:.0f}ms")
    with col2:
        st.metric("生成時間", f"{response.generation_time_ms:.0f}ms")
    with col3:
        st.metric("参考文献数", len(response.source_documents))
    
    # 英訳クエリ表示
    if 'english_query' in response.metadata:
        st.info(f"🔄 **検索に使用された英訳**: {response.metadata['english_query']}")
    
    # 回答表示
    st.markdown("### 💬 回答")
    st.markdown(response.answer)
    
    # 参考文献表示
    if response.source_documents:
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

def save_query_to_history(query: str, response: RAGResponse):
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
    
    # 検索実行
    col1, col2 = st.columns([1, 4])
    
    with col1:
        search_button = st.button("🔍 検索・回答生成", type="primary")
    
    with col2:
        if st.button("🗑️ 履歴をクリア"):
            st.session_state.query_history = []
            st.success("履歴をクリアしました")
    
    # クエリ実行
    if search_button and user_query.strip():
        
        if not st.session_state.rag_system:
            st.error("RAGシステムが初期化されていません。")
            return
        
        with st.spinner("文献を検索・分析中..."):
            start_time = time.time()
            
            try:
                # RAGクエリ実行
                response = st.session_state.rag_system.query(
                    user_query=user_query.strip(),
                    top_k=top_k,
                    similarity_threshold=similarity_threshold
                )
                
                execution_time = time.time() - start_time
                
                # 結果表示
                st.markdown("---")
                st.markdown(f"### 📋 質問: {user_query}")
                
                # パフォーマンス情報
                st.caption(f"⏱️ 総実行時間: {execution_time:.2f}秒")
                
                # レスポンス表示
                format_response_display(response)
                
                # 履歴に保存
                save_query_to_history(user_query, response)
                
            except Exception as e:
                st.error(f"❌ エラーが発生しました: {str(e)}")
                logger.error(f"Query execution failed: {e}")
    
    elif search_button:
        logger.info(f"user_query={user_query}")
        st.warning("質問を入力してください。")
    
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