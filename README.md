# Medical RAG System (Minimal Configuration)

## 概要

ローカル環境で動作する軽量な医療文献検索・回答生成システムです。
<img width="1920" height="1414" alt="image" src="https://github.com/user-attachments/assets/5fc32e6a-b8d1-4de9-a2e3-bc4b30b5dc8d" />

**主な特徴:**
- **技術スタック**: ChromaDB + Ollama (Llama-3.1-8B) + Streamlit
- **データソース**: PubMed医学文献
- **機能**: 医療文献の検索・回答生成

## 前提

### システム要件
- **Python**: 3.10+
- **OS**: Linux/Mac/Windows（WSLにて動作検証済）
- **メモリ**: 8GB以上推奨
- **ストレージ**: 10GB以上

### 事前準備
- Python 3.10+がインストール済み
- Ollamaがインストール済み
- インターネット接続（初回セットアップ時）

## インストール方法

### 1. リポジトリのクローン
```bash
git clone https://github.com/okayamah/medical-rag.git
cd medical-rag
```

### 2. 仮想環境の作成と有効化
```bash
# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate     # Windows
```

### 3. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 4. Ollamaモデルの取得
```bash
# llama3.1モデルをダウンロード（初回のみ）
ollama pull llama3.1:8b-instruct-q4_0

# 利用可能なモデル一覧の確認
ollama list

# モデルの詳細情報を確認
ollama show llama3.1:8b-instruct-q4_0
```

## 実行方法

### 1. Ollamaサーバーの起動
```bash
# Ollamaサーバーを起動
ollama serve
```

### 2. Streamlitアプリの起動
```bash
# 仮想環境をアクティベート後
streamlit run src/app.py
```

### 3. 動作確認
- ブラウザで http://localhost:8501 にアクセス
- アプリケーションが正常に表示されることを確認

## 注意事項

### システム動作確認
```bash
# Ollamaサーバーが起動しているか確認
lsof -i :11434

# API経由での動作確認
curl http://localhost:11434/api/tags
```

### 医療ドメイン特有の注意点
- **医学的助言ではない**: 本システムは医学的助言を目的としていません
- **参考文献の確認**: 回答には必ず参考文献を確認してください
- **医療従事者への相談**: 医療に関する判断は医療従事者にご相談ください

## ライセンス

このプロジェクトは MIT License の下で公開されています。

## 補足：システムフロー
```mermaid
graph TD
    %% ユーザーとメインアプリ
    User[👨‍⚕️ ユーザー] 
    App[🖥️ Streamlit WebUI]
    PubMed[📚 PubMed API<br/>医学論文データベース]
    LLM[🤖 Ollama]
    ChromaDB[(ChromaDB<br/>ベクトルデータベース)]
    
    %% Store プロセス
    subgraph Store ["Store - データ格納プロセス"]
        TextExtract[テキスト抽出<br/>論文抄録・本文]
        Chunk[チャンク分割<br/>500文字単位]
        Embedding[ベクトル化<br/>SentenceTransformers]
    end
    
    %% Retrieve プロセス
    subgraph Retrieve ["Retrieve - 検索プロセス"]
        Question[日本語質問<br/>例: 糖尿病の治療法は？]
        Translation[英語翻訳<br/>Llama-3.1による翻訳]
        QueryEmbedding[質問のエンベディング]
        VectorSearch[ベクトル検索<br/>類似文書の検索]
        Results[検索結果<br/>関連論文チャンク]
    end
    
    %% Augment プロセス
    subgraph Augment ["Augment - 拡張プロセス"]
        PromptBuild[プロンプト構築<br/>質問 + 検索結果 + 医療専用テンプレート]
    end
    
    %% Generate プロセス
    subgraph Generate ["Generate - 生成プロセス"]
        LLMCall[LLM呼び出し<br/>Ollama + Llama-3.1-8B]
        Response[回答生成<br/>エビデンスベースの医療回答]
    end
    
    %% データフロー（Store）
    PubMed ---> TextExtract
    TextExtract --> Chunk
    Chunk --> Embedding
    Embedding --> ChromaDB
    
    %% ユーザーフロー
    User -->|質問入力| App
    App -->|質問転送| Question
    
    %% Retrieveフロー
    Question --> Translation
    Translation -.-> LLM
    Translation -->|英語クエリ| QueryEmbedding
    QueryEmbedding --> VectorSearch
    QueryEmbedding -.-> LLM
    ChromaDB -.->|参照| VectorSearch
    VectorSearch --> Results
    
    %% Augmentフロー
    Results --> PromptBuild
    Question -.->|元の質問| PromptBuild
    
    %% Generateフロー
    PromptBuild --> LLMCall
    LLMCall -.-> LLM
    LLMCall --> Response
    
    %% 回答返却
    Response -->|回答返却| App
    App -->|回答表示| User
    
    %% スタイリング
    classDef storeBox fill:#fff7ed,stroke:#f97316,stroke-width:2px
    classDef retrieveBox fill:#f0fdf4,stroke:#16a34a,stroke-width:2px
    classDef augmentBox fill:#eff6ff,stroke:#1e40af,stroke-width:2px
    classDef generateBox fill:#f3e8ff,stroke:#7c3aed,stroke-width:2px
    classDef userBox fill:#e0f2fe,stroke:#0277bd,stroke-width:3px
    classDef dbBox fill:#fef3c7,stroke:#f59e0b,stroke-width:2px
    
    class User,App,PubMed userBox
    class ChromaDB,LLM dbBox
```
