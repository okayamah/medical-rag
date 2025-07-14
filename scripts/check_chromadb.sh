#!/bin/bash
# ChromaDB SQLite データ確認スクリプト

DB_PATH="chroma_db/chroma.sqlite3"

echo "🔍 ChromaDB Data Inspector"
echo "========================="

# 基本統計
echo "📊 Basic Statistics:"
echo "Documents: $(sqlite3 $DB_PATH 'SELECT COUNT(*) FROM embeddings;')"
echo "Metadata entries: $(sqlite3 $DB_PATH 'SELECT COUNT(*) FROM embedding_metadata;')"
echo "Collections: $(sqlite3 $DB_PATH 'SELECT COUNT(*) FROM collections;')"
echo

# メタデータキー一覧
echo "🔑 Available Metadata Keys:"
sqlite3 $DB_PATH -cmd ".mode list" "SELECT DISTINCT key FROM embedding_metadata ORDER BY key;" | sed 's/^/  - /'
echo

# サンプル文書の詳細
echo "📄 Sample Document (ID=1):"
sqlite3 $DB_PATH -cmd ".mode column" -cmd ".headers on" \
  "SELECT key, SUBSTR(string_value, 1, 50) as value_preview 
   FROM embedding_metadata 
   WHERE id=1 AND key IN ('pmid','title','journal','publication_date')
   ORDER BY key;"
echo

# PMID別文書数
echo "📚 PMIDs by chunk count:"
sqlite3 $DB_PATH -cmd ".mode column" -cmd ".headers on" \
  "SELECT 
     SUBSTR(string_value, 1, 10) as pmid,
     COUNT(*) as chunks
   FROM embedding_metadata 
   WHERE key='pmid' 
   GROUP BY string_value 
   ORDER BY COUNT(*) DESC ;"
echo

# 文書数
echo "📚 PMIDs by doc count:"
sqlite3 $DB_PATH -cmd ".mode column" -cmd ".headers on" \
  "SELECT 
     COUNT(DISTINCT SUBSTR(string_value, 1, 10)) as docs
   FROM embedding_metadata 
   WHERE key='pmid' ;"

echo "💡 Usage examples:"
echo "  sqlite3 $DB_PATH '.tables'"
echo "  sqlite3 $DB_PATH 'SELECT * FROM collections;'"
echo "  sqlite3 $DB_PATH -cmd '.headers on' -cmd '.mode column' 'SELECT key, COUNT(*) FROM embedding_metadata GROUP BY key;'"