"""
PubMed Data Collector
PubMed E-utilities APIを使用した医学文献データ収集システム
"""

import requests
import time
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PubMedArticle:
    """PubMed文献データのデータクラス"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: str
    doi: Optional[str] = None
    keywords: List[str] = None
    mesh_terms: List[str] = None
    publication_types: List[str] = None
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'pmid': self.pmid,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'journal': self.journal,
            'publication_date': self.publication_date,
            'doi': self.doi,
            'keywords': self.keywords or [],
            'mesh_terms': self.mesh_terms or [],
            'publication_types': self.publication_types or []
        }


class PubMedCollector:
    """PubMed APIクライアント"""
    
    def __init__(self, email: str = None, tool_name: str = "MedicalRAG"):
        """
        Args:
            email: 研究者のメールアドレス（API利用規約に必要）
            tool_name: ツール名
        """
        self.email = email
        self.tool_name = tool_name
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.session = requests.Session()
        
        # APIレート制限（1秒あたり3リクエスト）
        self.request_delay = 0.34  # 秒
        self.last_request_time = 0
        
    def _wait_for_rate_limit(self):
        """APIレート制限のための待機"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict) -> requests.Response:
        """APIリクエストの実行（レート制限・エラーハンドリング付き）"""
        self._wait_for_rate_limit()
        
        # 共通パラメータ追加
        if self.email:
            params['email'] = self.email
        params['tool'] = self.tool_name
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def search_articles(self, 
                       query: str, 
                       max_results: int = 500,
                       date_range: Optional[Tuple[str, str]] = None) -> List[str]:
        """
        PubMed検索を実行してPMIDリストを取得
        
        Args:
            query: 検索クエリ
            max_results: 最大取得件数
            date_range: 日付範囲 (開始日, 終了日) "YYYY/MM/DD"形式
            
        Returns:
            PMIDのリスト
        """
        logger.info(f"Searching PubMed for: {query}")
        
        url = f"{self.base_url}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'xml',
            'sort': 'relevance'
        }
        
        # 日付範囲指定
        if date_range:
            start_date, end_date = date_range
            params['datetype'] = 'pdat'
            params['mindate'] = start_date
            params['maxdate'] = end_date
        
        try:
            response = self._make_request(url, params)
            
            # XMLパース
            root = ET.fromstring(response.content)
            pmids = []
            
            for id_elem in root.findall('.//Id'):
                pmids.append(id_elem.text)
                
            logger.info(f"Found {len(pmids)} articles")
            return pmids
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def fetch_article_details(self, pmids: List[str]) -> List[PubMedArticle]:
        """
        PMIDリストから文献詳細情報を取得
        
        Args:
            pmids: PMIDのリスト
            
        Returns:
            PubMedArticleのリスト
        """
        if not pmids:
            return []
            
        logger.info(f"Fetching details for {len(pmids)} articles")
        
        url = f"{self.base_url}/efetch.fcgi"
        
        # バッチ処理（一度に最大200件）
        batch_size = 200
        all_articles = []
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            pmid_str = ','.join(batch_pmids)
            
            params = {
                'db': 'pubmed',
                'id': pmid_str,
                'retmode': 'xml',
                'rettype': 'abstract'
            }
            
            try:
                response = self._make_request(url, params)
                articles = self._parse_pubmed_xml(response.content)
                all_articles.extend(articles)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(pmids)-1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Failed to fetch batch {i//batch_size + 1}: {e}")
                continue
        
        logger.info(f"Successfully fetched {len(all_articles)} articles")
        return all_articles
    
    def _parse_pubmed_xml(self, xml_content: bytes) -> List[PubMedArticle]:
        """PubMed XMLレスポンスをパース"""
        articles = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article_elem in root.findall('.//PubmedArticle'):
                try:
                    article = self._extract_article_data(article_elem)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue
                    
        except ET.ParseError as e:
            logger.error(f"XML parsing failed: {e}")
            
        return articles
    
    def _extract_article_data(self, article_elem) -> Optional[PubMedArticle]:
        """単一文献のデータ抽出"""
        try:
            # PMID
            pmid_elem = article_elem.find('.//PMID')
            if pmid_elem is None:
                return None
            pmid = pmid_elem.text
            
            # タイトル
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            # アブストラクト
            abstract_parts = []
            for abstract_elem in article_elem.findall('.//AbstractText'):
                label = abstract_elem.get('Label', '')
                text = abstract_elem.text or ''
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = ' '.join(abstract_parts)
            
            # 著者
            authors = []
            for author_elem in article_elem.findall('.//Author'):
                lastname = author_elem.find('LastName')
                forename = author_elem.find('ForeName')
                if lastname is not None:
                    author_name = lastname.text
                    if forename is not None:
                        author_name = f"{forename.text} {author_name}"
                    authors.append(author_name)
            
            # ジャーナル
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            # 発行日
            pub_date_elem = article_elem.find('.//PubDate')
            publication_date = self._extract_publication_date(pub_date_elem)
            
            # DOI
            doi = None
            for article_id in article_elem.findall('.//ArticleId'):
                if article_id.get('IdType') == 'doi':
                    doi = article_id.text
                    break
            
            # MeSH用語
            mesh_terms = []
            for mesh_elem in article_elem.findall('.//MeshHeading/DescriptorName'):
                mesh_terms.append(mesh_elem.text)
            
            # キーワード
            keywords = []
            for keyword_elem in article_elem.findall('.//Keyword'):
                keywords.append(keyword_elem.text)
            
            # 発行タイプ
            publication_types = []
            for pubtype_elem in article_elem.findall('.//PublicationType'):
                publication_types.append(pubtype_elem.text)
            
            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=publication_date,
                doi=doi,
                keywords=keywords,
                mesh_terms=mesh_terms,
                publication_types=publication_types
            )
            
        except Exception as e:
            logger.error(f"Error extracting article data: {e}")
            return None
    
    def _extract_publication_date(self, pub_date_elem) -> str:
        """発行日の抽出"""
        if pub_date_elem is None:
            return ""
        
        year_elem = pub_date_elem.find('Year')
        month_elem = pub_date_elem.find('Month')
        day_elem = pub_date_elem.find('Day')
        
        year = year_elem.text if year_elem is not None else ""
        month = month_elem.text if month_elem is not None else "01"
        day = day_elem.text if day_elem is not None else "01"
        
        try:
            # 月名を数字に変換
            month_names = {
                'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
            }
            if month in month_names:
                month = month_names[month]
            
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        except:
            return year if year else ""
    
    def collect_medical_data(self, 
                           search_terms: List[str],
                           max_results_per_term: int = 100,
                           output_file: str = "data/sample_medical_data.json") -> Dict:
        """
        医療分野のデータ収集実行
        
        Args:
            search_terms: 検索キーワードリスト
            max_results_per_term: 各キーワードあたりの最大取得件数
            output_file: 出力ファイルパス
            
        Returns:
            収集統計情報
        """
        logger.info("Starting medical data collection")
        
        all_articles = []
        collection_stats = {
            'start_time': datetime.now().isoformat(),
            'search_terms': search_terms,
            'total_articles': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'articles_per_term': {}
        }
        
        for term in search_terms:
            logger.info(f"Processing search term: {term}")
            
            try:
                # 検索実行
                pmids = self.search_articles(
                    query=term,
                    max_results=max_results_per_term,
                    date_range=("2020/01/01", "2025/12/31")  # 最近5年間
                )
                
                if pmids:
                    # 詳細情報取得
                    articles = self.fetch_article_details(pmids)
                    all_articles.extend(articles)
                    
                    collection_stats['successful_searches'] += 1
                    collection_stats['articles_per_term'][term] = len(articles)
                    
                    logger.info(f"Collected {len(articles)} articles for '{term}'")
                else:
                    collection_stats['failed_searches'] += 1
                    collection_stats['articles_per_term'][term] = 0
                    logger.warning(f"No articles found for '{term}'")
                    
            except Exception as e:
                logger.error(f"Failed to process '{term}': {e}")
                collection_stats['failed_searches'] += 1
                collection_stats['articles_per_term'][term] = 0
        
        # 重複除去（PMID基準）
        unique_articles = {}
        for article in all_articles:
            unique_articles[article.pmid] = article
        
        final_articles = list(unique_articles.values())
        collection_stats['total_articles'] = len(final_articles)
        collection_stats['end_time'] = datetime.now().isoformat()
        
        # ファイル保存
        self._save_to_json(final_articles, collection_stats, output_file)
        
        logger.info(f"Data collection completed. {len(final_articles)} unique articles saved to {output_file}")
        return collection_stats
    
    def _save_to_json(self, articles: List[PubMedArticle], stats: Dict, output_file: str):
        """JSON形式でデータ保存"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': stats,
            'articles': [article.to_dict() for article in articles]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# 医療分野の検索キーワード
MEDICAL_SEARCH_TERMS = [
#   // 季節・環境関連の健康問題
  "heat stroke", "heat exhaustion", "heatwave", "dehydration", "sunburn",
  "cold", "flu", "influenza", "seasonal allergies", "hay fever",
  "winter depression", "seasonal affective disorder", "air pollution",
  "humidity", "temperature stress",
  
#   // 日常的な症状・不調
  "headache", "migraine", "back pain", "neck pain", "shoulder pain",
  "fatigue", "insomnia", "sleep disorders", "stress", "anxiety",
  "depression", "mood swings", "digestive problems", "constipation",
  "diarrhea", "nausea", "dizziness", "vertigo", "muscle pain",
  
#   // 生活習慣病・慢性疾患
  "diabetes", "high blood pressure", "cholesterol", "obesity", "weight loss",
  "heart disease", "stroke prevention", "arthritis", "osteoporosis",
  "metabolic syndrome", "fatty liver", "kidney stones", "gout",
  
#   // 感染症・ウイルス
  "COVID-19", "common cold", "pneumonia", "food poisoning", "stomach flu",
  "urinary tract infection", "skin infection", "athlete's foot",
  "herpes", "shingles", "vaccine", "vaccination", "immunity",
  
#   // 女性・男性特有の健康問題
  "menstruation", "menopause", "pregnancy", "fertility", "contraception",
  "breast health", "cervical cancer screening", "prostate health",
  "erectile dysfunction", "testosterone", "hormone therapy",
  
#   // 子供・高齢者の健康
  "child development", "growth", "immunization schedule", "teething",
  "childhood obesity", "ADHD", "autism", "elderly care", "dementia",
  "fall prevention", "medication management", "age-related diseases",
  
#   // 栄養・食事
  "nutrition", "vitamins", "minerals", "supplements", "protein",
  "calcium", "iron deficiency", "vitamin D", "omega-3", "fiber",
  "sugar intake", "salt reduction", "healthy eating", "diet", "fasting",
  
#   // 運動・フィットネス
  "exercise", "physical activity", "walking", "jogging", "strength training",
  "flexibility", "yoga", "sports injuries", "muscle strain", "joint health",
  "fitness", "cardio", "weight training", "stretching", "recovery",
  
#   // メンタルヘルス
  "mental health", "stress management", "relaxation", "meditation",
  "work-life balance", "burnout", "social anxiety", "panic attacks",
  "grief", "trauma", "self-care", "mindfulness", "therapy",
  
#   // 皮膚・美容健康
  "acne", "eczema", "dry skin", "aging skin", "wrinkles", "sun protection",
  "hair loss", "dandruff", "nail health", "cosmetics safety",
  "skincare routine", "moisturizer", "sunscreen",
  
#   // 眼・耳・口の健康
  "eye strain", "vision problems", "dry eyes", "contact lenses",
  "hearing loss", "tinnitus", "ear infection", "dental health",
  "tooth decay", "gum disease", "oral hygiene", "bad breath",
  
#   // 現代病・デジタル関連
  "computer vision syndrome", "text neck", "digital eye strain",
  "screen time", "blue light", "smartphone addiction", "gaming disorder",
  "repetitive strain injury", "carpal tunnel syndrome"
]

# // 日本語版キーワード（一般人向け）
MEDICAL_SEARCH_TERMS_ja = [
#   // 季節・環境関連の健康問題
  "熱中症", "熱射病", "脱水症状", "日焼け", "夏バテ",
  "風邪", "インフルエンザ", "花粉症", "アレルギー", "鼻炎",
  "冬季うつ", "季節性感情障害", "大気汚染", "湿度", "寒暖差疲労",
  
#   // 日常的な症状・不調
  "頭痛", "偏頭痛", "腰痛", "肩こり", "首の痛み",
  "疲労", "不眠症", "睡眠障害", "ストレス", "不安",
  "うつ病", "気分の落ち込み", "消化不良", "便秘",
  "下痢", "吐き気", "めまい", "立ちくらみ", "筋肉痛",
  
#   // 生活習慣病・慢性疾患
  "糖尿病", "高血圧", "コレステロール", "肥満", "ダイエット",
  "心臓病", "脳梗塞予防", "関節炎", "骨粗鬆症",
  "メタボリックシンドローム", "脂肪肝", "腎結石", "痛風",
  
#   // 感染症・ウイルス
  "新型コロナ", "普通感冒", "肺炎", "食中毒", "胃腸炎",
  "膀胱炎", "皮膚感染", "水虫", "口唇ヘルペス",
  "帯状疱疹", "ワクチン", "予防接種", "免疫力",
  
#   // 女性・男性特有の健康問題
  "生理", "更年期", "妊娠", "不妊", "避妊",
  "乳がん検診", "子宮がん検診", "前立腺", "ED",
  "男性ホルモン", "ホルモン補充療法", "PMS",
  
#   // 子供・高齢者の健康
  "子供の発達", "成長", "予防接種スケジュール", "歯が生える",
  "小児肥満", "ADHD", "自閉症", "介護", "認知症",
  "転倒予防", "薬の管理", "加齢による病気",
  
#   // 栄養・食事
  "栄養", "ビタミン", "ミネラル", "サプリメント", "タンパク質",
  "カルシウム", "鉄分不足", "ビタミンD", "オメガ3", "食物繊維",
  "糖分摂取", "減塩", "健康的な食事", "食事制限", "断食",
  
#   // 運動・フィットネス
  "運動", "身体活動", "ウォーキング", "ジョギング", "筋トレ",
  "柔軟性", "ヨガ", "スポーツ障害", "肉離れ", "関節の健康",
  "フィットネス", "有酸素運動", "筋力トレーニング", "ストレッチ", "回復",
  
#   // メンタルヘルス
  "精神的健康", "ストレス解消", "リラックス", "瞑想",
  "ワークライフバランス", "燃え尽き症候群", "社会不安", "パニック発作",
  "悲しみ", "トラウマ", "セルフケア", "マインドフルネス", "カウンセリング",
  
#   // 皮膚・美容健康
  "ニキビ", "湿疹", "乾燥肌", "肌の老化", "しわ", "紫外線対策",
  "抜け毛", "フケ", "爪の健康", "化粧品の安全性",
  "スキンケア", "保湿", "日焼け止め",
  
#   // 眼・耳・口の健康
  "眼精疲労", "視力低下", "ドライアイ", "コンタクトレンズ",
  "難聴", "耳鳴り", "中耳炎", "歯の健康",
  "虫歯", "歯周病", "口腔ケア", "口臭",
  
#   // 現代病・デジタル関連
  "VDT症候群", "スマホ首", "デジタル眼精疲労",
  "スクリーンタイム", "ブルーライト", "スマホ依存", "ゲーム障害",
  "反復性運動損傷", "手根管症候群", "テクノストレス"
]


def main():
    """メイン実行関数"""
    import os
    from config.settings import settings
    
    # 設定から値を取得
    email = settings.PUBMED_EMAIL or os.getenv('PUBMED_EMAIL')
    if not email:
        logger.warning("PUBMED_EMAIL not set. Some API features may be limited.")
    
    # データ収集実行
    collector = PubMedCollector(email=email)
    
    stats = collector.collect_medical_data(
        search_terms=MEDICAL_SEARCH_TERMS,
        max_results_per_term=50,  # テスト用に制限
        output_file="data/sample_data.json"
    )
    
    print("\n=== Collection Statistics ===")
    print(f"Total articles: {stats['total_articles']}")
    print(f"Successful searches: {stats['successful_searches']}")
    print(f"Failed searches: {stats['failed_searches']}")
    print("\nArticles per search term:")
    for term, count in stats['articles_per_term'].items():
        print(f"  {term}: {count}")


if __name__ == "__main__":
    main()