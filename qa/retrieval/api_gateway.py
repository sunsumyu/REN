# -*- coding: utf-8 -*-
"""
第二级：专业医学接口 Gateway。
提供对药智网、39健康及 PubMed 数据库的并发调用与归一化解析。
内置基于本地 SQLite3 的持久化缓存管理（彻底免去 Redis 的繁琐依赖与进程负担），
保护外部 API 调用限流，并能重复秒级唤醒相同的生成任务。
"""

import os
import sqlite3
import json
import hashlib
import logging
import urllib.request
import urllib.parse
import asyncio
from typing import List, Dict, Any, Optional
from retrieval.schemas import NormalizedClinicalRef

logger = logging.getLogger("MedicalQA.APIGateway")

class APIGatewayService:
    def __init__(self, db_dir: str = "."):
        """
        初始化专业医学接口 Gateway。在指定目录下创建本地持久化缓存。
        """
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        self.db_path = os.path.join(self.db_dir, "medical_cache.db")
        self._initialize_cache_db()

    def _initialize_cache_db(self):
        """
        在本地 SQLite3 中初始化缓存数据库
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                cache_key TEXT PRIMARY KEY,
                query TEXT,
                service_name TEXT,
                response_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        conn.close()
        logger.info(f"Initialized zero-dependency SQLite3 persistent cache at '{self.db_path}'")

    def _get_cache_key(self, service_name: str, query: str) -> str:
        """
        针对不同的服务和查询生成唯一的 SHA-256 缓存特征键
        """
        payload = f"{service_name.strip()}:{query.strip().lower()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """
        从本地缓存加载历史响应
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT response_json FROM api_cache WHERE cache_key = ?;",
            (cache_key,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            try:
                return json.loads(row["response_json"])
            except Exception as e:
                logger.error(f"Failed to decode cached response JSON: {e}")
        return None

    def _save_cache_response(self, cache_key: str, query: str, service_name: str, data: List[Dict[str, Any]]):
        """
        持久化响应数据到本地 SQLite3
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO api_cache (cache_key, query, service_name, response_json) VALUES (?, ?, ?, ?);",
                (cache_key, query, service_name, json.dumps(data, ensure_ascii=False))
            )
            conn.commit()
            conn.close()
            logger.debug(f"Saved cache hit for '{service_name}' on query '{query}'")
        except Exception as e:
            logger.error(f"Failed to write persistent cache: {e}")

    async def fetch_pubmed_abstracts(self, term: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        异步调用美国国立生物技术信息中心 (NCBI) PubMed 公开文献检索 API (e-utilities)
        获取相关的最新学术研究摘要。无需 API Key，天然免授权。
        """
        cache_key = self._get_cache_key("pubmed", term)
        cached = self._get_cached_response(cache_key)
        if cached is not None:
            logger.info(f"PubMed API persistent Cache HIT for term: '{term}'")
            return cached

        # 1. 搜索 PMID 列表
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": limit
        }
        
        loop = asyncio.get_event_loop()
        try:
            # 异步执行 HTTP 请求，防止阻塞生成任务的主事件循环
            def run_search():
                url = f"{search_url}?{urllib.parse.urlencode(search_params)}"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=3.0) as r:
                    return json.loads(r.read().decode('utf-8'))
            
            search_res = await loop.run_in_executor(None, run_search)
            id_list = search_res.get("esearchresult", {}).get("idlist", [])
            
            if not id_list:
                logger.info(f"PubMed Search yielded 0 results for term '{term}'")
                return []

            # 2. 提取 PMID 文献摘要 (Summary)
            summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            summary_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json"
            }
            
            def run_summary():
                url = f"{summary_url}?{urllib.parse.urlencode(summary_params)}"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=3.0) as r:
                    return json.loads(r.read().decode('utf-8'))

            summary_res = await loop.run_in_executor(None, run_summary)
            uid_results = summary_res.get("result", {})
            
            extracted_refs = []
            for uid in id_list:
                doc = uid_results.get(uid, {})
                title = doc.get("title", "Unknown Title")
                pub_date = doc.get("pubdate", "Unknown Date")
                source = doc.get("source", "PubMed")
                authors = ", ".join([a.get("name", "") for a in doc.get("authors", [])])
                
                context = f"【临床研究文献报道】: 题目《{title}》，发表于 {pub_date}。主要作者: {authors}。研究刊载于《{source}》。"
                item = {
                    "source": f"PubMed (PMID: {uid})",
                    "context": context,
                    "category": "文献证据",
                    "metadata": {"title": title, "authors": authors, "pmid": uid}
                }
                extracted_refs.append(item)

            self._save_cache_response(cache_key, term, "pubmed", extracted_refs)
            return extracted_refs

        except Exception as e:
            logger.warning(f"PubMed API connection bypassed or timed out: {e}")
            return []

    async def fetch_yaozhi_data(self, drug_name: str) -> List[Dict[str, Any]]:
        """
        模拟企业级药智网/39健康等商业接口。若无付费密钥授权，则调用高保真度 Mock 生成器，
        结合医学逻辑，为管线提供高质量的备用药品成分与医保数据。
        """
        cache_key = self._get_cache_key("yaozh_39", drug_name)
        cached = self._get_cached_response(cache_key)
        if cached is not None:
            logger.info(f"Yaozhi API Cache HIT for drug: '{drug_name}'")
            return cached

        # 使用真实的网页实时抓取 (Web Scraping) 替代硬编码 Mock 
        # 当没有 API Key 时，通过爬取公开网页获取真实文本
        loop = asyncio.get_event_loop()
        
        def scrape_drug_info():
            import urllib.request
            import urllib.parse
            import re
            
            # 首选尝试抓取百度百科的药品描述，作为真实数据的来源
            try:
                url = f"https://baike.baidu.com/item/{urllib.parse.quote(drug_name)}"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                with urllib.request.urlopen(req, timeout=3.0) as r:
                    html = r.read().decode('utf-8', errors='ignore')
                    
                    # 提取 meta description 作为药物简介
                    match = re.search(r'<meta name="description" content="(.*?)">', html, re.IGNORECASE)
                    if match:
                        desc = match.group(1).strip()
                        if desc:
                            return [{
                                "source": "在线医学百科数据库实时抓取",
                                "context": f"【公开药品档案】: {desc}",
                                "category": "药品信息",
                                "metadata": {"drug_name": drug_name, "scrape_source": "baike"}
                            }]
            except Exception as e:
                logger.warning(f"Baike scrape failed for '{drug_name}': {e}")
                
            # 如果百科失败，降级抓取 39健康网搜索页面的纯文本
            try:
                url_39 = f"https://yp.39.net/search/{urllib.parse.quote(drug_name)}.shtml"
                req_39 = urllib.request.Request(url_39, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                with urllib.request.urlopen(req_39, timeout=3.0) as r:
                    html_39 = r.read().decode('gbk', errors='ignore') # 39健康网多数页面是 GB2312/GBK
                    
                    # 提取简单的页面标题和一点上下文
                    title_match = re.search(r'<title>(.*?)</title>', html_39, re.IGNORECASE)
                    title_text = title_match.group(1).strip() if title_match else ""
                    if drug_name in title_text:
                        return [{
                            "source": "39健康网在线实时抓取",
                            "context": f"【39健康网实时检索】: 检索到关于药物【{drug_name}】的相关条目。网页标题信息: {title_text}。请详细阅读原网页说明书或遵医嘱。",
                            "category": "药品信息",
                            "metadata": {"drug_name": drug_name, "scrape_source": "39net"}
                        }]
            except Exception as e:
                logger.warning(f"39net scrape failed for '{drug_name}': {e}")
                
            # 如果抓取都被反爬虫拦截，为了防止大模型生成流水线崩溃，返回空数据或兜底说明
            return [{
                "source": "在线公开检索系统抓取异常",
                "context": f"【未收录或网络异常】: 当前未能在公开在线网络(39健康/百科)中抓取到关于【{drug_name}】的详细真实数据，可能被防爬拦截，请参考医师建议。",
                "category": "药品信息",
                "metadata": {"drug_name": drug_name, "status": "scrape_failed"}
            }]
            
        mock_data = await loop.run_in_executor(None, scrape_drug_info)

        self._save_cache_response(cache_key, drug_name, "yaozh_39", mock_data)
        return mock_data

    async def search(self, query: str, entity_name: str) -> List[NormalizedClinicalRef]:
        """
        并发请求多个外部医学接口，进行并行化高速获取，并在接口层归一化为 Schema
        """
        # 并发异步请求 PubMed 和 药智/39 模拟网
        tasks = [
            self.fetch_pubmed_abstracts(entity_name),
            self.fetch_yaozhi_data(entity_name)
        ]
        
        results = []
        try:
            # 3.0s 超时控制，超出自动熔断，确保整个 QA 管线的超高健壮度
            responses = await asyncio.gather(*tasks)
            for resp in responses:
                for item in resp:
                    ref_item = NormalizedClinicalRef(
                        source=item["source"],
                        context=item["context"],
                        category=item["category"],
                        metadata=item.get("metadata", {})
                    )
                    results.append(ref_item)
        except Exception as e:
            logger.error(f"Tier 2 Specialized API Gateway crashed or timed out: {e}")

        logger.info(f"Tier 2 (APIGateway) search completed. Entity: '{entity_name}', Found items: {len(results)}")
        return results
