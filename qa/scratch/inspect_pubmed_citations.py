# -*- coding: utf-8 -*-
import sqlite3
import json
import os

def main():
    db_path = "qa_datasets.db"
    if not os.path.exists(db_path):
        print(f"❌ Database '{db_path}' not found.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # 查找最新的生成记录
        cursor.execute("SELECT id, query, response_json FROM datasets ORDER BY id DESC LIMIT 5;")
        rows = cursor.fetchall()
        if not rows:
            print("ℹ️ No generated datasets found in DB yet.")
            return

        print(f"Checking the last {len(rows)} generated dataset items for PubMed citations:\n")
        for idx, row in enumerate(rows):
            row_id = row["id"]
            query = row["query"]
            response_json = row["response_json"]

            print(f"[{idx+1}] ID: {row_id} | Query: '{query}'")
            try:
                data = json.loads(response_json)
            except Exception as e:
                print(f"  ❌ Failed to parse JSON: {e}")
                continue

            # 1. 检查 refs 中是否有 PubMed references
            refs = data.get("refs", [])
            pubmed_refs_in_refs = [r for r in refs if "PubMed" in r.get("source", "")]
            print(f"  - Injected refs has {len(pubmed_refs_in_refs)} PubMed items.")
            for r in pubmed_refs_in_refs:
                print(f"    * Source: {r.get('source')}")

            # 2. 检查大模型生成的各个切面回答（planners）中是否引用了 PubMed PMID
            planners = data.get("planners", [])
            found_citations_in_planners = []
            for p in planners:
                planner_name = p.get("planner", "")
                answer_text = p.get("answer", "")
                
                # 检查回答中是否含有 PMID 字符
                if "PMID:" in answer_text or "pmid:" in answer_text.lower():
                    found_citations_in_planners.append(planner_name)
                    
            if found_citations_in_planners:
                print(f"  - ✅ Citation HIT in Planners: Model referenced PubMed in these facets: {found_citations_in_planners}")
            else:
                print("  - ❌ No PubMed citation found in planners.")

            # 3. 检查最终综合凝练的 answer summary 中是否有 PMID 引用
            summary = data.get("summary", "")
            if "PMID:" in summary or "pmid:" in summary.lower():
                print("  - ✅ Citation HIT in Summary: Model referenced PubMed in the synthesized summary!")
                # 提取包含 PMID: 的上下文行
                for line in summary.split("\n"):
                    if "pmid" in line.lower():
                        print(f"    * Citation context: {line.strip()}")
            else:
                print("  - ❌ No PubMed citation found in summary.")
                
            print("-" * 60)

    except Exception as e:
        print(f"❌ Error querying database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
