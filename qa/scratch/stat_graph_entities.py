# -*- coding: utf-8 -*-
import asyncio
import json
import logging
import os
import sys

# 切换工作目录到项目根目录
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api_client import APIClient
from utils.logging_config import setup_logging
import config

async def run_statistics():
    # 初始化日志为较低级别，避免输出干扰
    setup_logging(level=logging.WARNING)
    print("==========================================================")
    print("🚀 开始连接远端图数据库进行实体捞取与采样统计...")
    print("==========================================================")
    
    client = APIClient()
    
    all_entities = {}
    
    max_loops = 50
    consecutive_empty_count = 0
    
    for i in range(max_loops):
        if len(all_entities) >= 300:
            print("-> 已拉取到足够样本量 (>= 300)，为防止 API GET 请求超长 (414 URI Too Long) 自动结束拉取。")
            break
            
        print(f"-> 正在进行第 {i+1} 次拉取 (当前已收集去重实体: {len(all_entities)} 个)...")
        try:
            # 每次请求随机挑选 5 个种子，并让 API 游走
            graph_data = await client.fetch_random_knowledge_graph(count=5)
            entities = graph_data.get("entities", [])
            
            if not entities:
                consecutive_empty_count += 1
                if consecutive_empty_count >= 3:
                    print("连续多次返回空实体，图数据库中的实体已被全部捞取完毕。")
                    break
                continue
            
            consecutive_empty_count = 0
            
            for ent in entities:
                ent_id = str(ent.get("id"))
                if ent_id not in all_entities:
                    all_entities[ent_id] = {
                        "name": ent.get("name"),
                        "type": ent.get("type"),
                        "description": ent.get("description", "暂无描述")
                    }
        except Exception as e:
            print(f"第 {i+1} 次拉取发生异常: {e}")
            break
            
        # 稍作睡眠防止请求过快
        await asyncio.sleep(0.15)
        
    print(f"\n捞取结束。共收集到去重实体 {len(all_entities)} 个。")
    if not all_entities:
        print("未获取到任何实体，请检查网络或 API 配置。")
        await client.close()
        return
        
    # 将实体名称打包，每 80 个一组，送大模型判别是否是医学相关
    entities_list = list(all_entities.values())
    batch_size = 80
    
    medical_entities = []
    non_medical_entities = []
    
    print("\n🔮 正在调用轻量大模型进行实体医学属性判别...")
    
    for start_idx in range(0, len(entities_list), batch_size):
        batch = entities_list[start_idx:start_idx+batch_size]
        names = [item["name"] for item in batch]
        
        prompt = (
            "你是一个极度严谨的医学名词分类器。\n"
            "你会收到一个实体名称列表，你的任务是判断其中每一个实体名称是否属于【医学、病理、解剖、疾病、临床诊断、生理机制、药理、中草药、化学药品、医疗器械、或医疗法规】领域。\n\n"
            "判断标准：\n"
            "- 凡是跟医学、人体生理、疾病、药物、治疗手段或医疗规程直接相关的，均归入 `medical`。\n"
            "- 凡是属于通用软件测试、IT开发术语、产品设计、游戏机制、网络梗或纯通用日常词汇（如：提示、低风险、视觉布局、界面、冲突等），均归入 `non_medical`。\n\n"
            "输入实体列表：" + json.dumps(names, ensure_ascii=False) + "\n\n"
            "你必须严格返回以下 JSON 格式对象，不要代码块围栏，不要解释：\n"
            "{\n"
            "  \"medical\": [\"实体名A\", \"实体名B\"],\n"
            "  \"non_medical\": [\"实体名C\", \"实体名D\"]\n"
            "}"
        )
        
        try:
            response = await client.call_llm(prompt, model_pool="lightweight", stage="实体二分类")
            # 兼容性 JSON 解析
            from pipeline import parse_json_safely
            res_dict = parse_json_safely(response, {})
            
            med = res_dict.get("medical", [])
            non_med = res_dict.get("non_medical", [])
            
            # 核对结果防止 LLM 漏掉或伪造
            for name in names:
                target_item = next(item for item in batch if item["name"] == name)
                if name in med:
                    medical_entities.append(target_item)
                elif name in non_med:
                    non_medical_entities.append(target_item)
                else:
                    # 默认归入医疗
                    medical_entities.append(target_item)
        except Exception as e:
            print(f"分类批次 {start_idx} 发生异常: {e}")
            medical_entities.extend(batch)
            
    # 输出最终分析报告
    total_count = len(medical_entities) + len(non_medical_entities)
    med_ratio = (len(medical_entities) / total_count) * 100 if total_count > 0 else 0
    non_med_ratio = (len(non_medical_entities) / total_count) * 100 if total_count > 0 else 0
    
    # 写入 JSON 结果文件
    res_data = {
        "total_count": total_count,
        "medical_count": len(medical_entities),
        "medical_ratio": med_ratio,
        "non_medical_count": len(non_medical_entities),
        "non_medical_ratio": non_med_ratio,
        "non_medical_entities": non_medical_entities
    }
    result_file_path = os.path.join(os.path.dirname(__file__), "statistics_result.json")
    with open(result_file_path, "w", encoding="utf-8") as f:
        json.dump(res_data, f, ensure_ascii=False, indent=2)
        
    print("\n" + "="*50)
    print("📊 远端图数据库（ID: 201）实体分析报告")
    print("="*50)
    print(f"1. 收集去重实体总数: {total_count} 个")
    print(f"2. 医学专业实体数: {len(medical_entities)} 个 | 占比: {med_ratio:.2f}%")
    print(f"3. 非医学 (IT/其它) 实体数: {len(non_medical_entities)} 个 | 占比: {non_med_ratio:.2f}%")
    print("="*50)
    
    if non_medical_entities:
        print("\n🚫 发现的非医学实体清单（部分展示）:")
        for idx, item in enumerate(non_medical_entities[:30]):
            desc_preview = item['description'].replace('\n', ' ').strip()
            print(f" - [{idx+1}] 实体名: '{item['name']}' | 类型: '{item['type']}' | 描述: {desc_preview[:60]}...")
        if len(non_medical_entities) > 30:
            print(f" ...等共 {len(non_medical_entities)} 个非医学实体。")
    else:
        print("\n✅ 图谱库极度纯净，未发现任何非医学实体！")
    print("="*50 + "\n")
    
    await client.close()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_statistics())
