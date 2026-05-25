import asyncio
import json
import os
import sys
import logging

# Reconfigure stdout/stderr to use UTF-8 under Windows to prevent GBK UnicodeEncodeErrors when printing emojis
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from api_client import APIClient
from pipeline import MedicalQAPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "pipeline_execution.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("MedicalQA.Main")

async def run_generator():
    logger.info("Initializing Medical QA Multi-Round Dataset Generator...")
    
    # Check if API Key is set
    import config
    if not config.LLM_API_KEY:
        print("="*60)
        print("警告: 您的 LLM_API_KEY 环境变量未设置！")
        print("请在 .env 文件中配置您的大模型 API 密钥。例如:")
        print("LLM_API_KEY=Bearer 您的密钥")
        print("="*60)
        # Continue anyway, let the client try to send requests (maybe it's a public interface)
        
    client = APIClient()
    pipeline = MedicalQAPipeline(client)
    
    try:
        # Generate the multi-round trajectory dataset
        dataset = await pipeline.generate_multi_round_dataset()
        
        # Save output JSON
        output_file = os.path.join(os.path.dirname(__file__), "medical_qa_dataset.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
            
        print("\n" + "="*60)
        print("🎉 医药知识图谱多视角多轮问答数据集生成成功！")
        print(f"📁 数据集保存路径: {output_file}")
        print(f"📝 运行执行日志: {os.path.join(os.path.dirname(__file__), 'pipeline_execution.log')}")
        
        # Display a quick summary of the generated conversation
        print("\n=== 对话数据概览 ===")
        history_list = dataset.get('history')
        if not history_list:
            history_list = [dataset]
        print(f"第一轮问题 Q: {history_list[0].get('Q')}")
        print(f"共生成轮数: {len(dataset.get('history', [])) + 1} 轮")
        print(f"最终输出总结长度: {len(dataset.get('summary', ''))} 字符")
        print("="*60)
        
    except Exception as e:
        logger.critical(f"Pipeline execution encountered an unhandled exception: {e}", exc_info=True)
        print(f"\n❌ 生成失败: {e}", file=sys.stderr)
        print("请检查网络连通性、API Key 配置或日志文件 pipeline_execution.log", file=sys.stderr)
    finally:
        await client.close()

if __name__ == "__main__":
    # Handle event loop for Windows python environment safely
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_generator())
