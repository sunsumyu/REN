# -*- coding: utf-8 -*-
"""
用于对已生成的 Markdown 质检差异日志进行行号升序重排与完美编号排序的脚本。
"""

import os
import re
from pathlib import Path

def sort_markdown_log(file_path):
    target_path = Path(file_path)
    if not target_path.exists():
        print(f"❌ 找不到日志文件: {target_path}")
        return

    print(f"🔄 正在读取并重排日志文件: {target_path.name} ...")
    with open(target_path, 'r', encoding='utf-8') as f:
        content = f.read()

    header_marker = "## 🔍 详细提纯记录清单"
    if header_marker not in content:
        print("❌ 未在文件中找到清单标志头，请确认文件格式。")
        return

    header_part, blocks_part = content.split(header_marker, 1)
    header_part = header_part + header_marker + "\n\n"

    # 以 --- 分割各个净化块
    blocks = blocks_part.split("\n\n---\n\n")
    cleaned_blocks = [b.strip() for b in blocks if b.strip()]

    parsed_blocks = []
    for b in cleaned_blocks:
        # 正则匹配标题 ### [编号] 行 Y | 视角: **Z**
        match = re.search(r"###\s*\[(\d+)\]\s*行\s*(\d+)\s*\|\s*视角:\s*\*\*([^*]+)\*\*", b)
        if match:
            original_idx = int(match.group(1))
            line_num = int(match.group(2))
            facet = match.group(3).strip()
            parsed_blocks.append({
                "line_num": line_num,
                "facet": facet,
                "content": b
            })
        else:
            # 如果之前已经是经过本脚本重排过的格式，进行二次匹配兼容：### [编号] (对应数据集第 Y 行) | 视角: **Z**
            match_compat = re.search(r"###\s*\[(\d+)\]\s*\(对应数据集第\s*(\d+)\s*行\)\s*\|\s*视角:\s*\*\*([^*]+)\*\*", b)
            if match_compat:
                line_num = int(match_compat.group(2))
                facet = match_compat.group(3).strip()
                parsed_blocks.append({
                    "line_num": line_num,
                    "facet": facet,
                    "content": b
                })
            else:
                parsed_blocks.append({
                    "line_num": 9999,
                    "facet": "",
                    "content": b
                })

    # 按原始 JSONL 中的条目行号（line_num）升序排列，行号相同则按视角名字母排序
    parsed_blocks.sort(key=lambda x: (x["line_num"], x["facet"]))

    # 重新装配并格式化标题
    sorted_blocks = []
    for idx, b in enumerate(parsed_blocks):
        content = b["content"]
        new_heading = f"### [{idx+1}] (对应数据集第 {b['line_num']} 行) | 视角: **{b['facet']}**"
        
        # 替换老标题为格式化后的升序新标题
        content = re.sub(r"###\s*\[\d+\].*", new_heading, content)
        sorted_blocks.append(content)

    final_content = header_part + "\n\n---\n\n".join(sorted_blocks) + "\n\n---\n\n"

    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(final_content)

    print(f"🎉 排序重排成功！已保存至 {target_path}")

if __name__ == "__main__":
    # 对当前用户指定的特定日志文件进行排序
    log_file = "d:/REN/qa/logs/purification_run_20260527_092936.md"
    sort_markdown_log(log_file)
