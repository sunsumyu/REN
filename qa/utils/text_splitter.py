# -*- coding: utf-8 -*-
import re

def semantic_slice_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    基于语义和标点的递归字符切片算法（Recursive Character Text Splitting）。
    
    采用逐级降级的策略：先尝试双换行(\n\n)，再单换行(\n)，然后句号、叹号、问号，接着分号、逗号，
    实在无法在 chunk_size 限制内存下整块的，最后才触发机械硬切。
    并保证各个生成的切片块(chunk)之间保留指定数量(overlap)的字符作为上下文重叠。
    """
    separators = ["\n\n", "\n", "。", "！", "？", "；", "，", ""]
    
    def _split_to_pieces(text: str, sep_index: int) -> list[str]:
        if len(text) <= chunk_size or sep_index >= len(separators):
            return [text]
            
        separator = separators[sep_index]
        if not separator:
            # 最后的硬切降级
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
        # 对于标点符号，采用正则切分以保留标点在句子末尾
        if separator in ["。", "！", "？", "；", "，"]:
            parts = re.split(f"({separator})", text)
            splits = [parts[i] + (parts[i+1] if i+1 < len(parts) else "") for i in range(0, len(parts), 2) if parts[i] or (i+1 < len(parts) and parts[i+1])]
        else:
            raw_splits = text.split(separator)
            splits = []
            for idx, s in enumerate(raw_splits):
                if idx < len(raw_splits) - 1:
                    splits.append(s + separator)
                else:
                    splits.append(s)
            
        pieces = []
        for s in splits:
            if not s:
                continue
            if len(s) > chunk_size:
                pieces.extend(_split_to_pieces(s, sep_index + 1))
            else:
                pieces.append(s)
        return pieces

    # 1. 按照优先级分割成满足大小限制的最小基本句子块
    raw_pieces = _split_to_pieces(text, 0)
    
    # 2. 合并基本块并应用 overlap（重叠区）
    chunks = []
    current_pieces = []
    current_length = 0
    
    for piece in raw_pieces:
        if current_length + len(piece) > chunk_size and current_length > 0:
            # 当前大块已经塞满，先保存它
            chunks.append("".join(current_pieces))
            
            # 计算重叠区：尽量从后面完整的块去塞入 overlap
            overlap_pieces = []
            overlap_length = 0
            for p in reversed(current_pieces):
                if overlap_length + len(p) <= overlap:
                    overlap_pieces.insert(0, p)
                    overlap_length += len(p)
                else:
                    break
            
            # 如果一整个句子都塞不进 overlap，只好硬切最后一句的尾巴
            if overlap_length == 0 and len(current_pieces) > 0 and overlap > 0:
                forced_overlap = current_pieces[-1][-overlap:]
                if forced_overlap:
                    overlap_pieces = [forced_overlap]
                    overlap_length = len(forced_overlap)
                
            current_pieces = overlap_pieces
            current_length = overlap_length
            
        current_pieces.append(piece)
        current_length += len(piece)
        
    if current_pieces:
        chunks.append("".join(current_pieces))
        
    # 清理一些因切片导致的过短碎片
    return [c for c in chunks if len(c.strip()) > 10]
