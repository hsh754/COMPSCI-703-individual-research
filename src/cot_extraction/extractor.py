# 推理链抽取模块
# 用于将LLM输出按规则/正则拆分为逐步推理链

# TODO: 实现正则表达式或自定义规则的推理链拆解

import re
from typing import List, Optional

def clean_step(step: str) -> Optional[str]:
    # 简单清理：去掉多余空白，过滤过短内容
    s = step.strip()
    if len(s) < 3:
        return None
    return s

def extract_cot_steps(output_text: str, prompt_type: str) -> List[str]:
    """
    根据不同的prompt类型，从模型输出中提取推理步骤
    参数：
        output_text: 模型输出的完整文本
        prompt_type: prompt类型，可选值：'simple', 'templated', 'natural'
    返回：
        有效的推理步骤列表
    """
    steps = []
    
    # 预处理：移除可能的答案部分和重复的提取步骤标记
    output_text = re.sub(r'\n[A-E]\.?$', '', output_text)
    # 移除"提取的推理步骤："等标记后的重复内容
    output_text = re.split(r'提取的推理步骤[:：]', output_text)[0]
    
    if prompt_type == 'templated':
        # 使用新的正则表达式提取步骤
        step_pattern = re.compile(r'(Step\s*\d+:[\s\S]*?)(?=(?:Step\s*\d+:)|\Z)')
        matches = list(step_pattern.finditer(output_text))

        if matches:
            seen_contents = set()
            # 对每个 Step 段做二次拆分
            for match in matches:
                raw_seg = match.group(1).strip()
                # 提取并移除 “Step N:” 前缀
                step_num_match = re.match(r'Step\s*(\d+):', raw_seg)
                seg = re.sub(r'^Step\s*\d+:', '', raw_seg).strip()

                # 第二层：按句号、分号或子弹符号拆分成更细粒度的子句
                sub_frags = re.split(r'(?<=[\.;])\s+|[\-\u2022]\s*', seg)
                for frag in sub_frags:
                    frag = frag.strip()
                    if not frag:
                        continue
                    # 规范化去重
                    normalized = ' '.join(frag.split())
                    if normalized not in seen_contents:
                        steps.append(frag)
                        seen_contents.add(normalized)

        # 如果没有找到标准Step格式，尝试其他结构化标记
        if not steps:
            # 尝试匹配其他Step格式（可能没有编号）
            alt_step_pattern = re.compile(
                r'(?:Step|第.*?步|步骤)\s*:?\s*([^。\n]*[。\n])',
                re.DOTALL
            )
            matches = alt_step_pattern.finditer(output_text)
            seen_contents = set()
            for match in matches:
                content = match.group(1).strip()
                normalized_content = ' '.join(content.split())
                if content and normalized_content not in seen_contents:
                    steps.append(content)
                    seen_contents.add(normalized_content)
            
            # 如果还是没找到，尝试数字列表格式
            if not steps:
                number_list_pattern = re.compile(
                    r'(?:^|\n)\s*(?:\d+\.|\(\d+\))\s*([^\n]*(?:\n(?!\d+\.|\(\d+\))[^\n]*)*)',
                    re.MULTILINE
                )
                matches = number_list_pattern.finditer(output_text)
                for match in matches:
                    content = match.group(1).strip()
                    normalized_content = ' '.join(content.split())
                    if content and normalized_content not in seen_contents:
                        steps.append(content)
                        seen_contents.add(normalized_content)
                
            # 最后尝试匹配缩进或换行分隔的段落
            if not steps:
                # 移除空行和只包含数字的行
                lines = [line.strip() for line in output_text.split('\n')
                        if line.strip() and not re.match(r'^\s*\d+\s*$', line.strip())]
                seen_contents = set()
                for line in lines:
                    if len(line) > 5:  # 过滤过短的行
                        normalized_line = ' '.join(line.split())
                        if normalized_line not in seen_contents:
                            steps.append(line)
                            seen_contents.add(normalized_line)

    elif prompt_type == 'natural':
        # 1. 先把行首的 “- ” 或 “• ” 规范成 “Also,” 标记
        output_text = re.sub(r'^[\-\u2022]\s+', 'Also, ', output_text, flags=re.MULTILINE)

        # 2. 扩展 marker 列表，覆盖更多口语化表达
        markers = [
            r'First(?:ly)?,?', r'To\s+start\s+with,?', r'Firstly,?',
            r'Then(?:\s+we)?,?', r'Next,?', r'After\s+that,?',
            r'Also,?', r'Additionally,?', r'Moreover,?',
            r'Furthermore,?', r'Therefore,?', r'Finally,?',
            r'Lastly,?'
        ]
        marker_re = re.compile('(' + '|'.join(markers) + ')', re.IGNORECASE)

        # 3. 第一轮按 marker 拆分
        parts = marker_re.split(output_text)

        # 4. 对每个 marker 片段，再用宽松句子边界拆细
        seen_contents = set()
        for i in range(1, len(parts), 2):
            fragment = (parts[i] + parts[i + 1]).strip()
            # 按 “. ”、“? ”、“! ”、”; ” 之后的大写字母拆分
            sentences = re.split(r'(?<=[\.\?!;])\s+(?=[A-Z])', fragment)
            for s in sentences:
                step = s.strip()
                if len(step) < 4:
                    continue
                norm = ' '.join(step.split())
                if norm not in seen_contents:
                    steps.append(step)
                    seen_contents.add(norm)

        # 5. fallback：如果还是没拆出任何步骤，按普通句子边界再试一次
        if not steps:
            for s in re.split(r'(?<=[\.\?!])\s+(?=[A-Z])', output_text):
                step = s.strip()
                if len(step) < 4:
                    continue
                norm = ' '.join(step.split())
                if norm not in seen_contents:
                    steps.append(step)
                    seen_contents.add(norm)
                
    else:  # simple或其他格式
        # 1. 尝试句子分割
        sentences = re.split(r'[.!?]\s+', output_text)
        seen_contents = set()
        for s in sentences:
            s = s.strip()
            normalized_s = ' '.join(s.split())
            if s and normalized_s not in seen_contents:
                steps.append(s)
                seen_contents.add(normalized_s)
        
        # 2. 如果句子分割效果不好，尝试换行分割
        if len(steps) <= 1:
            lines = [s.strip() for s in output_text.split('\n')]
            seen_contents = set()
            for line in lines:
                normalized_line = ' '.join(line.split())
                if line and normalized_line not in seen_contents:
                    steps.append(line)
                    seen_contents.add(normalized_line)
    
    # 清理和验证每个步骤
    valid_steps = []
    seen_contents = set()
    for step in steps:
        cleaned = clean_step(step)
        if cleaned:
            normalized_cleaned = ' '.join(cleaned.split())
            # 避免重复步骤
            if normalized_cleaned not in seen_contents:
                valid_steps.append(cleaned)
                seen_contents.add(normalized_cleaned)
    
    # 如果没有提取到有效步骤，尝试将整个输出作为一个步骤
    if not valid_steps and output_text.strip():
        cleaned = clean_step(output_text)
        if cleaned:
            valid_steps.append(cleaned)
    
    return valid_steps