# quiz_module/question_generator.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import json
import random
import re
from typing import List, Dict, Any, Optional

# 生成配置
GENERATION_CONFIG = {
    "max_new_tokens": 1024,
    "temperature": 0.5,
    "top_p": 0.9,
    "do_sample": True,
}


def _build_question_gen_prompt(context: str, q_type: str = "choice", difficulty: str = "medium") -> List[Dict[str, str]]:
    """
    构建题目生成提示词 - 优化版
    
    Args:
        context: 上下文信息
        q_type: 题目类型 ("choice" 或 "boolean")
        difficulty: 难度等级 ("easy", "medium", "hard")
    """
    
    difficulty_instructions = {
        "easy": "基础概念和定义理解",
        "medium": "概念应用和知识综合",
        "hard": "深度分析和批判性思维"
    }
    
    if q_type == "choice":
        type_instruction = "一道四选一的选择题"
        json_format = """
{
  "question": "完整、独立的问题描述",
  "type": "choice",
  "options": [
    "A. 选项内容",
    "B. 选项内容",
    "C. 选项内容",
    "D. 选项内容"
  ],
  "correct_answer_letter": "A",
  "explanation": "详细解释正确答案的原因，并说明其他选项为何错误"
}
"""
        quality_rules = """
**选择题质量标准：**
- 干扰项要合理但明确错误，避免争议
- 选项长度和复杂度应相近
- 避免"以上都对/都错"等模糊选项
- 正确答案必须唯一且明确"""

    else:  # boolean
        type_instruction = "一道判断题"
        json_format = """
{
  "question": "一个明确的陈述句（可判断真假）",
  "type": "boolean",
  "correct_answer": true,
  "explanation": "详细说明这个陈述为何正确/错误，引用相关知识点"
}
"""
        quality_rules = """
**判断题质量标准：**
- 陈述必须明确、不含糊
- 避免双重否定或复杂逻辑
- 不使用"总是"、"永远"、"完全"等绝对词（除非确实如此）"""

    system_prompt = f"""你是一位经验丰富的教育测评专家，擅长设计高质量的学科测试题目。

**核心任务：**
基于提供的教学材料，设计{type_instruction}。
- **难度等级**: {difficulty_instructions[difficulty]}
- **知识来源**: 必须严格基于提供的上下文材料

**关键原则：**

1. **题目自洽性** ⭐
   - 题目表述必须完整、独立，不依赖额外背景
   - 学生无需阅读原始材料就能理解题目
   - 绝不使用"根据上文"、"材料中提到"、"以下哪个"等引用性表述

2. **知识聚焦** ⭐
   - 只考查学科核心知识（概念、原理、方法、应用）
   - 严禁考查元信息（章节号、页码、作者、参考文献等）
   - 题目应具有教学价值和实际意义

3. **答案准确性** ⭐
   - 正确答案必须在上下文中有明确依据
   - 不编造或推测材料外的信息
   - 如果材料信息不足，选择其他知识点

4. **表达规范** ⭐
   - 使用清晰、专业的学术语言
   - 避免歧义、模糊或过于口语化的表述
   - 数学公式和专业术语要准确

{quality_rules}

**输出格式要求：**
- 必须返回严格的JSON格式
- 不要添加任何额外说明或代码块标记
- 确保JSON完全有效（检查引号、逗号、括号）

**JSON格式示例：**
{json_format}"""

    user_message = f"""**教学材料：**
{context}

---

请基于上述材料，生成一道{difficulty_instructions[difficulty]}的{type_instruction}。

**要求：**
- 题目完全独立，不引用"材料"或"上文"
- 考查学科知识，非元信息
- 答案有明确依据
- 直接返回JSON，无额外内容"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]


def _parse_llm_json_output(response: str) -> Optional[Dict[str, Any]]:
    """
    从LLM输出中解析JSON - 增强版
    """
    try:
        response = response.strip()
        
        # 移除Markdown代码块
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        # 提取JSON对象
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            response = json_match.group(0)
        
        parsed = json.loads(response)
        
        q_type = parsed.get("type")

        # 布尔型格式转换
        if q_type == "boolean":
            if "correct_answer" in parsed:
                if not isinstance(parsed["correct_answer"], bool):
                    print(f"❌ 判断题答案必须是布尔值")
                    return None
                
                correct_bool = parsed["correct_answer"]
                parsed["options"] = ["正确", "错误"]
                parsed["correct_answer_index"] = 0 if correct_bool else 1
                del parsed["correct_answer"]
            
            elif "correct_answer_index" not in parsed:
                print(f"❌ 判断题缺少答案字段")
                return None
        
        # 选择题格式转换
        elif q_type == "choice":
            if "correct_answer_letter" in parsed:
                letter = parsed["correct_answer_letter"].upper().strip()
                letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                
                index = letter_map.get(letter)
                
                if index is None:
                    print(f"❌ 答案标识必须是A/B/C/D")
                    return None
                
                parsed["correct_answer_index"] = index
                del parsed["correct_answer_letter"]
            
            elif "correct_answer_index" not in parsed:
                print(f"❌ 选择题缺少答案字段")
                return None

        # 统一验证
        required_fields = ["question", "type", "options", "correct_answer_index", "explanation"]
        if not all(field in parsed for field in required_fields):
            missing = [f for f in required_fields if f not in parsed]
            print(f"❌ 缺少必需字段: {missing}")
            return None
        
        if not isinstance(parsed["options"], list) or len(parsed["options"]) == 0:
            print(f"❌ 选项格式错误")
            return None
        
        if not isinstance(parsed["correct_answer_index"], int):
            print(f"❌ 答案索引必须是整数")
            return None
        
        if parsed["correct_answer_index"] < 0 or parsed["correct_answer_index"] >= len(parsed["options"]):
            print(f"❌ 答案索引超出范围")
            return None
        
        return parsed
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        return None
    except Exception as e:
        print(f"❌ 解析异常: {e}")
        return None


def _validate_question_quality(question: Dict[str, Any]) -> tuple[bool, str]:
    """
    验证题目质量
    
    Returns:
        (is_valid, reason)
    """
    # 检查问题长度
    if len(question["question"]) < 10:
        return False, "问题过短"
    
    if len(question["question"]) > 500:
        return False, "问题过长"
    
    # 检查是否包含引用性词语（质量红线）
    forbidden_phrases = [
        "根据上文", "根据材料", "根据上述", "材料中", "文中", 
        "上面提到", "以下哪个", "该书", "本文", "作者认为"
    ]
    
    question_text = question["question"].lower()
    for phrase in forbidden_phrases:
        if phrase in question_text:
            return False, f"题目包含引用性表述: {phrase}"
    
    # 检查选项
    options = question["options"]
    if question["type"] == "choice":
        if len(options) != 4:
            return False, f"选择题应有4个选项，实际{len(options)}个"
        
        # 检查选项重复
        option_texts = [opt.split(". ", 1)[-1] if ". " in opt else opt for opt in options]
        if len(set(option_texts)) != len(option_texts):
            return False, "选项存在重复"
        
        # 检查选项长度
        for opt in options:
            if len(opt) < 2:
                return False, "选项过短"
    
    # 检查解释
    if len(question["explanation"]) < 20:
        return False, "解释过于简短"
    
    return True, "OK"


@torch.no_grad()
def generate_quiz_questions(
    retriever: BaseRetriever, 
    tokenizer: AutoTokenizer, 
    model: AutoModelForCausalLM, 
    device: str,
    num_choice: int = 3, 
    num_boolean: int = 2,
    difficulty: str = "medium",
    max_retries: int = 3,
    use_clustering: bool = True
) -> List[Dict[str, Any]]:
    """
    生成测验题目 - 支持主题聚类
    
    Args:
        retriever: 检索器
        tokenizer: 分词器
        model: 语言模型
        device: 设备
        num_choice: 选择题数量
        num_boolean: 判断题数量
        difficulty: 难度等级
        max_retries: 最大重试次数
        use_clustering: 是否使用主题聚类（推荐开启）
    
    Returns:
        题目列表
    """
    
    questions = []
    failed_count = 0
    
    num_to_generate = num_choice + num_boolean
    
    try:
        all_docs = []
        
        # 尝试从检索器获取文档
        if hasattr(retriever, 'retrievers') and len(retriever.retrievers) > 1:
            bm25_retriever = retriever.retrievers[1]
            if hasattr(bm25_retriever, 'documents'):
                print("✓ 使用文档池采样")
                all_docs = bm25_retriever.documents
        
        # 回退到查询
        if not all_docs:
            print("⚠️ 使用查询采样")
            base_queries = [
                "核心概念和关键知识点",
                "重要算法和方法",
                "基本原理和定理",
                "具体实现和技术细节",
                "不同方法的对比分析",
                "优缺点和适用场景",
                "高级技巧和注意事项"
            ]
            queries = random.sample(base_queries, k=min(len(base_queries), 3))
            
            seen_content = set()
            for query in queries:
                docs = retriever.invoke(query)
                for doc in docs:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(content_hash)

        if not all_docs:
            st.error("❌ 知识库为空")
            return []
        
        # === 核心改进：智能文档采样 ===
        if use_clustering and len(all_docs) > num_to_generate * 2:
            print(f"🎯 启用智能主题聚类采样...")
            
            try:
                from quiz_module.topic_clustering import smart_document_sampling
                
                # 使用K-Means聚类（更快）
                source_chunks = smart_document_sampling(
                    documents=all_docs,
                    num_samples=num_to_generate,
                    method="kmeans"
                )
                
                print(f"✓ 主题聚类采样完成，获得{len(source_chunks)}个高覆盖样本")
                
            except Exception as e:
                print(f"⚠️ 聚类采样失败: {e}，回退到随机采样")
                # 聚类失败，立即回退到随机采样
                if len(all_docs) < num_to_generate:
                    print(f"⚠️ 文档不足，进行有放回采样")
                    source_chunks = random.choices(all_docs, k=num_to_generate)
                else:
                    source_chunks = random.sample(all_docs, k=num_to_generate)
        
        # 降级到随机采样 (如果禁用了聚类，或者文档数不足)
        else:
            print("✓ 使用随机采样（聚类未启用或文档不足）")
            if len(all_docs) < num_to_generate:
                print(f"⚠️ 文档不足，进行有放回采样")
                source_chunks = random.choices(all_docs, k=num_to_generate)
            else:
                source_chunks = random.sample(all_docs, k=num_to_generate)
            
    except Exception as e:
        st.error(f"❌ 检索失败: {e}")
        return []

    total_steps = num_choice + num_boolean
    progress_bar = st.progress(0, text="🎯 开始生成题目...")
    
    # 生成选择题
    for i in range(num_choice):
        chunk = source_chunks[i]
        success = False
        
        for retry in range(max_retries):
            try:
                messages = _build_question_gen_prompt(chunk.page_content, "choice", difficulty)
                
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(device)
                
                outputs = model.generate(**inputs, **GENERATION_CONFIG)
                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if "assistant" in response_text:
                    response_text = response_text.split("assistant")[-1].strip()
                
                parsed_q = _parse_llm_json_output(response_text)
                
                if parsed_q:
                    is_valid, reason = _validate_question_quality(parsed_q)
                    if is_valid:
                        questions.append(parsed_q)
                        success = True
                        break
                    else:
                        print(f"⚠️ 质量不合格 (重试 {retry+1}/{max_retries}): {reason}")
                else:
                    print(f"⚠️ 解析失败 (重试 {retry+1}/{max_retries})")
                
            except Exception as e:
                print(f"⚠️ 生成异常 (重试 {retry+1}/{max_retries}): {e}")
        
        if not success:
            failed_count += 1
            print(f"❌ 选择题 {i+1} 失败")
        
        progress_text = f"📝 生成中... ({i+1}/{total_steps}) 选择题"
        if failed_count > 0:
            progress_text += f" [失败: {failed_count}]"
        progress_bar.progress((i + 1) / total_steps, text=progress_text)

    # 生成判断题
    for i in range(num_boolean):
        chunk = source_chunks[num_choice + i]
        success = False
        
        for retry in range(max_retries):
            try:
                messages = _build_question_gen_prompt(chunk.page_content, "boolean", difficulty)
                
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(device)
                
                outputs = model.generate(**inputs, **GENERATION_CONFIG)
                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if "assistant" in response_text:
                    response_text = response_text.split("assistant")[-1].strip()
                
                parsed_q = _parse_llm_json_output(response_text)
                
                if parsed_q:
                    is_valid, reason = _validate_question_quality(parsed_q)
                    if is_valid:
                        questions.append(parsed_q)
                        success = True
                        break
                    else:
                        print(f"⚠️ 质量不合格 (重试 {retry+1}/{max_retries}): {reason}")
                else:
                    print(f"⚠️ 解析失败 (重试 {retry+1}/{max_retries})")
                    
            except Exception as e:
                print(f"⚠️ 生成异常 (重试 {retry+1}/{max_retries}): {e}")
        
        if not success:
            failed_count += 1
            print(f"❌ 判断题 {i+1} 失败")
        
        progress_text = f"📝 生成中... ({num_choice + i + 1}/{total_steps}) 判断题"
        if failed_count > 0:
            progress_text += f" [失败: {failed_count}]"
        progress_bar.progress((num_choice + i + 1) / total_steps, text=progress_text)

    progress_bar.empty()
    
    # 结果统计
    success_count = len(questions)
    if success_count > 0:
        st.success(f"✅ 成功生成 {success_count} 道题目")
        if failed_count > 0:
            st.warning(f"⚠️ {failed_count} 道题目失败")
    else:
        st.error("❌ 题目生成失败，请重试")
    
    return questions