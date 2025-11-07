# quiz_module/evaluator.py

from typing import List, Dict, Any, Optional
from collections import Counter
import re


def grade_quiz(questions: List[Dict[str, Any]], user_answers_list: List[Optional[str]]) -> Dict[str, Any]:
    """
    主判分函数（处理 st.form 提交的原始答案）
    
    Args:
        questions: 原始的题目列表 (来自 st.session_state.quiz_questions)
        user_answers_list: 用户在st.form中提交的答案字符串列表 (来自 st.radio)
                          例如: ["A. 选项内容", "True", None, "C. 其他选项"]
    
    Returns:
        包含得分和详细信息的字典
        {
            "total": int,
            "correct": int,
            "wrong": int,
            "unanswered": int,
            "score_percentage": float,
            "results": List[Dict],
            "knowledge_gaps": List[str],
            "statistics": Dict
        }
    """
    
    # 1. 验证输入
    if len(questions) != len(user_answers_list):
        raise ValueError(f"题目数量({len(questions)})与答案数量({len(user_answers_list)})不匹配")
    
    # 2. 将用户的答案字符串转换为索引
    user_answers_map = {}
    unanswered_count = 0
    
    for i, q in enumerate(questions):
        user_ans_str = user_answers_list[i]
        
        if user_ans_str is None or user_ans_str == "":
            # 用户未作答
            user_answers_map[i] = -1
            unanswered_count += 1
            continue
        
        try:
            # 找到用户选择的字符串在 options 列表中的索引
            user_ans_index = q['options'].index(user_ans_str)
            user_answers_map[i] = user_ans_index
        except ValueError:
            # 容错处理：尝试模糊匹配（去除选项前缀 "A. "）
            matched = False
            cleaned_user_ans = _clean_option_text(user_ans_str)
            
            for idx, opt in enumerate(q['options']):
                if _clean_option_text(opt) == cleaned_user_ans:
                    user_answers_map[i] = idx
                    matched = True
                    break
            
            if not matched:
                # 实在匹配不到，标记为未答
                user_answers_map[i] = -1
                unanswered_count += 1
                print(f"⚠️ 警告：第{i+1}题的答案无法匹配: '{user_ans_str}'")
    
    # 3. 调用核心计算函数
    result = calculate_score(user_answers_map, questions)
    
    # 4. 补充未答题统计
    result['unanswered'] = unanswered_count
    
    return result


def _clean_option_text(option: str) -> str:
    """
    清理选项文本（移除 A. B. True False 等前缀）
    """
    # 移除常见前缀
    option = re.sub(r'^[A-D]\.\s*', '', option)
    option = option.strip()
    return option


def calculate_score(user_answers: Dict[int, int], questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算测验得分（核心算法）
    
    Args:
        user_answers: {question_index: selected_option_index}
                     -1 表示未作答
        questions: 题目列表
    
    Returns:
        详细的评分报告字典
    """
    total = len(questions)
    correct = 0
    wrong = 0
    results = []
    wrong_types = []  # 记录错题类型（用于知识盲区分析）
    
    for i, question in enumerate(questions):
        user_answer_index = user_answers.get(i, -1)
        correct_answer_index = question["correct_answer_index"]
        
        # 判断正误
        is_correct = (user_answer_index == correct_answer_index) and (user_answer_index != -1)
        
        if is_correct:
            correct += 1
        elif user_answer_index != -1:
            wrong += 1
            wrong_types.append(question.get('type', 'unknown'))
        
        # 存储详细结果
        results.append({
            "question_index": i,
            "question": question["question"],
            "type": question.get("type", "unknown"),
            "options": question["options"],
            "user_answer": user_answer_index,
            "correct_answer": correct_answer_index,
            "is_correct": is_correct,
            "is_unanswered": (user_answer_index == -1),
            "explanation": question["explanation"]
        })
    
    # 计算得分
    score_percentage = (correct / total * 100) if total > 0 else 0
    
    # 统计信息
    statistics = _calculate_statistics(results, wrong_types)
    
    # 识别知识盲区
    knowledge_gaps = _identify_knowledge_gaps(results)
    
    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "unanswered": total - correct - wrong,
        "score_percentage": round(score_percentage, 2),
        "results": results,
        "knowledge_gaps": knowledge_gaps,
        "statistics": statistics
    }


def _calculate_statistics(results: List[Dict], wrong_types: List[str]) -> Dict[str, Any]:
    """
    计算统计信息
    """
    # 按题型统计
    choice_correct = sum(1 for r in results if r['type'] == 'choice' and r['is_correct'])
    choice_total = sum(1 for r in results if r['type'] == 'choice')
    
    boolean_correct = sum(1 for r in results if r['type'] == 'boolean' and r['is_correct'])
    boolean_total = sum(1 for r in results if r['type'] == 'boolean')
    
    # 错题类型分布
    wrong_type_count = Counter(wrong_types)
    
    return {
        "choice_accuracy": (choice_correct / choice_total * 100) if choice_total > 0 else 0,
        "boolean_accuracy": (boolean_correct / boolean_total * 100) if boolean_total > 0 else 0,
        "wrong_type_distribution": dict(wrong_type_count),
        "average_difficulty": _estimate_difficulty(results)
    }


def _estimate_difficulty(results: List[Dict]) -> str:
    """
    估计测验难度（基于正确率）
    """
    correct_rate = sum(1 for r in results if r['is_correct']) / len(results) if results else 0
    
    if correct_rate >= 0.8:
        return "简单"
    elif correct_rate >= 0.5:
        return "中等"
    else:
        return "困难"


def _identify_knowledge_gaps(results: List[Dict]) -> List[str]:
    """
    识别知识盲区（从错题中提取关键概念）
    
    这是一个简化版本，后续可以结合NLP进行更精确的提取
    """
    gaps = []
    wrong_questions = [r for r in results if not r['is_correct'] and not r['is_unanswered']]
    
    if not wrong_questions:
        return []
    
    # 简单的关键词提取（后续可以使用知识图谱优化）
    keywords = set()
    common_terms = ['是', '的', '了', '在', '和', '与', '或', '如何', '什么', '为什么', '？', '，', '。']
    
    for item in wrong_questions:
        question = item['question']
        # 提取问题中的关键词（简化版）
        words = [w for w in question.split() if w not in common_terms and len(w) > 2]
        keywords.update(words[:3])  # 最多取前3个关键词
    
    # 构建知识盲区描述
    if keywords:
        gaps.append(f"涉及以下概念：{', '.join(list(keywords)[:5])}")
    
    # 按题型分类
    wrong_by_type = {}
    for item in wrong_questions:
        q_type = "选择题" if item['type'] == 'choice' else "判断题"
        wrong_by_type[q_type] = wrong_by_type.get(q_type, 0) + 1
    
    for q_type, count in wrong_by_type.items():
        gaps.append(f"{q_type}错误较多（{count}题）")
    
    return gaps


def get_performance_level(score_percentage: float) -> Dict[str, str]:
    """
    根据得分获取表现等级
    
    Returns:
        {"level": str, "emoji": str, "color": str, "message": str}
    """
    if score_percentage >= 90:
        return {
            "level": "优秀",
            "emoji": "🏆",
            "color": "green",
            "message": "出色的表现！你对这部分内容掌握得非常好！"
        }
    elif score_percentage >= 80:
        return {
            "level": "良好",
            "emoji": "🥈",
            "color": "blue",
            "message": "不错的成绩！继续保持，你离优秀只差一步了！"
        }
    elif score_percentage >= 70:
        return {
            "level": "中等",
            "emoji": "🥉",
            "color": "orange",
            "message": "还不错！再多练习一下就能更上一层楼！"
        }
    elif score_percentage >= 60:
        return {
            "level": "及格",
            "emoji": "📘",
            "color": "yellow",
            "message": "基础还可以，建议加强薄弱环节的学习。"
        }
    else:
        return {
            "level": "需加强",
            "emoji": "📕",
            "color": "red",
            "message": "不要气馁！找到知识盲区，系统学习后一定能进步！"
        }


def format_detailed_results(report_data: Dict[str, Any]) -> str:
    """
    格式化详细的答题结果（用于导出或显示）
    
    Returns:
        Markdown格式的详细报告
    """
    results = report_data['results']
    
    output = []
    output.append("# 测验详细结果\n")
    output.append(f"## 总体表现")
    output.append(f"- 总题数: {report_data['total']}")
    output.append(f"- 答对: {report_data['correct']}")
    output.append(f"- 答错: {report_data['wrong']}")
    output.append(f"- 未答: {report_data['unanswered']}")
    output.append(f"- 得分: {report_data['score_percentage']}%\n")
    
    # 答题详情
    output.append("## 答题详情\n")
    
    for i, result in enumerate(results, 1):
        status = "✅ 正确" if result['is_correct'] else ("⭕ 未答" if result['is_unanswered'] else "❌ 错误")
        output.append(f"### 第 {i} 题 - {status}")
        output.append(f"**问题:** {result['question']}\n")
        
        output.append("**选项:**")
        for idx, opt in enumerate(result['options']):
            marker = ""
            if idx == result['user_answer']:
                marker = " 👈 您的答案"
            if idx == result['correct_answer']:
                marker += " ✅ 正确答案"
            output.append(f"- {opt}{marker}")
        
        output.append(f"\n**解析:** {result['explanation']}\n")
        output.append("---\n")
    
    return "\n".join(output)