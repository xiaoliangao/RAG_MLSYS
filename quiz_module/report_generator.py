# quiz_module/report_generator.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
import os

# 生成配置
GENERATION_CONFIG = {
    "max_new_tokens": 1024,
    "temperature": 0.6,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}


@torch.no_grad()
def generate_study_feedback(
    tokenizer: AutoTokenizer, 
    model: AutoModelForCausalLM, 
    device: str, 
    report_data: Dict[str, Any]
) -> str:
    """
    生成个性化学习反馈 - 优化版
    
    Args:
        tokenizer: 分词器
        model: 语言模型
        device: 设备
        report_data: 测验报告数据
    
    Returns:
        Markdown格式的学习反馈
    """
    
    # 获取错题
    wrong_answers = [r for r in report_data['results'] if not r['is_correct']]
    
    # 全对情况
    if not wrong_answers:
        return generate_perfect_score_feedback(report_data)
    
    # 准备错题上下文
    context = _prepare_wrong_answers_context(wrong_answers, report_data)
    
    # 构建提示词 - 优化版
    system_prompt = """你是一位经验丰富的学习顾问，擅长分析学生的测验表现并提供有针对性的学习建议。

**角色定位：**
- 专业但亲切，像一位关心学生成长的导师
- 客观分析问题，但始终保持鼓励和建设性的态度
- 提供具体可行的改进方案，而非空洞的建议

**反馈原则：**
1. 肯定优势，指出不足，给予方向
2. 从错题中提炼核心问题（概念理解、知识应用等）
3. 建议具体、可操作，有明确的学习路径
4. 语言简洁明了，避免过度专业术语"""

    user_message = f"""{context}

**请生成一份学习反馈报告，包含以下内容：**

**1. 整体评价** (2-3句话)
- 总体表现如何（得分{report_data['score_percentage']:.1f}%）
- 答对{report_data['correct']}/{report_data['total']}题的水平定位
- 简要的鼓励或肯定

**2. 知识盲区分析** (列出2-3个核心问题)
- 从错题中提炼出的知识薄弱点
- 每个点用一句话概括
- 按重要性排序

**3. 针对性建议** (3-4条具体建议)
- 每条建议针对一个知识盲区
- 说明"应该做什么"和"如何做"
- 可以推荐具体的学习方法或资源

**4. 下一步行动**
- 引导学生使用"AI助教"功能深入学习错题
- 给出具体的提问示例

**格式要求：**
- 使用Markdown格式
- 使用标题、列表等结构化元素
- 语言友好、专业、有温度
- 总长度控制在300-400字

直接输出报告内容，不要有额外说明。"""

    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": user_message}
    ]
    
    try:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        outputs = model.generate(**inputs, **GENERATION_CONFIG)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        return response
        
    except Exception as e:
        print(f"⚠️ AI反馈生成失败: {e}")
        return generate_fallback_feedback(report_data)


def _prepare_wrong_answers_context(wrong_answers: List[Dict[str, Any]], report_data: Dict[str, Any]) -> str:
    """准备错题分析上下文 - 优化版"""
    
    context = f"""**学生测验情况：**
- 总题数: {report_data['total']}
- 答对: {report_data['correct']}
- 答错: {report_data['wrong']}
- 得分: {report_data['score_percentage']:.1f}%

**错题详情：**
"""
    
    for i, item in enumerate(wrong_answers, 1):
        context += f"\n【错题 {i}】\n"
        context += f"题目: {item['question']}\n"
        
        # 获取答案
        try:
            user_ans_idx = item.get('user_answer', -1)
            if user_ans_idx == -1:
                user_ans_text = "未作答"
            else:
                user_ans_text = item['options'][user_ans_idx]
            
            correct_ans_text = item['options'][item['correct_answer']]
            
        except (IndexError, KeyError):
            user_ans_text = "无效"
            correct_ans_text = "无效"
        
        context += f"学生答案: {user_ans_text}\n"
        context += f"正确答案: {correct_ans_text}\n"
        context += f"解析: {item['explanation']}\n"
    
    return context


def generate_perfect_score_feedback(report_data: Dict[str, Any]) -> str:
    """全对时的祝贺反馈 - 优化版"""
    
    from .evaluator import get_performance_level
    performance = get_performance_level(report_data['score_percentage'])
    
    return f"""## {performance['emoji']} 完美表现！

恭喜你全部答对！({report_data['correct']}/{report_data['total']}题)

### 📊 成绩分析
- **得分**: {report_data['score_percentage']:.1f}%
- **评级**: {performance['level']}
- {performance['message']}

### 💪 你的优势
- **知识扎实**: 对本部分内容的核心概念掌握牢固
- **理解深入**: 能够准确区分相似概念，理解细微差别
- **应用熟练**: 将理论知识应用到具体问题时游刃有余

### 🚀 进阶建议
1. **挑战更高难度**: 尝试"困难"级别的测验，拓展知识边界
2. **深化理解**: 在AI助教中探讨更深层的原理和数学推导
3. **实践应用**: 将学到的知识应用到实际项目或案例中
4. **知识迁移**: 思考这些概念在其他领域的应用

### 🤖 继续探索
既然基础已经很扎实，不妨在「AI助教」中尝试这些问题：
- "能否从数学角度深入解释...的原理？"
- "...方法在实际项目中有哪些注意事项？"
- "对比...和...的底层实现有何不同？"

保持这份热情和专注，继续加油！ 🌟"""


def generate_fallback_feedback(report_data: Dict[str, Any]) -> str:
    """降级反馈（当AI生成失败时）- 优化版"""
    
    from .evaluator import get_performance_level
    performance = get_performance_level(report_data['score_percentage'])
    
    feedback = f"""## {performance['emoji']} 测验反馈

### 📊 总体表现
- **得分**: {report_data['score_percentage']:.1f}%
- **评级**: {performance['level']}
- **正确**: {report_data['correct']}/{report_data['total']} 题
- **错误**: {report_data['wrong']} 题

{performance['message']}

### 🎯 知识盲区
"""
    
    # 分析错题类型
    wrong_answers = [r for r in report_data['results'] if not r['is_correct']]
    
    if wrong_answers:
        # 简单分析（基于题目关键词）
        knowledge_areas = set()
        for item in wrong_answers[:3]:
            question = item['question']
            # 提取可能的知识点关键词
            if '算法' in question or '方法' in question:
                knowledge_areas.add("算法和方法的理解")
            if '原理' in question or '为什么' in question:
                knowledge_areas.add("基础原理的掌握")
            if '应用' in question or '场景' in question:
                knowledge_areas.add("知识的实际应用")
            if '对比' in question or '区别' in question:
                knowledge_areas.add("概念的辨析能力")
        
        if knowledge_areas:
            for area in list(knowledge_areas)[:3]:
                feedback += f"- {area}\n"
        else:
            feedback += "- 请仔细复习错题涉及的知识点\n"
    else:
        feedback += "- 表现优秀，无明显知识盲区\n"
    
    feedback += """
### 💡 学习建议

**复习策略：**
1. **精读错题解析**: 不要只看答案，理解为什么这样做
2. **追根溯源**: 回到教材，找到相关知识点完整学习
3. **举一反三**: 思考类似的问题应该如何解决

**提升方法：**
1. **使用AI助教**: 针对不理解的错题，在助教中深入提问
2. **主动练习**: 完成相关习题，巩固薄弱知识点
3. **定期回顾**: 过几天再次测验，检验学习效果

### 🤖 推荐行动

👉 立即前往「AI助教」，尝试这样提问：
- "请详细解释[错题中的概念]"
- "为什么[错误选项]不正确？"
- "能举个[知识点]的实际应用例子吗？"

每次测验都是进步的机会，继续努力！ 💪
"""
    
    return feedback


def prepare_chart_data(report_data: Dict[str, Any]) -> pd.DataFrame:
    """为可视化准备答题分布数据"""
    data = {
        "类别": ["✅ 答对", "❌ 答错"],
        "数量": [
            report_data['correct'], 
            report_data['wrong']
        ]
    }
    
    if report_data.get('unanswered', 0) > 0:
        data["类别"].append("⭕ 未答")
        data["数量"].append(report_data['unanswered'])
    
    return pd.DataFrame(data)


def prepare_type_accuracy_data(report_data: Dict[str, Any]) -> pd.DataFrame:
    """准备题型准确率数据"""
    results = report_data.get('results', [])
    
    if not results:
        return None
    
    # 统计各题型
    choice_correct = sum(1 for r in results if r.get('type') == 'choice' and r['is_correct'])
    choice_total = sum(1 for r in results if r.get('type') == 'choice')
    
    boolean_correct = sum(1 for r in results if r.get('type') == 'boolean' and r['is_correct'])
    boolean_total = sum(1 for r in results if r.get('type') == 'boolean')
    
    data = {
        "题型": [],
        "准确率": []
    }
    
    if choice_total > 0:
        data["题型"].append("📋 选择题")
        data["准确率"].append(choice_correct / choice_total * 100)
    
    if boolean_total > 0:
        data["题型"].append("❓ 判断题")
        data["准确率"].append(boolean_correct / boolean_total * 100)
    
    if not data["题型"]:
        return None
    
    return pd.DataFrame(data)


def export_report_to_text(report_data: Dict[str, Any], feedback: str) -> str:
    """导出文本格式报告"""
    
    lines = []
    lines.append("=" * 80)
    lines.append(" " * 28 + "📊 学习测验报告")
    lines.append("=" * 80)
    lines.append(f"\n📅 生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
    
    # 成绩概览
    lines.append("┌" + "─" * 78 + "┐")
    lines.append("│" + " " * 30 + "成绩概览" + " " * 40 + "│")
    lines.append("├" + "─" * 78 + "┤")
    lines.append(f"│  总题数: {report_data['total']:<3}  正确: {report_data['correct']:<3}  错误: {report_data['wrong']:<3}  得分: {report_data['score_percentage']:.1f}%" + " " * 20 + "│")
    
    from .evaluator import get_performance_level
    performance = get_performance_level(report_data['score_percentage'])
    lines.append(f"│  评级: {performance['level']}" + " " * 60 + "│")
    lines.append("└" + "─" * 78 + "┘")
    lines.append("")
    
    # AI学习反馈
    lines.append("=" * 80)
    lines.append(" " * 28 + "🤖 AI 学习反馈")
    lines.append("=" * 80)
    lines.append(feedback)
    lines.append("")
    
    # 详细题目
    lines.append("=" * 80)
    lines.append(" " * 30 + "📋 答题详情")
    lines.append("=" * 80)
    
    for result in report_data['results']:
        idx = result['question_index']
        lines.append(f"\n{'━' * 80}")
        lines.append(f"第 {idx + 1} 题")
        lines.append(f"题目: {result['question']}")
        lines.append("\n选项:")
        
        for opt in result['options']:
            lines.append(f"  {opt}")
        
        correct_idx = result['correct_answer']
        user_idx = result.get('user_answer', -1)
        
        lines.append("")
        if result.get('is_unanswered', False):
            lines.append(f"您的答案: ⭕ 未作答")
            lines.append(f"正确答案: {result['options'][correct_idx]}")
        elif result['is_correct']:
            lines.append(f"您的答案: {result['options'][user_idx]} ✅")
        else:
            lines.append(f"您的答案: {result['options'][user_idx]} ❌")
            lines.append(f"正确答案: {result['options'][correct_idx]}")
        
        lines.append(f"\n解析: {result['explanation']}")
    
    lines.append("\n" + "=" * 80)
    lines.append(" " * 32 + "报告结束")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def export_report_to_pdf(report_data: Dict[str, Any], feedback: str) -> BytesIO:
    """
    导出PDF格式报告 - 字体优化版
    
    Args:
        report_data: 测验报告数据
        feedback: AI学习反馈
    
    Returns:
        BytesIO: PDF文件字节流
    """
    
    buffer = BytesIO()
    
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # 智能中文字体检测和注册
    chinese_font_registered = False
    font_paths = [
        # macOS
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
        # Windows
        'C:/Windows/Fonts/simhei.ttf',
        'C:/Windows/Fonts/simsun.ttc',
        'C:/Windows/Fonts/msyh.ttc',
        # Linux
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                chinese_font_registered = True
                break
        except Exception as e:
            continue
    
    # 根据字体可用性设置样式
    if chinese_font_registered:
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName='ChineseFont',
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName='ChineseFont',
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName='ChineseFont',
            fontSize=10,
            leading=14
        )
    else:
        # 降级到默认字体（可能无法正确显示中文）
        import streamlit as st
        st.warning("⚠️ 未找到中文字体，PDF中的中文可能显示为方框")
        title_style = styles['Heading1']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']
    
    # 标题
    title = Paragraph("学习测验报告", title_style)
    story.append(title)
    story.append(Spacer(1, 0.5*cm))
    
    # 生成时间
    timestamp = Paragraph(
        f"生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}",
        normal_style
    )
    story.append(timestamp)
    story.append(Spacer(1, 0.5*cm))
    
    # 成绩概览
    story.append(Paragraph("成绩概览", heading_style))
    
    from .evaluator import get_performance_level
    performance = get_performance_level(report_data['score_percentage'])
    
    summary_data = [
        ['项目', '数值'],
        ['总题数', str(report_data['total'])],
        ['正确', str(report_data['correct'])],
        ['错误', str(report_data['wrong'])],
        ['得分', f"{report_data['score_percentage']:.1f}%"],
        ['评级', performance['level']]
    ]
    
    summary_table = Table(summary_data, colWidths=[8*cm, 8*cm])
    summary_table.setStyle(TableStyle([
        # 1. 【修复】为整个表格设置中文字体
        ('FONTNAME', (0, 0), (-1, -1), 'ChineseFont'),
        
        # 2. 表头样式
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        
        # 3. 内容样式
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        
        # 4. 网格线
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 1*cm))
    
    # AI学习反馈
    story.append(Paragraph("AI 学习反馈", heading_style))
    
    feedback_lines = feedback.split('\n')
    for line in feedback_lines:
        if line.strip():
            story.append(Paragraph(line, normal_style))
            story.append(Spacer(1, 0.2*cm))
    
    story.append(Spacer(1, 1*cm))
    
    # 答题详情
    story.append(Paragraph("答题详情", heading_style))
    
    for result in report_data['results']:
        idx = result['question_index']
        
        q_title = Paragraph(f"第 {idx + 1} 题", heading_style)
        story.append(q_title)
        
        question_text = Paragraph(f"<b>题目:</b> {result['question']}", normal_style)
        story.append(question_text)
        story.append(Spacer(1, 0.3*cm))
        
        for opt in result['options']:
            story.append(Paragraph(f"  {opt}", normal_style))
        
        story.append(Spacer(1, 0.3*cm))
        
        correct_idx = result['correct_answer']
        user_idx = result.get('user_answer', -1)
        
        if result.get('is_unanswered', False):
            story.append(Paragraph("您的答案: 未作答", normal_style))
            story.append(Paragraph(f"正确答案: {result['options'][correct_idx]}", normal_style))
            status = Paragraph("<font color='orange'>⭕ 未作答</font>", normal_style)
        elif result['is_correct']:
            story.append(Paragraph(f"您的答案: {result['options'][user_idx]}", normal_style))
            status = Paragraph("<font color='green'>✅ 正确</font>", normal_style)
        else:
            story.append(Paragraph(f"您的答案: {result['options'][user_idx]}", normal_style))
            story.append(Paragraph(f"正确答案: {result['options'][correct_idx]}", normal_style))
            status = Paragraph("<font color='red'>❌ 错误</font>", normal_style)
        
        story.append(status)
        story.append(Spacer(1, 0.3*cm))
        
        story.append(Paragraph(f"<b>解析:</b> {result['explanation']}", normal_style))
        story.append(Spacer(1, 0.8*cm))
    
    doc.build(story)
    
    buffer.seek(0)
    return buffer