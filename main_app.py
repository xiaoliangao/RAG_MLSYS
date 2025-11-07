# main_app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import os
import uuid
import shutil
import hashlib
from typing import List, Optional, Dict, Any
from quiz_module import question_generator, evaluator, report_generator
from datetime import datetime
import altair as alt
import core_processing
import core_indexing
from threading import Thread
import json
import re
# --- 全局配置 ---
EMBEDDING_MODEL_NAME = "./models/bge-large-zh-v1.5"
LLM_MODEL_PATH = "./models/Qwen2.5-7B-Instruct"
STATIC_VECTOR_DB_PATH = "./vector_db"
TEMP_UPLOAD_DIR = "./temp_uploads"
SESSION_DB_BASE = "./vector_db"

# 【新增】反馈数据库路径
FEEDBACK_DB_PATH = "./feedback_db"

# 【新增】Few-shot 示例
FEW_SHOT_EXAMPLES = [
    {
        "question": "什么是反向传播算法？",
        "answer": """**反向传播算法**是训练神经网络的核心算法，用于高效计算每个参数的梯度。

**核心流程：**

1. **前向传播**
   - 输入数据逐层通过网络
   - 每层进行线性变换和激活函数计算
   - 最终得到预测输出

2. **计算损失**
   - 对比预测值与真实标签
   - 使用损失函数（如交叉熵、MSE）量化误差

3. **反向传播**
   - 从输出层开始，向输入层逐层传递
   - 利用链式法则计算每个参数的梯度
   - ∂L/∂w = ∂L/∂y × ∂y/∂w

4. **参数更新**
   - 使用梯度下降优化器更新权重
   - w_new = w_old - learning_rate × gradient

**关键优势：** 通过缓存前向传播的中间结果，避免重复计算，大幅提升训练效率。"""
    },
    {
        "question": "Batch Normalization如何工作？",
        "answer": """**Batch Normalization（批归一化）**是一种强大的正则化技术，能显著改善深度网络训练。

**工作机制：**

1. **标准化**
   - 对每个batch的激活值进行标准化
   - 使其均值为0，方差为1
   - x_norm = (x - μ_batch) / √(σ²_batch + ε)

2. **缩放和平移**
   - 引入可学习参数γ（scale）和β（shift）
   - y = γ × x_norm + β
   - 允许网络恢复原始表示能力

**主要优势：**

- **加速收敛**：稳定激活分布，允许使用更大学习率
- **减少梯度消失/爆炸**：规范化激活值范围
- **正则化效应**：batch间的随机性产生类似dropout的效果
- **降低对初始化的敏感度**：使网络更容易训练

**应用场景：** 通常放置在线性层之后、激活函数之前。"""
    }
]

GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
}


# ==================== 辅助函数 ====================

def _display_question_result(result: Dict[str, Any], expanded: bool = False):
    """显示单个题目的答题结果"""
    idx = result['question_index']
    question = result['question']
    options = result['options']
    user_ans_idx = result.get('user_answer', -1)
    correct_ans_idx = result['correct_answer']
    is_correct = result['is_correct']
    is_unanswered = result.get('is_unanswered', False)
    explanation = result['explanation']
    
    if is_correct:
        status_badge = "✅ **正确**"
    elif is_unanswered:
        status_badge = "⭕ **未作答**"
    else:
        status_badge = "❌ **错误**"
    
    with st.expander(f"第 {idx+1} 题 - {status_badge}", expanded=expanded):
        st.markdown(f"**题目:** {question}")
        
        st.markdown("**选项:**")
        for i, opt in enumerate(options):
            if is_unanswered:
                if i == correct_ans_idx:
                    st.markdown(f"- :green[**{opt}**] ✅ (正确答案)")
                else:
                    st.markdown(f"- {opt}")
            elif i == user_ans_idx and i == correct_ans_idx:
                st.markdown(f"- :green[**{opt}**] ✅ (您的答案，正确)")
            elif i == user_ans_idx:
                st.markdown(f"- :red[**{opt}**] ❌ (您的答案)")
            elif i == correct_ans_idx:
                st.markdown(f"- :green[**{opt}**] ✅ (正确答案)")
            else:
                st.markdown(f"- {opt}")
        
        st.markdown("**📖 解析:**")
        st.info(explanation)


# ==================== 模型加载（全局缓存）====================

@st.cache_resource
def load_llm(model_path: str):
    """加载大语言模型"""
    print("🧠 开始加载大语言模型...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda':
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            model = model.to(device)
        
        model.eval()
        print("✓ 大语言模型加载完成")
        return tokenizer, model, device
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        raise


@st.cache_resource
def load_embedding_model(model_path: str):
    """加载Embedding模型"""
    print("📊 开始加载Embedding模型...")
    embedding_model = core_indexing.initialize_embedding_model(model_path)
    print("✓ Embedding模型加载完成")
    return embedding_model


# ==================== 检索器创建 ====================

class EnsembleRetriever:
    """混合检索器：向量检索 + BM25"""
    def __init__(self, retrievers, weights=None):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)

    def invoke(self, query: str) -> List[Document]:
        all_docs = []
        for retriever, w in zip(self.retrievers, self.weights):
            try:
                docs = retriever.invoke(query)
            except Exception:
                docs = retriever.get_relevant_documents(query)
            all_docs.extend(docs * int(w * 10))

        unique_docs = {d.page_content: d for d in all_docs}
        return list(unique_docs.values())


def create_retriever_from_db(db: Chroma, embedding_model: HuggingFaceEmbeddings):
    """从Chroma数据库创建混合检索器"""
    
    vector_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    
    try:
        all_data = db.get()
        if all_data and all_data.get('documents'):
            docs = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(
                    all_data['documents'],
                    all_data.get('metadatas', [{}] * len(all_data['documents']))
                )
            ]
            
            bm25_retriever = BM25Retriever.from_documents(docs)
            bm25_retriever.k = 6
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.6, 0.4]
            )
            print("✓ 混合检索器创建成功 (向量 + BM25)")
            return ensemble_retriever
    except Exception as e:
        print(f"⚠ BM25创建失败: {e}，使用纯向量检索")
    
    return vector_retriever


def load_static_retriever(db_path: str, embedding_model: HuggingFaceEmbeddings):
    """加载静态知识库的检索器"""
    
    if not os.path.exists(db_path):
        return None
    
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model
    )
    
    return create_retriever_from_db(db, embedding_model)


# ==================== RAG问答功能 (【替换】增强版) ====================

def generate_queries(original_query, num_queries=2):
    """智能查询扩展"""
    queries = [original_query]
    
    # 补充疑问词
    if not original_query.startswith(("什么", "如何", "为什么", "请问", "能否", "怎么")):
        queries.append(f"什么是{original_query}")
    
    # 添加解释性查询
    if "解释" not in original_query and "介绍" not in original_query:
        queries.append(f"请解释{original_query}")
    
    # 添加领域前缀
    domain_keywords = ["机器学习", "深度学习", "神经网络", "算法"]
    has_domain = any(kw in original_query for kw in domain_keywords)
    
    if not has_domain and len(queries) < num_queries + 1:
        queries.append(f"深度学习中的{original_query}")
    
    return queries[:num_queries + 1]


def smart_context_selection(docs, query, max_docs=4):
    """智能上下文选择：多维度评分"""
    if len(docs) <= max_docs:
        return docs
    
    query_terms = set(query.lower().split())
    
    scored_docs = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        
        # 1. 关键词匹配得分
        keyword_score = sum(1 for term in query_terms if term in content_lower)
        
        # 2. 文档长度得分（更完整的信息）
        length_score = min(len(doc.page_content) / 1000, 2.0)
        
        # 3. 文档多样性（避免重复）
        diversity_score = 1.0
        
        total_score = keyword_score * 2 + length_score + diversity_score
        scored_docs.append((total_score, doc))
    
    scored_docs.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored_docs[:max_docs]]


def extract_dialogue_context(messages, max_history=3):
    """提取多轮对话上下文"""
    if len(messages) < 3:
        return None
    
    recent_messages = messages[-(2*max_history):]
    
    context_parts = []
    for i in range(0, len(recent_messages), 2):
        if i+1 < len(recent_messages):
            user_msg = recent_messages[i]["content"][:150]
            assistant_msg = recent_messages[i+1]["content"][:150]
            context_parts.append(f"Q: {user_msg}\nA: {assistant_msg}")
    
    return "\n\n".join(context_parts) if context_parts else None


def retrieve_with_enhancements(retriever, query: str, k: int = 4, enable_expansion: bool = True):
    """增强检索 (来自 module_rag_assistant.py)"""
    try:
        all_docs = []
        seen_content = set()
        
        if enable_expansion:
            queries = generate_queries(query, num_queries=2)
        else:
            queries = [query]
        
        for q in queries:
            docs = retriever.invoke(q)
            
            for doc in docs:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(content_hash)
        
        final_docs = smart_context_selection(all_docs, query, max_docs=k)
        
        context_parts = []
        sources = []
        
        for i, doc in enumerate(final_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            
            context_parts.append(f"[文档 {i}]\n{doc.page_content}")
            sources.append(f"{source} (页码: {page})")
        
        context = "\n\n".join(context_parts)
        
        return context, sources, final_docs
        
    except Exception as e:
        st.error(f"检索出错: {e}")
        return "", [], []


def build_enhanced_prompt(context, question, dialogue_history=None, 
                         use_fewshot=True, use_multi_turn=True):
    """构建优化的prompt"""
    
    system_prompt = """你是一位经验丰富的机器学习与深度学习专家教师。你的使命是帮助学习者深入理解复杂的技术概念。

**教学原则：**

1. **准确性是基础**
   - 严格基于提供的参考资料回答
   - 不编造或臆测超出资料范围的内容
   - 遇到资料不足时，诚实说明并建议查阅方向

2. **结构化表达**
   - 使用清晰的标题和层次组织内容
   - 先概述核心概念，再展开细节
   - 善用**加粗**、编号列表、分点说明

3. **深入浅出**
   - 复杂概念先给出直观解释
   - 适时使用类比和实例帮助理解
   - 必要时指出数学原理，但保持可读性

4. **理论联系实践**
   - 说明概念的实际应用场景
   - 指出常见误区和注意事项
   - 提供进一步学习的方向

5. **对话连贯性**（多轮对话时）
   - 参考之前讨论的内容
   - 逐步深入，避免重复
   - 回答时呼应学习者的问题脉络

**回答风格：** 专业而友好，像一位耐心的导师与学生面对面交流。"""

    # Few-shot示例
    fewshot_text = ""
    if use_fewshot:
        fewshot_text = "\n\n**参考示例：**\n"
        for i, example in enumerate(FEW_SHOT_EXAMPLES[:2], 1):
            fewshot_text += f"\n【示例 {i}】\n问：{example['question']}\n答：{example['answer'][:300]}...\n"
    
    # 对话历史
    history_section = ""
    if use_multi_turn and dialogue_history:
        history_section = f"\n\n**之前的对话：**\n{dialogue_history}\n"
    
    user_message = f"""{fewshot_text}

**参考资料：**
{context}{history_section}

---

**当前问题：** {question}

请基于参考资料，提供一个专业、准确且易于理解的回答。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    return messages


def generate_response_stream(tokenizer, model, device, messages):
    """流式生成响应"""
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        generation_kwargs = {
            **inputs,
            **GENERATION_CONFIG,
            "streamer": streamer,
        }
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text_chunk in streamer:
            yield text_chunk
            
    except Exception as e:
        yield f"生成出错: {e}"


def save_feedback(question, answer, feedback_type, comment=""):
    """保存用户反馈"""
    try:
        os.makedirs(FEEDBACK_DB_PATH, exist_ok=True)
        
        feedback_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer[:200], # 只保存简略答案
            "type": feedback_type,
            "comment": comment
        }
        
        feedback_file = os.path.join(
            FEEDBACK_DB_PATH,
            f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        )
        
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        st.error(f"保存反馈失败: {e}")
        return False


# ==================== 侧边栏配置 (【替换】增强版) ====================

def render_sidebar():
    """渲染侧边栏配置"""
    with st.sidebar:
        st.header("⚙️ 系统设置")
        
        st.divider()
        
        # --- RAG 问答设置 ---
        st.subheader("🤖 AI助教设置")
        
        enable_query_expansion = st.checkbox(
            "启用查询扩展",
            value=True,
            help="自动生成相关查询，提高检索覆盖率"
        )
        
        enable_multi_turn = st.checkbox(
            "多轮对话优化",
            value=True,
            help="在对话中考虑历史上下文"
        )
        
        if enable_multi_turn:
            max_history_turns = st.slider(
                "对话历史轮数",
                min_value=1,
                max_value=5,
                value=3
            )
        else:
            max_history_turns = 0
        
        use_fewshot = st.checkbox(
            "Few-shot示例",
            value=True,
            help="在prompt中包含示例回答"
        )
        
        k_documents = st.slider(
            "检索文档数量",
            min_value=2,
            max_value=8,
            value=4,
            help="每次检索返回的文档数量"
        )
        
        st.divider()
        
        # --- 生成参数 ---
        st.subheader("🎚️ 生成参数")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="控制回答的随机性，越高越多样化"
        )
        
        GENERATION_CONFIG['temperature'] = temperature
        
        st.divider()
        
        # --- 知识库状态 (来自原 main_app.py) ---
        st.subheader("📚 知识库状态")
        if st.session_state.get('quiz_retriever'):
            st.success("✅ 出题库已加载")
        else:
            st.info("⏳ 出题库未加载")
            
        if st.session_state.get('rag_retriever'):
            st.success("✅ 问答库已加载")
        else:
            st.info("⏳ 问答库未加载")
        
        st.divider()
        
        # --- 会话控制 ---
        st.subheader("🔄 会话控制")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空对话", use_container_width=True):
                st.session_state.rag_messages = []
                st.rerun()

        with col2:
            if st.button("🔄 重置会话", use_container_width=True):
                if st.session_state.session_db_path and "session_" in st.session_state.session_db_path:
                    if os.path.exists(st.session_state.session_db_path):
                        shutil.rmtree(st.session_state.session_db_path)
                
                for key in list(st.session_state.keys()):
                    if key != 'models_loaded':
                        del st.session_state[key]
                
                st.rerun()
        
        # 【修改】返回所有RAG配置
        return (enable_query_expansion, k_documents, 
                enable_multi_turn, max_history_turns, use_fewshot)


# ==================== 主应用 ====================

def main():
    st.set_page_config(
        page_title="个性化学习测验系统",
        page_icon="📘",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📘 个性化学习测验系统")
    st.caption("上传教材 → 智能出题 → 自动评分 → AI答疑")
    
    # --- 初始化全局模型 ---
    if 'models_loaded' not in st.session_state:
        with st.status("系统初始化中...", expanded=True) as status:
            st.write("🧠 加载语言模型...")
            tokenizer, model, device = load_llm(LLM_MODEL_PATH)
            st.session_state.llm_tokenizer = tokenizer
            st.session_state.llm_model = model
            st.session_state.device = device
            
            st.write("📊 加载检索模型...")
            st.session_state.embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
            
            st.session_state.models_loaded = True
            status.update(label="✅ 系统初始化完成", state="complete", expanded=False)
    
    # --- 初始化会话状态 ---
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if 'session_db_path' not in st.session_state:
        st.session_state.session_db_path = None
    
    # 【修改】使用双检索器架构
    if 'quiz_retriever' not in st.session_state:
        st.session_state.quiz_retriever = None  # 仅PDF，用于出题
        
    if 'rag_retriever' not in st.session_state:
        st.session_state.rag_retriever = None  # 混合知识库，用于问答
    
    # 【新增】缓存默认教材检索器
    if 'static_retriever' not in st.session_state:
        st.session_state.static_retriever = None
    
    if 'quiz_questions' not in st.session_state:
        st.session_state.quiz_questions = []
    
    if 'quiz_report' not in st.session_state:
        st.session_state.quiz_report = None
    
    if 'rag_messages' not in st.session_state:
        st.session_state.rag_messages = []
        
    # 【新增】为推荐 2 做准备
    if 'queued_rag_question' not in st.session_state:
        st.session_state.queued_rag_question = None
    
    # --- 侧边栏配置 (【修改】接收5个返回值) ---
    (enable_query_expansion, k_documents, 
     enable_multi_turn, max_history_turns, use_fewshot) = render_sidebar()
    
    # --- 创建标签页 ---
    tab_upload, tab_quiz, tab_report, tab_rag = st.tabs([
        "📚 上传教材",
        "📝 开始测验",
        "📊 学习报告",
        "🤖 AI助教"
    ])
    
    # ==================== 标签页1：上传教材 ====================
    with tab_upload:
        st.header("📚 上传学习教材")
        st.info("💡 **双知识库架构**：上传PDF后将创建两个知识库 - 一个专门用于出题（仅PDF），另一个用于AI问答（默认教材+PDF混合）")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "选择PDF文件",
                type="pdf",
                help="支持机器学习、深度学习相关教材"
            )
            
            if uploaded_file is not None:
                st.success(f"✓ 已选择: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
                
                # 显示预估处理时间
                estimated_time = max(1, int(uploaded_file.size / (1024 * 1024)))  # 粗略估计：1MB/分钟
                st.caption(f"⏱️ 预估处理时间: {estimated_time}-{estimated_time*2} 分钟")
                
                if st.button("🚀 开始处理", type="primary", use_container_width=True):
                    # 保存文件
                    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
                    temp_path = os.path.join(TEMP_UPLOAD_DIR, f"{st.session_state.session_id}.pdf")
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # 提交后台任务
                    from background_processor import submit_pdf_task
                    
                    task_id = submit_pdf_task(temp_path, uploaded_file.name, st.session_state.session_id)
                    st.session_state.processing_task_id = task_id
                    
                    st.success("✅ 上传成功！您的教材正在后台处理中...")
                    st.info(f"""
💡 **处理说明：**
- 教材正在后台处理，这大约需要 {estimated_time}-{estimated_time*2} 分钟
- 您可以关闭此页面，稍后再回来查看
- 处理完成后将自动通知您
- 任务ID: `{task_id[:16]}...`
                    """)
                    st.rerun()
            
            # 显示处理状态
            if 'processing_task_id' in st.session_state:
                st.divider()
                st.markdown("### 📊 处理状态")
                
                from background_processor import get_task_status, TaskStatus
                
                task = get_task_status(st.session_state.processing_task_id)
                
                if task:
                    # 状态显示
                    if task.status == TaskStatus.PROCESSING:
                        st.info(f"⏳ 处理中... {task.progress}%")
                        st.progress(task.progress / 100)
                        st.caption(f"当前: {task.message}")
                        
                        # 自动刷新
                        if st.button("🔄 刷新状态"):
                            st.rerun()
                        
                        st.caption("💡 页面将自动刷新，请稍候...")
                        
                    elif task.status == TaskStatus.COMPLETED:
                        st.success("✅ 处理完成！")
                        st.balloons()
                        
                        # 【核心修改】加载双知识库
                        if task.db_path:
                            embedding_model = st.session_state.embedding_model
                            
                            # 1. 【新增】确保默认教材检索器已加载
                            if st.session_state.static_retriever is None:
                                print("📚 首次加载默认教材检索器...")
                                static_retriever = load_static_retriever(STATIC_VECTOR_DB_PATH, embedding_model)
                                st.session_state.static_retriever = static_retriever
                            else:
                                static_retriever = st.session_state.static_retriever
                            
                            # 2. 创建"仅PDF"的检索器（用于出题）
                            session_db = Chroma(
                                persist_directory=task.db_path,
                                embedding_function=embedding_model
                            )
                            session_retriever = create_retriever_from_db(session_db, embedding_model)
                            
                            # 3. 【修改】设置出题检索器（仅PDF）
                            st.session_state.quiz_retriever = session_retriever
                            print("✓ 出题检索器已设置 (仅PDF)")
                            
                            # 4. 【新增】创建混合检索器（默认 + PDF，用于RAG问答）
                            if static_retriever:
                                hybrid_rag_retriever = EnsembleRetriever(
                                    retrievers=[static_retriever, session_retriever],
                                    weights=[0.5, 0.5]  # 可调整权重
                                )
                                st.session_state.rag_retriever = hybrid_rag_retriever
                                print("✓ 混合RAG检索器创建成功 (默认教材 + PDF)")
                            else:
                                # 如果默认库加载失败，RAG也只能用PDF
                                st.session_state.rag_retriever = session_retriever
                                print("⚠ 默认库未加载，RAG使用PDF库")
                            
                            st.session_state.session_db_path = task.db_path
                            st.session_state.rag_messages = []
                            
                            st.success(f"🎉 双知识库已就绪！共生成 {task.chunk_count} 个知识块")
                            st.info("""
📚 **知识库说明：**
- 🎯 出题知识库：仅使用您上传的PDF
- 🤖 问答知识库：混合默认教材 + 您的PDF（更全面）
                            """)
                            st.info("👉 请切换到 **「开始测验」** 或 **「AI助教」** 标签页")
                            
                            # 清理任务状态
                            del st.session_state.processing_task_id
                        
                    elif task.status == TaskStatus.FAILED:
                        st.error(f"❌ 处理失败: {task.error}")
                        
                        if st.button("🔄 重新尝试"):
                            del st.session_state.processing_task_id
                            st.rerun()
                    
                    elif task.status == TaskStatus.PENDING:
                        st.info("⏰ 等待处理...")
                        if st.button("🔄 刷新状态"):
                            st.rerun()
        
        with col2:
            st.markdown("### 使用默认教材")
            st.caption("包含经典ML/DL教材")
            
            if st.button("📖 加载默认教材", use_container_width=True):
                with st.spinner("正在加载..."):
                    embedding_model = st.session_state.embedding_model
                    
                    retriever = load_static_retriever(STATIC_VECTOR_DB_PATH, embedding_model)
                    
                    if retriever:
                        st.session_state.session_db_path = STATIC_VECTOR_DB_PATH
                        
                        # 【修改】缓存默认教材检索器
                        st.session_state.static_retriever = retriever
                        
                        # 【修改】两个检索器都指向默认教材
                        st.session_state.quiz_retriever = retriever
                        st.session_state.rag_retriever = retriever
                        
                        st.session_state.rag_messages = []
                        
                        st.success("✅ 默认教材加载成功")
                        st.info("👉 可以开始使用测验或问答功能")
                    else:
                        st.error("❌ 加载失败")
    
    # ==================== 标签页2：开始测验 ====================
    with tab_quiz:
        st.header("📝 个性化测验")
        
        # 【修改】检查出题知识库
        if st.session_state.quiz_retriever is None:
            st.warning("⚠️ 请先上传或加载教材")
        else:
            # 显示当前使用的知识库类型
            if st.session_state.session_db_path == STATIC_VECTOR_DB_PATH:
                st.success("✓ 出题知识库：默认教材")
            else:
                st.success("✓ 出题知识库：您上传的PDF")
            
            if 'quiz_stage' not in st.session_state:
                st.session_state.quiz_stage = 'config'
            
            # ==================== 配置测验 ====================
            if st.session_state.quiz_stage == 'config':
                st.subheader("🎯 测验配置")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 题目设置")
                    num_choice = st.slider("选择题数量", 1, 10, 3)
                    num_boolean = st.slider("判断题数量", 1, 10, 2)
                    total_questions = num_choice + num_boolean
                    st.info(f"📝 总题数: **{total_questions}** 道")
                
                with col2:
                    st.markdown("#### 难度设置")
                    difficulty = st.select_slider(
                        "选择难度",
                        options=["easy", "medium", "hard"],
                        value="medium",
                        format_func=lambda x: {"easy": "🟢 简单", "medium": "🟡 中等", "hard": "🔴 困难"}[x]
                    )
                    
                    st.markdown("""
                    - 🟢 **简单**: 基础概念
                    - 🟡 **中等**: 概念应用
                    - 🔴 **困难**: 深度分析
                    """)
                
                st.divider()
                
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                
                with col_btn2:
                    if st.button("🚀 生成测验", type="primary", use_container_width=True):
                        with st.spinner("🎯 正在生成题目..."):
                            try:
                                # 【修改】使用出题知识库
                                questions = question_generator.generate_quiz_questions(
                                    retriever=st.session_state.quiz_retriever,
                                    tokenizer=st.session_state.llm_tokenizer,
                                    model=st.session_state.llm_model,
                                    device=st.session_state.device,
                                    num_choice=num_choice,
                                    num_boolean=num_boolean,
                                    difficulty=difficulty,
                                    max_retries=3
                                )
                                
                                if questions and len(questions) > 0:
                                    st.session_state.quiz_questions = questions
                                    st.session_state.quiz_stage = 'answering'
                                    st.session_state.quiz_report = None
                                    st.success(f"✅ 成功生成 {len(questions)} 道题目")
                                    st.rerun()
                                else:
                                    st.error("❌ 生成失败，请重试")
                            
                            except Exception as e:
                                st.error(f"❌ 出错: {e}")
            
            # ==================== 答题中 ====================
            elif st.session_state.quiz_stage == 'answering':
                questions = st.session_state.quiz_questions
                
                col_info1, col_info2, col_info3 = st.columns([2, 2, 1])
                
                with col_info1:
                    st.metric("📝 题目总数", f"{len(questions)} 道")
                
                with col_info2:
                    choice_count = sum(1 for q in questions if q['type'] == 'choice')
                    boolean_count = len(questions) - choice_count
                    st.metric("📋 题型", f"选择 {choice_count} / 判断 {boolean_count}")
                
                with col_info3:
                    if st.button("🔄 重新生成"):
                        st.session_state.quiz_stage = 'config'
                        st.session_state.quiz_questions = []
                        st.rerun()
                
                st.divider()
                
                with st.form("quiz_form"):
                    st.markdown("### 📝 请作答")
                    
                    user_answers_list = []
                    
                    for i, question in enumerate(questions):
                        q_type_emoji = "📋" if question["type"] == "choice" else "❓"
                        q_type_text = "选择题" if question["type"] == "choice" else "判断题"
                        
                        st.markdown(f"#### {q_type_emoji} 第 {i+1} 题 ({q_type_text})")
                        st.markdown(f"**{question['question']}**")
                        
                        options = question["options"]
                        
                        selected = st.radio(
                            f"请选择答案（第{i+1}题）",
                            options=options,
                            key=f"q_{i}",
                            index=None,
                            label_visibility="collapsed"
                        )
                        
                        user_answers_list.append(selected)
                        st.divider()
                    
                    col_submit1, col_submit2, col_submit3 = st.columns([1, 2, 1])
                    
                    with col_submit2:
                        submitted = st.form_submit_button(
                            "📊 提交测验",
                            type="primary",
                            use_container_width=True
                        )
                    
                    if submitted:
                        unanswered_count = user_answers_list.count(None)
                        
                        if unanswered_count > 0:
                            st.warning(f"⚠️ 还有 {unanswered_count} 道题未作答")
                        
                        try:
                            score_data = evaluator.grade_quiz(questions, user_answers_list)
                            st.session_state.quiz_report = score_data
                            st.session_state.quiz_stage = 'completed'
                            st.balloons()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 判分失败: {e}")
            
            # ==================== 已完成 ====================
            elif st.session_state.quiz_stage == 'completed':
                if st.session_state.quiz_report is None:
                    st.error("❌ 找不到测验结果")
                    st.session_state.quiz_stage = 'config'
                    st.rerun()
                
                report = st.session_state.quiz_report
                
                st.subheader("🎉 测验完成")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 总分", f"{report['score_percentage']:.1f}%")
                
                with col2:
                    st.metric("✅ 正确", f"{report['correct']} 题")
                
                with col3:
                    st.metric("❌ 错误", f"{report['wrong']} 题")
                
                with col4:
                    if report.get('unanswered', 0) > 0:
                        st.metric("⭕ 未答", f"{report['unanswered']} 题")
                    else:
                        st.metric("📝 总数", f"{report['total']} 题")
                
                from quiz_module.evaluator import get_performance_level
                performance = get_performance_level(report['score_percentage'])
                
                st.markdown(f"### 评级: :{performance['color']}[{performance['emoji']} {performance['level']}]")
                st.info(performance['message'])
                
                st.divider()
                
                st.subheader("📋 答题详情")
                
                correct_results = [r for r in report['results'] if r['is_correct']]
                wrong_results = [r for r in report['results'] if not r['is_correct'] and not r.get('is_unanswered', False)]
                
                tab_wrong, tab_correct, tab_all = st.tabs([
                    f"❌ 错题 ({len(wrong_results)})",
                    f"✅ 正确 ({len(correct_results)})",
                    f"📝 全部 ({report['total']})"
                ])
                
                with tab_wrong:
                    if len(wrong_results) == 0:
                        st.success("🎉 没有错题！")
                    else:
                        for result in wrong_results:
                            _display_question_result(result, expanded=True)
                
                with tab_correct:
                    if len(correct_results) == 0:
                        st.warning("😅 加油！")
                    else:
                        for result in correct_results:
                            _display_question_result(result, expanded=False)
                
                with tab_all:
                    for result in report['results']:
                        _display_question_result(result, expanded=not result['is_correct'])
                
                st.divider()
                
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                
                with col_btn1:
                    if st.button("📊 查看报告", use_container_width=True):
                        st.info("👉 请切换到「学习报告」标签页")
                
                with col_btn2:
                    if st.button("🔄 重新测验", use_container_width=True):
                        st.session_state.quiz_stage = 'config'
                        st.session_state.quiz_questions = []
                        st.session_state.quiz_report = None
                        st.rerun()
                
                with col_btn3:
                    if st.button("💾 导出结果", use_container_width=True):
                        import json
                        
                        export_data = {
                            "timestamp": datetime.now().isoformat(),
                            "score": report,
                            "questions": st.session_state.quiz_questions
                        }
                        
                        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
                        
                        st.download_button(
                            label="📥 下载 (JSON)",
                            data=json_str,
                            file_name=f"quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )

    # ==================== 标签页3：学习报告 ====================
    with tab_report:
        st.header("📊 学习报告")
        
        if st.session_state.quiz_report is None:
            st.info("📝 完成测验后将显示详细报告")
            
            st.markdown("""
            **报告内容：**
            - 📈 成绩和评级
            - 🎯 知识点掌握度分析
            - 💡 薄弱知识点识别
            - 📚 个性化学习建议
            - 📊 可视化图表
            """)
        else:
            report = st.session_state.quiz_report
            
            st.success("✓ 报告已生成")
            
            st.subheader("📈 测验概览")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("📝 总题数", f"{report['total']}")
            
            with col2:
                st.metric("✅ 正确", f"{report['correct']}", delta=f"+{report['correct']}")
            
            with col3:
                st.metric("❌ 错误", f"{report['wrong']}", delta=f"-{report['wrong']}" if report['wrong'] > 0 else "0")
            
            with col4:
                st.metric("💯 得分", f"{report['score_percentage']:.1f}%")
            
            with col5:
                from quiz_module.evaluator import get_performance_level
                performance = get_performance_level(report['score_percentage'])
                st.metric("🏆 评级", f"{performance['emoji']} {performance['level']}")
            
            st.divider()
            
            st.subheader("📊 数据可视化")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 📋 答题分布")
                
                chart_df = report_generator.prepare_chart_data(report)
                
                if not chart_df.empty:
                    chart = alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X('类别', axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('数量'),
                        tooltip=['类别', '数量']
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("暂无数据")
            
            with col_chart2:
                st.markdown("#### 🎯 题型准确率")
                
                type_df = report_generator.prepare_type_accuracy_data(report)
                
                if type_df is not None and not type_df.empty:
                    chart = alt.Chart(type_df).mark_bar().encode(
                        x=alt.X('题型', axis=alt.Axis(labelAngle=0)),
                        y=alt.Y('准确率', title='准确率 (%)'),
                        tooltip=['题型', '准确率']
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("暂无数据")
            
            st.divider()
            
            st.subheader("🤖 AI 学习反馈")
            
            if 'ai_feedback' not in st.session_state or st.session_state.get('feedback_report_id') != id(report):
                with st.spinner("🧠 AI正在分析..."):
                    try:
                        feedback = report_generator.generate_study_feedback(
                            tokenizer=st.session_state.llm_tokenizer,
                            model=st.session_state.llm_model,
                            device=st.session_state.device,
                            report_data=report
                        )
                        
                        st.session_state.ai_feedback = feedback
                        st.session_state.feedback_report_id = id(report)
                        
                    except Exception as e:
                        st.error(f"❌ 生成失败: {e}")
                        feedback = report_generator.generate_fallback_feedback(report)
                        st.session_state.ai_feedback = feedback
            else:
                feedback = st.session_state.ai_feedback
            
            suggested_questions = re.findall(r'["“](.*?)[”"]', feedback)
            
            parts = re.split(r'["“].*?[”"]', feedback)

            st.markdown(parts[0])

            if suggested_questions:
                for i, question in enumerate(suggested_questions):
                    # 为每个问题创建一个唯一的key
                    button_key = f"suggest_q_{i}"
                    
                    # 创建按钮，点击后执行跳转逻辑
                    if st.button(f"🤖 助教：{question}", key=button_key, use_container_width=True):
                        st.session_state.queued_rag_question = question
                        st.success(f"已将问题发送到AI助教！请切换标签页查看。")
                        
                    # 显示按钮后的文本部分
                    if (i + 1) < len(parts):
                        st.markdown(parts[i+1])
            else:
                # 如果没有提取到问题，就显示完整的反馈
                if len(parts) > 1:
                     st.markdown("".join(parts[1:]))
                     
            st.divider()
            
            col_action1, col_action2, col_action3 = st.columns(3)
            
            with col_action1:
                if st.button("🤖 前往AI助教", use_container_width=True):
                    st.info("👉 请切换到「AI助教」标签页")
            
            with col_action2:
                if st.button("🔄 重新生成", use_container_width=True):
                    if 'ai_feedback' in st.session_state:
                        del st.session_state.ai_feedback
                    if 'feedback_report_id' in st.session_state:
                        del st.session_state.feedback_report_id
                    st.rerun()
            
            with col_action3:
                export_format = st.selectbox(
                    "导出格式",
                    options=["TXT", "PDF"],
                    key="export_format"
                )
                
                if st.button("📥 导出报告", use_container_width=True):
                    try:
                        feedback_text = st.session_state.get('ai_feedback', '未生成')
                        
                        if export_format == "TXT":
                            text_report = report_generator.export_report_to_text(
                                report_data=report,
                                feedback=feedback_text
                            )
                            
                            st.download_button(
                                label="💾 下载 (TXT)",
                                data=text_report,
                                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        elif export_format == "PDF":
                            pdf_buffer = report_generator.export_report_to_pdf(
                                report_data=report,
                                feedback=feedback_text
                            )
                            
                            st.download_button(
                                label="💾 下载 (PDF)",
                                data=pdf_buffer,
                                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                    
                    except Exception as e:
                        st.error(f"导出失败: {e}")
    
    # ==================== 标签页4：AI助教 (【替换】增强版) ====================
    with tab_rag:
        st.header("🤖 AI智能助教")
        
        # 【修改】检查问答知识库
        if st.session_state.rag_retriever is None:
            st.warning("⚠️ 请先上传或加载教材")
        else:
            # 显示当前使用的知识库类型
            if st.session_state.session_db_path == STATIC_VECTOR_DB_PATH:
                st.info("💡 基于 **默认教材** 回答问题")
            else:
                st.success("💡 基于 **默认教材 + 您上传的PDF（混合知识库）** 回答问题")
            
            # 【新增】检查是否有来自报告页的排队问题
            if st.session_state.get("queued_rag_question"):
                # 获取问题并立即清除队列
                user_question = st.session_state.queued_rag_question
                del st.session_state.queued_rag_question
                
                # 【关键】将这个问题模拟为用户刚刚的输入
                st.session_state.rag_messages.append({"role": "user", "content": user_question})
                
                # 立即执行一次RAG流程 (复制粘贴下方的流式逻辑)
                with st.chat_message("user"):
                    st.markdown(user_question)
                
                with st.chat_message("assistant"):
                    status_container = st.empty()
                    with status_container.status("🔍 正在检索...", expanded=False):
                        context, sources, docs = retrieve_with_enhancements(
                            st.session_state.rag_retriever,
                            user_question,
                            k=k_documents,
                            enable_expansion=enable_query_expansion
                        )
                    
                    if not docs:
                        full_response = "抱歉，未找到相关信息。请尝试换个方式提问。"
                        st.markdown(full_response)
                        st.session_state.rag_messages.append({
                            "role": "assistant", "content": full_response, "sources": [], "question": user_question
                        })
                    else:
                        dialogue_history = None
                        if enable_multi_turn:
                            with status_container.status("💭 分析对话...", expanded=False):
                                dialogue_history = extract_dialogue_context(
                                    st.session_state.rag_messages[:-1],
                                    max_history=max_history_turns
                                )
                        
                        with status_container.status("✍️ 正在生成...", expanded=False):
                            messages = build_enhanced_prompt(
                                context, user_question, dialogue_history, use_fewshot, enable_multi_turn
                            )
                            response_placeholder = st.empty()
                            full_response = ""
                            try:
                                for chunk in generate_response_stream(
                                    st.session_state.llm_tokenizer, st.session_state.llm_model, st.session_state.device, messages
                                ):
                                    full_response += chunk
                                    response_placeholder.markdown(full_response + "▌")
                                response_placeholder.markdown(full_response)
                                status_container.empty()
                            except Exception as e:
                                st.error(f"❌ 生成出错: {e}")
                                full_response = "抱歉，生成时遇到问题。"
                                response_placeholder.markdown(full_response)
                        
                        st.session_state.rag_messages.append({
                            "role": "assistant", "content": full_response, "sources": sources, "question": user_question
                        })
                        st.rerun() # 立即重跑以显示新消息和反馈按钮

            # 显示聊天历史
            for i, message in enumerate(st.session_state.rag_messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # 仅为助教消息显示 来源 和 反馈
                    if message["role"] == "assistant":
                        # 引用来源
                        if "sources" in message and message["sources"]:
                            with st.expander("📚 引用来源"):
                                for j, source in enumerate(message["sources"], 1):
                                    st.text(f"{j}. {source}")
                        
                        # 反馈按钮 (使用唯一的key)
                        st.caption("反馈")
                        col_like, col_dislike, _ = st.columns([1, 1, 8])
                        
                        with col_like:
                            if st.button("👍", key=f"like_{i}"):
                                save_feedback(
                                    message.get("question", ""), # 获取对应的问题
                                    message["content"],
                                    "helpful"
                                )
                                st.toast("感谢反馈！")
                        
                        with col_dislike:
                            if st.button("👎", key=f"dislike_{i}"):
                                save_feedback(
                                    message.get("question", ""), # 获取对应的问题
                                    message["content"],
                                    "unhelpful"
                                )
                                st.toast("感谢反馈！")
            
            # 用户输入
            if user_question := st.chat_input("💭 请输入问题..."):
                st.session_state.rag_messages.append({
                    "role": "user",
                    "content": user_question
                })
                
                with st.chat_message("user"):
                    st.markdown(user_question)
                
                # 开始流式回答
                with st.chat_message("assistant"):
                    status_container = st.empty() # 用于显示状态
                    
                    # 1. 检索
                    with status_container.status("🔍 正在检索...", expanded=False):
                        context, sources, docs = retrieve_with_enhancements(
                            st.session_state.rag_retriever,
                            user_question,
                            k=k_documents,
                            enable_expansion=enable_query_expansion
                        )
                    
                    if not docs:
                        full_response = "抱歉，未找到相关信息。请尝试换个方式提问。"
                        st.markdown(full_response)
                        
                        # 保存无答案的回答
                        st.session_state.rag_messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "sources": [],
                            "question": user_question
                        })
                    else:
                        # 2. 提取对话历史
                        dialogue_history = None
                        if enable_multi_turn:
                            with status_container.status("💭 分析对话...", expanded=False):
                                dialogue_history = extract_dialogue_context(
                                    st.session_state.rag_messages[:-1], # 排除当前问题
                                    max_history=max_history_turns
                                )
                        
                        # 3. 生成回答
                        with status_container.status("✍️ 正在生成...", expanded=False):
                            messages = build_enhanced_prompt(
                                context,
                                user_question,
                                dialogue_history=dialogue_history,
                                use_fewshot=use_fewshot,
                                use_multi_turn=enable_multi_turn
                            )
                            
                            response_placeholder = st.empty()
                            full_response = ""
                            
                            try:
                                for chunk in generate_response_stream(
                                    st.session_state.llm_tokenizer,
                                    st.session_state.llm_model,
                                    st.session_state.device,
                                    messages
                                ):
                                    full_response += chunk
                                    response_placeholder.markdown(full_response + "▌")
                                
                                response_placeholder.markdown(full_response) # 最终显示
                                status_container.empty() # 清空状态
                                
                            except Exception as e:
                                st.error(f"❌ 生成出错: {e}")
                                full_response = "抱歉，生成时遇到问题。"
                                response_placeholder.markdown(full_response)
                        
                        # 4. 保存完整回答到历史
                        st.session_state.rag_messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "sources": sources,
                            "question": user_question # 保存对应的问题，用于反馈
                        })
                        
                        # 5. 立即重新运行以显示反馈按钮
                        st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"应用运行出错: {e}")
        import traceback
        st.code(traceback.format_exc())