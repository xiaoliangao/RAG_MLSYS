# module_rag_assistant.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from threading import Thread
from typing import List, Dict, Tuple
import json
from datetime import datetime
import hashlib
import os

# --- 全局配置 ---
VECTOR_DB_PATH = "./vector_db"
EMBEDDING_MODEL_NAME = "./models/bge-large-zh-v1.5"
LLM_MODEL_PATH = "./models/Qwen2.5-7B-Instruct"
FEEDBACK_DB_PATH = "./feedback_db"

GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
}

# Few-shot示例 - 优化版
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

# --- 混合检索器 ---

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

# --- 缓存加载函数 ---

@st.cache_resource
def load_retriever(db_path, model_name):
    """加载检索引擎"""
    with st.spinner("正在加载检索引擎..."):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        db = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_model
        )
        
        vector_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )
        
        try:
            all_data = db.get()
            if all_data and all_data.get('documents'):
                docs = [Document(page_content=doc, metadata=meta) 
                       for doc, meta in zip(all_data['documents'], 
                                           all_data.get('metadatas', [{}]*len(all_data['documents'])))]
                bm25_retriever = BM25Retriever.from_documents(docs)
                bm25_retriever.k = 6
                
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    weights=[0.6, 0.4]
                )
                st.success("✓ 混合检索器已就绪")
                return ensemble_retriever
        except Exception as e:
            st.warning(f"使用向量检索")
        
        return vector_retriever


@st.cache_resource
def load_llm(model_path):
    """加载LLM模型"""
    with st.spinner("正在加载语言模型..."):
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
            return tokenizer, model, device
            
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            raise


# --- 查询优化 ---

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


# --- 检索函数 ---

def retrieve_with_enhancements(retriever, query, k=4, enable_expansion=True):
    """增强检索"""
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


# --- Prompt构建 - 优化版 ---

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
            "answer": answer[:200],
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


# --- 侧边栏 ---

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.header("⚙️ 系统设置")
        
        st.divider()
        
        st.subheader("🎯 检索优化")
        
        enable_query_expansion = st.checkbox(
            "查询扩展",
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
        
        st.divider()
        
        st.subheader("🎛️ 生成参数")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="控制回答的创造性"
        )
        
        top_p = st.slider(
            "Top-p",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=512,
            max_value=4096,
            value=2048,
            step=256
        )
        
        GENERATION_CONFIG['temperature'] = temperature
        GENERATION_CONFIG['top_p'] = top_p
        GENERATION_CONFIG['max_new_tokens'] = max_tokens
        
        st.divider()
        
        st.subheader("🔍 检索配置")
        
        k_documents = st.slider(
            "检索文档数",
            min_value=2,
            max_value=8,
            value=4
        )
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🔄 重生成", use_container_width=True):
                if len(st.session_state.get('messages', [])) >= 2:
                    st.session_state.messages = st.session_state.messages[:-1]
                    st.rerun()
        
        return (k_documents, enable_query_expansion, enable_multi_turn, 
                max_history_turns, use_fewshot)


# --- 主应用 ---

def main():
    st.set_page_config(
        page_title="ML/DL AI助教",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 ML/DL AI智能助教")
    st.caption("基于检索增强生成的智能问答系统")
    
    (k_documents, enable_query_expansion, enable_multi_turn, 
     max_history_turns, use_fewshot) = render_sidebar()
    
    # 初始化模型
    if 'models_loaded' not in st.session_state:
        with st.status("正在初始化...", expanded=True) as status:
            try:
                st.write("📥 加载检索引擎...")
                retriever = load_retriever(VECTOR_DB_PATH, EMBEDDING_MODEL_NAME)
                st.session_state.retriever = retriever
                
                st.write("🧠 加载语言模型...")
                tokenizer, model, device = load_llm(LLM_MODEL_PATH)
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.session_state.device = device
                
                st.session_state.models_loaded = True
                status.update(label="✅ 系统准备就绪", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="❌ 初始化失败", state="error", expanded=True)
                st.error(f"错误: {e}")
                st.stop()
    
    # 初始化聊天记录
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 欢迎消息
    if len(st.session_state.messages) == 0:
        with st.chat_message("assistant"):
            st.markdown("""
👋 您好！我是您的ML/DL学习助教。

**我能帮您：**
- 📚 解释机器学习和深度学习概念
- 🔍 基于教材提供准确的技术解答
- 💡 提供学习建议和知识点梳理
- 🗣️ 进行连贯的多轮对话交流

**功能特性：**
- 🎯 **智能检索** - 混合向量检索+关键词匹配
- 🔄 **查询优化** - 自动扩展查询提高覆盖率
- 💬 **对话记忆** - 理解上下文，连贯交流
- 📖 **引用来源** - 每个回答都标注参考资料

**提问建议：**
- "什么是注意力机制？"
- "对比Adam和SGD优化器的优缺点"
- "如何解决梯度消失问题？"
- "继续解释其中的数学原理"（多轮对话）

现在就开始提问吧！ 🚀
            """)
    
    # 显示聊天历史
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant":
                if "sources" in message:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        with st.expander("📚 查看引用来源"):
                            for j, source in enumerate(message["sources"], 1):
                                st.text(f"{j}. {source}")
                    
                    with col2:
                        st.caption("反馈")
                        col_like, col_dislike = st.columns(2)
                        
                        with col_like:
                            if st.button("👍", key=f"like_{i}"):
                                save_feedback(
                                    message.get("question", ""),
                                    message["content"],
                                    "helpful"
                                )
                                st.toast("感谢反馈！")
                        
                        with col_dislike:
                            if st.button("👎", key=f"dislike_{i}"):
                                save_feedback(
                                    message.get("question", ""),
                                    message["content"],
                                    "unhelpful"
                                )
                                st.toast("感谢反馈！")
    
    # 处理用户输入
    if user_question := st.chat_input("💭 请输入问题..."):
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })
        
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # 生成回答
        with st.chat_message("assistant"):
            status_container = st.empty()
            
            # 检索
            with status_container.status("🔍 正在检索...", expanded=False) as status:
                context, sources, docs = retrieve_with_enhancements(
                    st.session_state.retriever,
                    user_question,
                    k=k_documents,
                    enable_expansion=enable_query_expansion
                )
                
                status_info = []
                if enable_query_expansion:
                    status_info.append("✓ 查询扩展")
                status_info.append("✓ 混合检索")
                
                st.write(", ".join(status_info))
                st.write(f"✓ 检索到 {len(docs)} 个文档")
            
            if not docs:
                st.error("❌ 未找到相关信息")
                full_response = "抱歉，未找到相关信息。请尝试换个方式提问。"
                st.markdown(full_response)
            else:
                # 提取对话历史
                dialogue_history = None
                if enable_multi_turn and len(st.session_state.messages) > 2:
                    with status_container.status("💭 分析对话...", expanded=False):
                        dialogue_history = extract_dialogue_context(
                            st.session_state.messages[:-1],
                            max_history=max_history_turns
                        )
                        if dialogue_history:
                            st.write(f"✓ 包含 {max_history_turns} 轮对话")
                
                # 生成回答
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
                            st.session_state.tokenizer,
                            st.session_state.model,
                            st.session_state.device,
                            messages
                        ):
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")
                        
                        response_placeholder.markdown(full_response)
                        status_container.empty()
                        
                    except Exception as e:
                        st.error(f"❌ 生成出错: {e}")
                        full_response = "抱歉，生成时遇到问题。"
                        response_placeholder.markdown(full_response)
                
                # 显示来源和反馈
                if sources:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        with st.expander("📚 查看引用来源"):
                            for i, source in enumerate(sources, 1):
                                st.text(f"{i}. {source}")
                    
                    with col2:
                        st.caption("反馈")
                        col_like, col_dislike = st.columns(2)
                        
                        with col_like:
                            if st.button("👍", key=f"new_like"):
                                save_feedback(user_question, full_response, "helpful")
                                st.toast("感谢反馈！")
                        
                        with col_dislike:
                            if st.button("👎", key=f"new_dislike"):
                                save_feedback(user_question, full_response, "unhelpful")
                                st.toast("感谢反馈！")
            
            # 保存到历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "question": user_question
            })
            
            st.rerun()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"应用运行出错: {e}")
        import traceback
        st.code(traceback.format_exc())