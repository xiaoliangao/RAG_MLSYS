# core_indexing.py
import os
import pickle
import uuid
from typing import List, Optional
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import torch
import shutil

# --- 配置参数 ---
EMBEDDING_MODEL_NAME = "./models/bge-large-zh-v1.5"
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 512


# ==================== 核心动态构建函数 ====================

def build_session_vector_db(
    chunks: List[Document],
    session_id: str,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    base_db_path: str = "./vector_db"
) -> tuple[str, Chroma]:
    """
    为特定会话创建独立的向量数据库
    
    Args:
        chunks: 知识块列表
        session_id: 会话ID（用于隔离不同用户/文件）
        embedding_model: 预加载的embedding模型（如果为None则创建新的）
        base_db_path: 数据库基础路径
    
    Returns:
        tuple: (数据库路径, 数据库对象)
    """
    # 1. 定义会话专属的数据库路径
    session_db_path = os.path.join(base_db_path, f"session_{session_id}")
    
    # 2. 清理旧数据库（如果存在）
    if os.path.exists(session_db_path):
        shutil.rmtree(session_db_path)
        print(f"  清理旧数据库: {session_db_path}")
    
    os.makedirs(session_db_path, exist_ok=True)
    print(f"开始为会话 [{session_id}] 创建向量数据库")
    print(f"  目标路径: {session_db_path}")
    
    # 3. 初始化或使用传入的embedding模型
    if embedding_model is None:
        print("  初始化Embedding模型...")
        embedding_model = initialize_embedding_model()
    else:
        print("  使用已加载的Embedding模型")
    
    # 4. 预处理知识块
    print(f"  预处理 {len(chunks)} 个知识块...")
    filtered_chunks = filter_chunks(chunks)
    processed_chunks = truncate_long_chunks(filtered_chunks)
    print(f"  ✓ 预处理完成，保留 {len(processed_chunks)} 个有效片段")
    
    # 5. 创建向量数据库
    print("  开始向量化...")
    db = create_vector_db(processed_chunks, embedding_model, session_db_path)
    
    print(f"✓ 会话 [{session_id}] 的向量数据库创建成功！")
    print(f"  共索引 {len(processed_chunks)} 个知识片段")
    
    return session_db_path, db


def load_session_vector_db(
    session_id: str,
    embedding_model: Optional[HuggingFaceEmbeddings] = None,
    base_db_path: str = "./vector_db"
) -> Optional[Chroma]:
    """
    加载已存在的会话数据库
    
    Args:
        session_id: 会话ID
        embedding_model: embedding模型
        base_db_path: 数据库基础路径
    
    Returns:
        Chroma数据库对象，如果不存在则返回None
    """
    session_db_path = os.path.join(base_db_path, f"session_{session_id}")
    
    if not os.path.exists(session_db_path):
        print(f"会话 [{session_id}] 的数据库不存在")
        return None
    
    if embedding_model is None:
        embedding_model = initialize_embedding_model()
    
    db = Chroma(
        persist_directory=session_db_path,
        embedding_function=embedding_model
    )
    
    print(f"✓ 加载会话 [{session_id}] 的数据库")
    print(f"  向量总数: {db._collection.count()}")
    
    return db


# ==================== Embedding模型管理 ====================

def initialize_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    初始化Embedding模型（应该被@st.cache_resource缓存）
    
    Args:
        model_name: 模型路径
    
    Returns:
        HuggingFaceEmbeddings对象
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"初始化 Embedding 模型: {model_name}")
    print(f"  设备: {device}")
    
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  可用显存: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("✓ Embedding模型加载成功")
    return embedding_model


# ==================== 数据预处理 ====================

def filter_chunks(chunks: List[Document]) -> List[Document]:
    """
    过滤知识块，移除不需要向量化的内容
    """
    original_count = len(chunks)
    
    # 只保留正文内容
    filtered_chunks = [
        chunk for chunk in chunks 
        if chunk.metadata.get('page_type', 'content') == 'content'
    ]
    
    filtered_count = len(filtered_chunks)
    removed_count = original_count - filtered_count
    
    if removed_count > 0:
        print(f"  过滤: 移除 {removed_count} 个非正文片段 ({removed_count/original_count*100:.1f}%)")
    
    return filtered_chunks


def truncate_long_chunks(chunks: List[Document]) -> List[Document]:
    """
    截断过长的知识块
    """
    truncated = []
    truncated_count = 0
    max_chars = MAX_SEQ_LENGTH * 2
    
    for chunk in chunks:
        if len(chunk.page_content) > max_chars:
            chunk = chunk.copy(update={
                "page_content": chunk.page_content[:max_chars] + "..."
            })
            chunk.metadata["truncated"] = True
            truncated_count += 1
        truncated.append(chunk)
    
    if truncated_count > 0:
        print(f"  截断: {truncated_count} 个过长片段")
    
    return truncated


# ==================== 向量数据库创建 ====================

def create_vector_db(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    db_path: str
) -> Chroma:
    """
    创建向量数据库
    
    Args:
        chunks: 知识块列表
        embedding_model: embedding模型
        db_path: 数据库存储路径
    
    Returns:
        Chroma数据库对象
    """
    try:
        # 分批处理（避免内存溢出）
        if len(chunks) > 500:
            db = _create_db_in_batches(chunks, embedding_model, db_path)
        else:
            db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=db_path,
                collection_metadata={"hnsw:space": "cosine"}
            )
        
        return db
        
    except Exception as e:
        print(f"  ✗ 创建向量数据库失败: {e}")
        raise


def _create_db_in_batches(
    chunks: List[Document],
    embedding_model: HuggingFaceEmbeddings,
    db_path: str,
    batch_size: int = 500
) -> Chroma:
    """
    分批创建向量数据库
    """
    db = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="  向量化进度"):
        batch = chunks[i:i+batch_size]
        
        # 确保每个文档有唯一ID
        new_batch = [
            Document(
                page_content=d.page_content,
                metadata={**d.metadata, "id": str(uuid.uuid4())}
            )
            for d in batch
        ]
        
        if db is None:
            db = Chroma.from_documents(
                documents=new_batch,
                embedding=embedding_model,
                persist_directory=db_path,
                collection_metadata={"hnsw:space": "cosine"}
            )
        else:
            db.add_documents(new_batch)
        
        # 清理GPU缓存
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    return db


# ==================== 批量处理（保留原功能）====================

class OptimizedVectorDBBuilder:
    """
    优化的向量数据库构建器（用于批量处理）
    """
    
    def __init__(self, model_name: str, db_path: str):
        self.model_name = model_name
        self.db_path = db_path
        self.embedding_model = None
        self.device = None
    
    def initialize_embedding_model(self):
        """初始化Embedding模型"""
        self.embedding_model = initialize_embedding_model(self.model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.embedding_model
    
    def filter_chunks(self, chunks: List[Document]) -> List[Document]:
        """预过滤chunks"""
        return filter_chunks(chunks)
    
    def truncate_long_chunks(self, chunks: List[Document]) -> List[Document]:
        """截断过长chunks"""
        return truncate_long_chunks(chunks)
    
    def create_vector_db(self, chunks: List[Document]) -> Chroma:
        """创建向量数据库"""
        print(f"\n正在创建向量数据库于: {self.db_path}")
        
        os.makedirs(self.db_path, exist_ok=True)
        
        db = create_vector_db(chunks, self.embedding_model, self.db_path)
        
        print(f"✓ 向量数据库创建成功！")
        return db
    
    def load_and_test(self, test_queries: Optional[List[str]] = None):
        """加载数据库并测试"""
        print(f"\n测试向量数据库: {self.db_path}")
        
        db = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embedding_model
        )
        
        collection = db._collection
        print(f"  向量总数: {collection.count()}")
        
        if test_queries is None:
            test_queries = [
                "什么是支持向量机？",
                "神经网络的反向传播算法",
                "如何防止过拟合？"
            ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n测试 {i}: {query}")
            
            results = db.similarity_search(
                query,
                k=3,
                filter={'page_type': 'content'}
            )
            
            for j, doc in enumerate(results, 1):
                print(f"  [{j}] {doc.metadata.get('source', 'N/A')} "
                      f"(页 {doc.metadata.get('page', 'N/A')})")
                print(f"      {doc.page_content[:100]}...")
        
        return db


# ==================== 工具函数 ====================

def load_chunks_from_pickle(file_path: str) -> Optional[List[Document]]:
    """从pickle文件加载chunks"""
    try:
        with open(file_path, 'rb') as f:
            chunks = pickle.load(f)
        print(f"✓ 加载 {len(chunks)} 个知识片段")
        return chunks
    except FileNotFoundError:
        print(f"✗ 文件不存在: {file_path}")
        return None
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return None


# ==================== 主程序（保留用于测试）====================

if __name__ == "__main__":
    """
    仅用于测试，生产环境请调用 build_session_vector_db()
    """
    print("="*60)
    print("  向量数据库构建工具（测试模式）")
    print("="*60)
    
    CHUNK_PICKLE_FILE = "./processed_chunks/chunks.pkl"
    VECTOR_DB_PATH = "./vector_db"
    
    # 1. 加载chunks
    all_chunks = load_chunks_from_pickle(CHUNK_PICKLE_FILE)
    
    if all_chunks is None:
        print("请先运行 01_process_data.py 生成chunks文件")
        exit(1)
    
    # 2. 初始化构建器
    builder = OptimizedVectorDBBuilder(EMBEDDING_MODEL_NAME, VECTOR_DB_PATH)
    builder.initialize_embedding_model()
    
    # 3. 检查数据库是否已存在
    db_exists = os.path.exists(os.path.join(VECTOR_DB_PATH, "chroma.sqlite3"))
    
    if db_exists:
        print(f"\n⚠️  数据库已存在: {VECTOR_DB_PATH}")
        choice = input("是否重新创建? (y/n): ").strip().lower()
        
        if choice != 'y':
            print("跳过创建，测试现有数据库...")
            builder.load_and_test()
            exit(0)
        else:
            shutil.rmtree(VECTOR_DB_PATH)
            print("✓ 已删除旧数据库")
    
    # 4. 预处理chunks
    filtered_chunks = builder.filter_chunks(all_chunks)
    processed_chunks = builder.truncate_long_chunks(filtered_chunks)
    
    # 5. 创建向量数据库
    db = builder.create_vector_db(processed_chunks)
    
    # 6. 测试
    builder.load_and_test()
    
    print(f"\n{'='*60}")
    print("✅ 测试完成！")
    print(f"{'='*60}")