# core_processing.py
import os
import re
from typing import List, Optional
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ==================== 核心处理函数 ====================

def process_single_pdf(pdf_path: str, source_name: Optional[str] = None) -> List[Document]:
    """
    处理单个PDF文件（核心函数，用于动态处理上传的文件）
    
    Args:
        pdf_path: PDF文件路径 
        source_name: 来源名称（用于元数据标记）
    
    Returns:
        List[Document]: 处理后的知识块列表
    """
    if source_name is None:
        source_name = os.path.basename(pdf_path)
    
    print(f"正在处理: {source_name}")
    
    try:
        # 1. 加载PDF
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        print(f"  ✓ 成功加载 {len(documents)} 个页面")
        
        # 2. 清洗文档
        cleaned_documents = clean_document_content(documents)
        print(f"  ✓ 清洗完成，保留 {len(cleaned_documents)} 个有效页面")
        
        # 3. 文本分块
        all_chunks = split_text_into_chunks(cleaned_documents)
        print(f"  ✓ 分块完成，生成 {len(all_chunks)} 个知识片段")
        
        # 4. 添加来源元数据（关键：用于后续追踪）
        for chunk in all_chunks:
            chunk.metadata['source'] = source_name
            chunk.metadata['original_path'] = pdf_path
        
        return all_chunks
        
    except Exception as e:
        print(f"  ✗ 处理失败: {e}")
        return []


def process_directory(directory_path: str) -> List[Document]:
    """
    批量处理目录下的所有PDF（保留原有功能）
    
    Args:
        directory_path: 知识库目录路径
    
    Returns:
        List[Document]: 所有文件的知识块列表
    """
    print(f"正在从 '{directory_path}' 批量加载PDF文件...")
    
    # 使用目录加载器
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    
    documents = loader.load()
    print(f"成功加载了 {len(documents)} 个页面。")
    
    # 清洗和分块
    cleaned_documents = clean_document_content(documents)
    all_chunks = split_text_into_chunks(cleaned_documents)
    
    return all_chunks


# ==================== 文档清洗 ====================

def clean_document_content(documents: List[Document]) -> List[Document]:
    """
    对文档内容进行增强清洗
    """
    print("开始清洗文档内容...")
    cleaned_documents = []
    
    for doc in documents:
        text = doc.page_content
        
        # 预先检测页面类型
        page_type = 'content'
        if is_table_of_contents(text):
            page_type = 'toc'
            doc.metadata['page_type'] = 'table_of_contents'
        elif is_glossary_or_index(text):
            page_type = 'glossary'
            doc.metadata['page_type'] = 'glossary'
        elif is_reference_page(text):
            page_type = 'reference'
            doc.metadata['page_type'] = 'reference'
        else:
            doc.metadata['page_type'] = 'content'
        
        # 清洗规则
        text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
        
        if page_type == 'content':
            text = re.sub(r'\(\s*(\d+\.\d+)\s*\)', r'【公式\1】', text)
            text = re.sub(r'(?<![.\s])(定理|引理|证明|推论|命题)\s*(\d+\.\d+)?(?!\s*\.)', 
                         r'\n\n【\1\2】\n', text)
        
        if page_type == 'toc':
            text = re.sub(r'【(定理|引理|证明|推论|命题)[^】]*】', '', text)
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        if page_type == 'content':
            ocr_fixes = {'BA': '为', 'ME': '使', 'sk': '求'}
            for wrong, correct in ocr_fixes.items():
                text = re.sub(rf'\b{wrong}\b', correct, text)
            text = re.sub(r'([=≈≠≤≥<>])', r' \1 ', text)
        
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'^\s*第 \d+ 章.*?\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        doc.page_content = text.strip()
        
        # 过滤过短内容
        if len(doc.page_content) > 100:
            cleaned_documents.append(doc)
    
    print(f"清洗完成。剩余 {len(cleaned_documents)} 个有效页面。")
    return cleaned_documents


# ==================== 页面类型检测 ====================

def is_table_of_contents(text: str) -> bool:
    """检测目录页"""
    dots_pattern = r'\.\s*\.\s*\.\s*\.\s*\d+'
    matches = re.findall(dots_pattern, text)
    if len(matches) > 5:
        return True
    
    toc_pattern = r'[\u4e00-\u9fff\w\s]+\.\s*\.\s*\.\s*\d+'
    toc_matches = re.findall(toc_pattern, text)
    if len(toc_matches) > 5:
        return True
    
    return False


def is_glossary_or_index(text: str) -> bool:
    """检测词汇表或索引页"""
    pattern = r'[\w\u4e00-\u9fff]+\s+\d+(,\s*\d+|–\d+){3,}'
    matches = re.findall(pattern, text)
    if len(matches) > 10:
        return True
    
    word_number_pattern = r'\b[A-Za-z]+\s+[A-Za-z\s]+\d+'
    word_matches = re.findall(word_number_pattern, text)
    if len(word_matches) > 15:
        return True
    
    return False


def is_reference_page(text: str) -> bool:
    """检测参考文献页"""
    reference_patterns = [
        r'\[\d+\]\s*[A-Z]',
        r'et\s+al\.',
        r'\([12]\d{3}\)\.',
        r'^[A-Z][a-z]+,\s*[A-Z\.]',
    ]
    
    matches = 0
    lines = text.split('\n')
    
    for line in lines:
        if any(re.search(p, line.strip()) for p in reference_patterns):
            matches += 1
    
    if len(lines) > 0:
        match_ratio = matches / len(lines)
        if matches > 5 or (match_ratio > 0.3 and len(lines) > 5):
            return True
    
    return False


# ==================== 文本分块 ====================

def split_text_into_chunks(documents: List[Document]) -> List[Document]:
    """
    将文档切分为知识片段
    """
    print("开始进行文本分块...")
    
    separators = [
        "\n\n【定理",
        "\n\n【引理",
        "\n\n【证明",
        "\n\n【公式",
        "\n\n\n",
        "\n\n",
        "。\n",
        "；\n",
        "\n",
        "。",
        "，",
        " ",
        ""
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=separators,
        keep_separator=True,
        length_function=len,
    )
    
    all_chunks = text_splitter.split_documents(documents)
    all_chunks = post_process_chunks(all_chunks)
    
    print(f"成功将文档切分为 {len(all_chunks)} 个知识片段。")
    return all_chunks


def post_process_chunks(chunks: List[Document]) -> List[Document]:
    """后处理分块结果"""
    processed_chunks = []
    skip_next = False
    
    for i in range(len(chunks)):
        if skip_next:
            skip_next = False
            continue
        
        current_chunk = chunks[i]
        content = current_chunk.page_content.strip()
        
        # 过滤过短片段
        if len(content) < 100:
            continue
        
        page_type = current_chunk.metadata.get('page_type', 'content')
        
        if page_type != 'content':
            current_chunk.metadata['is_special_page'] = True
        
        # 检查是否需要合并
        if i < len(chunks) - 1 and page_type == 'content':
            next_chunk = chunks[i + 1]
            next_page_type = next_chunk.metadata.get('page_type', 'content')
            
            if next_page_type == 'content':
                if should_merge_with_next(content, next_chunk.page_content):
                    merged_content = content + "\n" + next_chunk.page_content
                    current_chunk.page_content = merged_content
                    skip_next = True
        
        processed_chunks.append(current_chunk)
    
    return processed_chunks


def should_merge_with_next(current_content: str, next_content: str) -> bool:
    """判断是否应该与下一块合并"""
    if re.search(r'[=+\-*/]$', current_content.strip()):
        return True
    
    if re.search(r'[，；,;]$', current_content.strip()):
        return True
    
    if '【证明' in current_content and '证毕' not in current_content:
        if not next_content.strip().startswith('【'):
            return True
    
    if re.search(r'[（([]$', current_content.strip()):
        return True
    
    return False


# ==================== 质量分析 ====================

def analyze_chunk_quality(chunks: List[Document]) -> dict:
    """分析分块质量"""
    stats = {
        'total_chunks': len(chunks),
        'avg_length': 0,
        'min_length': float('inf'),
        'max_length': 0,
        'formula_chunks': 0,
        'theorem_chunks': 0,
        'proof_chunks': 0,
        'incomplete_chunks': 0,
        'toc_chunks': 0,
        'glossary_chunks': 0,
        'reference_chunks': 0,
        'content_chunks': 0,
    }
    
    total_length = 0
    incomplete_patterns = [
        r'[=+\-*/]$',
        r'[，,]$',
        r'[（(]$',
    ]
    
    for chunk in chunks:
        content = chunk.page_content
        length = len(content)
        
        total_length += length
        stats['min_length'] = min(stats['min_length'], length)
        stats['max_length'] = max(stats['max_length'], length)
        
        page_type = chunk.metadata.get('page_type', 'content')
        if page_type == 'table_of_contents':
            stats['toc_chunks'] += 1
        elif page_type == 'glossary':
            stats['glossary_chunks'] += 1
        elif page_type == 'reference':
            stats['reference_chunks'] += 1
        else:
            stats['content_chunks'] += 1
            
            if '【公式' in content or re.search(r'[=≈≠≤≥]', content):
                stats['formula_chunks'] += 1
            
            if '【定理' in content:
                stats['theorem_chunks'] += 1
            
            if '【证明' in content:
                stats['proof_chunks'] += 1
            
            if any(re.search(pattern, content.strip()) for pattern in incomplete_patterns):
                stats['incomplete_chunks'] += 1
    
    stats['avg_length'] = total_length / len(chunks) if chunks else 0
    
    return stats


def print_quality_report(stats: dict):
    """打印质量报告"""
    print("\n" + "="*50)
    print(" 【分块质量详细报告】 ")
    print("="*50)
    
    print(f"\n📊 基础统计:")
    print(f"  • 总片段数: {stats['total_chunks']}")
    print(f"  • 平均长度: {stats['avg_length']:.0f} 字符")
    print(f"  • 最小长度: {stats['min_length']} 字符")
    print(f"  • 最大长度: {stats['max_length']} 字符")
    
    print(f"\n📄 页面类型分布:")
    print(f"  • 正文内容: {stats['content_chunks']} "
          f"({stats['content_chunks']/stats['total_chunks']*100:.1f}%)")
    print(f"  • 目录页: {stats['toc_chunks']} "
          f"({stats['toc_chunks']/stats['total_chunks']*100:.1f}%)")
    print(f"  • 词汇表/索引: {stats['glossary_chunks']} "
          f"({stats['glossary_chunks']/stats['total_chunks']*100:.1f}%)")
    print(f"  • 参考文献: {stats['reference_chunks']} "
          f"({stats['reference_chunks']/stats['total_chunks']*100:.1f}%)")
    
    print("="*50)


# ==================== 工具函数 ====================

def save_chunks_to_file(chunks: List[Document], output_dir: str = "./processed_chunks"):
    """保存知识块到文件"""
    import json
    import pickle
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为JSON
    chunks_data = []
    for i, chunk in enumerate(chunks):
        chunks_data.append({
            'id': i,
            'content': chunk.page_content,
            'metadata': chunk.metadata
        })
    
    json_path = os.path.join(output_dir, "chunks.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON格式已保存到: {json_path}")
    
    # 保存为Pickle
    pickle_path = os.path.join(output_dir, "chunks.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"✓ Pickle格式已保存到: {pickle_path}")
    
    return json_path, pickle_path


# ==================== 主程序（保留用于测试）====================

if __name__ == "__main__":
    """
    仅用于测试，生产环境请调用 process_single_pdf() 或 process_directory()
    """
    print("文档处理模块 - 测试模式")
    print("="*60)
    
    KB_PATH = "./knowledge_base"
    
    # 测试批量处理
    raw_documents = process_directory(KB_PATH)
    
    # 质量分析
    stats = analyze_chunk_quality(raw_documents)
    print_quality_report(stats)
    
    # 保存
    save_option = input("\n是否保存分块结果？(y/n): ").strip().lower()
    if save_option == 'y':
        save_chunks_to_file(raw_documents)