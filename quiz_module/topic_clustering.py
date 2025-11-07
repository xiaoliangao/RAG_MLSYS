# quiz_module/topic_clustering.py
"""
主题聚类模块 - 智能分层抽样
确保测验题目覆盖文档的不同主题，而非随机抽取
"""

import numpy as np
from typing import List, Dict, Any
from langchain_core.documents import Document
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import random


def cluster_documents_simple(
    documents: List[Document], 
    num_clusters: int = 10,
    random_state: int = 42
) -> Dict[int, List[Document]]:
    """
    简单版：使用K-Means对文档进行主题聚类
    
    Args:
        documents: 文档列表
        num_clusters: 聚类数量（主题数）
        random_state: 随机种子
    
    Returns:
        {cluster_id: [documents]} 字典
    """
    
    if len(documents) < num_clusters:
        # 文档太少，直接返回
        print(f"⚠️ 文档数({len(documents)})少于聚类数({num_clusters})，使用简单分组")
        return {i: [doc] for i, doc in enumerate(documents)}
    
    try:
        # 提取文本
        texts = [doc.page_content for doc in documents]
        
        # TF-IDF向量化（轻量级）
        vectorizer = TfidfVectorizer(
            max_features=500,  # 限制特征数
            stop_words=None,   # 中文无内置停用词
            min_df=2,          # 至少出现2次
            max_df=0.8         # 最多80%文档
        )
        
        X = vectorizer.fit_transform(texts)
        
        # K-Means聚类
        kmeans = KMeans(
            n_clusters=num_clusters,
            random_state=random_state,
            n_init=10
        )
        
        labels = kmeans.fit_predict(X)
        
        # 按聚类分组
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(documents[idx])
        
        # 统计
        cluster_sizes = {k: len(v) for k, v in clusters.items()}
        print(f"✓ K-Means聚类完成: {len(clusters)}个主题")
        print(f"  主题分布: {cluster_sizes}")
        
        return clusters
        
    except Exception as e:
        print(f"❌ 聚类失败: {e}，使用随机分组")
        return _random_grouping(documents, num_clusters)


def cluster_documents_llm(
    documents: List[Document],
    tokenizer,
    model,
    device: str,
    num_topics: int = 10,
    sample_size: int = 20
) -> Dict[str, List[Document]]:
    """
    LLM版：使用大模型识别主题并分组
    
    Args:
        documents: 文档列表
        tokenizer: 分词器
        model: 语言模型
        device: 设备
        num_topics: 期望的主题数量
        sample_size: 用于主题识别的样本数
    
    Returns:
        {topic_name: [documents]} 字典
    """
    
    if len(documents) < sample_size:
        sample_docs = documents
    else:
        sample_docs = random.sample(documents, sample_size)
    
    # 准备上下文
    context = "\n\n---\n\n".join([
        f"[片段 {i+1}]\n{doc.page_content[:300]}"  # 只取前300字
        for i, doc in enumerate(sample_docs)
    ])
    
    # 构建提示词
    system_prompt = f"""你是一个文档分析专家，擅长识别教材的核心主题。

请分析以下{len(sample_docs)}个文档片段，总结出{num_topics}个核心主题。

要求：
1. 每个主题用3-5个字的短语概括
2. 主题应该互不重叠
3. 按重要性排序
4. 直接返回主题列表，每行一个"""

    user_message = f"""文档片段：

{context}

---

请提取{num_topics}个核心主题，每行一个。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        import torch
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        # 解析主题
        topics = [line.strip() for line in response.split('\n') if line.strip()]
        topics = topics[:num_topics]  # 限制数量
        
        if not topics:
            raise Exception("未识别到主题")
        
        print(f"✓ LLM识别到{len(topics)}个主题: {topics}")
        
        # 将文档分配到主题（简单的关键词匹配）
        topic_docs = {topic: [] for topic in topics}
        topic_docs["其他"] = []
        
        for doc in documents:
            content_lower = doc.page_content.lower()
            matched = False
            
            for topic in topics:
                topic_keywords = topic.lower().split()
                if any(kw in content_lower for kw in topic_keywords):
                    topic_docs[topic].append(doc)
                    matched = True
                    break
            
            if not matched:
                topic_docs["其他"].append(doc)
        
        # 移除空主题
        topic_docs = {k: v for k, v in topic_docs.items() if v}
        
        # 统计
        topic_sizes = {k: len(v) for k, v in topic_docs.items()}
        print(f"  主题分布: {topic_sizes}")
        
        return topic_docs
        
    except Exception as e:
        print(f"❌ LLM主题识别失败: {e}，回退到K-Means")
        clusters = cluster_documents_simple(documents, num_topics)
        # 转换为字符串key
        return {f"主题{k}": v for k, v in clusters.items()}


def stratified_sample_documents(
    clusters: Dict[Any, List[Document]],
    num_samples: int
) -> List[Document]:
    """
    分层抽样：从每个聚类中抽取文档
    
    Args:
        clusters: {cluster_id: [documents]}
        num_samples: 需要的样本总数
    
    Returns:
        抽样的文档列表
    """
    
    sampled_docs = []
    cluster_list = list(clusters.items())
    
    if not cluster_list:
        return []
    
    # 计算每个聚类应该抽取的数量
    samples_per_cluster = num_samples // len(cluster_list)
    remainder = num_samples % len(cluster_list)
    
    for i, (cluster_id, docs) in enumerate(cluster_list):
        # 前几个聚类多抽1个（分配余数）
        n_samples = samples_per_cluster + (1 if i < remainder else 0)
        
        if len(docs) <= n_samples:
            # 聚类文档不够，全部使用
            sampled_docs.extend(docs)
        else:
            # 随机抽样
            sampled_docs.extend(random.sample(docs, n_samples))
    
    # 打乱顺序
    random.shuffle(sampled_docs)
    
    print(f"✓ 分层抽样完成: 从{len(cluster_list)}个主题抽取{len(sampled_docs)}个文档")
    
    return sampled_docs[:num_samples]


def _random_grouping(documents: List[Document], num_groups: int) -> Dict[int, List[Document]]:
    """随机分组（降级方案）"""
    groups = {i: [] for i in range(num_groups)}
    
    for i, doc in enumerate(documents):
        group_id = i % num_groups
        groups[group_id].append(doc)
    
    return groups


def smart_document_sampling(
    documents: List[Document],
    num_samples: int,
    method: str = "kmeans",
    **kwargs
) -> List[Document]:
    """
    智能文档抽样（统一接口）
    
    Args:
        documents: 文档列表
        num_samples: 需要的样本数
        method: "kmeans" 或 "llm"
        **kwargs: 传递给具体方法的参数
    
    Returns:
        抽样的文档列表
    """
    
    if len(documents) <= num_samples:
        print(f"⚠️ 文档数({len(documents)})不足，返回全部")
        return documents
    
    num_clusters = min(num_samples, len(documents) // 2)  # 聚类数不超过样本数
    
    if method == "llm":
        clusters = cluster_documents_llm(
            documents,
            tokenizer=kwargs.get('tokenizer'),
            model=kwargs.get('model'),
            device=kwargs.get('device'),
            num_topics=num_clusters
        )
    else:  # kmeans
        clusters = cluster_documents_simple(documents, num_clusters)
    
    return stratified_sample_documents(clusters, num_samples)