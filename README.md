# 🎓 MLTutor - 机器学习 / 深度学习 个性化学习与 RAG 智能助教

> 集成「检索增强问答 (RAG) + 智能出题测验 + 自动学习反馈」的一体化学习助手。支持 PDF 教材上传、异步构建专属知识库、混合检索、结构化题目生成与学习报告导出。

---

## 📋 项目简介

MLTutor 面向希望系统学习 ML/DL 的自学者与教学辅助场景。它通过双知识库架构与混合检索策略，在同一个 Streamlit 界面中提供：

| 模块 | 功能 | 说明 |
|------|------|------|
| 上传与构建 | 异步处理 PDF | 后台线程解析 → 分块 → 向量化 → 专属会话向量库 |
| RAG 问答 | 语义检索 + BM25 混合 | 查询扩展 + 上下文评分 + Few-shot + 多轮记忆 |
| 测验系统 | 选择题 + 判断题自动生成 | LLM 生成结构化 JSON，聚类采样保证主题覆盖 |
| 评测与报告 | 自动判分与可视化 | 题型准确率 / 错题分类 / 个性化 AI 学习反馈 |
| 知识反馈 | 用户点赞/点踩 | 保存简略反馈 JSON 供后续质量迭代 |

---

## ✨ 核心特性（真实代码支持）

1. � 双检索架构：上传 PDF 后同时生成「出题专用库」与「问答混合库（默认教材 + 上传内容）」
2. 🧠 混合检索策略：Chroma 向量检索 + BM25 关键词检索加权融合，去重+评分选取上下文
3. � 查询增强：自动生成同义 / 解释 / 领域前缀拓展查询，提高召回率
4. 🧩 智能分块：多规则清洗、公式/定理标注、跨块语义合并、超长截断与质量分析
5. �️ 会话隔离：每次上传构建独立 `session_<id>` 向量库，不污染基础教材库
6. 📝 题目高质量约束：严格 JSON 格式 + 质量校验（长度/唯一答案/干扰项/自洽性）+ 多次重试
7. � 主题覆盖采样：支持 KMeans 聚类分层抽样，避免集中某单一章节
8. 🤖 学习反馈生成：错题上下文 → LLM 产出结构化 Markdown 建议；失败回退保底逻辑
9. � 导出能力：支持测验结果与学习报告 TXT / PDF (自动检测中文字体)
10. 🧪 健壮降级：聚类失败自动回退随机采样；检索失败回退向量-only；反馈生成失败使用 fallback 模板

---

## 🏗️ 系统架构概览

```
            ┌─────────────┐
 用户上传PDF →│ background  │→ 异步任务：解析 / 清洗 / 分块 / 向量化
              │ processor   │
              └─────┬──────┘
                    │ session_db_path
        ┌───────────▼───────────┐
        │   core_indexing        │  会话向量库构建 / 过滤 / 截断
        └───────┬───────────────┘
                │ Chroma + Embeddings
     ┌──────────▼──────────┐
     │  EnsembleRetriever  │  (向量检索 + BM25 加权)
     └─────┬─────────┬─────┘
           │         │
      RAG 问答    测验出题（聚类抽样）
           │         │
    Prompt 构建   题目生成(JSON)
           │         │
     LLM 响应      判分 / 报告 / AI反馈
           │
       用户交互 (Streamlit 多标签页)
```

---

## 📂 实际目录结构（当前仓库）

```
rag_mlsys/
├── main_app.py               # 主入口：上传 / 测验 / 报告 / AI助教 四合一界面
├── module_rag_assistant.py   # 独立简化版 RAG 助教（可单独运行）
├── background_processor.py   # 异步任务：处理上传 PDF 构建会话向量库
├── core_processing.py        # PDF 批量 / 单文件处理 + 清洗 + 分块 + 质量分析
├── core_indexing.py          # 向量库构建 / 会话隔离 / 批量模式 / 加载测试
├── quiz_module/              # 测验子系统：出题 / 判分 / 报告 / 聚类
│   ├── question_generator.py
│   ├── evaluator.py
│   ├── report_generator.py
│   └── topic_clustering.py
├── down.py                   # 模型下载脚本（bge embedding + Qwen 7B）
├── knowledge_base/           # 放置基础教材 PDF（被忽略，只保留目录）
├── vector_db/                # 持久化向量数据库（动态生成，忽略内容）
├── processed_chunks/         # 分块缓存 (chunks.json / chunks.pkl)
├── feedback_db/              # 用户反馈 JSON（忽略内容）
├── task_status/              # 后台任务状态记录文件
├── temp_uploads/             # 临时上传 PDF
├── models/                   # 本地模型权重（需自行下载）
├── requirements.txt
└── README.md
```

> 提示：`.gitignore` 已配置忽略大模型 / PDF / 生成数据，仅保留结构。需要手动下载模型。

---

## 🔧 核心模块详解

| 模块 | 关键函数 / 类 | 说明 |
|------|---------------|------|
| `core_processing.py` | `process_single_pdf`, `clean_document_content`, `split_text_into_chunks` | PDF 加载 → 内容清洗 → 智能分块（定理/公式标注 + 合并 + OCR 修复）|
| `core_indexing.py` | `build_session_vector_db`, `initialize_embedding_model` | 过滤非正文 / 截断过长 / 分批构建 Chroma / 会话隔离 |
| `background_processor.py` | `ProcessingTask`, `BackgroundProcessor` | 后台线程轮询 PENDING 任务，阶段进度写入 JSON |
| `main_app.py` | `render_sidebar`, `retrieve_with_enhancements`, `build_enhanced_prompt` | 查询扩展 / 多轮对话抽取 / Few-shot 示例融合 |
| `quiz_module/question_generator.py` | `_build_question_gen_prompt`, `_validate_question_quality` | 高质量题目 JSON 结构 + 失败重试 |
| `quiz_module/topic_clustering.py` | `cluster_documents_simple`, `smart_document_sampling` | KMeans 聚类 + 分层抽样保证覆盖面 |
| `quiz_module/evaluator.py` | `grade_quiz`, `calculate_score`, `get_performance_level` | 判分统计 / 错题分析 / 盲区抽取 |
| `quiz_module/report_generator.py` | `generate_study_feedback`, `export_report_to_pdf` | 个性化 AI 学习反馈 + 中英文字体兼容 |

---

## 🚀 快速开始

### 环境需求
| 组件 | 推荐 |
|------|------|
| Python | 3.9+ (>=3.8 亦可) |
| RAM | ≥8GB（处理大 PDF 更稳） |
| GPU | 推荐 (Qwen 7B 推理更快；CPU 可用但较慢) |

### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/rag_mlsys.git
cd rag_mlsys
```

### 2. 创建并激活虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# Windows: venv\Scripts\activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 下载模型（Embedding + LLM）
```bash
python down.py
```
> 若下载失败请检查网络或使用镜像。CPU 环境无法加载 bfloat16 时会自动使用 float32。

### 5. 准备基础教材（可选）
将 PDF 放入 `knowledge_base/`，可执行：
```bash
python core_processing.py     # 批量处理并输出 chunks.json / chunks.pkl
python core_indexing.py       # 构建/测试基础向量库（首次可跳过交互）
```
若只依赖上传即时构建，可跳过此步，直接在界面上传。

### 6. 运行主应用（推荐 Streamlit 方式）
```bash
streamlit run main_app.py
```
或：
```bash
python main_app.py
```

### 7. 使用流程
1. 打开标签页「上传教材」上传 PDF → 后台异步处理（可刷新查看进度）
2. 处理完成后自动生成两个检索器（出题库 / 混合问答库）
3. 在「开始测验」配置题量与难度 → LLM 生成题目 → 提交判分
4. 「学习报告」查看统计图与 AI 个性化反馈 → 一键将建议问题发送到助教
5. 「AI助教」进行多轮深度问答，查看引用来源与提交反馈

---

## 🧪 测验系统工作流细节
1. 文档采样：优先聚类 + 分层抽样 → 覆盖不同主题
2. Prompt 构建：强约束 + 禁止“根据上文”类引用，题面自洽
3. LLM 生成：多次重试直至通过 JSON / 质量校验
4. 判分统计：正确 / 错误 / 未答 + 题型正确率 + 难度估计
5. 错题分析：关键词抽取 + 构造知识盲区摘要
6. 学习反馈：错题上下文 → LLM 输出结构化建议（失败回退模板）
7. 报告导出：支持 TXT / PDF（自动字体检测）

---

## � RAG 检索增强策略
| 步骤 | 说明 |
|------|------|
| 查询扩展 | 自动补充“什么是 / 请解释 / 领域前缀”多版本查询 |
| 文档合并 | 多查询结果去重 + 评分（关键词命中 / 长度 / 多样性）|
| Few-shot | 加载内置示例（反向传播 / BatchNorm）增强回答风格 |
| 多轮上下文 | 最近 N 轮对话压缩为 Q/A 片段融入 prompt |
| 反馈存储 | 每条回答支持 👍👎，截断保存 answer 片段到 `feedback_db` |

---

## 🛡️ 健壮性与降级设计
| 场景 | 降级策略 |
|------|---------|
| BM25 构建失败 | 仅使用向量检索 |
| 聚类失败 / 文档少 | 回退随机采样或有放回采样 |
| 学习反馈生成异常 | 使用 fallback 模板替代 |
| 无检索结果 | 返回提示语 + 不生成误导性回答 |
| GPU 不可用 | 自动切换到 CPU float32 |

---

## 📊 运行性能（建议）
- Qwen2.5-7B-Instruct 建议 GPU (≥12GB 显存)；CPU 下首轮加载可能数分钟
- Embedding 模型 bge-large-zh-v1.5 体积较小，可 CPU 加载
- 上传大 PDF（>50MB）建议分卷；后台处理进度可视查看

---

## 🛣️ Roadmap（后续规划）
- [ ] 知识图谱与概念关联可视化
- [ ] FastAPI 服务化接口 / 多用户管理
- [ ] 缓存热门查询与向量检索结果
- [ ] 题目难度自适应（IRT / 知识追踪）
- [ ] 英文/多语言支持
- [ ] 更细粒度错题知识点抽取 (NLP + Tagging)

---

## 🤝 贡献指南
欢迎 Issue / PR！
1. Fork 仓库
2. 新建分支：`git checkout -b feature/xxx`
3. 提交代码：`git commit -m 'feat: xxx'`
4. 推送：`git push origin feature/xxx`
5. 发起 Pull Request（请附功能说明 / 截图）

代码规范建议：
- 保持中文注释 + 英文函数名一致性
- 新增模块请补充 README 对应章节或添加内联文档
- 如引入新依赖请更新 `requirements.txt`

---

## ❓ FAQ
| 问题 | 解决方案 |
|------|----------|
| 加载模型很慢 | 首次下载 / 解压；可预先运行 `python down.py` 并使用国内镜像 |
| GPU 显存不足 | 降级为 CPU；或使用更小模型（自行替换路径）|
| 题目经常失败 | 提高 `max_retries`；减少难度为 `easy`；检查基础文档是否足够 |
| 中文 PDF 乱码 | 确认使用 PyMuPDF 加载成功；尝试重新生成或转换编码 |
| PDF 特殊公式未识别 | 手动预处理 PDF 或后续添加 LaTeX 解析增强 |
| 向量库重复 | 每次上传会覆盖同一会话目录；如需持久保留可改写命名策略 |

---

## 📄 许可证
MIT License — 可自由使用与二次开发（请保留原版权声明）。

---

## 🙏 致谢
开源生态：LangChain · Chroma · HuggingFace · PyTorch · scikit-learn · ReportLab

---

## ⭐ Star 支持
如果此项目对你有帮助，欢迎点亮 Star 给予支持与反馈！

---

## 🧩 下一步建议（供使用者）
1. 试运行一个小型 PDF，验证全流程
2. 在测验报告中把错题发送到助教进一步深挖
3. 根据反馈 JSON 聚合统计（可后续新增脚本）
4. 尝试替换其他中文 Embedding 模型（例如 m3e 或 bge-m3）

祝学习进步！