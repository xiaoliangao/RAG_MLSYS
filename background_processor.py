# background_processor.py
"""
后台PDF处理模块 - 异步任务处理
解决大文件上传时UI卡死的问题
"""

import os
import json
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import core_processing
import core_indexing


class TaskStatus:
    """任务状态枚举"""
    PENDING = "pending"      # 等待处理
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"   # 完成
    FAILED = "failed"        # 失败


class ProcessingTask:
    """PDF处理任务"""
    
    def __init__(self, task_id: str, pdf_path: str, filename: str, session_id: str):
        self.task_id = task_id
        self.pdf_path = pdf_path
        self.filename = filename
        self.session_id = session_id
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.message = "等待处理"
        self.error = None
        self.db_path = None
        self.chunk_count = 0
        self.start_time = None
        self.end_time = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于保存状态）"""
        return {
            "task_id": self.task_id,
            "pdf_path": self.pdf_path,
            "filename": self.filename,
            "session_id": self.session_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "db_path": self.db_path,
            "chunk_count": self.chunk_count,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None
        }


class BackgroundProcessor:
    """后台处理器 - 单例模式"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.tasks = {}  # task_id -> ProcessingTask
        self.task_dir = Path("./task_status")
        self.task_dir.mkdir(exist_ok=True)
        self.worker_thread = None
        self._initialized = True
    
    def submit_task(self, task: ProcessingTask) -> str:
        """提交一个新任务"""
        self.tasks[task.task_id] = task
        self._save_task_status(task)
        
        # 如果worker线程不存在或已结束，启动新的
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
        
        return task.task_id
    
    def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """获取任务状态"""
        if task_id in self.tasks:
            return self.tasks[task_id]
        
        # 尝试从文件加载
        return self._load_task_status(task_id)
    
    def _worker(self):
        """后台工作线程"""
        while True:
            # 查找待处理的任务
            pending_task = None
            for task in self.tasks.values():
                if task.status == TaskStatus.PENDING:
                    pending_task = task
                    break
            
            if pending_task is None:
                # 没有待处理任务，休眠后再检查
                time.sleep(2)
                continue
            
            # 处理任务
            self._process_task(pending_task)
    
    def _process_task(self, task: ProcessingTask):
        """处理单个任务"""
        try:
            task.status = TaskStatus.PROCESSING
            task.start_time = datetime.now()
            task.progress = 5
            task.message = "开始处理PDF..."
            self._save_task_status(task)
            
            # 步骤1: 处理PDF
            task.progress = 10
            task.message = "正在解析PDF文档..."
            self._save_task_status(task)
            
            chunks = core_processing.process_single_pdf(task.pdf_path, task.filename)
            
            if not chunks:
                raise Exception("PDF处理失败，未生成任何文本块")
            
            task.chunk_count = len(chunks)
            task.progress = 50
            task.message = f"PDF解析完成，生成{len(chunks)}个文本块"
            self._save_task_status(task)
            
            # 步骤2: 创建向量数据库
            task.progress = 60
            task.message = "正在创建向量数据库..."
            self._save_task_status(task)
            
            # 需要embedding模型（从session state获取）
            # 注意：这里需要传入embedding模型
            # 在实际使用时，需要从主线程传入
            from langchain_huggingface import HuggingFaceEmbeddings
            import torch
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            embedding_model = HuggingFaceEmbeddings(
                model_name="./models/bge-large-zh-v1.5",
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            db_path, db_object = core_indexing.build_session_vector_db(
                chunks,
                task.session_id,
                embedding_model,
                "./vector_db"
            )
            
            task.db_path = db_path
            task.progress = 100
            task.message = "处理完成！"
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            
            self._save_task_status(task)
            
            print(f"✅ 任务 {task.task_id} 处理完成")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.message = f"处理失败: {str(e)}"
            task.end_time = datetime.now()
            self._save_task_status(task)
            
            print(f"❌ 任务 {task.task_id} 处理失败: {e}")
    
    def _save_task_status(self, task: ProcessingTask):
        """保存任务状态到文件"""
        try:
            status_file = self.task_dir / f"{task.task_id}.json"
            with open(status_file, 'w', encoding='utf-8') as f:
                json.dump(task.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存任务状态失败: {e}")
    
    def _load_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """从文件加载任务状态"""
        try:
            status_file = self.task_dir / f"{task_id}.json"
            if not status_file.exists():
                return None
            
            with open(status_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            task = ProcessingTask(
                task_id=data["task_id"],
                pdf_path=data["pdf_path"],
                filename=data["filename"],
                session_id=data["session_id"]
            )
            task.status = data["status"]
            task.progress = data["progress"]
            task.message = data["message"]
            task.error = data.get("error")
            task.db_path = data.get("db_path")
            task.chunk_count = data.get("chunk_count", 0)
            
            return task
            
        except Exception as e:
            print(f"⚠️ 加载任务状态失败: {e}")
            return None


# 全局处理器实例
processor = BackgroundProcessor()


def submit_pdf_task(pdf_path: str, filename: str, session_id: str) -> str:
    """
    提交PDF处理任务（便捷函数）
    
    Args:
        pdf_path: PDF文件路径
        filename: 文件名
        session_id: 会话ID
    
    Returns:
        task_id: 任务ID
    """
    task_id = f"task_{session_id}_{int(time.time())}"
    task = ProcessingTask(task_id, pdf_path, filename, session_id)
    processor.submit_task(task)
    return task_id


def get_task_status(task_id: str) -> Optional[ProcessingTask]:
    """
    获取任务状态（便捷函数）
    
    Args:
        task_id: 任务ID
    
    Returns:
        ProcessingTask或None
    """
    return processor.get_task_status(task_id)