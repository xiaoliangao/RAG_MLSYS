# version_manager.py
"""
知识库版本管理工具
负责记录每次PDF构建产生的向量库版本，并支持查询与使用记录
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

VERSIONS_FILE = Path("./vector_db/version_history.json")


def _ensure_storage() -> None:
    """确保版本记录文件所在目录存在"""
    VERSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_versions() -> List[Dict[str, Any]]:
    """加载全部版本记录，按创建时间倒序"""
    _ensure_storage()
    if not VERSIONS_FILE.exists():
        return []
    
    try:
        with open(VERSIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                return []
    except Exception:
        return []
    
    return sorted(
        data,
        key=lambda item: item.get("created_at", ""),
        reverse=True,
    )


def _save_versions(versions: List[Dict[str, Any]]) -> None:
    """保存版本记录"""
    _ensure_storage()
    with open(VERSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(versions, f, ensure_ascii=False, indent=2)


def register_version(
    filename: str,
    db_path: str,
    chunk_count: int,
    task_id: str,
    session_id: str,
) -> Dict[str, Any]:
    """记录一次新的知识库构建结果"""
    versions = load_versions()
    created_at = datetime.now().isoformat()
    version_id = str(uuid.uuid4())
    
    entry = {
        "version_id": version_id,
        "session_id": session_id,
        "task_id": task_id,
        "filename": filename,
        "display_name": Path(filename).stem,
        "db_path": db_path,
        "chunk_count": chunk_count,
        "created_at": created_at,
        "last_used_at": created_at,
    }
    
    versions = [
        v for v in versions
        if v.get("session_id") != session_id and v.get("db_path") != db_path
    ]
    versions.insert(0, entry)
    _save_versions(versions)
    return entry


def get_version(version_id: str) -> Optional[Dict[str, Any]]:
    """根据ID获取版本"""
    for version in load_versions():
        if version.get("version_id") == version_id:
            return version
    return None


def mark_version_used(version_id: str) -> None:
    """更新版本的最近使用时间"""
    versions = load_versions()
    updated = False
    for version in versions:
        if version.get("version_id") == version_id:
            version["last_used_at"] = datetime.now().isoformat()
            updated = True
            break
    if updated:
        _save_versions(versions)


def remove_version(version_id: str) -> None:
    """移除指定版本（暂未在UI暴露，预留）"""
    versions = load_versions()
    new_versions = [v for v in versions if v.get("version_id") != version_id]
    if len(new_versions) != len(versions):
        _save_versions(new_versions)
