# learning_tracker.py
"""
学习行为追踪模块
记录测验历史并提供聚合所需的数据
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ANALYTICS_DIR = Path("./analytics")
QUIZ_HISTORY_FILE = ANALYTICS_DIR / "quiz_history.json"
MAX_HISTORY = 200


def _ensure_dir() -> None:
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)


def load_quiz_history(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """加载测验历史，按时间升序返回"""
    _ensure_dir()
    if not QUIZ_HISTORY_FILE.exists():
        return []
    
    try:
        with open(QUIZ_HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
            if not isinstance(history, list):
                history = []
    except Exception:
        history = []
    
    history.sort(key=lambda x: x.get("timestamp", ""))
    if limit:
        return history[-limit:]
    return history


def _save_history(history: List[Dict[str, Any]]) -> None:
    _ensure_dir()
    with open(QUIZ_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def record_quiz_attempt(report: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """记录一次测验结果"""
    history = load_quiz_history()
    entry = {
        "attempt_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "score_percentage": float(report.get("score_percentage", 0.0)),
        "correct": int(report.get("correct", 0)),
        "wrong": int(report.get("wrong", 0)),
        "unanswered": int(report.get("unanswered", 0)),
        "total": int(report.get("total", 0)),
        "difficulty": metadata.get("difficulty", "medium"),
        "num_choice": metadata.get("num_choice", 0),
        "num_boolean": metadata.get("num_boolean", 0),
        "version_id": metadata.get("version_id"),
        "version_name": metadata.get("version_name"),
        "knowledge_tags": metadata.get("knowledge_tags") or _extract_keywords(report),
        "type_breakdown": _type_breakdown(report),
    }
    history.append(entry)
    history = history[-MAX_HISTORY:]
    _save_history(history)
    return entry


def summarize_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """给前端提供的汇总指标"""
    if not history:
        return {
            "attempts": 0,
            "average_score": 0.0,
            "best_score": 0.0,
            "recent_score": 0.0,
        }
    
    scores = [h.get("score_percentage", 0.0) for h in history]
    return {
        "attempts": len(history),
        "average_score": sum(scores) / len(scores),
        "best_score": max(scores),
        "recent_score": scores[-1],
    }


def aggregate_knowledge_tags(history: List[Dict[str, Any]], topk: int = 6) -> List[tuple[str, int]]:
    """统计最常出现的薄弱知识点标签"""
    counter: Counter = Counter()
    for entry in history:
        for tag in entry.get("knowledge_tags", []):
            if tag:
                counter[tag] += 1
    return counter.most_common(topk)


def aggregate_difficulty_performance(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """按难度统计平均得分"""
    buckets: Dict[str, List[float]] = {}
    for entry in history:
        difficulty = entry.get("difficulty", "unknown")
        buckets.setdefault(difficulty, []).append(entry.get("score_percentage", 0.0))
    
    return {
        diff: (sum(scores) / len(scores)) if scores else 0.0
        for diff, scores in buckets.items()
    }


def _extract_keywords(report: Dict[str, Any], max_terms: int = 6) -> List[str]:
    """从错题题干中提取关键概念"""
    wrong_questions = [
        item for item in report.get("results", [])
        if not item.get("is_correct") and not item.get("is_unanswered")
    ]
    tokens: Counter = Counter()
    for item in wrong_questions:
        text = item.get("question", "")
        zh_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        en_terms = re.findall(r"[A-Za-z]{3,}", text)
        for term in zh_terms + en_terms:
            tokens[term] += 1
    most_common = [term for term, _ in tokens.most_common(max_terms)]
    return most_common


def _type_breakdown(report: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """统计不同题型的正确数量"""
    stats: Dict[str, Dict[str, int]] = {}
    for item in report.get("results", []):
        q_type = item.get("type", "unknown")
        stats.setdefault(q_type, {"total": 0, "correct": 0})
        stats[q_type]["total"] += 1
        if item.get("is_correct"):
            stats[q_type]["correct"] += 1
    return stats
