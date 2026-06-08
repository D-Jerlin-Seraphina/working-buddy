import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from backend.models.database import Conversation, Employee

logger = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(self, db: Session):
        self.db = db

    def get_conversation_history(self, employee_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        conversations = self.db.query(Conversation).filter(
            Conversation.employee_id == employee_id
        ).order_by(Conversation.week_number.desc()).limit(limit).all()
        
        history = []
        for conv in reversed(conversations):
            if conv.user_response:
                history.append({
                    "week": conv.week_number,
                    "date": conv.completed_date.isoformat() if conv.completed_date else None,
                    "user_response": conv.user_response,
                    "ai_response": conv.ai_response,
                    "sentiment": conv.sentiment,
                    "emotion": conv.emotion,
                    "problem_category": conv.problem_category,
                    "topics": conv.topics or []
                })
        return history

    def get_recent_context(self, employee_id: int, weeks: int = 3) -> str:
        history = self.get_conversation_history(employee_id, limit=weeks)
        
        if not history:
            return ""
        
        context_parts = ["Previous conversations:"]
        for h in history:
            context_parts.append(f"Week {h['week']}: Employee said: \"{h['user_response'][:200]}...\"")
            if h.get('problem_category') and h['problem_category'] != 'none':
                context_parts.append(f"  (Noted: {h['problem_category'].replace('_', ' ')})")
        
        return "\n".join(context_parts)

    def get_recurring_topics(self, employee_id: int) -> Dict[str, int]:
        history = self.get_conversation_history(employee_id, limit=20)
        
        topic_counts = {}
        for h in history:
            for topic in h.get('topics', []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))

    def get_sentiment_trend(self, employee_id: int, weeks: int = 8) -> List[Dict[str, Any]]:
        conversations = self.db.query(Conversation).filter(
            Conversation.employee_id == employee_id,
            Conversation.sentiment_score.isnot(None)
        ).order_by(Conversation.week_number).all()
        
        trend = []
        for conv in conversations[-weeks:]:
            trend.append({
                "week": conv.week_number,
                "sentiment_score": conv.sentiment_score,
                "sentiment": conv.sentiment,
                "emotion": conv.emotion,
                "risk_score": self._estimate_week_risk(conv)
            })
        return trend

    def _estimate_week_risk(self, conv: Conversation) -> float:
        risk = 50.0
        
        if conv.sentiment_score is not None:
            risk += (0.5 - conv.sentiment_score) * 40
        
        if conv.stress_level is not None:
            risk += conv.stress_level * 30
        
        if conv.voice_confidence is not None:
            risk += (1 - conv.voice_confidence) * 20
        
        if conv.problem_category and conv.problem_category != 'none':
            risk += 10
        
        return max(0, min(100, risk))

    def get_employee_summary(self, employee_id: int) -> Dict[str, Any]:
        employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
        if not employee:
            return {}
        
        history = self.get_conversation_history(employee_id)
        recurring = self.get_recurring_topics(employee_id)
        trend = self.get_sentiment_trend(employee_id)
        
        total_conversations = len(history)
        completed_conversations = len([h for h in history if h['user_response']])
        
        avg_sentiment = 0
        if trend:
            avg_sentiment = sum(t['sentiment_score'] for t in trend) / len(trend)
        
        primary_concern = list(recurring.keys())[0] if recurring else "none"
        
        return {
            "employee_id": employee.id,
            "name": employee.name,
            "joining_date": employee.joining_date.isoformat() if employee.joining_date else None,
            "total_conversations": total_conversations,
            "completed_conversations": completed_conversations,
            "participation_rate": completed_conversations / max(1, total_conversations),
            "avg_sentiment": avg_sentiment,
            "recurring_topics": recurring,
            "primary_concern": primary_concern,
            "sentiment_trend": trend,
            "weeks_active": max([h['week'] for h in history]) if history else 0
        }


def get_conversation_memory(db: Session) -> ConversationMemory:
    return ConversationMemory(db)