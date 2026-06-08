import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    category: str
    title: str
    description: str
    priority: str
    action_items: List[str]
    owner: str


RECOMMENDATIONS = {
    "training_issue": Recommendation(
        category="training_issue",
        title="Additional Training Support Needed",
        description="Employee is struggling with tools, systems, or processes",
        priority="High",
        action_items=[
            "Assign a dedicated buddy for tool training",
            "Schedule 1-on-1 training sessions for difficult systems",
            "Provide access to video tutorials and documentation",
            "Set up weekly check-ins on training progress",
            "Consider extending the formal onboarding period"
        ],
        owner="Manager + Learning & Development"
    ),
    "manager_issue": Recommendation(
        category="manager_issue",
        title="Manager Communication Gap",
        description="Employee reports issues with manager communication, availability, or support",
        priority="High",
        action_items=[
            "Schedule mediated 1-on-1 between employee and manager",
            "Set clear expectations for meeting cadence and response times",
            "Provide manager coaching on supportive leadership",
            "Establish escalation path for employee concerns",
            "Consider manager feedback survey from team"
        ],
        owner="HR Business Partner + Manager's Manager"
    ),
    "team_issue": Recommendation(
        category="team_issue",
        title="Team Integration Challenge",
        description="Employee feels isolated, excluded, or disconnected from teammates",
        priority="Medium",
        action_items=[
            "Assign an onboarding buddy from the team",
            "Schedule regular team social interactions (coffee chats, lunches)",
            "Include employee in relevant team meetings and decisions",
            "Facilitate introductions to key stakeholders",
            "Monitor team dynamics in retrospectives"
        ],
        owner="Manager + Team Lead"
    ),
    "workload_issue": Recommendation(
        category="workload_issue",
        title="Workload Overwhelm",
        description="Employee reports excessive workload, unrealistic deadlines, or burnout signs",
        priority="High",
        action_items=[
            "Conduct immediate workload audit with manager",
            "Reprioritize or delegate non-critical tasks",
            "Set realistic deadlines and expectations",
            "Enforce meeting-free blocks for deep work",
            "Monitor for burnout signs weekly",
            "Consider temporary resource augmentation"
        ],
        owner="Manager + HR"
    ),
    "career_growth_issue": Recommendation(
        category="career_growth_issue",
        title="Career Development Gap",
        description="Employee doesn't see growth opportunities or skill utilization",
        priority="Medium",
        action_items=[
            "Create Individual Development Plan (IDP)",
            "Identify stretch projects or shadowing opportunities",
            "Schedule quarterly career conversations",
            "Connect with internal mobility programs",
            "Allocate learning budget for relevant skills"
        ],
        owner="Manager + HR + L&D"
    ),
    "compensation_concern": Recommendation(
        category="compensation_concern",
        title="Compensation Concern",
        description="Employee has raised concerns about pay, benefits, or equity",
        priority="High",
        action_items=[
            "Review compensation against market benchmarks",
            "Schedule transparent compensation conversation",
            "Clarify total rewards package (benefits, equity, bonuses)",
            "Document path for salary review/promotion",
            "Escalate to compensation team if adjustment needed"
        ],
        owner="HR + Compensation Team"
    ),
    "work_life_balance_issue": Recommendation(
        category="work_life_balance_issue",
        title="Work-Life Balance Concern",
        description="Employee struggling with boundaries, flexibility, or personal time",
        priority="Medium",
        action_items=[
            "Discuss flexible working arrangements",
            "Review and respect after-hours communication norms",
            "Encourage use of PTO and mental health days",
            "Assess meeting load and async alternatives",
            "Connect with EAP resources if needed"
        ],
        owner="Manager + HR"
    ),
    "culture_issue": Recommendation(
        category="culture_issue",
        title="Cultural Misalignment",
        description="Employee feels misaligned with team/organizational culture",
        priority="Medium",
        action_items=[
            "Conduct stay interview to understand specific concerns",
            "Connect with Employee Resource Groups",
            "Review team norms and psychological safety",
            "Escalate to Culture/DEI team if systemic",
            "Consider team restructuring if toxic dynamics"
        ],
        owner="HR + DEI Team + Leadership"
    )
}


class RecommendationEngine:
    def __init__(self):
        self.recommendations = RECOMMENDATIONS

    def get_recommendations(
        self,
        problem_categories: List[str],
        risk_level: str,
        sentiment_trend: str,
        employee_context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        recs = []
        
        for category in problem_categories:
            if category in self.recommendations:
                rec = self.recommendations[category]
                recs.append({
                    "category": rec.category,
                    "title": rec.title,
                    "description": rec.description,
                    "priority": rec.priority,
                    "action_items": rec.action_items,
                    "owner": rec.owner
                })
        
        if not recs and risk_level == "High Risk":
            recs.append({
                "category": "general",
                "title": "General Retention Intervention",
                "description": "High risk detected without specific category",
                "priority": "High",
                "action_items": [
                    "Schedule immediate stay interview with HR",
                    "Review all aspects of employee experience",
                    "Assign executive sponsor for retention",
                    "Create 30-day action plan with weekly check-ins"
                ],
                "owner": "HR Business Partner"
            })
        
        if sentiment_trend == "declining":
            for rec in recs:
                if rec["priority"] != "High":
                    rec["priority"] = "High"
        
        return recs

    def get_all_recommendations(self) -> Dict[str, Dict[str, Any]]:
        return {
            cat: {
                "title": rec.title,
                "description": rec.description,
                "priority": rec.priority,
                "action_items": rec.action_items,
                "owner": rec.owner
            }
            for cat, rec in self.recommendations.items()
        }


def get_recommendation_engine() -> RecommendationEngine:
    return RecommendationEngine()