"""
Generate synthetic employee + conversation data for training and demo.
Usage: python -m data.generate_synthetic [--employees 100] [--seed 42]
"""
import random
import sys
import os
import argparse
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.database import init_db, SessionLocal
from backend.models.database import Employee, Conversation, RiskAssessment, Alert, RiskLevel

DEPARTMENTS = ["Engineering", "Product", "Sales", "HR", "Marketing", "Finance", "Operations", "Design"]
ROLES = {
    "Engineering": ["Software Engineer", "Senior Engineer", "DevOps Engineer", "QA Engineer"],
    "Product": ["Product Manager", "Product Analyst", "UX Designer"],
    "Sales": ["Sales Representative", "Account Executive", "Sales Manager"],
    "HR": ["HR Coordinator", "Recruiter", "HR Business Partner"],
    "Marketing": ["Marketing Specialist", "Content Writer", "Growth Analyst"],
    "Finance": ["Financial Analyst", "Accountant", "Finance Manager"],
    "Operations": ["Operations Analyst", "Project Manager", "Operations Manager"],
    "Design": ["UI Designer", "UX Researcher", "Graphic Designer"],
}
FIRST_NAMES = ["Alex", "Jordan", "Morgan", "Taylor", "Casey", "Riley", "Drew", "Quinn",
               "Blake", "Avery", "Sam", "Jamie", "Chris", "Robin", "Dana", "Logan",
               "Priya", "Ravi", "Ananya", "Deepak", "Neha", "Arjun", "Anjali", "Vikram"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris",
              "Patel", "Kumar", "Sharma", "Mehta", "Singh", "Kapoor", "Gupta", "Shah"]

SAMPLE_RESPONSES = {
    "positive": [
        "Things are going really well! I feel supported by my team and manager.",
        "I'm enjoying the work a lot. The culture here is great and I'm learning tons.",
        "Really happy with my role. The onboarding has been smooth and everyone's been helpful.",
        "I love what I'm doing. The challenges are exciting and I can see myself growing here."
    ],
    "negative": [
        "Honestly, I'm struggling. The tools are confusing and I don't feel I'm getting enough support.",
        "It's been tough. My manager is hard to reach and I don't always know what's expected of me.",
        "I feel a bit overwhelmed. There's too much work and not enough guidance.",
        "I'm not sure this role is what I expected. The culture feels a bit off.",
        "I'm stressed. The workload is way too high and the deadlines are unrealistic."
    ],
    "neutral": [
        "Things are okay. Some days are better than others.",
        "I'm managing. Still figuring some things out but making progress.",
        "It's been a learning curve. Not bad, but could be better.",
        "Getting by. Would appreciate more feedback from my manager."
    ]
}

AI_RESPONSES = [
    "Thanks for sharing that. I'll flag this for your manager's attention.",
    "I appreciate your honesty. Let's make sure the right people see this.",
    "That's helpful to know. We want to make sure you're well supported.",
    "I hear you. Your wellbeing matters to us and we'll follow up on this."
]

PROBLEM_CATEGORIES = [
    "training_issue", "manager_issue", "team_issue", "workload_issue",
    "career_growth_issue", "compensation_concern", "work_life_balance_issue", "culture_issue"
]


def generate_employees(n: int, db) -> list:
    employees = []
    for i in range(n):
        dept = random.choice(DEPARTMENTS)
        role = random.choice(ROLES[dept])
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        emp = Employee(
            employee_id=f"EMP{str(i + 1001).zfill(4)}",
            name=name,
            email=f"emp{i + 1001}@company.com",
            department=dept,
            role=role,
            joining_date=datetime.utcnow() - timedelta(days=random.randint(7, 120)),
            manager_id=f"MGR{random.randint(100, 110):03d}",
            is_active=True
        )
        db.add(emp)
        employees.append(emp)
    db.commit()
    for emp in employees:
        db.refresh(emp)
    print(f"  ✓ Created {len(employees)} employees")
    return employees


def generate_conversations(employees: list, db) -> None:
    count = 0
    for emp in employees:
        weeks_active = max(1, (datetime.utcnow() - emp.joining_date).days // 7)
        check_in_weeks = [w for w in [1, 2, 4, 6, 8, 10, 12, 14] if w <= weeks_active]
        risk_profile = random.choices(["low", "medium", "high"], weights=[0.5, 0.3, 0.2])[0]

        for week in check_in_weeks:
            weights = {"low": [0.6, 0.3, 0.1], "medium": [0.3, 0.4, 0.3], "high": [0.1, 0.3, 0.6]}[risk_profile]
            polarity = random.choices(["positive", "neutral", "negative"], weights=weights)[0]

            sentiment_score = (
                random.uniform(0.65, 0.95) if polarity == "positive" else
                random.uniform(0.4, 0.6) if polarity == "neutral" else
                random.uniform(0.1, 0.4)
            )
            emotion_map = {"positive": "happy", "neutral": "satisfied", "negative": "frustrated"}
            problem_cat = random.choice(PROBLEM_CATEGORIES) if polarity == "negative" else "none"

            conv = Conversation(
                employee_id=emp.id,
                week_number=week,
                scheduled_date=emp.joining_date + timedelta(weeks=week),
                completed_date=emp.joining_date + timedelta(weeks=week, hours=random.randint(0, 48)),
                transcript=random.choice(SAMPLE_RESPONSES[polarity]),
                user_response=random.choice(SAMPLE_RESPONSES[polarity]),
                ai_response=random.choice(AI_RESPONSES),
                sentiment=polarity,
                sentiment_score=round(sentiment_score, 3),
                emotion=emotion_map[polarity],
                emotion_confidence=round(random.uniform(0.6, 0.95), 3),
                voice_confidence=round(random.uniform(0.4, 1.0), 3),
                stress_level=round(random.uniform(0.0, 0.8) if polarity == "negative" else random.uniform(0.0, 0.3), 3),
                speaking_speed=round(random.uniform(1.5, 4.5), 2),
                pitch_variation=round(random.uniform(0.05, 0.4), 3),
                energy_level=round(random.uniform(0.005, 0.12), 4),
                hesitation_count=random.randint(0, 8),
                pause_duration=round(random.uniform(0.0, 3.0), 2),
                problem_category=problem_cat,
                problem_confidence=round(random.uniform(0.5, 0.95), 3),
                topics=[problem_cat] if problem_cat != "none" else ["general"]
            )
            db.add(conv)
            count += 1
    db.commit()
    print(f"  ✓ Created {count} conversations")


def generate_risk_assessments(employees: list, db) -> None:
    count = 0
    for emp in employees:
        convs = db.query(Conversation).filter(Conversation.employee_id == emp.id).all()
        for conv in convs:
            s = conv.sentiment_score or 0.5
            risk_score = max(0, min(100,
                (1 - s) * 40 +
                (conv.stress_level or 0.3) * 30 +
                (1 - (conv.voice_confidence or 0.7)) * 20 +
                (10 if conv.problem_category != "none" else 0) +
                random.gauss(0, 5)
            ))

            if risk_score >= 60:
                level = RiskLevel.HIGH
            elif risk_score >= 30:
                level = RiskLevel.MEDIUM
            else:
                level = RiskLevel.LOW

            db.add(RiskAssessment(
                employee_id=emp.id,
                week_number=conv.week_number,
                risk_score=round(risk_score, 2),
                risk_level=level,
                sentiment_trend=round(s, 3),
                emotion_trend=round(conv.emotion_confidence or 0.7, 3),
                participation_rate=1.0,
                voice_confidence_avg=round(conv.voice_confidence or 0.7, 3),
                primary_concern=conv.problem_category,
                recommendations=[]
            ))
            count += 1

            if level == RiskLevel.HIGH:
                db.add(Alert(
                    employee_id=emp.id,
                    alert_type="HIGH_RISK",
                    severity="high",
                    message=f"Employee {emp.name} flagged at high attrition risk ({risk_score:.1f}%)",
                    risk_score=round(risk_score, 2),
                    details={"week": conv.week_number, "primary_concern": conv.problem_category},
                    is_read=random.choice([True, False]),
                    is_resolved=random.choice([True, False, False])
                ))
    db.commit()
    print(f"  ✓ Created {count} risk assessments")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--employees", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"\n🚀 Generating synthetic data ({args.employees} employees)...\n")

    init_db()
    db = SessionLocal()
    try:
        existing = db.query(Employee).count()
        if existing > 0:
            print(f"  ⚠️  Database already has {existing} employees.")
            print("  Delete data/retention.db to regenerate.")
            return

        employees = generate_employees(args.employees, db)
        generate_conversations(employees, db)
        generate_risk_assessments(employees, db)

        print(f"\n✅ Done! Populated data/retention.db")
        print("   Start API:       uvicorn backend.main:app --reload")
        print("   Start dashboard: streamlit run dashboard/app.py")
    finally:
        db.close()


if __name__ == "__main__":
    main()
