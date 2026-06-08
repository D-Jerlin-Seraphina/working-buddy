import logging
import random
from typing import Dict, Any, List, Optional
from backend.conversation.memory import ConversationMemory

logger = logging.getLogger(__name__)

WEEK_QUESTIONS = {
    1: [
        "Hey! How has your first week been so far?",
        "Welcome to the team! What's been the most surprising thing about your first week?",
        "How are you settling in? Anything unexpected come up?"
    ],
    2: [
        "How has week two been treating you?",
        "What part of your work have you enjoyed most so far?",
        "Any challenges popping up as you get more familiar with things?"
    ],
    3: [
        "How are things going three weeks in?",
        "Starting to feel more comfortable with the day-to-day?",
        "What's been the biggest learning curve so far?"
    ],
    4: [
        "Month one done! How comfortable are you with your team now?",
        "If you could change one thing about your onboarding experience, what would it be?",
        "How's the team dynamic feeling at this point?"
    ],
    6: [
        "Six weeks in - what part of your day motivates you the most?",
        "When do you feel frustrated or stuck lately?",
        "Do you feel like you're getting the hang of your role?"
    ],
    8: [
        "Two months! Do you feel your skills are being utilized well?",
        "What are you looking forward to learning next?",
        "Any areas where you'd like more exposure or responsibility?"
    ],
    10: [
        "How's everything going at the 10-week mark?",
        "Do you see a clear path for growth here?",
        "What would make your experience better right now?"
    ],
    12: [
        "Three months in! How does reality compare to your expectations when you joined?",
        "What's been the highlight so far?",
        "Is there anything you'd tell your week-one self?"
    ],
    14: [
        "How are things progressing? Still feeling engaged?",
        "Any new challenges or opportunities come up recently?",
        "What support would be most helpful right now?"
    ]
}

EMPATHY_RESPONSES = {
    "frustrated": [
        "That sounds really frustrating. I hear you.",
        "I can see why that would be annoying. Thanks for sharing that.",
        "Frustrating is right. That's not okay."
    ],
    "anxious": [
        "That makes sense - new environments can be overwhelming.",
        "It's totally normal to feel that way early on.",
        "I appreciate you being honest about that."
    ],
    "confused": [
        "Getting used to new tools and processes takes time.",
        "That's a common pain point. You're not alone there.",
        "Makes sense - there's a lot to absorb."
    ],
    "disengaged": [
        "Thanks for telling me. That's important to know.",
        "I hear you. Let's talk about what might help.",
        "Appreciate the honesty - that takes courage."
    ],
    "burned_out": [
        "That sounds really tough. You shouldn't have to feel that way.",
        "Burnout is real and it matters. Thanks for flagging it.",
        "I'm glad you said something. That's not sustainable."
    ],
    "happy": [
        "That's great to hear!",
        "Love that things are going well.",
        "Awesome - keep that momentum going."
    ],
    "satisfied": [
        "Good to hear it's working out.",
        "That's what we like to hear.",
        "Sounds like you're in a good spot."
    ]
}

FOLLOW_UP_QUESTIONS = {
    "training_issue": [
        "Is there any specific tool or system that's been particularly difficult?",
        "What would make the learning curve easier for you?",
        "Have you had a chance to go through any training materials yet?"
    ],
    "manager_issue": [
        "How has communication been with your manager?",
        "Do you feel like you're getting the direction you need?",
        "What would a better manager relationship look like for you?"
    ],
    "team_issue": [
        "How connected do you feel with your teammates?",
        "Have there been opportunities to collaborate or just chat?",
        "What would help you feel more included?"
    ],
    "workload_issue": [
        "Which part of the workload has been taking most of your energy?",
        "Are the deadlines feeling realistic?",
        "What would a manageable workload look like for you?"
    ],
    "career_growth_issue": [
        "Are there any skills or projects you'd like more exposure to?",
        "What does growth look like for you in this role?",
        "Have you had conversations about your career path?"
    ],
    "compensation_concern": [
        "Is this about salary, benefits, or something else?",
        "Have you had a chance to review your compensation package?",
        "What would feel fair to you?"
    ],
    "work_life_balance_issue": [
        "What's been encroaching on your personal time?",
        "Have you been able to take time off when needed?",
        "What flexibility would help most?"
    ],
    "culture_issue": [
        "What about the environment feels off?",
        "Do you feel like you can be yourself at work?",
        "What would a better culture look like to you?"
    ]
}

CLOSING_RESPONSES = [
    "Thanks for sharing that with me. I'll make sure the right people see this.",
    "Appreciate you taking the time to chat. Your experience matters.",
    "Good talking with you. I'm here if anything else comes up.",
    "Thanks for being open. That helps us make things better."
]


class ResponseGenerator:
    def __init__(self, db_session=None):
        self.memory = ConversationMemory(db_session) if db_session else None

    def generate_question(self, week_number: int, employee_id: int = None) -> str:
        questions = WEEK_QUESTIONS.get(week_number, WEEK_QUESTIONS[14])
        base_question = random.choice(questions)
        
        if employee_id and self.memory:
            context = self.memory.get_recent_context(employee_id, weeks=2)
            if context:
                recurring = self.memory.get_recurring_topics(employee_id)
                if recurring:
                    top_topic = list(recurring.keys())[0]
                    if top_topic != "none":
                        followup = self._get_contextual_followup(top_topic)
                        if followup:
                            return f"{base_question} {followup}"
        
        return base_question

    def _get_contextual_followup(self, topic: str) -> Optional[str]:
        followups = FOLLOW_UP_QUESTIONS.get(topic, [])
        if followups:
            return random.choice(followups)
        return None

    def generate_response(
        self,
        user_input: str,
        sentiment: str,
        emotion: str,
        problem_category: str,
        week_number: int,
        employee_id: int = None
    ) -> str:
        parts = []
        
        empathy = self._get_empathy_response(emotion, sentiment)
        if empathy:
            parts.append(empathy)
        
        followup = self._get_problem_followup(problem_category, user_input)
        if followup:
            parts.append(followup)
        
        if not parts:
            parts.append(random.choice([
                "Thanks for sharing that.",
                "I appreciate you telling me.",
                "That's helpful to know."
            ]))
        
        closing = random.choice(CLOSING_RESPONSES)
        parts.append(closing)
        
        return " ".join(parts)

    def _get_empathy_response(self, emotion: str, sentiment: str) -> str:
        if emotion in EMPATHY_RESPONSES:
            return random.choice(EMPATHY_RESPONSES[emotion])
        
        if sentiment == "negative":
            return random.choice(EMPATHY_RESPONSES["frustrated"])
        elif sentiment == "positive":
            return random.choice(EMPATHY_RESPONSES["happy"])
        
        return ""

    def _get_problem_followup(self, problem_category: str, user_input: str) -> Optional[str]:
        if problem_category == "none" or problem_category not in FOLLOW_UP_QUESTIONS:
            return None
        
        followups = FOLLOW_UP_QUESTIONS[problem_category]
        return random.choice(followups)

    def generate_weekly_checkin_message(self, week_number: int, employee_name: str) -> str:
        greetings = [
            f"Hey {employee_name}!",
            f"Hi {employee_name},",
            f"Hello {employee_name}!"
        ]
        
        question = self.generate_question(week_number)
        
        return f"{random.choice(greetings)} {question}"


def get_response_generator(db_session=None) -> ResponseGenerator:
    return ResponseGenerator(db_session)