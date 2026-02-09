"""
Clinical Assessment Agent (ReAct-based)

High-level agent for clinical risk assessment using ReAct pattern.
Provides both single-shot and conversational interfaces.

"""

from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import json
import re

from src.config.prompts import (
    get_clinical_assessment_extraction_prompt,
    get_clinical_patient_extraction_prompt
)

from src.agents.react_agent import ReActAgent, StreamingReActAgent, AgentResult, AgentTrace
from src.agents.tools import ClinicalTools
from src.config.logging_config import get_logger

logger = get_logger("clinical_agent")


class RiskLevel(Enum):
    """Clinical risk levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INSUFFICIENT = "insufficient_info"


class ConversationState(Enum):
    """States in the clinical assessment conversation."""
    GREETING = "greeting"
    GATHERING = "gathering"
    CLARIFYING = "clarifying"
    ASSESSING = "assessing"
    COMPLETE = "complete"
    ERROR = "error"


class UrgencyLevel(Enum):
    """Urgency levels for referral."""
    IMMEDIATE = "immediate"
    URGENT_2_WEEK = "urgent_2_week"
    URGENT = "urgent"
    SOON = "soon"
    ROUTINE = "routine"


@dataclass
class ClinicalAssessment:
    """Complete clinical assessment result."""
    assessment_id: str
    timestamp: str
    risk_level: RiskLevel
    urgency: UrgencyLevel
    summary: str
    recommended_actions: List[str] = field(default_factory=list)
    investigations: List[str] = field(default_factory=list)
    referral_pathway: Optional[str] = None
    reasoning_trace: Optional[AgentTrace] = None
    citations: List[str] = field(default_factory=list)
    matched_criteria: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    differential_considerations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    steps_taken: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "timestamp": self.timestamp,
            "risk_level": self.risk_level.value,
            "urgency": self.urgency.value,
            "summary": self.summary,
            "recommended_actions": self.recommended_actions,
            "investigations": self.investigations,
            "referral_pathway": self.referral_pathway,
            "citations": self.citations,
            "matched_criteria": self.matched_criteria,
            "red_flags": self.red_flags,
            "differential_considerations": self.differential_considerations,
            "confidence": self.confidence,
            "tools_used": self.tools_used,
            "steps_taken": self.steps_taken,
            "reasoning_trace": self.reasoning_trace.to_dict() if self.reasoning_trace else None
        }
    
    def get_urgency_display(self) -> str:
        displays = {
            UrgencyLevel.IMMEDIATE: "🔴 Immediate action required",
            UrgencyLevel.URGENT_2_WEEK: "🔴 2-week suspected cancer pathway",
            UrgencyLevel.URGENT: "🟠 Urgent referral recommended",
            UrgencyLevel.SOON: "🟡 Referral within 4-6 weeks",
            UrgencyLevel.ROUTINE: "🟢 Routine referral/monitoring",
        }
        return displays.get(self.urgency, self.urgency.value)


@dataclass
class PatientInfo:
    """Patient information for assessment."""
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: List[str] = field(default_factory=list)
    symptom_duration: Optional[str] = None
    risk_factors: List[str] = field(default_factory=list)
    medical_history: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)
    presenting_complaint: str = ""
    additional_notes: str = ""
    
    def to_task_description(self) -> str:
        parts = ["Assess this patient for potential cancer referral according to NG12 guidelines:"]
        
        if self.age:
            parts.append(f"- Age: {self.age} years")
        if self.sex:
            parts.append(f"- Sex: {self.sex}")
        if self.presenting_complaint:
            parts.append(f"- Presenting complaint: {self.presenting_complaint}")
        if self.symptoms:
            parts.append(f"- Symptoms: {', '.join(self.symptoms)}")
        if self.symptom_duration:
            parts.append(f"- Duration: {self.symptom_duration}")
        if self.risk_factors:
            parts.append(f"- Risk factors: {', '.join(self.risk_factors)}")
        if self.medical_history:
            parts.append(f"- Medical history: {', '.join(self.medical_history)}")
        if self.family_history:
            parts.append(f"- Family history: {', '.join(self.family_history)}")
        if self.additional_notes:
            parts.append(f"- Additional notes: {self.additional_notes}")
        
        parts.append("\nProvide: risk level, urgency, recommended actions, and cite relevant NG12 sections.")
        
        return "\n".join(parts)
    
    def to_summary(self) -> str:
        """Get a summary of current patient info."""
        parts = []
        if self.age:
            parts.append(f"Age: {self.age}")
        if self.sex:
            parts.append(f"Sex: {self.sex}")
        if self.symptoms:
            parts.append(f"Symptoms: {', '.join(self.symptoms)}")
        if self.symptom_duration:
            parts.append(f"Duration: {self.symptom_duration}")
        if self.risk_factors:
            parts.append(f"Risk factors: {', '.join(self.risk_factors)}")
        
        if not parts:
            return "No patient information gathered yet."
        return "; ".join(parts)


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Full conversation context for multi-turn assessment."""
    session_id: str
    state: ConversationState = ConversationState.GREETING
    history: List[ConversationTurn] = field(default_factory=list)
    patient_info: PatientInfo = field(default_factory=PatientInfo)
    assessment_result: Optional[ClinicalAssessment] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    info_gathered: Dict[str, bool] = field(default_factory=lambda: {
        "age": False,
        "symptoms": False,
        "duration": False,
        "risk_factors": False,
    })
    
    # Context summarization and caching
    _last_extraction_turn: int = 0  # Track when we last extracted info
    _context_summary: str = ""  # Summarized conversation history
    _max_history_before_summarize: int = 8  # Summarize after this many turns
    
    def add_turn(self, role: str, content: str, **metadata):
        self.history.append(ConversationTurn(
            role=role,
            content=content,
            metadata=metadata,
        ))
    
    def get_history_text(self, max_turns: int = 10) -> str:
        """Get conversation history, using summary if available."""
        if self._context_summary and len(self.history) > self._max_history_before_summarize:
            # Use summary + recent turns
            recent = self.history[-4:]
            recent_text = "\n\n".join(
                f"{'User' if t.role == 'user' else 'Assistant'}: {t.content}"
                for t in recent
            )
            return f"[Previous conversation summary: {self._context_summary}]\n\nRecent conversation:\n{recent_text}"
        else:
            # Full history
            recent = self.history[-max_turns:] if len(self.history) > max_turns else self.history
            lines = []
            for turn in recent:
                role = "User" if turn.role == "user" else "Assistant"
                lines.append(f"{role}: {turn.content}")
            return "\n\n".join(lines)
    
    def needs_extraction(self) -> bool:
        """Check if we need to run LLM extraction (new user messages since last extraction)."""
        user_turns = sum(1 for t in self.history if t.role == "user")
        return user_turns > self._last_extraction_turn
    
    def mark_extraction_done(self):
        """Mark that extraction has been performed."""
        self._last_extraction_turn = sum(1 for t in self.history if t.role == "user")
    
    def update_summary(self, summary: str):
        """Update the context summary."""
        self._context_summary = summary
    
    def has_minimum_info(self) -> bool:
        return self.info_gathered.get("symptoms", False)
    
    def get_missing_info(self) -> List[str]:
        return [k for k, v in self.info_gathered.items() if not v]


# Alias for backward compatibility
ConversationSession = ConversationContext


class ClinicalAgent:
    """
    High-level clinical assessment agent using ReAct pattern.
    
    Key features:
    - LLM-based patient info extraction (handles complex cases)
    - Proper context preservation across turns
    - Can answer questions about current context
    - Transparent reasoning traces
    """
    
    def __init__(
        self,
        retriever,
        llm,
        max_steps: int = 8,
        verbose: bool = False
    ):
        self.retriever = retriever
        self.llm = llm
        self.max_steps = max_steps
        self.verbose = verbose
        
        self.tools = ClinicalTools(retriever, llm)
        
        self.react_agent = ReActAgent(
            tools=self.tools,
            llm=llm,
            max_steps=max_steps,
            verbose=verbose
        )
        
        self.streaming_agent = StreamingReActAgent(
            tools=self.tools,
            llm=llm,
            max_steps=max_steps,
            verbose=verbose
        )
        
        self._sessions: Dict[str, ConversationContext] = {}
    
    def assess(self, patient: PatientInfo) -> ClinicalAssessment:
        """Perform clinical assessment."""
        logger.info(f"Starting assessment for patient (age: {patient.age})")
        
        task = patient.to_task_description()
        result = self.react_agent.run(task)
        assessment = self._parse_agent_result(result, patient)
        
        logger.info(f"Assessment complete: {assessment.risk_level.value}, {assessment.urgency.value}")
        return assessment
    
    def assess_streaming(self, patient: PatientInfo) -> Generator[Dict[str, Any], None, None]:
        """Perform assessment with streaming output."""
        task = patient.to_task_description()
        
        final_result = None
        for step in self.streaming_agent.run_streaming(task):
            yield step
            if step.get("type") == "final_answer":
                final_result = step
        
        if final_result:
            assessment = self._parse_final_answer(
                final_result.get("answer", ""),
                final_result.get("trace"),
                patient
            )
            yield {"type": "assessment", "assessment": assessment.to_dict()}
    
    def assess_quick(
        self,
        symptoms: List[str],
        age: Optional[int] = None,
        sex: Optional[str] = None
    ) -> ClinicalAssessment:
        """Quick assessment from symptoms only."""
        patient = PatientInfo(
            age=age,
            sex=sex,
            symptoms=symptoms,
            presenting_complaint=", ".join(symptoms)
        )
        return self.assess(patient)
    
    def _parse_agent_result(self, result: AgentResult, patient: PatientInfo) -> ClinicalAssessment:
        if not result.success or not result.answer:
            return ClinicalAssessment(
                assessment_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now().isoformat(),
                risk_level=RiskLevel.INSUFFICIENT,
                urgency=UrgencyLevel.ROUTINE,
                summary=result.error or "Assessment could not be completed",
                reasoning_trace=result.trace,
                confidence=0.0
            )
        
        return self._parse_final_answer(
            result.answer,
            result.trace,
            patient,
            tools_used=result.get_tools_used()
        )
    
    def _parse_final_answer(
        self,
        answer: str,
        trace: Optional[AgentTrace],
        patient: PatientInfo,
        tools_used: Optional[List[str]] = None
    ) -> ClinicalAssessment:
        """Parse final answer into structured assessment."""
        from src.llm.gemini import ResponseFormat
        
        extraction_prompt = get_clinical_assessment_extraction_prompt(answer)

        response = self.llm.generate(
            prompt=extraction_prompt,
            response_format=ResponseFormat.JSON,
            temperature=0.0
        )
        
        data = response.parsed or {}
        
        risk_map = {
            "critical": RiskLevel.CRITICAL,
            "high": RiskLevel.HIGH,
            "moderate": RiskLevel.MODERATE,
            "low": RiskLevel.LOW,
            "insufficient_info": RiskLevel.INSUFFICIENT
        }
        risk_level = risk_map.get(data.get("risk_level", "").lower(), RiskLevel.INSUFFICIENT)
        
        urgency_map = {
            "immediate": UrgencyLevel.IMMEDIATE,
            "urgent_2_week": UrgencyLevel.URGENT_2_WEEK,
            "urgent": UrgencyLevel.URGENT,
            "soon": UrgencyLevel.SOON,
            "routine": UrgencyLevel.ROUTINE
        }
        urgency = urgency_map.get(data.get("urgency", "").lower(), UrgencyLevel.ROUTINE)
        
        return ClinicalAssessment(
            assessment_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            risk_level=risk_level,
            urgency=urgency,
            summary=data.get("summary", answer[:200]),
            recommended_actions=data.get("recommended_actions", []),
            investigations=data.get("investigations", []),
            referral_pathway=data.get("referral_pathway"),
            reasoning_trace=trace,
            citations=data.get("citations", []),
            red_flags=data.get("red_flags", []),
            confidence=float(data.get("confidence", 0.7)),
            tools_used=tools_used or [],
            steps_taken=len(trace.steps) if trace else 0
        )
    
    def get_available_tools(self) -> List[Dict]:
        return self.tools.get_tools_schema()
    
    # ==========================================
    # Conversation Management
    # ==========================================
    
    def start_session(self, session_id: Optional[str] = None) -> ConversationContext:
        """Start a new conversation session."""
        session_id = session_id or str(uuid.uuid4())[:8]
        
        context = ConversationContext(session_id=session_id)
        
        greeting = """Hello! I'm here to help assess a patient for potential cancer referral 
according to NICE NG12 guidelines.

Please tell me about the patient. What symptoms are they presenting with?
Also helpful: age, sex, duration of symptoms, and any risk factors."""
        
        context.add_turn("assistant", greeting)
        self._sessions[session_id] = context
        
        logger.info(f"Started new session: {session_id}")
        return context
    
    def get_session(self, session_id: str) -> Optional[ConversationContext]:
        """Get existing session by ID."""
        return self._sessions.get(session_id)
    
    def chat(self, context: ConversationContext, user_message: str) -> str:
        """Process a user message and generate response."""
        logger.info(f"Chat in session {context.session_id}: {user_message[:50]}...")
        
        # Add user message to history
        context.add_turn("user", user_message)
        
        # 1. Check if this is a question about current context FIRST (uses cached data, no LLM)
        if self._is_context_question(user_message):
            response = self._answer_context_question(user_message, context)
            context.add_turn("assistant", response)
            return response
        
        # 2. Extract patient info ONLY if there's new user input since last extraction
        if context.needs_extraction():
            self._extract_patient_info_llm(context)
            context.mark_extraction_done()
            
            # Summarize context if history is getting long
            if len(context.history) > context._max_history_before_summarize:
                self._summarize_context(context)
        
        # 3. Check if user explicitly wants assessment
        explicit_assessment = self._is_assessment_request(user_message)
        
        # 4. Decide response
        if explicit_assessment and self._has_sufficient_info(context):
            # User requested assessment and we have enough info
            context.state = ConversationState.ASSESSING
            assessment = self.assess(context.patient_info)
            context.assessment_result = assessment
            context.state = ConversationState.COMPLETE
            response = self._format_assessment_response(assessment)
        elif explicit_assessment and not self._has_sufficient_info(context):
            # User wants assessment but we need more info
            missing = context.get_missing_info()
            response = "I'd like to run the assessment, but I need a bit more information first:\n" + \
                      "\n".join(f"- {self._get_question_for_field(m)}" for m in missing[:2])
            context.state = ConversationState.GATHERING
        elif self._has_all_key_info(context):
            # We have comprehensive info - offer to run assessment
            patient_summary = context.patient_info.to_summary()
            response = f"Thank you. I have the following information:\n\n**{patient_summary}**\n\n" + \
                      "Would you like me to run the NG12 cancer risk assessment now? " + \
                      "(Say 'yes' or 'assess' to proceed, or provide additional information)"
            context.state = ConversationState.CLARIFYING
        else:
            # Still gathering - ask for missing info
            missing = context.get_missing_info()
            response = self._ask_for_missing_info(missing)
            context.state = ConversationState.GATHERING
        
        context.add_turn("assistant", response)
        return response
    
    def _summarize_context(self, context: ConversationContext):
        """Summarize conversation history to reduce token usage."""
        from src.llm.gemini import ResponseFormat
        
        # Get current patient info as base
        patient = context.patient_info
        
        summary = f"Patient: {patient.age or '?'}yo {patient.sex or '?'}, "
        summary += f"symptoms: {', '.join(patient.symptoms) if patient.symptoms else 'unknown'}"
        if patient.symptom_duration:
            summary += f" for {patient.symptom_duration}"
        if patient.risk_factors:
            summary += f". Risk factors: {', '.join(patient.risk_factors)}"
        
        context.update_summary(summary)
        logger.info(f"Context summarized: {summary}")
    
    def _has_all_key_info(self, context: ConversationContext) -> bool:
        """Check if we have all key information (age + symptoms)."""
        info = context.patient_info
        return (info.age is not None and 
                len(info.symptoms) > 0 and 
                info.symptom_duration is not None)
    
    def _get_question_for_field(self, field: str) -> str:
        """Get question text for a missing field."""
        questions = {
            "age": "What is the patient's age?",
            "symptoms": "What symptoms is the patient experiencing?",
            "duration": "How long have these symptoms been present?",
            "risk_factors": "Are there any relevant risk factors (smoking, family history, etc.)?",
        }
        return questions.get(field, f"Can you provide the patient's {field}?")
    
    def _is_context_question(self, message: str) -> bool:
        """Check if user is asking about current context."""
        message_lower = message.lower().strip()
        
        # Must contain a question indicator
        question_words = ['what', 'how', 'tell', 'remind', 'summarize', 'summary', 'show', 'give']
        has_question_word = any(qw in message_lower for qw in question_words)
        
        if not has_question_word:
            return False
        
        # Direct context questions
        context_patterns = [
            r"what (is|was|are|were) the (patient'?s? )?age",
            r"what'?s the age",
            r"how old",
            r"what (is|was|are|were) the symptom",
            r"what symptom",
            r"what (is|was|are|were) the (patient'?s? )?(info|information|detail)",
            r"what do (you|we) (know|have)",
            r"what (is|are) in (the )?context",
            r"tell me (about )?the patient",
            r"summarize",
            r"summary",
            r"what have i told",
            r"what did i (say|tell)",
            r"remind me",
            r"show me (the )?(patient|info|details)",
            # Findings and results
            r"what (is|are|was|were) the (key )?(finding|result|conclusion|outcome)",
            r"(key )?finding",
            r"what did (you|the assessment) (find|conclude|determine)",
            r"give me (the )?(finding|result|summary|conclusion)",
            r"(show|tell) me (the )?(finding|result)",
        ]
        
        import re
        for pattern in context_patterns:
            if re.search(pattern, message_lower):
                return True
        
        return False
    
    def _is_assessment_request(self, message: str) -> bool:
        """Check if user explicitly wants to run assessment."""
        message_lower = message.lower().strip()
        
        # Direct assessment keywords
        assessment_keywords = [
            "assess", "evaluate", "analyze", "analyse", "check risk",
            "what is the risk", "run assessment", "do assessment",
            "cancer risk", "referral", "should i refer",
            "give assessment", "provide assessment", "start assessment",
        ]
        
        # Affirmative responses (when we've asked if they want assessment)
        affirmative_keywords = [
            "yes", "yeah", "yep", "sure", "ok", "okay", "go ahead",
            "please", "proceed", "do it", "run it", "yes please",
        ]
        
        # Check for assessment keywords
        if any(kw in message_lower for kw in assessment_keywords):
            return True
        
        # Check for short affirmative response
        if len(message_lower.split()) <= 3:
            if any(kw in message_lower for kw in affirmative_keywords):
                return True
        
        return False
    
    def _answer_context_question(self, message: str, context: ConversationContext) -> str:
        """Answer a question about the current context using cached data (no LLM call)."""
        patient = context.patient_info
        message_lower = message.lower()
        
        # Specific questions - answer from cached data
        if "age" in message_lower:
            if patient.age:
                return f"The patient's age is **{patient.age} years old**."
            else:
                return "I don't have the patient's age yet. Could you please provide it?"
        
        if "symptom" in message_lower:
            if patient.symptoms:
                return f"The patient's symptoms are: **{', '.join(patient.symptoms)}**"
            else:
                return "I don't have any symptoms recorded yet. What symptoms is the patient experiencing?"
        
        if "duration" in message_lower or "how long" in message_lower:
            if patient.symptom_duration:
                return f"The symptoms have been present for **{patient.symptom_duration}**."
            else:
                return "I don't have the duration of symptoms yet. How long have the symptoms been present?"
        
        if "risk" in message_lower or "smoking" in message_lower or "factor" in message_lower:
            if patient.risk_factors:
                return f"The patient's risk factors are: **{', '.join(patient.risk_factors)}**"
            else:
                return "No risk factors have been recorded. Are there any relevant risk factors (smoking, family history, etc.)?"
        
        if "sex" in message_lower or "gender" in message_lower:
            if patient.sex:
                return f"The patient's sex is **{patient.sex}**."
            else:
                return "I don't have the patient's sex recorded yet."
        
        # Questions about findings/results
        if any(kw in message_lower for kw in ["finding", "result", "assessment", "conclusion", "diagnosis"]):
            if context.assessment_result:
                result = context.assessment_result
                return self._format_key_findings(result)
            else:
                return "No assessment has been completed yet. Would you like me to run the assessment now? (Say 'yes' or 'assess')"
        
        # General context question - return full summary
        summary = patient.to_summary()
        if summary == "No patient information gathered yet.":
            return "I don't have any patient information yet. Please tell me about the patient's symptoms, age, and any relevant history."
        
        response = f"**Current patient information:**\n\n{summary}"
        
        # Add assessment result if available
        if context.assessment_result:
            result = context.assessment_result
            response += f"\n\n**Previous Assessment:**\n- Risk Level: {result.risk_level.value.upper()}\n- Urgency: {result.get_urgency_display()}"
        
        response += "\n\nIs there anything else you'd like to add or clarify?"
        return response
    
    def _format_key_findings(self, result: ClinicalAssessment) -> str:
        """Format key findings from assessment in a readable way."""
        lines = [
            "**Key Findings from Assessment:**\n",
            f"📊 **Risk Level:** {result.risk_level.value.upper()}",
            f"⏰ **Urgency:** {result.get_urgency_display()}",
            f"\n📝 **Summary:** {result.summary}",
        ]
        
        if result.red_flags:
            lines.append(f"\n⚠️ **Red Flags:** {', '.join(result.red_flags)}")
        
        if result.recommended_actions:
            lines.append(f"\n✅ **Key Actions:** {'; '.join(result.recommended_actions[:2])}")
        
        if result.citations:
            lines.append(f"\n📚 **Guidelines:** {', '.join(result.citations[:2])}")
        
        return "\n".join(lines)
    
    def _extract_patient_info_llm(self, context: ConversationContext):
        """
        Use LLM to extract patient info from conversation.
        This handles complex cases like negations ("no smoking history").
        """
        from src.llm.gemini import ResponseFormat
        
        # Get conversation history
        history_text = context.get_history_text(max_turns=10)
        
        # Current patient info (to preserve)
        current = context.patient_info
        
        prompt = get_clinical_patient_extraction_prompt(
            conversation_history=history_text,
            age=str(current.age) if current.age else "Unknown",
            sex=str(current.sex) if current.sex else "Unknown",
            symptoms=", ".join(current.symptoms) if current.symptoms else "None recorded",
            symptom_duration=str(current.symptom_duration) if current.symptom_duration else "Unknown",
            risk_factors=", ".join(current.risk_factors) if current.risk_factors else "None recorded"
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                response_format=ResponseFormat.JSON,
                temperature=0.0
            )
            
            data = response.parsed or {}
            
            # Update patient info
            if data.get("age") is not None:
                context.patient_info.age = data["age"]
                context.info_gathered["age"] = True
            
            if data.get("sex"):
                context.patient_info.sex = data["sex"]
            
            if data.get("symptoms"):
                # Merge with existing symptoms
                existing = set(context.patient_info.symptoms)
                new_symptoms = set(data["symptoms"])
                context.patient_info.symptoms = list(existing | new_symptoms)
                if context.patient_info.symptoms:
                    context.info_gathered["symptoms"] = True
            
            if data.get("symptom_duration"):
                context.patient_info.symptom_duration = data["symptom_duration"]
                context.info_gathered["duration"] = True
            
            if data.get("risk_factors"):
                # Replace risk factors (LLM handles negations)
                context.patient_info.risk_factors = data["risk_factors"]
                context.info_gathered["risk_factors"] = True
            elif "risk_factors" in data and data["risk_factors"] == []:
                # Explicitly no risk factors
                context.patient_info.risk_factors = []
                context.info_gathered["risk_factors"] = True
            
            if data.get("notes"):
                context.patient_info.additional_notes = data["notes"]
                
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            # Fall back to simple extraction
            self._extract_patient_info_simple(context)
    
    def _extract_patient_info_simple(self, context: ConversationContext):
        """Simple regex-based extraction as fallback."""
        # Get last few messages
        recent_messages = " ".join(
            turn.content for turn in context.history[-4:] 
            if turn.role == "user"
        ).lower()
        
        # Age extraction
        age_patterns = [
            r'(?:age|aged)\s*(?:is|:)?\s*(\d{1,3})',
            r'(\d{1,3})\s*(?:year|yr|yo)',
            r'patient\s+(?:is\s+)?(\d{1,3})',
        ]
        for pattern in age_patterns:
            match = re.search(pattern, recent_messages)
            if match:
                try:
                    age = int(match.group(1))
                    if 0 <= age <= 120:
                        context.patient_info.age = age
                        context.info_gathered["age"] = True
                        break
                except:
                    pass
        
        # Symptom extraction
        symptoms_map = {
            'cough': ['cough', 'coughing'],
            'fever': ['fever', 'febrile', 'temperature'],
            'weight loss': ['weight loss', 'losing weight'],
            'fatigue': ['fatigue', 'tired', 'tiredness'],
            'pain': ['pain', 'ache'],
            'bleeding': ['bleeding', 'blood'],
            'lump': ['lump', 'mass', 'swelling'],
            'breathlessness': ['breath', 'dyspnea', 'breathless'],
        }
        
        for symptom, keywords in symptoms_map.items():
            for kw in keywords:
                if kw in recent_messages and symptom not in context.patient_info.symptoms:
                    context.patient_info.symptoms.append(symptom)
                    context.info_gathered["symptoms"] = True
        
        # Duration
        duration_match = re.search(r'(\d+)\s*(day|week|month|year)s?', recent_messages)
        if duration_match:
            context.patient_info.symptom_duration = duration_match.group(0)
            context.info_gathered["duration"] = True
    
    def _has_sufficient_info(self, context: ConversationContext) -> bool:
        """Check if we have sufficient info for assessment."""
        return context.has_minimum_info()
    
    def _ask_for_missing_info(self, missing: List[str]) -> str:
        """Ask for missing information."""
        prompts = {
            "age": "What is the patient's age?",
            "symptoms": "What symptoms is the patient experiencing?",
            "duration": "How long have these symptoms been present?",
            "risk_factors": "Are there any relevant risk factors (smoking, family history, etc.)?",
        }
        
        questions = [prompts.get(m, f"Can you provide the patient's {m}?") for m in missing[:2]]
        
        return "To better assess this patient, I need more information:\n" + \
               "\n".join(f"- {q}" for q in questions)
    
    def _format_assessment_response(self, assessment: ClinicalAssessment) -> str:
        """Format assessment result as natural language response with readable reasoning."""
        lines = [
            "**Assessment Complete**\n",
            f"**Risk Level:** {assessment.risk_level.value.upper()}",
            f"**Urgency:** {assessment.get_urgency_display()}",
            f"\n**Summary:** {assessment.summary}",
        ]
        
        if assessment.recommended_actions:
            lines.append("\n**Recommended Actions:**")
            for action in assessment.recommended_actions:
                lines.append(f"- {action}")
        
        if assessment.investigations:
            lines.append("\n**Investigations to Consider:**")
            for inv in assessment.investigations:
                lines.append(f"- {inv}")
        
        if assessment.red_flags:
            lines.append("\n**⚠️ Red Flags:**")
            for flag in assessment.red_flags:
                lines.append(f"- {flag}")
        
        if assessment.citations:
            lines.append("\n**Guideline References:**")
            for citation in assessment.citations[:3]:
                lines.append(f"- {citation}")
        
        # Add formatted reasoning trace if available
        if assessment.reasoning_trace and assessment.reasoning_trace.steps:
            lines.append("\n---")
            lines.append("\n**Clinical Reasoning:**")
            lines.append(self._format_reasoning_trace(assessment.reasoning_trace))
        
        lines.append("\n*This assessment is based on NG12 guidelines and is intended to support, not replace, clinical judgment.*")
        
        return "\n".join(lines)
    
    def _format_reasoning_trace(self, trace: AgentTrace) -> str:
        """Format reasoning trace in human-readable format."""
        lines = []
        
        for i, step in enumerate(trace.steps, 1):
            # Format thought
            thought_summary = step.thought[:150] + "..." if len(step.thought) > 150 else step.thought
            lines.append(f"\n**Step {i}:** {thought_summary}")
            
            # Format action if present
            if step.action:
                action_display = {
                    "search_guidelines": "🔍 Searched guidelines",
                    "check_red_flags": "⚠️ Checked red flags",
                    "calculate_risk": "📊 Calculated risk",
                    "get_referral_pathway": "🏥 Retrieved referral pathway",
                    "extract_symptoms": "📋 Extracted symptoms",
                    "lookup_cancer_criteria": "📖 Looked up cancer criteria",
                    "get_section": "📄 Retrieved guideline section",
                }.get(step.action, f"🔧 {step.action}")
                
                lines.append(f"   → {action_display}")
                
                # Summarize observation if present
                if step.observation:
                    obs_preview = step.observation[:100] + "..." if len(step.observation) > 100 else step.observation
                    # Clean up JSON formatting for readability
                    obs_preview = obs_preview.replace('{', '').replace('}', '').replace('"', '')
                    lines.append(f"   ✓ Found: {obs_preview}")
        
        # Add final answer indicator
        if trace.final_answer:
            lines.append(f"\n**Conclusion reached after {len(trace.steps)} reasoning steps.**")
        
        return "\n".join(lines)


class ConversationalClinicalAgent:
    """Conversational wrapper around ClinicalAgent."""
    
    def __init__(self, clinical_agent: ClinicalAgent):
        self.agent = clinical_agent
        self.sessions: Dict[str, ConversationSession] = {}
    
    def start_session(self) -> ConversationSession:
        session = self.agent.start_session()
        self.sessions[session.session_id] = session
        return session
    
    def chat(self, session_id: str, message: str) -> str:
        session = self.sessions.get(session_id) or self.agent.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return self.agent.chat(session, message)
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        return self.sessions.get(session_id) or self.agent.get_session(session_id)


def create_clinical_agent(
    retriever=None,
    llm=None,
    mock: bool = False
) -> ClinicalAgent:
    """Factory function to create a clinical agent."""
    from src.llm.gemini import get_llm
    from src.services.retrieval import ClinicalRetriever
    from src.ingestion.embedder import get_embedder
    from src.ingestion.vector_store import VectorStore
    
    if llm is None:
        llm = get_llm(mock=mock)
    
    if retriever is None:
        embedder = get_embedder(mock=mock)
        vector_store = VectorStore()
        retriever = ClinicalRetriever(vector_store, embedder)
    
    return ClinicalAgent(retriever, llm)
