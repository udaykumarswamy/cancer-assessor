"""
Clinical Assessment Service

Orchestrates the clinical risk assessment workflow:
1. Patient intake and symptom extraction
2. Guideline retrieval
3. Risk assessment with LLM reasoning
4. Recommendation generation with citations

Interview Discussion Points:
---------------------------
1. Assessment workflow:
   - Structured patient intake → symptom normalization
   - Targeted retrieval based on symptoms + demographics
   - LLM reasoning with retrieved guidelines
   - Structured output with citations

2. Risk stratification:
   - Based on NG12 urgency levels
   - Considers symptom combinations
   - Age-adjusted thresholds

3. Citation importance:
   - Every recommendation must cite NG12
   - Enables verification and audit
   - Builds trust in AI recommendations
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from src.config.logging_config import get_logger
from src.services.retrieval import ClinicalRetriever, RetrievalContext
from src.llm.gemini import GeminiLLM, ResponseFormat, LLMResponse

logger = get_logger("assessment")


class RiskLevel(Enum):
    """Clinical risk levels aligned with NG12."""
    CRITICAL = "critical"           # Immediate action required
    HIGH = "high"                   # 2-week pathway referral
    MODERATE = "moderate"           # Urgent but not 2-week
    LOW = "low"                     # Routine referral or monitoring
    INSUFFICIENT_INFO = "insufficient_info"  # Need more information


class UrgencyLevel(Enum):
    """Recommended urgency for action."""
    IMMEDIATE = "immediate"         # Same day
    URGENT_2_WEEK = "urgent_2_week" # Suspected cancer pathway
    URGENT = "urgent"               # Within 2 weeks but not cancer pathway
    SOON = "soon"                   # Within 4-6 weeks
    ROUTINE = "routine"             # Standard referral timeline
    MONITOR = "monitor"             # Watchful waiting


@dataclass
class PatientContext:
    """
    Patient information for assessment.
    
    Attributes:
        patient_id: Unique identifier
        age: Patient age in years
        sex: Patient sex (male/female/other)
        presenting_complaint: Main reason for consultation
        symptoms: List of symptoms
        symptom_duration: How long symptoms present
        medical_history: Relevant medical history
        risk_factors: Known risk factors
        family_history: Relevant family history
        medications: Current medications
        additional_notes: Any additional context
    """
    patient_id: str = ""
    age: Optional[int] = None
    sex: Optional[str] = None
    presenting_complaint: str = ""
    symptoms: List[str] = field(default_factory=list)
    symptom_duration: Optional[str] = None
    medical_history: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    additional_notes: str = ""
    
    def to_prompt_text(self) -> str:
        """Format patient context for LLM prompt."""
        lines = []
        
        if self.age:
            lines.append(f"- Age: {self.age} years")
        if self.sex:
            lines.append(f"- Sex: {self.sex}")
        if self.presenting_complaint:
            lines.append(f"- Presenting complaint: {self.presenting_complaint}")
        if self.symptoms:
            lines.append(f"- Symptoms: {', '.join(self.symptoms)}")
        if self.symptom_duration:
            lines.append(f"- Duration: {self.symptom_duration}")
        if self.medical_history:
            lines.append(f"- Medical history: {', '.join(self.medical_history)}")
        if self.risk_factors:
            lines.append(f"- Risk factors: {', '.join(self.risk_factors)}")
        if self.family_history:
            lines.append(f"- Family history: {', '.join(self.family_history)}")
        if self.additional_notes:
            lines.append(f"- Additional notes: {self.additional_notes}")
        
        return "\n".join(lines)


@dataclass
class AssessmentResult:
    """
    Complete assessment result.
    
    Contains risk assessment, recommendations, and supporting evidence.
    """
    # Core assessment
    risk_level: RiskLevel
    urgency: UrgencyLevel
    summary: str
    
    # Recommendations
    recommended_actions: List[str] = field(default_factory=list)
    investigations: List[str] = field(default_factory=list)
    referral_pathway: Optional[str] = None
    
    # Evidence and reasoning
    reasoning: str = ""
    citations: List[str] = field(default_factory=list)
    matched_criteria: List[str] = field(default_factory=list)
    
    # Possible diagnoses
    differential_considerations: List[str] = field(default_factory=list)
    
    # Metadata
    confidence: float = 0.0
    assessment_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_used: str = ""
    
    # Red flags
    red_flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "assessment_id": self.assessment_id,
            "risk_level": self.risk_level.value,
            "urgency": self.urgency.value,
            "summary": self.summary,
            "recommended_actions": self.recommended_actions,
            "investigations": self.investigations,
            "referral_pathway": self.referral_pathway,
            "reasoning": self.reasoning,
            "citations": self.citations,
            "matched_criteria": self.matched_criteria,
            "differential_considerations": self.differential_considerations,
            "red_flags": self.red_flags,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "model_used": self.model_used,
        }
    
    def get_urgency_display(self) -> str:
        """Get human-readable urgency."""
        displays = {
            UrgencyLevel.IMMEDIATE: "🔴 Immediate action required",
            UrgencyLevel.URGENT_2_WEEK: "🔴 2-week suspected cancer pathway",
            UrgencyLevel.URGENT: "🟠 Urgent referral recommended",
            UrgencyLevel.SOON: "🟡 Referral within 4-6 weeks",
            UrgencyLevel.ROUTINE: "🟢 Routine referral",
            UrgencyLevel.MONITOR: "⚪ Monitor and review",
        }
        return displays.get(self.urgency, str(self.urgency.value))


class ClinicalAssessmentService:
    """
    Service for performing clinical risk assessments.
    
    Orchestrates:
    1. Patient context processing
    2. Guideline retrieval
    3. LLM-based reasoning
    4. Structured recommendation generation
    
    Usage:
        service = ClinicalAssessmentService(retriever, llm)
        
        patient = PatientContext(
            age=55,
            symptoms=["persistent cough", "weight loss"],
            risk_factors=["smoker"]
        )
        
        result = service.assess(patient)
        print(result.summary)
        print(result.recommended_actions)
    """
    
    SYSTEM_INSTRUCTION = """You are a clinical decision support assistant helping healthcare professionals 
assess patients according to NICE NG12 guidelines for suspected cancer recognition and referral.

Your role is to:
1. Analyze patient symptoms and risk factors
2. Match against NG12 criteria for cancer referral
3. Provide evidence-based recommendations with specific citations
4. Clearly indicate urgency levels

Important guidelines:
- Always cite specific NG12 sections for recommendations
- Be clear about urgency (2-week pathway vs routine referral)
- Note when information is insufficient for assessment
- Flag any red flags or concerning symptom combinations
- This is decision SUPPORT - final decisions are made by clinicians"""

    ASSESSMENT_SCHEMA = {
        "risk_level": "critical|high|moderate|low|insufficient_info",
        "urgency": "immediate|urgent_2_week|urgent|soon|routine|monitor",
        "summary": "Brief assessment summary",
        "recommended_actions": ["List of recommended actions"],
        "investigations": ["Recommended investigations/tests"],
        "referral_pathway": "Specific referral pathway if applicable",
        "reasoning": "Clinical reasoning explanation",
        "matched_criteria": ["NG12 criteria that were matched"],
        "citations": ["[NG12 Section X.X, p.Y] format"],
        "differential_considerations": ["Possible diagnoses to consider"],
        "red_flags": ["Any red flags identified"],
        "confidence": 0.0
    }
    
    def __init__(
        self,
        retriever: ClinicalRetriever,
        llm: GeminiLLM,
        top_k_retrieval: int = 5,
    ):
        """
        Initialize the assessment service.
        
        Args:
            retriever: ClinicalRetriever instance
            llm: GeminiLLM instance
            top_k_retrieval: Number of chunks to retrieve
        """
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k_retrieval
    
    def assess(
        self,
        patient: PatientContext,
        include_reasoning: bool = True,
    ) -> AssessmentResult:
        """
        Perform full clinical assessment.
        
        Args:
            patient: Patient context
            include_reasoning: Include detailed reasoning in response
            
        Returns:
            AssessmentResult with recommendations
        """
        logger.info(f"Starting assessment for patient {patient.patient_id or 'unknown'}")
        
        # Step 1: Build query from patient context
        query = self._build_assessment_query(patient)
        logger.debug(f"Assessment query: {query[:100]}...")
        
        # Step 2: Retrieve relevant guidelines
        context = self.retriever.retrieve_for_patient(
            query=query,
            patient_age=patient.age,
            symptoms=patient.symptoms,
            top_k=self.top_k,
        )
        
        logger.info(f"Retrieved {len(context.results)} relevant guideline chunks")
        
        if context.is_empty:
            logger.warning("No relevant guidelines found")
            return self._insufficient_info_result(patient, "No matching guidelines found")
        
        # Step 3: Generate assessment with LLM
        result = self._generate_assessment(patient, context, include_reasoning)
        
        logger.info(f"Assessment complete: {result.risk_level.value} risk, {result.urgency.value}")
        
        return result
    
    def assess_symptoms(
        self,
        symptoms: List[str],
        age: Optional[int] = None,
        sex: Optional[str] = None,
    ) -> AssessmentResult:
        """
        Quick assessment from symptoms only.
        
        Args:
            symptoms: List of symptoms
            age: Patient age
            sex: Patient sex
            
        Returns:
            AssessmentResult
        """
        patient = PatientContext(
            age=age,
            sex=sex,
            symptoms=symptoms,
            presenting_complaint=", ".join(symptoms),
        )
        return self.assess(patient)
    
    def extract_symptoms(self, free_text: str) -> Dict[str, Any]:
        """
        Extract structured symptoms from free text.
        
        Args:
            free_text: Unstructured clinical notes
            
        Returns:
            Dict with extracted symptoms, duration, etc.
        """
        logger.info("Extracting symptoms from free text")
        
        extraction_prompt = f"""Extract clinical information from the following text.

Text: {free_text}

Extract:
1. All symptoms mentioned
2. Duration of symptoms if mentioned
3. Any risk factors mentioned
4. Any red flags

Respond in JSON format."""

        schema = {
            "symptoms": ["list of symptoms"],
            "duration": "duration string or null",
            "risk_factors": ["list of risk factors"],
            "red_flags": ["any concerning findings"],
            "age_mentioned": "age if mentioned or null",
            "sex_mentioned": "sex if mentioned or null"
        }
        
        response = self.llm.generate(
            prompt=extraction_prompt,
            response_format=ResponseFormat.JSON,
            json_schema=schema,
            temperature=0.0,  # Deterministic extraction
        )
        
        if response.parsed:
            return response.parsed
        
        return {"symptoms": [], "error": "Failed to extract symptoms"}
    
    def _build_assessment_query(self, patient: PatientContext) -> str:
        """Build search query from patient context."""
        parts = []
        
        # Add presenting complaint
        if patient.presenting_complaint:
            parts.append(patient.presenting_complaint)
        
        # Add symptoms
        if patient.symptoms:
            parts.append(" ".join(patient.symptoms[:5]))
        
        # Add age context for guideline matching
        if patient.age:
            if patient.age >= 40:
                parts.append("aged 40 and over")
            if patient.age >= 50:
                parts.append("aged 50 and over")
        
        # Add risk factors
        if patient.risk_factors:
            parts.append(" ".join(patient.risk_factors[:3]))
        
        return " ".join(parts) if parts else "cancer referral criteria"
    
    def _generate_assessment(
        self,
        patient: PatientContext,
        context: RetrievalContext,
        include_reasoning: bool,
    ) -> AssessmentResult:
        """Generate assessment using LLM."""
        
        # Build prompt
        prompt = f"""Assess this patient according to NG12 guidelines.

## Patient Information:
{patient.to_prompt_text()}

## Relevant NG12 Guidelines:
{context.get_context_text(max_chunks=self.top_k)}

## Instructions:
1. Identify which NG12 criteria apply to this patient
2. Determine the appropriate urgency level
3. Provide specific recommendations with citations
4. Note any red flags or concerning features
5. Indicate confidence in the assessment

Provide your assessment as a JSON object."""

        # Generate
        response = self.llm.generate(
            prompt=prompt,
            system_instruction=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat.JSON,
            json_schema=self.ASSESSMENT_SCHEMA,
            temperature=0.1,  # Low temperature for consistency
        )
        
        # Parse response
        return self._parse_assessment_response(response, patient, context)
    
    def _parse_assessment_response(
        self,
        response: LLMResponse,
        patient: PatientContext,
        context: RetrievalContext,
    ) -> AssessmentResult:
        """Parse LLM response into AssessmentResult."""
        
        if not response.parsed:
            logger.warning("Failed to parse LLM response as JSON")
            return self._insufficient_info_result(patient, "Failed to generate assessment")
        
        data = response.parsed
        
        # Map risk level
        risk_map = {
            "critical": RiskLevel.CRITICAL,
            "high": RiskLevel.HIGH,
            "moderate": RiskLevel.MODERATE,
            "low": RiskLevel.LOW,
            "insufficient_info": RiskLevel.INSUFFICIENT_INFO,
        }
        risk_level = risk_map.get(
            data.get("risk_level", "").lower(),
            RiskLevel.INSUFFICIENT_INFO
        )
        
        # Map urgency
        urgency_map = {
            "immediate": UrgencyLevel.IMMEDIATE,
            "urgent_2_week": UrgencyLevel.URGENT_2_WEEK,
            "urgent": UrgencyLevel.URGENT,
            "soon": UrgencyLevel.SOON,
            "routine": UrgencyLevel.ROUTINE,
            "monitor": UrgencyLevel.MONITOR,
        }
        urgency = urgency_map.get(
            data.get("urgency", "").lower(),
            UrgencyLevel.ROUTINE
        )
        
        # Build result
        import uuid
        return AssessmentResult(
            assessment_id=str(uuid.uuid4())[:8],
            risk_level=risk_level,
            urgency=urgency,
            summary=data.get("summary", "Assessment generated"),
            recommended_actions=data.get("recommended_actions", []),
            investigations=data.get("investigations", []),
            referral_pathway=data.get("referral_pathway"),
            reasoning=data.get("reasoning", ""),
            citations=data.get("citations", context.get_citations()),
            matched_criteria=data.get("matched_criteria", []),
            differential_considerations=data.get("differential_considerations", []),
            red_flags=data.get("red_flags", []),
            confidence=float(data.get("confidence", 0.7)),
            model_used=response.model,
        )
    
    def _insufficient_info_result(
        self,
        patient: PatientContext,
        reason: str,
    ) -> AssessmentResult:
        """Create result when assessment cannot be completed."""
        import uuid
        return AssessmentResult(
            assessment_id=str(uuid.uuid4())[:8],
            risk_level=RiskLevel.INSUFFICIENT_INFO,
            urgency=UrgencyLevel.ROUTINE,
            summary=f"Unable to complete assessment: {reason}",
            recommended_actions=["Gather additional clinical information"],
            reasoning=reason,
            confidence=0.0,
        )
