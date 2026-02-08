"""
Clinical Agent Tools

Tool definitions for the ReAct clinical assessment agent.
Each tool has a clear schema, description, and implementation.

Interview Discussion Points:
---------------------------
1. Why tool-based architecture?
   - Modular: Each capability is a separate tool
   - Auditable: Every action is logged
   - Extensible: Easy to add new tools
   - Testable: Tools can be unit tested

2. Tool design principles:
   - Single responsibility
   - Clear input/output schemas
   - Descriptive names and descriptions
   - Error handling built-in
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # string, integer, boolean, array
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Any = None


@dataclass
class Tool:
    """
    Tool definition for the agent.
    
    Attributes:
        name: Unique tool identifier
        description: What the tool does (used by LLM to decide when to use)
        parameters: List of parameters the tool accepts
        function: The actual function to execute
        returns: Description of return value
    """
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable
    returns: str = "Result of the tool execution"
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema for LLM."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            
            properties[param.name] = prop
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        try:
            result = self.function(**kwargs)
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    
    def to_string(self) -> str:
        """Format for inclusion in prompt."""
        if self.success:
            if isinstance(self.result, dict):
                return json.dumps(self.result, indent=2)
            return str(self.result)
        return f"Error: {self.error}"


class ClinicalTools:
    """
    Collection of tools for clinical assessment.
    
    Tools:
    1. search_guidelines - Search NG12 for relevant recommendations
    2. get_section - Get content from specific NG12 section
    3. extract_symptoms - Extract structured symptoms from text
    4. check_red_flags - Check for clinical red flags
    5. calculate_risk - Calculate risk level based on criteria
    6. get_referral_pathway - Get appropriate referral pathway
    7. lookup_cancer_criteria - Get criteria for specific cancer type
    """
    
    def __init__(self, retriever, llm):
        """
        Initialize clinical tools.
        
        Args:
            retriever: ClinicalRetriever instance
            llm: LLM instance for extraction tasks
        """
        self.retriever = retriever
        self.llm = llm
        self._tools: Dict[str, Tool] = {}
        self._register_tools()
    
    def _register_tools(self):
        """Register all available tools."""
        
        # Tool 1: Search Guidelines
        self._tools["search_guidelines"] = Tool(
            name="search_guidelines",
            description="""Search the NG12 clinical guidelines for relevant recommendations.
Use this to find guideline content related to specific symptoms, cancer types, or clinical scenarios.
Returns ranked chunks from the guidelines with relevance scores.""",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query describing symptoms, conditions, or clinical question"
                ),
                ToolParameter(
                    name="cancer_type",
                    type="string",
                    description="Optional: Filter by cancer type",
                    required=False,
                    enum=["lung", "breast", "colorectal", "prostate", "skin", "bladder", 
                          "brain", "head_neck", "thyroid", "oesophageal", "pancreatic",
                          "liver", "ovarian", "cervical", "endometrial", "renal",
                          "testicular", "lymphoma", "leukaemia", "myeloma", "sarcoma"]
                ),
                ToolParameter(
                    name="urgent_only",
                    type="boolean",
                    description="Only return urgent (2-week pathway) recommendations",
                    required=False,
                    default=False
                )
            ],
            function=self._search_guidelines,
            returns="List of relevant guideline excerpts with citations and scores"
        )
        
        # Tool 2: Get Section
        self._tools["get_section"] = Tool(
            name="get_section",
            description="""Get the full content of a specific NG12 section by section number.
Use this when you need to read a complete section referenced in guidelines.
Section numbers follow pattern like 1.1, 1.3.2, etc.""",
            parameters=[
                ToolParameter(
                    name="section_number",
                    type="string",
                    description="Section number (e.g., '1.1' for lung cancer, '1.3' for upper GI)"
                )
            ],
            function=self._get_section,
            returns="Full content of the specified section"
        )
        
        # Tool 3: Extract Symptoms
        self._tools["extract_symptoms"] = Tool(
            name="extract_symptoms",
            description="""Extract structured symptom information from free text clinical notes.
Use this to parse unstructured patient descriptions into structured data.
Identifies symptoms, duration, severity, and associated factors.""",
            parameters=[
                ToolParameter(
                    name="clinical_text",
                    type="string",
                    description="Free text clinical notes or patient description"
                )
            ],
            function=self._extract_symptoms,
            returns="Structured symptoms with duration, severity, and risk factors"
        )
        
        # Tool 4: Check Red Flags
        self._tools["check_red_flags"] = Tool(
            name="check_red_flags",
            description="""Check a list of symptoms against known clinical red flags for cancer.
Use this to identify concerning symptom combinations that warrant urgent attention.
Returns matched red flags with their clinical significance.""",
            parameters=[
                ToolParameter(
                    name="symptoms",
                    type="array",
                    description="List of symptoms to check"
                ),
                ToolParameter(
                    name="patient_age",
                    type="integer",
                    description="Patient age in years",
                    required=False
                ),
                ToolParameter(
                    name="patient_sex",
                    type="string",
                    description="Patient sex",
                    required=False,
                    enum=["male", "female"]
                )
            ],
            function=self._check_red_flags,
            returns="List of matched red flags with urgency levels"
        )
        
        # Tool 5: Calculate Risk
        self._tools["calculate_risk"] = Tool(
            name="calculate_risk",
            description="""Calculate cancer risk level based on symptoms, age, and risk factors.
Use this to determine the urgency level for a patient based on NG12 criteria.
Returns risk level (critical/high/moderate/low) with justification.""",
            parameters=[
                ToolParameter(
                    name="symptoms",
                    type="array",
                    description="List of presenting symptoms"
                ),
                ToolParameter(
                    name="age",
                    type="integer",
                    description="Patient age in years"
                ),
                ToolParameter(
                    name="risk_factors",
                    type="array",
                    description="Known risk factors (smoking, family history, etc.)",
                    required=False
                ),
                ToolParameter(
                    name="cancer_type",
                    type="string",
                    description="Suspected cancer type if known",
                    required=False
                )
            ],
            function=self._calculate_risk,
            returns="Risk level with urgency and justification"
        )
        
        # Tool 6: Get Referral Pathway
        self._tools["get_referral_pathway"] = Tool(
            name="get_referral_pathway",
            description="""Get the appropriate referral pathway based on suspected cancer and risk level.
Use this to determine the correct NHS referral route for the patient.
Returns pathway details including timeframe and required actions.""",
            parameters=[
                ToolParameter(
                    name="cancer_type",
                    type="string",
                    description="Suspected cancer type"
                ),
                ToolParameter(
                    name="urgency",
                    type="string",
                    description="Determined urgency level",
                    enum=["immediate", "urgent_2_week", "urgent", "routine"]
                )
            ],
            function=self._get_referral_pathway,
            returns="Referral pathway with timeline and actions"
        )
        
        # Tool 7: Lookup Cancer Criteria
        self._tools["lookup_cancer_criteria"] = Tool(
            name="lookup_cancer_criteria",
            description="""Look up the specific NG12 referral criteria for a cancer type.
Use this to get the exact symptoms and thresholds that trigger referral.
Returns the criteria list with age thresholds and symptom combinations.""",
            parameters=[
                ToolParameter(
                    name="cancer_type",
                    type="string",
                    description="Cancer type to look up criteria for"
                )
            ],
            function=self._lookup_cancer_criteria,
            returns="Referral criteria for the specified cancer type"
        )
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_tools_schema(self) -> List[Dict]:
        """Get JSON schema for all tools (for LLM)."""
        return [tool.to_schema() for tool in self._tools.values()]
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with given parameters."""
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}"
            )
        
        result = tool.execute(**kwargs)
        return ToolResult(
            tool_name=tool_name,
            success=result["success"],
            result=result.get("result"),
            error=result.get("error")
        )
    
    # Tool implementations
    
    def _search_guidelines(
        self,
        query: str,
        cancer_type: Optional[str] = None,
        urgent_only: bool = False
    ) -> Dict[str, Any]:
        """Search NG12 guidelines."""
        context = self.retriever.retrieve_for_patient(
            query=query,
            suspected_cancer=cancer_type,
            urgent_only=urgent_only,
            top_k=5
        )
        
        results = []
        for r in context.results:
            results.append({
                "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                "citation": r.citation,
                "page": r.page,
                "urgency": r.urgency,
                "score": round(r.score, 3)
            })
        
        return {
            "query": query,
            "results_count": len(results),
            "results": results
        }
    
    def _get_section(self, section_number: str) -> Dict[str, Any]:
        """Get content from a specific section."""
        context = self.retriever.retrieve_by_section(section_number, top_k=10)
        
        if context.is_empty:
            return {"error": f"Section {section_number} not found"}
        
        content = "\n\n".join(r.text for r in context.results)
        return {
            "section": section_number,
            "content": content,
            "chunks_count": len(context.results)
        }
    
    def _extract_symptoms(self, clinical_text: str) -> Dict[str, Any]:
        """Extract symptoms from clinical text using LLM."""
        from src.llm.gemini import ResponseFormat
        
        prompt = f"""Extract clinical information from this text:

"{clinical_text}"

Return a JSON object with:
- symptoms: list of symptoms mentioned
- duration: how long symptoms have been present (if mentioned)
- severity: severity indicators (mild/moderate/severe)
- risk_factors: any risk factors mentioned
- red_flags: any concerning findings
- age: patient age if mentioned
- sex: patient sex if mentioned"""

        response = self.llm.generate(
            prompt=prompt,
            response_format=ResponseFormat.JSON,
            temperature=0.0
        )
        
        return response.parsed or {"symptoms": [], "error": "Extraction failed"}
    
    def _check_red_flags(
        self,
        symptoms: List[str],
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None
    ) -> Dict[str, Any]:
        """Check symptoms against red flag criteria."""
        
        # Red flag definitions based on NG12
        RED_FLAGS = {
            "haemoptysis": {
                "urgency": "urgent_2_week",
                "condition": "lung_cancer",
                "note": "Coughing up blood - urgent chest X-ray needed"
            },
            "unexplained weight loss": {
                "urgency": "urgent_2_week",
                "condition": "multiple",
                "note": "Unintentional weight loss >5% in 3 months"
            },
            "persistent hoarseness": {
                "urgency": "urgent_2_week",
                "condition": "laryngeal_cancer",
                "note": "Voice changes lasting >3 weeks"
            },
            "dysphagia": {
                "urgency": "urgent_2_week",
                "condition": "oesophageal_cancer",
                "note": "Difficulty swallowing - urgent endoscopy"
            },
            "breast lump": {
                "urgency": "urgent_2_week",
                "condition": "breast_cancer",
                "note": "New breast lump - urgent referral"
            },
            "rectal bleeding": {
                "urgency": "urgent_2_week",
                "condition": "colorectal_cancer",
                "note": "Rectal bleeding with change in bowel habit"
            },
            "visible haematuria": {
                "urgency": "urgent_2_week",
                "condition": "bladder_cancer",
                "note": "Blood in urine - urgent urology referral"
            },
            "testicular lump": {
                "urgency": "urgent_2_week",
                "condition": "testicular_cancer",
                "note": "Non-painful testicular enlargement"
            },
            "post-menopausal bleeding": {
                "urgency": "urgent_2_week",
                "condition": "endometrial_cancer",
                "note": "Any bleeding after menopause"
            }
        }
        
        matched_flags = []
        symptoms_lower = [s.lower() for s in symptoms]
        
        for symptom in symptoms_lower:
            for flag, details in RED_FLAGS.items():
                if flag in symptom or symptom in flag:
                    matched_flags.append({
                        "symptom": symptom,
                        "red_flag": flag,
                        **details
                    })
        
        # Age-based red flags
        if patient_age and patient_age >= 40:
            for symptom in symptoms_lower:
                if "cough" in symptom and "persistent" in symptom:
                    matched_flags.append({
                        "symptom": symptom,
                        "red_flag": "persistent cough in patient ≥40",
                        "urgency": "consider",
                        "condition": "lung_cancer",
                        "note": "Consider chest X-ray if smoker or ex-smoker"
                    })
        
        return {
            "symptoms_checked": symptoms,
            "red_flags_found": len(matched_flags),
            "red_flags": matched_flags,
            "requires_urgent_action": any(f["urgency"] == "urgent_2_week" for f in matched_flags)
        }
    
    def _calculate_risk(
        self,
        symptoms: List[str],
        age: int,
        risk_factors: Optional[List[str]] = None,
        cancer_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate risk level based on NG12 criteria."""
        
        risk_factors = risk_factors or []
        risk_score = 0
        reasons = []
        
        # Check red flags first
        red_flag_result = self._check_red_flags(symptoms, age)
        if red_flag_result["requires_urgent_action"]:
            return {
                "risk_level": "high",
                "urgency": "urgent_2_week",
                "score": 100,
                "reasons": [f["note"] for f in red_flag_result["red_flags"]],
                "recommendation": "Refer using suspected cancer pathway"
            }
        
        # Age factor
        if age >= 40:
            risk_score += 20
            reasons.append(f"Patient age {age} (≥40)")
        if age >= 50:
            risk_score += 10
            reasons.append(f"Patient age {age} (≥50)")
        
        # Symptom count
        if len(symptoms) >= 3:
            risk_score += 15
            reasons.append(f"Multiple symptoms ({len(symptoms)})")
        
        # Risk factors
        high_risk_factors = ["smoker", "smoking", "family history", "previous cancer"]
        for rf in risk_factors:
            if any(hr in rf.lower() for hr in high_risk_factors):
                risk_score += 20
                reasons.append(f"Risk factor: {rf}")
        
        # Determine level
        if risk_score >= 60:
            risk_level = "high"
            urgency = "urgent_2_week"
            recommendation = "Consider suspected cancer pathway referral"
        elif risk_score >= 40:
            risk_level = "moderate"
            urgency = "urgent"
            recommendation = "Urgent investigation recommended"
        elif risk_score >= 20:
            risk_level = "low"
            urgency = "routine"
            recommendation = "Routine referral or monitoring"
        else:
            risk_level = "low"
            urgency = "routine"
            recommendation = "Monitor and review if symptoms persist"
        
        return {
            "risk_level": risk_level,
            "urgency": urgency,
            "score": risk_score,
            "reasons": reasons,
            "recommendation": recommendation
        }
    
    def _get_referral_pathway(
        self,
        cancer_type: str,
        urgency: str
    ) -> Dict[str, Any]:
        """Get referral pathway details."""
        
        PATHWAYS = {
            "lung": {
                "urgent_2_week": {
                    "pathway": "Suspected lung cancer pathway",
                    "timeline": "Appointment within 2 weeks",
                    "actions": [
                        "Request urgent chest X-ray",
                        "Refer to respiratory physician",
                        "Include smoking history in referral"
                    ]
                }
            },
            "breast": {
                "urgent_2_week": {
                    "pathway": "Suspected breast cancer pathway", 
                    "timeline": "Appointment within 2 weeks",
                    "actions": [
                        "Refer to breast clinic",
                        "Include examination findings",
                        "Note any family history"
                    ]
                }
            },
            "colorectal": {
                "urgent_2_week": {
                    "pathway": "Suspected lower GI cancer pathway",
                    "timeline": "Appointment within 2 weeks",
                    "actions": [
                        "Request FIT test if appropriate",
                        "Refer for colonoscopy",
                        "Include bowel habit changes"
                    ]
                }
            }
        }
        
        cancer_lower = cancer_type.lower()
        if cancer_lower in PATHWAYS and urgency in PATHWAYS[cancer_lower]:
            return PATHWAYS[cancer_lower][urgency]
        
        # Default pathway
        return {
            "pathway": f"Suspected {cancer_type} cancer pathway",
            "timeline": "2 weeks" if urgency == "urgent_2_week" else "Routine",
            "actions": [
                f"Refer to appropriate specialist",
                "Include full clinical history",
                "Document examination findings"
            ]
        }
    
    def _lookup_cancer_criteria(self, cancer_type: str) -> Dict[str, Any]:
        """Look up referral criteria for a cancer type."""
        
        # Search guidelines for criteria
        context = self.retriever.retrieve(
            query=f"{cancer_type} cancer referral criteria symptoms",
            top_k=5
        )
        
        criteria_text = "\n".join(r.text for r in context.results)
        
        return {
            "cancer_type": cancer_type,
            "source": "NG12 Guidelines",
            "criteria": criteria_text if criteria_text else f"No specific criteria found for {cancer_type}"
        }
