"""
Gemini LLM Module

Wrapper for Google's Gemini 1.5 Pro model via Vertex AI.
Provides structured prompting, response parsing, and error handling.

Why this architecture?:
---------------------------
1. Why Gemini 2.5 Pro?
   - Long context window (1M+ tokens) for complex clinical cases
   - Strong reasoning capabilities for medical guidelines
   - Native JSON mode for structured outputs
   - Cost-effective for production use

2. Temperature settings:
   - 0.1 for clinical assessments (consistency critical)
   - 0.3 for explanations (slight creativity)
   - 0.0 for JSON extraction (deterministic)

3. Safety settings:
   - Medical content requires careful safety configuration
   - Block harmful content but allow clinical terminology
"""

import json
import time
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum

# Try to import Vertex AI
try:
    import vertexai
    from vertexai.generative_models import (
        GenerativeModel,
        GenerationConfig,
        HarmCategory,
        HarmBlockThreshold,
        Content,
        Part
    )
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

# Try to import settings
# try:
#     from src.config.settings import settings
# except ImportError:
#     class DefaultSettings:
#         GCP_PROJECT_ID = "your-gcp-project-id"
#         GCP_LOCATION = "us-central1"
#         LLM_MODEL = "gemini-1.5-pro"
#         LLM_TEMPERATURE = 0.1
#     settings = DefaultSettings()
from src.config.settings import settings
from src.config.logging_config import get_logger

logger = get_logger("llm")


class ResponseFormat(Enum):
    """Output format for LLM responses."""
    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class LLMResponse:
    """
    Structured response from LLM.
    
    Attributes:
        content: Raw response text
        parsed: Parsed JSON (if JSON format requested)
        model: Model used
        usage: Token usage statistics
        latency_ms: Response latency in milliseconds
        finish_reason: Why generation stopped
    """
    content: str
    parsed: Optional[Dict[str, Any]] = None
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    finish_reason: str = ""
    
    @property
    def is_valid_json(self) -> bool:
        """Check if response contains valid JSON."""
        return self.parsed is not None
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from parsed JSON."""
        if self.parsed:
            return self.parsed.get(key, default)
        return default


class GeminiLLM:
    """
    Wrapper for Gemini 2.5 Pro via Vertex AI.
    
    Features:
    - Lazy initialization
    - Structured output (JSON mode)
    - Configurable safety settings
    - Automatic retries
    - Response parsing
    
    Usage:
        llm = GeminiLLM()
        
        # Simple text generation
        response = llm.generate("Explain cancer staging")
        
        # JSON output
        response = llm.generate(
            prompt="Extract symptoms from: Patient has cough and fever",
            response_format=ResponseFormat.JSON,
            json_schema={"symptoms": ["list of symptoms"]}
        )
        print(response.parsed)  # {"symptoms": ["cough", "fever"]}
    """
    
    # Safety settings for medical content
    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    } if VERTEX_AI_AVAILABLE else {}
    
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize Gemini LLM.
        
        Args:
            project_id: GCP project ID
            location: GCP region
            model_name: Model name (e.g., "gemini-2.5-pro")
            temperature: Default temperature (0.0-1.0)
        """
        self.project_id = project_id or settings.GCP_PROJECT_ID
        self.location = location or settings.GCP_LOCATION
        self.model_name = model_name or settings.LLM_MODEL
        self.temperature = temperature or settings.LLM_TEMPERATURE
        
        self._model = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Initialize Vertex AI lazily."""
        if not VERTEX_AI_AVAILABLE:
            raise RuntimeError(
                "Vertex AI not installed. Run: pip install google-cloud-aiplatform"
            )
        
        if not self._initialized:
            logger.info(f"Initializing Gemini ({self.model_name})...")
            vertexai.init(
                project=self.project_id,
                location=self.location
            )
            self._model = GenerativeModel(
                self.model_name,
                safety_settings=self.SAFETY_SETTINGS
            )
            self._initialized = True
    
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        response_format: ResponseFormat = ResponseFormat.TEXT,
        json_schema: Optional[Dict] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        stop_sequences: Optional[List[str]] = None,
    ) -> LLMResponse:
        """
        Generate a response from Gemini.
        
        Args:
            prompt: User prompt
            system_instruction: System prompt (role/context)
            response_format: Desired output format
            json_schema: Expected JSON structure (for JSON format)
            temperature: Override default temperature
            max_tokens: Maximum output tokens
            stop_sequences: Sequences that stop generation
            
        Returns:
            LLMResponse with content and metadata
        """
        self._ensure_initialized()
        
        # Build generation config
        gen_config = GenerationConfig(
            temperature=temperature or self.temperature,
            max_output_tokens=max_tokens,
            stop_sequences=stop_sequences or [],
        )
        
        # Add JSON mode if requested
        if response_format == ResponseFormat.JSON:
            gen_config.response_mime_type = "application/json"
            if json_schema:
                prompt = self._add_json_schema_to_prompt(prompt, json_schema)
        
        # Build the full prompt
        full_prompt = prompt
        if system_instruction:
            full_prompt = f"{system_instruction}\n\n{prompt}"
        
        # Generate
        start_time = time.time()
        try:
            response = self._model.generate_content(
                full_prompt,
                generation_config=gen_config
            )
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract content - handle multiple parts
            content = ""
            try:
                # Try to use response.text first
                content = response.text if response.text else ""
            except Exception:
                # If response.text fails (multiple parts), manually concatenate parts
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        parts = candidate.content.parts
                        content = "".join(part.text for part in parts if hasattr(part, 'text'))
            
            # Parse JSON if requested
            parsed = None
            if response_format == ResponseFormat.JSON and content:
                parsed = self._parse_json(content)
            
            # Build response
            return LLMResponse(
                content=content,
                parsed=parsed,
                model=self.model_name,
                usage={
                    "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0,
                },
                latency_ms=latency_ms,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else "unknown"
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def generate_with_context(
        self,
        query: str,
        context_chunks: List[str],
        system_instruction: Optional[str] = None,
        response_format: ResponseFormat = ResponseFormat.TEXT,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response with RAG context.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            system_instruction: System prompt
            response_format: Output format
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse
        """
        # Build context section
        context_text = "\n\n---\n\n".join(context_chunks)
        
        # Build RAG prompt
        prompt = f"""Based on the following clinical guideline excerpts, answer the query.

## Retrieved Guidelines:
{context_text}

## Query:
{query}

## Instructions:
- Use ONLY information from the guidelines above
- Cite specific sections when making recommendations
- If the guidelines don't address the query, say so clearly
- Be precise about urgency levels and timeframes
"""
        
        return self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            response_format=response_format,
            **kwargs
        )
    
    def _add_json_schema_to_prompt(self, prompt: str, schema: Dict) -> str:
        """Add JSON schema instruction to prompt."""
        schema_str = json.dumps(schema, indent=2)
        return f"""{prompt}

Respond with a JSON object matching this schema:
```json
{schema_str}
```

Return ONLY the JSON object, no other text."""
    
    def _parse_json(self, content: str) -> Optional[Dict]:
        """Parse JSON from response content."""
        try:
            # Try direct parse
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            import re
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON object in text
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        
        logger.warning(f"Failed to parse JSON from response")
        return None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate)."""
        # Gemini uses ~4 chars per token for English
        return len(text) // 4


class MockLLM:
    """
    Mock LLM for testing without Vertex AI credentials.
    
    Returns predefined responses based on prompt patterns.
    """
    
    def __init__(self, **kwargs):
        self.model_name = "mock-gemini"
        self.temperature = 0.1
    
    def generate(
        self,
        prompt: str,
        response_format: ResponseFormat = ResponseFormat.TEXT,
        **kwargs
    ) -> LLMResponse:
        """Generate mock response."""
        
        # Check for JSON format
        if response_format == ResponseFormat.JSON:
            mock_json = self._get_mock_json(prompt)
            return LLMResponse(
                content=json.dumps(mock_json),
                parsed=mock_json,
                model=self.model_name,
                usage={"prompt_tokens": 100, "completion_tokens": 50},
                latency_ms=100,
                finish_reason="STOP"
            )
        
        # Text response
        mock_text = self._get_mock_text(prompt)
        return LLMResponse(
            content=mock_text,
            model=self.model_name,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            latency_ms=100,
            finish_reason="STOP"
        )
    
    def generate_with_context(self, query: str, context_chunks: List[str], **kwargs) -> LLMResponse:
        """Generate mock RAG response."""
        return self.generate(query, **kwargs)
    
    def _get_mock_json(self, prompt: str) -> Dict:
        """Return mock JSON based on prompt."""
        prompt_lower = prompt.lower()
        
        if "symptom" in prompt_lower or "extract" in prompt_lower:
            return {
                "symptoms": ["cough", "weight loss", "fatigue"],
                "duration": "3 weeks",
                "severity": "moderate"
            }
        
        if "assessment" in prompt_lower or "risk" in prompt_lower:
            return {
                "risk_level": "high",
                "urgency": "urgent_2_week",
                "recommended_action": "Refer using suspected cancer pathway",
                "relevant_guidelines": ["1.1.1", "1.1.2"],
                "reasoning": "Patient presents with symptoms requiring urgent evaluation"
            }
        
        return {"status": "ok", "message": "Mock response"}
    
    def _get_mock_text(self, prompt: str) -> str:
        """Return mock text based on prompt."""
        return """Based on the clinical guidelines, this patient should be considered for 
urgent referral via the suspected cancer pathway. The combination of symptoms 
(persistent cough, unexplained weight loss) in a patient over 40 who has smoked 
meets the criteria for urgent chest X-ray within 2 weeks.

[NG12 Section 1.1.1, p.10]"""
    
    def count_tokens(self, text: str) -> int:
        """Count tokens (approximate)."""
        return len(text) // 4


def get_llm(mock: bool = False) -> Union[GeminiLLM, MockLLM]:
    """
    Factory function to get LLM instance.
    
    Args:
        mock: If True, return mock LLM for testing
        
    Returns:
        LLM instance
    """
    if mock:
        return MockLLM()
    return GeminiLLM()
