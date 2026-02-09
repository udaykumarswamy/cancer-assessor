"""
Prompt management system for dynamic prompt loading from prompts.md.

This module provides utilities to load and manage all prompts used throughout
the clinical assessment system from a centralized markdown file.
"""

import re
from pathlib import Path
from typing import Dict, Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)




CHAT_GUIDELINE_SYSTEM_PROMPT = """
You are a clinical guidelines assistant specialising in NICE NG12
(Suspected cancer: recognition and referral).

Your role is to answer healthcare professionals' questions about the NG12 guidelines
using ONLY the retrieved guideline passages provided to you.

RULES:
1. Answer ONLY from the provided guideline context. If the context does not contain
   the answer, say so clearly — do NOT fabricate information.
2. ALWAYS cite specific NG12 sections and page numbers using [NG12 Section X.X, p.XX].
3. Use clear, professional clinical language.
4. When multiple sections are relevant, reference all of them.
5. If the question is ambiguous, address the most likely interpretation.
6. Be precise about referral urgency pathways.
7. Do NOT ask for patient details.
8. Redirect patient assessments to the Assessment tab.
9. Keep answers concise and clinically relevant.
10. Use conversation history when relevant.
"""

CHAT_GUIDELINE_PROMPT = """Answer the following question about NICE NG12 guidelines.

## Question:
{question}

## Conversation History:
{history}

## Retrieved NG12 Guideline Passages:
{context}

## Instructions:
- Answer using ONLY the guideline passages above.
- Cite specific sections and page numbers inline: [NG12 Section X.X, p.XX].
- If the passages do not contain enough information, say so clearly.
- Be precise about urgency levels and referral pathways.
- Keep the answer focused and clinically relevant.
- Output MUST be plain text only.
- Do NOT return JSON, lists, or structured objects.

# Your answer: """



class PromptLoader:
    """Load and manage prompts from prompts.md file."""
    
    def __init__(self, prompts_file: Optional[Path] = None):
        """
        Initialize the prompt loader.
        
        Args:
            prompts_file: Path to prompts.md file. 
                         If None, uses default location relative to this file.
        """
        if prompts_file is None:
            # Default to prompts.md in project root
            this_file = Path(__file__).resolve()
            prompts_file = this_file.parent.parent.parent / "prompts.md"
        
        self.prompts_file = Path(prompts_file)
        self._prompts: Dict[str, str] = {}
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """Parse prompts.md and extract all prompts."""
        if not self.prompts_file.exists():
            logger.warning(f"Prompts file not found: {self.prompts_file}")
            return
        
        with open(self.prompts_file, 'r') as f:
            content = f.read()
        
        # Pattern to match ### PROMPT_KEY sections
        # Captures everything until the next ### or end of file
        pattern = r"### (\w+)\n((?:(?!### ).*\n?)*)"
        
        matches = re.finditer(pattern, content, re.MULTILINE)
        
        for match in matches:
            key = match.group(1)
            text = match.group(2).strip()
            
            # Remove trailing markdown metadata if present
            # (like "Response schema:" sections that are not part of prompt)
            if "\nResponse schema:" in text or "\n{" in text.split('\n')[-2:]:
                # Keep everything as-is, markdown sections provide context
                pass
            
            self._prompts[key] = text
        
        logger.info(f"Loaded {len(self._prompts)} prompts from {self.prompts_file}")
    
    def get(self, prompt_key: str, default: Optional[str] = None) -> str:
        """
        Get a prompt by key.
        
        Args:
            prompt_key: Key of the prompt (e.g., 'REACT_SYSTEM_PROMPT')
            default: Default value if prompt not found
        
        Returns:
            The prompt text
        
        Raises:
            KeyError: If prompt not found and no default provided
        """
        if prompt_key not in self._prompts:
            if default is not None:
                logger.warning(f"Prompt '{prompt_key}' not found, using default")
                return default
            raise KeyError(f"Prompt '{prompt_key}' not found in {self.prompts_file}")
        
        return self._prompts[prompt_key]
    
    def get_all(self) -> Dict[str, str]:
        """Get all loaded prompts."""
        return self._prompts.copy()
    
    def list_prompts(self) -> list:
        """List all available prompt keys."""
        return list(self._prompts.keys())
    
    def reload(self) -> None:
        """Reload prompts from file (useful for development)."""
        self._prompts.clear()
        self._load_prompts()


# Global loader instance
_loader: Optional[PromptLoader] = None


def get_prompt_loader(prompts_file: Optional[Path] = None) -> PromptLoader:
    """
    Get or create the global prompt loader instance.
    
    Args:
        prompts_file: Path to prompts.md (only used on first call)
    
    Returns:
        PromptLoader instance
    """
    global _loader
    if _loader is None:
        _loader = PromptLoader(prompts_file)
    return _loader


def get_prompt(prompt_key: str, **kwargs) -> str:
    """
    Convenience function to get a prompt with placeholder substitution.
    
    Args:
        prompt_key: Key of the prompt to retrieve
        **kwargs: Keyword arguments for .format() on the prompt template
    
    Returns:
        The prompt text with placeholders filled in
    
    Example:
        get_prompt('REACT_PROMPT', task='Assess patient', 
                  tools_description='...', previous_steps='...')
    """
    loader = get_prompt_loader()
    prompt = loader.get(prompt_key)
    
    if kwargs:
        try:
            return prompt.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing required parameter for prompt '{prompt_key}': {e}")
            raise
    
    return prompt


def list_available_prompts() -> list:
    """List all available prompt keys."""
    loader = get_prompt_loader()
    return loader.list_prompts()


def get_chat_guideline_prompt(question: str, context: str, history: str = "") -> str:
    """Format the guideline Q&A prompt with retrieved context."""
    return CHAT_GUIDELINE_PROMPT.format(
        question=question,
        context=context,
        history=history if history else "No previous conversation.",
    )

# def get_chat_guideline_system_prompt() -> str:
#     """Get the system prompt for the guideline Q&A agent."""
#     return get_prompt('CHAT_GUIDELINE_SYSTEM_PROMPT')

# Convenience functions for each prompt type
def get_react_system_prompt() -> str:
    """Get the React agent system prompt."""
    return get_prompt('REACT_SYSTEM_PROMPT')


def get_react_prompt(task: str, tools_description: str, previous_steps: str) -> str:
    """Get the React agent continuation prompt with values filled in."""
    return get_prompt('REACT_PROMPT', task=task, 
                     tools_description=tools_description, 
                     previous_steps=previous_steps)


def get_assessment_system_instruction() -> str:
    """Get the assessment service system instruction."""
    return get_prompt('ASSESSMENT_SYSTEM_INSTRUCTION')


def get_assessment_extraction_prompt(text: str) -> str:
    """Get the assessment extraction prompt with text filled in."""
    return get_prompt('ASSESSMENT_EXTRACTION_PROMPT', text=text)


def get_assessment_prompt(patient_info: str, guidelines_context: str) -> str:
    """Get the full assessment prompt with patient and guideline info."""
    return get_prompt('ASSESSMENT_PROMPT', patient_info=patient_info, 
                     guidelines_context=guidelines_context)


def get_clinical_assessment_extraction_prompt(text: str) -> str:
    """Get the clinical assessment extraction prompt."""
    return get_prompt('CLINICAL_ASSESSMENT_EXTRACTION_PROMPT', text=text)


def get_clinical_patient_extraction_prompt(
    conversation_history: str,
    age: str,
    sex: str,
    symptoms: str,
    symptom_duration: str,
    risk_factors: str
) -> str:
    """Get the clinical patient information extraction prompt."""
    return get_prompt('CLINICAL_PATIENT_EXTRACTION_PROMPT',
                     conversation_history=conversation_history,
                     age=age,
                     sex=sex,
                     symptoms=symptoms,
                     symptom_duration=symptom_duration,
                     risk_factors=risk_factors)


def get_tools_symptom_extraction_prompt(text: str) -> str:
    """Get the tools symptom extraction prompt."""
    return get_prompt('TOOLS_SYMPTOM_EXTRACTION_PROMPT', text=text)


if __name__ == "__main__":
    # Example usage
    loader = get_prompt_loader()
    print("Available prompts:")
    for prompt_key in loader.list_prompts():
        print(f"  - {prompt_key}")
