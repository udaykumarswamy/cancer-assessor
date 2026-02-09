"""
Semantic Chunker for Clinical Guidelines (Gemini Compatible)

This module implements intelligent text chunking optimized for clinical
documents like NICE NG12 guidelines with rich metadata extraction.

Key Features:
- Gemini-compatible token counting (~4 chars/token)
- Consistent chunk sizes with proper boundary handling
- Rich clinical metadata extraction (cancer types, symptoms, urgency, etc.)
- Section hierarchy for accurate citations
- Content type classification

Why this architecture?:
---------------------------
1. Why semantic chunking over fixed-size?
   - Clinical recommendations must not be split mid-sentence
   - "Refer urgently if X" must stay with "AND patient is over 40"
   - Tables contain related criteria that should stay together
   - Section headers provide context for retrieval

2. Token counting for Gemini:
   - Gemini uses different tokenizer than GPT/Claude
   - Approximation: ~4 characters per token for English text
   - More accurate than tiktoken for Gemini models
   - Can use google.generativeai.count_tokens() for exact count

3. Chunking parameters trade-offs:
   - Larger chunks = more context but lower retrieval precision
   - Smaller chunks = precise retrieval but may lose context
   - 512 tokens is sweet spot for clinical recommendations
   - 100 token overlap catches criteria spanning boundaries

4. Rich metadata benefits:
   - Filter by cancer type, urgency, age threshold
   - Faster retrieval with pre-computed filters
   - Better citation generation with section hierarchy
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum
from datetime import datetime

from src.ingestion.pdf_parser import ParsedDocument, ParsedPage


class UrgencyLevel(Enum):
    """Clinical urgency levels from NG12."""
    URGENT_2_WEEK = "urgent_2_week"      # Suspected cancer pathway
    URGENT = "urgent"                      # Urgent but not 2-week
    ROUTINE = "routine"                    # Routine referral
    CONSIDER = "consider"                  # Consider referral
    MONITOR = "monitor"                    # Monitor/watchful waiting
    UNKNOWN = "unknown"


class ContentType(Enum):
    """Types of clinical content."""
    RECOMMENDATION = "recommendation"
    CRITERIA = "criteria"
    TABLE = "table"
    BACKGROUND = "background"
    DEFINITION = "definition"
    PATHWAY = "pathway"
    CONTENT = "content"


@dataclass
class ClinicalMetadata:
    """
    Rich clinical metadata extracted from chunk content.
    
    Enables powerful filtering during retrieval:
    - Find all urgent lung cancer recommendations
    - Get criteria for patients over 40
    - Filter by specific symptoms
    """
    # Cancer types mentioned
    cancer_types: List[str] = field(default_factory=list)
    
    # Symptoms mentioned
    symptoms: List[str] = field(default_factory=list)
    
    # Age thresholds (e.g., "40+", "<18")
    age_thresholds: List[str] = field(default_factory=list)
    
    # Timeframes mentioned (e.g., "2_week", "3_month")
    timeframes: List[str] = field(default_factory=list)
    
    # Urgency level
    urgency: UrgencyLevel = UrgencyLevel.UNKNOWN
    
    # Clinical actions (refer, offer, consider, etc.)
    actions: List[str] = field(default_factory=list)
    
    # Risk factors mentioned
    risk_factors: List[str] = field(default_factory=list)
    
    # Investigations/tests mentioned
    investigations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "cancer_types": ",".join(self.cancer_types) if self.cancer_types else "",
            "symptoms": ",".join(self.symptoms) if self.symptoms else "",
            "age_thresholds": ",".join(self.age_thresholds) if self.age_thresholds else "",
            "timeframes": ",".join(self.timeframes) if self.timeframes else "",
            "urgency": self.urgency.value,
            "actions": ",".join(self.actions) if self.actions else "",
            "risk_factors": ",".join(self.risk_factors) if self.risk_factors else "",
            "investigations": ",".join(self.investigations) if self.investigations else "",
        }


@dataclass
class Chunk:
    """
    A chunk of text with rich metadata for retrieval and citation.
    
    Attributes:
        chunk_id: Unique identifier (hash-based for deduplication)
        text: The chunk content
        page_start: First page this chunk appears on
        page_end: Last page this chunk appears on (may span pages)
        section: Section header this chunk belongs to
        section_hierarchy: Full path for citations
        content_type: Type of content (recommendation, table, etc.)
        token_count: Approximate token count (Gemini compatible)
        char_count: Character count
        clinical_metadata: Extracted clinical information
        chunk_index: Position in document
        prev_chunk_id: Previous chunk for context
        next_chunk_id: Next chunk for context
        semantic_density: How information-dense (0-1)
        created_at: Timestamp
    """
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    section: Optional[str] = None
    section_hierarchy: List[str] = field(default_factory=list)
    content_type: ContentType = ContentType.CONTENT
    token_count: int = 0
    char_count: int = 0
    clinical_metadata: ClinicalMetadata = field(default_factory=ClinicalMetadata)
    chunk_index: int = 0
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    semantic_density: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ChromaDB storage."""
        base = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section or "",
            "section_hierarchy": "|".join(self.section_hierarchy),
            "content_type": self.content_type.value,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "chunk_index": self.chunk_index,
            "prev_chunk_id": self.prev_chunk_id or "",
            "next_chunk_id": self.next_chunk_id or "",
            "semantic_density": self.semantic_density,
            "created_at": self.created_at,
            "source": "NG12",
            # Boolean flags for quick filtering
            "has_recommendation": self.content_type == ContentType.RECOMMENDATION,
            "has_table": self.content_type == ContentType.TABLE,
            "is_urgent": self.clinical_metadata.urgency in [UrgencyLevel.URGENT, UrgencyLevel.URGENT_2_WEEK],
        }
        # Add flattened clinical metadata
        base.update(self.clinical_metadata.to_dict())
        return base
    
    def get_citation(self) -> str:
        """Generate citation string for this chunk."""
        if self.section:
            return f"[NG12 Section {self.section}, p.{self.page_start}]"
        return f"[NG12 p.{self.page_start}]"


class ClinicalMetadataExtractor:
    """
    Extracts rich clinical metadata from text.
    
    Uses regex patterns to identify:
    - Cancer types
    - Symptoms
    - Age thresholds
    - Urgency levels
    - Clinical actions
    - Investigations
    """
    
    # Cancer type patterns (lowercase for matching)
    CANCER_TYPES = [
        "lung cancer", "breast cancer", "colorectal cancer", "prostate cancer",
        "skin cancer", "melanoma", "pancreatic cancer", "ovarian cancer",
        "brain tumour", "brain tumor", "leukaemia", "leukemia", "lymphoma",
        "bladder cancer", "kidney cancer", "liver cancer", "oesophageal cancer",
        "esophageal cancer", "stomach cancer", "gastric cancer", "thyroid cancer",
        "head and neck cancer", "cervical cancer", "endometrial cancer",
        "testicular cancer", "bone cancer", "sarcoma", "myeloma", "mesothelioma",
        "laryngeal cancer", "upper gi cancer", "lower gi cancer"
    ]
    
    # Common symptoms (lowercase)
    SYMPTOMS = [
        "haemoptysis", "hemoptysis", "cough", "weight loss", "fatigue",
        "breathlessness", "shortness of breath", "chest pain", "hoarseness",
        "dysphagia", "difficulty swallowing", "abdominal pain", "bloating",
        "rectal bleeding", "blood in stool", "blood in urine", "haematuria",
        "breast lump", "nipple discharge", "skin lesion", "mole", "lump",
        "headache", "seizure", "night sweats", "fever", "jaundice",
        "lymphadenopathy", "swollen lymph nodes", "bone pain", "back pain",
        "unexplained bleeding", "persistent", "unexplained"
    ]
    
    # Investigations/tests
    INVESTIGATIONS = [
        "chest x-ray", "chest x ray", "cxr", "ct scan", "ct", "mri", 
        "ultrasound", "mammogram", "mammography", "colonoscopy", "endoscopy",
        "biopsy", "blood test", "psa test", "psa", "fbc", "full blood count",
        "lfts", "liver function", "ca125", "ca 125", "cea", "tumour markers",
        "tumor markers", "pet scan", "bone scan", "dermoscopy", "cystoscopy",
        "bronchoscopy", "gastroscopy", "sigmoidoscopy", "urine test",
        "faecal immunochemical test", "fit test"
    ]
    
    # Risk factors
    RISK_FACTORS = [
        "smoking", "smoker", "tobacco", "family history", "previous cancer",
        "obesity", "overweight", "alcohol", "exposure", "radiation",
        "genetic", "hereditary", "hpv", "hepatitis", "ulcerative colitis",
        "crohn", "barrett", "h pylori", "asbestos", "occupational"
    ]
    
    # Age patterns
    AGE_PATTERN = re.compile(
        r'(?:aged?\s*)?(\d+)\s*(?:years?)?\s*(?:and\s*)?(?:or\s*)?(over|under|above|below|older|younger|\+|and over)',
        re.IGNORECASE
    )
    
    # Timeframe patterns
    TIMEFRAME_PATTERN = re.compile(
        r'(\d+)\s*[\-\s]*(week|day|month|year)s?',
        re.IGNORECASE
    )
    
    # Urgency patterns (order matters - check specific first)
    URGENCY_PATTERNS = [
        (UrgencyLevel.URGENT_2_WEEK, [
            r'suspected\s+cancer\s+pathway',
            r'2[\s\-]?week',
            r'two[\s\-]?week',
            r'within\s+2\s+weeks?',
            r'appointment\s+within\s+2\s+weeks?',
        ]),
        (UrgencyLevel.URGENT, [
            r'refer\s+urgently',
            r'urgent\s+referral',
            r'immediate\s+referral',
            r'very\s+urgent',
        ]),
        (UrgencyLevel.CONSIDER, [
            r'consider\s+referral',
            r'consider\s+referring',
            r'consider\s+an?\s+urgent',
            r'consider\s+direct\s+access',
        ]),
        (UrgencyLevel.ROUTINE, [
            r'routine\s+referral',
            r'non[\-\s]?urgent\s+referral',
        ]),
    ]
    
    # Action patterns
    ACTION_PATTERNS = [
        (r'\bOffer\b', 'offer'),
        (r'\bRefer\b', 'refer'),
        (r'\bConsider\b', 'consider'),
        (r'\bArrange\b', 'arrange'),
        (r'\bDiscuss\b', 'discuss'),
        (r'\bDo\s+not\s+offer\b', 'do_not_offer'),
        (r'\bDo\s+not\s+refer\b', 'do_not_refer'),
        (r'\bCarry\s+out\b', 'carry_out'),
        (r'\bUse\b', 'use'),
    ]
    
    def extract(self, text: str) -> ClinicalMetadata:
        """Extract all clinical metadata from text."""
        text_lower = text.lower()
        
        return ClinicalMetadata(
            cancer_types=self._extract_list(text_lower, self.CANCER_TYPES),
            symptoms=self._extract_list(text_lower, self.SYMPTOMS),
            age_thresholds=self._extract_ages(text),
            timeframes=self._extract_timeframes(text),
            urgency=self._extract_urgency(text_lower),
            actions=self._extract_actions(text),
            risk_factors=self._extract_list(text_lower, self.RISK_FACTORS),
            investigations=self._extract_list(text_lower, self.INVESTIGATIONS),
        )
    
    def _extract_list(self, text: str, items: List[str]) -> List[str]:
        """Find items from list in text."""
        found = []
        for item in items:
            if item in text:
                normalized = item.replace(" ", "_").replace("-", "_")
                if normalized not in found:
                    found.append(normalized)
        return found
    
    def _extract_ages(self, text: str) -> List[str]:
        """Find age thresholds."""
        found = []
        for match in self.AGE_PATTERN.finditer(text):
            age = match.group(1)
            direction = match.group(2).lower() if match.group(2) else "over"
            if direction in ["over", "above", "older", "+", "and over"]:
                threshold = f"{age}+"
            else:
                threshold = f"<{age}"
            if threshold not in found:
                found.append(threshold)
        return found
    
    def _extract_timeframes(self, text: str) -> List[str]:
        """Find timeframes."""
        found = []
        for match in self.TIMEFRAME_PATTERN.finditer(text):
            num = match.group(1)
            unit = match.group(2).lower()
            timeframe = f"{num}_{unit}"
            if timeframe not in found:
                found.append(timeframe)
        return found
    
    def _extract_urgency(self, text: str) -> UrgencyLevel:
        """Determine urgency level."""
        for level, patterns in self.URGENCY_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return level
        return UrgencyLevel.UNKNOWN
    
    def _extract_actions(self, text: str) -> List[str]:
        """Find clinical actions."""
        found = []
        for pattern, action in self.ACTION_PATTERNS:
            if re.search(pattern, text):
                if action not in found:
                    found.append(action)
        return found


class SemanticChunker:
    """
    Semantic chunker optimized for clinical guideline documents.
    
    Features:
    - Gemini-compatible token counting (~4 chars/token)
    - Consistent chunk sizes with proper overlap
    - Rich clinical metadata extraction
    - Preserves clinical recommendations as units
    - Keeps tables together when possible
    - Section hierarchy for citations
    
    Usage:
        chunker = SemanticChunker(chunk_size=512, chunk_overlap=100)
        chunks = chunker.chunk_document(parsed_doc)
        
        for chunk in chunks:
            print(f"{chunk.chunk_id}: {chunk.token_count} tokens")
            print(f"Urgency: {chunk.clinical_metadata.urgency.value}")
    """
    
    # Patterns for detecting semantic boundaries
    SECTION_PATTERN = re.compile(
        r'^#{1,4}\s+(\d+(?:\.\d+)*)\s+(.+)$',
        re.MULTILINE
    )
    
    RECOMMENDATION_PATTERN = re.compile(
        r'(?:^>\s*\*\*|^\*\*(?:Offer|Consider|Refer|Do not|Discuss|Arrange)|^Offer\s|^Consider\s|^Refer\s)',
        re.MULTILINE | re.IGNORECASE
    )
    
    TABLE_PATTERN = re.compile(r'^\|.*\|$', re.MULTILINE)
    
    LIST_PATTERN = re.compile(r'^[\-•\*]\s', re.MULTILINE)
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
        chars_per_token: float = 4.0,  
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Number of tokens to overlap between chunks
            min_chunk_size: Minimum chunk size (avoid tiny fragments)
            chars_per_token: Characters per token (4.0 for Gemini, 3.5 for GPT)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.chars_per_token = chars_per_token
        
        # Max characters per chunk (for consistent sizing)
        self.max_chars = int(chunk_size * chars_per_token)
        self.overlap_chars = int(chunk_overlap * chars_per_token)
        self.min_chars = int(min_chunk_size * chars_per_token)
        
        # Metadata extractor
        self.metadata_extractor = ClinicalMetadataExtractor()
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens - Gemini compatible approximation.
        
        Gemini uses ~4 characters per token for English text.
        This is more accurate for Gemini than tiktoken.
        """
        return max(1, int(len(text) / self.chars_per_token))
    
    def chunk_document(self, document: ParsedDocument) -> List[Chunk]:
        """
        Chunk an entire parsed document.
        
        Args:
            document: ParsedDocument from the PDF parser
            
        Returns:
            List of Chunk objects with rich metadata
        """
        chunks = []
        chunk_index = 0
        
        # Process content pages only (skip TOC, frontmatter)
        content_pages = document.get_content_pages()
        
        if not content_pages:
            content_pages = document.pages
        
        # Current chunk state
        current_text = ""
        current_pages = []
        current_section = None
        section_hierarchy = []
        
        for page in content_pages:
            page_num = page.page_number
            text = page.markdown
            
            # Update section if available
            if page.sections:
                current_section = page.sections[0]
                section_hierarchy = self._extract_section_hierarchy(current_section)
            
            # Split into semantic paragraphs
            paragraphs = self._split_into_paragraphs(text)
            
            for para in paragraphs:
                para_chars = len(para)
                current_chars = len(current_text)
                
                # Check if adding paragraph exceeds limit
                if current_chars + para_chars > self.max_chars and current_text.strip():
                    # Create chunk from current text
                    chunk = self._create_chunk(
                        text=current_text.strip(),
                        pages=current_pages,
                        section=current_section,
                        section_hierarchy=section_hierarchy.copy(),
                        index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_text)
                    current_text = overlap_text
                    if overlap_text:
                        current_text += "\n\n"
                    current_text += para
                    current_pages = [page_num]
                else:
                    # Add paragraph to current chunk
                    if current_text:
                        current_text += "\n\n" + para
                    else:
                        current_text = para
                    
                    if page_num not in current_pages:
                        current_pages.append(page_num)
        
        # Don't forget the last chunk
        if current_text.strip() and len(current_text) >= self.min_chars:
            chunk = self._create_chunk(
                text=current_text.strip(),
                pages=current_pages,
                section=current_section,
                section_hierarchy=section_hierarchy.copy(),
                index=chunk_index
            )
            chunks.append(chunk)
        
        # Link chunks (prev/next references)
        self._link_chunks(chunks)
        
        return chunks
    
    def chunk_text(
        self,
        text: str,
        page_number: int = 0,
        section: Optional[str] = None
    ) -> List[Chunk]:
        """
        Chunk a single text string.
        
        Useful for testing or processing individual sections.
        """
        chunks = []
        paragraphs = self._split_into_paragraphs(text)
        
        current_text = ""
        chunk_index = 0
        section_hierarchy = self._extract_section_hierarchy(section) if section else []
        
        for para in paragraphs:
            para_chars = len(para)
            current_chars = len(current_text)
            
            if current_chars + para_chars > self.max_chars and current_text.strip():
                chunk = self._create_chunk(
                    text=current_text.strip(),
                    pages=[page_number],
                    section=section,
                    section_hierarchy=section_hierarchy,
                    index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                
                overlap_text = self._get_overlap_text(current_text)
                current_text = overlap_text + "\n\n" + para if overlap_text else para
            else:
                current_text = current_text + "\n\n" + para if current_text else para
        
        if current_text.strip() and len(current_text) >= self.min_chars:
            chunk = self._create_chunk(
                text=current_text.strip(),
                pages=[page_number],
                section=section,
                section_hierarchy=section_hierarchy,
                index=chunk_index
            )
            chunks.append(chunk)
        
        self._link_chunks(chunks)
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into semantic paragraphs.
        
        Keeps related content together:
        - Tables as single units
        - Recommendations with their criteria
        - Lists with their items
        """
        # Protect tables from being split
        text = self._protect_tables(text)
        
        # Split on double newlines
        raw_paragraphs = re.split(r'\n\n+', text)
        
        paragraphs = []
        current_group = []
        
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this starts a new semantic unit
            is_new_section = bool(self.SECTION_PATTERN.match(para))
            is_recommendation = bool(self.RECOMMENDATION_PATTERN.search(para))
            
            if is_new_section and current_group:
                paragraphs.append("\n\n".join(current_group))
                current_group = [para]
            elif is_recommendation:
                if current_group:
                    paragraphs.append("\n\n".join(current_group))
                current_group = [para]
            else:
                current_group.append(para)
            
            # Check if group is getting too large (80% of max)
            group_text = "\n\n".join(current_group)
            if len(group_text) > self.max_chars * 0.8:
                paragraphs.append(group_text)
                current_group = []
        
        if current_group:
            paragraphs.append("\n\n".join(current_group))
        
        return paragraphs
    
    def _protect_tables(self, text: str) -> str:
        """Keep markdown tables as single units."""
        lines = text.split('\n')
        result = []
        table_lines = []
        in_table = False
        
        for line in lines:
            is_table_line = line.strip().startswith('|') and line.strip().endswith('|')
            
            if is_table_line:
                if not in_table:
                    if result and result[-1].strip():
                        result.append('')
                in_table = True
                table_lines.append(line)
            else:
                if in_table:
                    result.append('\n'.join(table_lines))
                    result.append('')
                    table_lines = []
                    in_table = False
                result.append(line)
        
        if table_lines:
            result.append('\n'.join(table_lines))
        
        return '\n'.join(result)
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Extract overlap text from the end of a chunk.
        
        Uses character-based approach for consistency.
        """
        if len(text) <= self.overlap_chars:
            return text
        
        overlap_text = text[-self.overlap_chars:]
        
        # Try to start at sentence boundary
        sentence_match = re.search(r'^[^.!?]*[.!?]\s+', overlap_text)
        if sentence_match:
            overlap_text = overlap_text[sentence_match.end():]
        
        return overlap_text.strip()
    
    def _extract_section_hierarchy(self, section: Optional[str]) -> List[str]:
        """Extract section hierarchy from section string."""
        if not section:
            return []
        
        match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)$', section)
        if not match:
            return [section]
        
        numbers = match.group(1)
        parts = numbers.split('.')
        
        hierarchy = []
        for i in range(len(parts)):
            prefix = '.'.join(parts[:i+1])
            hierarchy.append(prefix)
        
        return hierarchy
    
    def _generate_chunk_id(self, text: str, page: int, index: int) -> str:
        """Generate unique chunk ID using hash."""
        content = f"{text[:100]}_{page}_{index}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"ng12_p{page}_{index:04d}_{hash_val}"
    
    def _classify_content(self, text: str) -> ContentType:
        """Classify the type of content in a chunk."""
        text_lower = text.lower()
        
        # Check for recommendation language
        if self.RECOMMENDATION_PATTERN.search(text):
            return ContentType.RECOMMENDATION
        
        # Check for tables
        if self.TABLE_PATTERN.search(text):
            return ContentType.TABLE
        
        # Check for criteria/thresholds
        if any(phrase in text_lower for phrase in [
            "refer urgently", "2 week", "urgent referral",
            "if the patient", "years or over", "years and over"
        ]):
            return ContentType.CRITERIA
        
        # Check for background/context
        if any(phrase in text_lower for phrase in [
            "evidence shows", "studies have", "research indicates",
            "the following factors", "risk factors include"
        ]):
            return ContentType.BACKGROUND
        
        # Check for definitions
        if any(phrase in text_lower for phrase in [
            "is defined as", "means", "refers to"
        ]):
            return ContentType.DEFINITION
        
        return ContentType.CONTENT
    
    def _calculate_semantic_density(self, text: str) -> float:
        """
        Calculate semantic density (information richness).
        
        Higher density = more clinical keywords per character.
        """
        clinical_terms = (
            ClinicalMetadataExtractor.CANCER_TYPES +
            ClinicalMetadataExtractor.SYMPTOMS +
            ClinicalMetadataExtractor.INVESTIGATIONS
        )
        
        text_lower = text.lower()
        matches = sum(1 for term in clinical_terms if term in text_lower)
        
        # Normalize by text length (per 100 chars)
        density = matches / (len(text) / 100) if text else 0
        return min(1.0, density)
    
    def _create_chunk(
        self,
        text: str,
        pages: List[int],
        section: Optional[str],
        section_hierarchy: List[str],
        index: int
    ) -> Chunk:
        """Create a Chunk object with all metadata."""
        content_type = self._classify_content(text)
        clinical_meta = self.metadata_extractor.extract(text)
        
        return Chunk(
            chunk_id=self._generate_chunk_id(text, pages[0], index),
            text=text,
            page_start=min(pages),
            page_end=max(pages),
            section=section,
            section_hierarchy=section_hierarchy,
            content_type=content_type,
            token_count=self.count_tokens(text),
            char_count=len(text),
            clinical_metadata=clinical_meta,
            chunk_index=index,
            semantic_density=self._calculate_semantic_density(text),
        )
    
    def _link_chunks(self, chunks: List[Chunk]) -> None:
        """Link chunks with prev/next references."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.prev_chunk_id = chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                chunk.next_chunk_id = chunks[i + 1].chunk_id


def chunk_document(document: ParsedDocument, **kwargs) -> List[Chunk]:
    """
    Convenience function to chunk a document with default settings.
    
    Args:
        document: ParsedDocument from PDF parser
        **kwargs: Override default chunker settings
            - chunk_size: Target size in tokens (default: 512)
            - chunk_overlap: Overlap in tokens (default: 100)
            - min_chunk_size: Minimum size in tokens (default: 50)
            - chars_per_token: Chars per token (default: 4.0 for Gemini)
        
    Returns:
        List of Chunk objects with rich clinical metadata
    """
    chunker = SemanticChunker(**kwargs)
    return chunker.chunk_document(document)
