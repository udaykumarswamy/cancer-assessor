import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import logging
import sys

# Import marker components
from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.models import create_model_dict

# PyMuPDF for lightweight parsing fallback
import fitz  

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config.logging_config import get_logger


logger = get_logger('marker-pdf-parser', level="DEBUG")

@dataclass
class ParsedPage:
    """
    Represents extracted content from a single PDF page.
    
    Attributes:
        page_number: 1-indexed page number (matches PDF viewer)
        markdown: Extracted content in Markdown format
        sections: Detected section headers on this page
        has_tables: Whether page contains tabular content
        content_type: Classification (toc, frontmatter, content, appendix)
    """
    page_number: int
    markdown: str
    sections: list[str] = field(default_factory=list)
    has_tables: bool = False
    content_type: str = "content"  # toc, frontmatter, content, appendix


@dataclass 
class ParsedDocument:
    """
    Complete extracted document with all pages and metadata.
    
    Attributes:
        source: Original file path
        title: Document title
        total_pages: Number of pages in document
        pages: List of parsed page content
        full_markdown: Complete document as markdown
        toc: Table of contents if detected
    """
    source: str
    title: str
    total_pages: int
    pages: list[ParsedPage]
    full_markdown: str
    toc: Optional[dict] = None
    
    def get_content_pages(self) -> list[ParsedPage]:
        """Get pages excluding TOC and front matter."""
        return [p for p in self.pages if p.content_type == "content"]
    
    def get_sections(self) -> list[tuple[str, int]]:
        """Get all section headers with their page numbers."""
        sections = []
        for page in self.pages:
            for section in page.sections:
                sections.append((section, page.page_number))
        return sections


class MarkerPDFParser:
    """
    PDF Parser using Marker for high-quality extraction.
    
    Marker uses deep learning models to understand document layout,
    making it ideal for complex clinical documents like NICE guidelines.
    
    Usage:
        parser = MarkerPDFParser()
        doc = parser.parse("/path/to/ng12.pdf")
        
        # Access full markdown
        print(doc.full_markdown)
        
        # Access by page
        for page in doc.get_content_pages():
            print(f"Page {page.page_number}: {len(page.markdown)} chars")
    
    Configuration:
        The parser can be configured via environment variables:
        - MARKER_BATCH_SIZE: Batch size for processing (default: 4)
        - MARKER_MAX_PAGES: Max pages to process (default: None = all)
    """
    
    # Regex patterns for structure detection
    SECTION_HEADER_PATTERN = re.compile(
        r'^(#{1,3})\s+(\d+(?:\.\d+)*)\s+(.+)$',  # Markdown headers with section numbers
        re.MULTILINE
    )
    
    # Also match non-markdown section patterns (from raw text)
    RAW_SECTION_PATTERN = re.compile(
        r'^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s]+)',  # "1.2 Section Title"
        re.MULTILINE
    )
    
    RECOMMENDATION_PATTERN = re.compile(
        r'\b(Offer|Consider|Refer urgently?|Do not offer|Discuss|Arrange)\b',
        re.IGNORECASE
    )
    
    TABLE_PATTERN = re.compile(
        r'\|.+\|',  # Markdown table row
        re.MULTILINE
    )
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the parser.
        
        Args:
            use_gpu: Whether to use GPU acceleration (faster but requires CUDA)
        """
        self.use_gpu = use_gpu
        self._models = None
    
    def _load_models(self):
        """Lazy load Marker models."""
        if self._models is None:
            logger.debug("Loading Marker models...(this may take a moment)")
            
            self._models = create_model_dict()
        return self._models
    
    def parse(self, pdf_path: Path | str) -> ParsedDocument:
        """
        Parse a PDF file using Marker.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedDocument with extracted content and metadata
            
        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If PDF cannot be parsed
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Parsing PDF: {pdf_path.name}")
        
      
        # Load models, its one time will cache for future calls in our local instance
        models = self._load_models()
        
        # Configure converter
        config_parser = ConfigParser({})
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=models,
            processor_list=None,
            renderer=None
        )
        
        # Convert PDF to markdown
        logger.debug("Extracting content from PDF using Marker...")
        rendered = converter(str(pdf_path))
        
        full_markdown = rendered.markdown
        metadata = rendered.metadata
        
        # Parse the markdown into pages
        pages = self._split_into_pages(full_markdown, metadata)
        
        # Extract document title
        title = self._extract_title(full_markdown, metadata)
        logger.debug(f" Extracted {len(pages)} pages, {len(full_markdown):,} characters")
        
        return ParsedDocument(
            source=str(pdf_path),
            title=title,
            total_pages=len(pages),
            pages=pages,
            full_markdown=full_markdown,
            toc=metadata.get("toc") if metadata else None
        )
    
    def parse_lightweight(self, pdf_path: Path | str) -> ParsedDocument:
        """
        Parse PDF without Marker models - uses PyMuPDF directly.
        
        Fallback for environments without full Marker dependencies.
        Faster but less accurate for complex layouts.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ParsedDocument with extracted content
        """
       
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.debug(f"Using PyMuPDF for lightweight parsing...: {pdf_path.name}")
        
        doc = fitz.open(str(pdf_path))
        pages = []
        all_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            # Convert to basic markdown
            markdown = self._text_to_markdown(text, page_num + 1)
            
            # Detect sections
            sections = self._extract_sections(markdown)
            
            # Detect tables (basic heuristic)
            has_tables = bool(re.search(r'\t.*\t', text)) or "│" in text
            
            # Classify page type
            content_type = self._classify_page(markdown, page_num + 1, len(doc))
            
            pages.append(ParsedPage(
                page_number=page_num + 1,
                markdown=markdown,
                sections=sections,
                has_tables=has_tables,
                content_type=content_type
            ))
            
            all_text.append(f"<!-- Page {page_num + 1} -->\n{markdown}")
        
        doc.close()
        
        # Extract title from first page
        title = self._extract_title_from_text(all_text[0] if all_text else "")
        
        logger.debug(f" Extracted {len(pages)} pages, {len(all_text):,} characters")
        
        return ParsedDocument(
            source=str(pdf_path),
            title=title,
            total_pages=len(pages),
            pages=pages,
            full_markdown="\n\n---\n\n".join(all_text)
        )
    
    def _split_into_pages(
        self, 
        markdown: str, 
        metadata: Optional[dict]
    ) -> list[ParsedPage]:
        """
        Split full markdown into page-level chunks.
        
        Marker may include page markers in the output, which we use
        to reconstruct page boundaries.
        """
        # Look for page markers in the markdown
        # Marker sometimes adds these: <!-- Page X --> or similar
        page_pattern = re.compile(r'(?:<!-+\s*Page\s*(\d+)\s*-+>|^---$)', re.MULTILINE)
        
        matches = list(page_pattern.finditer(markdown))
        
        if len(matches) > 1:
            # Use detected page boundaries
            pages = []
            for i, match in enumerate(matches):
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
                page_text = markdown[start:end].strip()
                
                page_num = int(match.group(1)) if match.group(1) else i + 1
                pages.append(self._create_page(page_text, page_num))
            return pages
        
        # Fallback: Estimate pages by content length
        # NG12 PDF has ~50 pages, average ~2500 chars per page in markdown
        estimated_pages = max(1, len(markdown) // 2500)
        chunk_size = len(markdown) // estimated_pages
        
        pages = []
        pos = 0
        page_num = 1
        
        while pos < len(markdown):
            end = min(pos + chunk_size, len(markdown))
            
            # Try to break at paragraph boundary
            if end < len(markdown):
                next_para = markdown.find('\n\n', end - 200, end + 200)
                if next_para != -1:
                    end = next_para
            
            page_text = markdown[pos:end].strip()
            if page_text:
                pages.append(self._create_page(page_text, page_num))
                page_num += 1
            
            pos = end
        
        return pages
    
    def _create_page(self, text: str, page_num: int) -> ParsedPage:
        """Create a ParsedPage with detected metadata."""
        sections = self._extract_sections(text)
        has_tables = bool(self.TABLE_PATTERN.search(text))
        content_type = self._classify_page(text, page_num, 50)  
        
        return ParsedPage(
            page_number=page_num,
            markdown=text.strip(),
            sections=sections,
            has_tables=has_tables,
            content_type=content_type
        )
    
    def _extract_sections(self, text: str) -> list[str]:
        """Extract section headers from markdown text."""
        sections = []
        
        # Try markdown headers first
        for match in self.SECTION_HEADER_PATTERN.finditer(text):
            level, number, title = match.groups()
            sections.append(f"{number} {title.strip()}")
        
        # also try raw section patterns
        if not sections:
            for match in self.RAW_SECTION_PATTERN.finditer(text):
                number, title = match.groups()
                sections.append(f"{number} {title.strip()}")
        
        return sections
    
    def _classify_page(self, text: str, page_num: int, total_pages: int) -> str:
        """Classify page type based on content."""
        text_lower = text.lower()
        
        if page_num <= 2:
            return "frontmatter"
        
        if "contents" in text_lower[:500] and page_num < 10:
            if text_lower.count('...') > 3 or text_lower.count('. . .') > 3:
                return "toc"
        
        if "appendix" in text_lower[:200]:
            return "appendix"
        
        if page_num > total_pages - 5 and "reference" in text_lower[:200]:
            return "references"
        
        return "content"
    
    def _extract_title(self, markdown: str, metadata: Optional[dict]) -> str:
        """Extract document title."""
        # Try metadata first
        if metadata and "title" in metadata:
            return metadata["title"]
        
        # Look for H1 in markdown
        h1_match = re.search(r'^#\s+(.+)$', markdown, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()
        
        # Look for the NG12 title pattern
        ng12_match = re.search(r'Suspected cancer.*recognition.*referral', markdown[:2000], re.IGNORECASE)
        if ng12_match:
            return "NG12 Suspected Cancer: Recognition and Referral"
        
        return "NG12 Suspected Cancer Guidelines"
    
    def _extract_title_from_text(self, text: str) -> str:
        """Extract title from plain text (lightweight mode)."""
        lines = text.strip().split('\n')
        for line in lines[:15]:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                if not line.startswith(('Page', 'ISBN', 'Copyright', 'www.', '©')):
                    if 'cancer' in line.lower() or 'NG12' in line:
                        return line
        return "NG12 Suspected Cancer Guidelines"
    
    def _text_to_markdown(self, text: str, page_num: int) -> str:
        """
        Convert plain text to basic markdown.
        
        Used in lightweight mode when full Marker isn't available.
        """
        lines = text.split('\n')
        markdown_lines = []
        in_list = False
        
        for line in lines:
            line = line.rstrip()
            
            if not line.strip():
                if in_list:
                    in_list = False
                markdown_lines.append('')
                continue
            
            stripped = line.strip()
            
            # Detect potential headers (numbered sections)
            section_match = re.match(r'^(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z\s,]+)$', stripped)
            if section_match:
                number, title = section_match.groups()
                level = min(number.count('.') + 2, 4)  # ## for 1.1, ### for 1.1.1
                markdown_lines.append(f"{'#' * level} {number} {title}")
                continue
            
            # Detect bullet points
            if stripped.startswith(('•', '-', '●', '○', '▪')):
                markdown_lines.append(f"- {stripped[1:].strip()}")
                in_list = True
                continue
            
            # Detect recommendations (important for clinical docs)
            if self.RECOMMENDATION_PATTERN.match(stripped):
                markdown_lines.append(f"> **{stripped}**")
                continue
            
            # Regular paragraph
            markdown_lines.append(stripped)
        
        return '\n'.join(markdown_lines)


# Convenience function for simple usage
def parse_pdf(pdf_path: Path | str, lightweight: bool = False) -> ParsedDocument:
    """
    Parse a PDF file and return structured content.
    
    Args:
        pdf_path: Path to the PDF file
        lightweight: If True, use simpler extraction (no deep learning models)
        
    Returns:
        ParsedDocument with extracted content
        

    """
    parser = MarkerPDFParser()
    if lightweight:
        return parser.parse_lightweight(pdf_path)
    return parser.parse(pdf_path)
