#!/usr/bin/env python3
"""
NG12 PDF Download Script

Downloads the official NICE NG12 Cancer Guidelines PDF.

Usage:
    python scripts/download_ng12.py
    
    # Or with custom path
    python scripts/download_ng12.py --output /custom/path/ng12.pdf

Interview Discussion Points:
---------------------------
1. Why download at build time vs. runtime?
   - PDF is static (versioned guidelines)
   - Avoids network dependency during inference
   - Can verify integrity (checksum) at build
   - Docker layer caching benefits

2. Error handling strategy:
   - Retry with exponential backoff
   - Verify file integrity after download
   - Graceful failure with clear messages
"""

import argparse
import hashlib
import sys
from pathlib import Path
import logging
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config.settings import settings
from src.config.logging_config import get_logger


logger = get_logger('download_ng12',level=logging.DEBUG)

# Known checksum for integrity verification (update if PDF version changes)
# This helps catch corrupted downloads or if NICE updates the document
EXPECTED_SHA256 = "3260089fbca1f86f55285c3d438de08be0fa85b7eb2de62b07155ede03bc9b90"  # Set after first successful download


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def download_file(url: str, output_path: Path) -> None:
    """
    Download a file with retry logic.
    
    Uses streaming to handle large files efficiently.
    
    Args:
        url: Source URL
        output_path: Destination file path
        
    Raises:
        httpx.HTTPError: If download fails after retries
    """
    logger.info(f"Starting download of NG12 PDF from {url} to {output_path}")
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Stream download for memory efficiency
    '''
    Memory-efficient download, so that we reduce memory footprint while downloading large files.
    '''
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        
        # Get total size for progress indication
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                
                # Simple progress indicator
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    logger.info(f"Download progress: {progress:.1f}%")
    


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_pdf(file_path: Path) -> bool:
    """
    Verify the downloaded PDF is valid.
    
    Checks:
    1. File exists and has content
    2. Starts with PDF magic bytes
    3. Optional: SHA256 checksum (if known)
    
    Returns:
        True if PDF appears valid
    """
    if not file_path.exists():
        logger.error("File does not exist")
        return False
    
    file_size = file_path.stat().st_size
    if file_size < 1000:  # PDF should be much larger
        logger.error(f"File size too small to be a valid PDF, file size :{file_size} bytes")
        return False
    
    # Check PDF magic bytes
    '''
    to handle invalid PDF files, we check for the magic bytes at the start of the file.
    PDF files start with "%PDF-" signature. (bytes: 25 50 44 46 2D)
    '''
    with open(file_path, "rb") as f:
        magic = f.read(5)
        if magic != b"%PDF-":
            logger.error(f"Invalid PDF header :{magic}")
            return False
    
    # Verify checksum if known
    if EXPECTED_SHA256:
        actual_sha256 = calculate_sha256(file_path)
        if actual_sha256 != EXPECTED_SHA256:
            logger.warning("PDF checksum does not match expected value.")
            logger.warning(f"   Expected: {EXPECTED_SHA256}")
            logger.warning(f"   Actual:   {actual_sha256}")
            logger.warning("   (PDF may have been updated by NICE)")
            # Don't fail - just warn, as NICE may have updated
    
    logger.info(f"PDF verification successful, file size: {file_size}, bytes")
    return True


def main():
    logger.info("NG12 PDF Download Script Started")
    parser = argparse.ArgumentParser(
        description="Download the NICE NG12 Cancer Guidelines PDF"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=settings.pdf_path,
        help="Output file path"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if file exists"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=settings.NG12_PDF_URL,
        help="Override PDF URL"
    )
    
    args = parser.parse_args()
    
    # Check if already downloaded
    if args.output.exists() and not args.force:
        logger.info(f"PDF already exists at :{args.output}, skipping download")
        if verify_pdf(args.output):
            logge.info("   Use --force to re-download")
            logger.info("if you want to force download:  Use --force to re-download")
            return 0
        else:
            logger.info("Existing file invalid, re-downloading...")
    
    try:
        download_file(args.url, args.output)
        
        if verify_pdf(args.output):
            # Print checksum for future reference
            sha256 = calculate_sha256(args.output)
            logger.info(f"SHA256 checksum: {sha256}")
            logger.info("Download and verification completed successfully")
            return 0
        else:
            logger.error("Downloaded file verification failed")
            return 1
            
    except httpx.HTTPError as e:
        logger.error(f"Download failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
