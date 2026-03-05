#!/usr/bin/env python3
"""
Citation Validation Tool

This script extracts citations from LaTeX files, retrieves DOI information from refs.bib,
downloads PDFs, and converts them to text for validation purposes.
"""

import re
import os
import sys
import json
import random
import requests
import time
import logging
import asyncio
import bibtexparser
import pikepdf
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from collections import defaultdict
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CitationExtractor:
    """Extract citations and their contexts from LaTeX files."""
    
    def __init__(self):
        # Common citation patterns in LaTeX
        self.citation_patterns = [
            r'\\cite\{([^}]+)\}',
            r'\\citep\{([^}]+)\}',
            r'\\citet\{([^}]+)\}',
            r'\\citeauthor\{([^}]+)\}',
            r'\\citeyear\{([^}]+)\}',
            r'\\citealp\{([^}]+)\}'
        ]
        
    def extract_citations_with_context(self, tex_content: str, context_chars: int = 200) -> List[Dict]:
        """Extract citations with surrounding context."""
        citations = []
        
        # Split content into sentences (simplified approach)
        sentences = re.split(r'(?<=[.!?])\s+', tex_content)
        
        for sentence in sentences:
            # Check each citation pattern
            for pattern in self.citation_patterns:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    citation_keys = match.group(1).split(',')
                    citation_keys = [key.strip() for key in citation_keys]
                    
                    # Find position in original text for broader context
                    start_pos = tex_content.find(sentence)
                    context_start = max(0, start_pos - context_chars)
                    context_end = min(len(tex_content), start_pos + len(sentence) + context_chars)
                    
                    for key in citation_keys:
                        citations.append({
                            'key': key,
                            'sentence': sentence.strip(),
                            'context': tex_content[context_start:context_end].strip(),
                            'citation_type': pattern.split('\\\\')[1].split('{')[0]
                        })
        
        return citations

class BibParser:
    """Parse bibliography file to extract DOI information."""
    
    def __init__(self, bib_file: str):
        self.bib_file = bib_file
        self.entries = {}
        self.parse_bib()
    
    def parse_bib(self):
        """Parse the bibliography file."""
        with open(self.bib_file, 'r', encoding='utf-8') as f:
            bib_database = bibtexparser.load(f)
        
        for entry in bib_database.entries:
            key = entry.get('ID', '')
            self.entries[key] = {
                'doi': entry.get('doi', '').strip(),
                'title': entry.get('title', '').strip(),
                'author': entry.get('author', '').strip(),
                'year': entry.get('year', '').strip(),
                'url': entry.get('url', '').strip()
            }
    
    def get_doi(self, citation_key: str) -> Optional[str]:
        """Get DOI for a citation key."""
        if citation_key in self.entries:
            return self.entries[citation_key].get('doi', None)
        return None
    
    def get_entry(self, citation_key: str) -> Optional[Dict]:
        """Get full entry for a citation key."""
        return self.entries.get(citation_key, None)

class DownloadResult(BaseModel):
    success: bool
    message: str
    file_path: Optional[str] = None
    doi: str

class PDFExtractionResult(BaseModel):
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None
    page_count: Optional[int] = None
    file_path: str

class PaperDownloader:
    def __init__(self):
        self.all_uas_url = ('https://gist.githubusercontent.com/ozturkoktay/f1073b3038cab632c16231ef73353d7c/raw'
                           '/cf847b76a142955b1410c8bcef3aabe221a63db1/user-agents.txt')
        self.retries = 5
        self.session = requests.Session()
    
    def select_random_user_agent(self) -> str:
        """Selects a random user agent from the list of user agents."""
        start_time = time.time()
        try:
            logger.debug("Fetching user agents list...")
            req = requests.get(self.all_uas_url, timeout=10)
            if req.status_code == 200:
                user_agents: List[str] = req.text.split('\n')
                selected_ua = random.choice([ua.strip() for ua in user_agents if ua.strip()])
                logger.debug(f"User agent selected in {time.time() - start_time:.2f}s")
                return selected_ua
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch user agents: {e}, using fallback")
        # Fallback user agent
        logger.debug(f"Using fallback user agent (took {time.time() - start_time:.2f}s)")
        return 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    
    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Wrapper for requests with retry logic and random user agents."""
        start_time = time.time()
        headers = kwargs.get('headers', {})
        headers['User-Agent'] = self.select_random_user_agent()
        kwargs['headers'] = headers
        
        for attempt in range(self.retries + 1):
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.retries + 1} - {method} {url}")
                response = self.session.request(method, url, timeout=10, **kwargs)
                response.raise_for_status()
                logger.debug(f"Request successful in {time.time() - start_time:.2f}s (status: {response.status_code})")
                return response
            except requests.exceptions.RequestException as e:
                if attempt < self.retries:
                    sleep_time = 1.0 * 2 ** attempt
                    logger.warning(f"Request failed (attempt {attempt + 1}): {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All {self.retries + 1} requests to {url} failed after {time.time() - start_time:.2f}s")
                    raise Exception(f'{self.retries + 1} requests to {url} failed: {str(e)}')
    
    def get_scihub_urls(self) -> List[str]:
        """Get list of available Sci-Hub URLs from multiple sources."""
        start_time = time.time()
        logger.info("Discovering Sci-Hub mirror URLs...")
        scihub_urls = []
        
        # Method 1: Try sci-hub.pub for dynamic discovery
        try:
            logger.debug("Attempting dynamic discovery from sci-hub.pub...")
            response = self._request('GET', 'https://www.sci-hub.pub/')
            soup = BeautifulSoup(response.content, 'html.parser')
            discovered_urls = [link.get('href') for link in soup.select("a")]
            discovered_urls = [s for s in discovered_urls if s and "sci-hub" in s]
            scihub_urls.extend(discovered_urls)
            logger.debug(f"Found {len(discovered_urls)} URLs from dynamic discovery")
        except Exception as e:
            logger.warning(f"Dynamic discovery failed: {e}")
            pass
        
        # Method 2: Add known working mirrors (frequently updated)
        known_mirrors = [
            'https://sci-hub.se/',
            'https://sci-hub.st/',
            'https://sci-hub.ru/',
            'https://sci-hub.ren/',
            'https://sci-hub.cc/',
            'https://sci-hub.lu/',
            'https://sci-hub.ee/',
            'https://sci-hub.pm/',
            'https://sci-hub.tf/',
            'https://sci-hub.wf/',
        ]
        scihub_urls.extend(known_mirrors)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in scihub_urls:
            if url and url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        total_urls = len(unique_urls) or 1  # Fallback count
        logger.info(f"Mirror discovery completed in {time.time() - start_time:.2f}s - found {total_urls} unique URLs")
        
        return unique_urls or ['https://sci-hub.se/']  # Fallback if all fail
    
    def get_scihub_url(self) -> str:
        """Get a single working Sci-Hub URL."""
        urls = self.get_scihub_urls()
        return random.choice(urls)
    
    def convert_to_url(self, url_string: str, base_url: str = None) -> str:
        """Convert the pdf url to a url that can be downloaded."""
        if url_string.startswith('//'):
            return f"https:{url_string}"
        elif url_string.startswith('/') and base_url:
            # Handle relative URLs by combining with base URL
            from urllib.parse import urljoin
            return urljoin(base_url, url_string)
        else:
            return url_string
    
    def download_file(self, url: str, filename: str, ctx=None) -> None:
        """Downloads the file from the url with progress bar."""
        start_time = time.time()
        logger.info(f"Starting download: {filename}")
        
        response = self._request('GET', url, stream=True)
        
        # Get file size for progress bar
        total_size = int(response.headers.get('Content-Length', 0))
        chunk_size = 1024
        downloaded_size = 0
        
        logger.info(f"File size: {total_size} bytes ({total_size / 1024 / 1024:.2f} MB)")
        
        with open(filename, 'wb') as file:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)
                            downloaded_size += len(chunk)
                            pbar.update(len(chunk))
                            
                            # Log progress every 10% for large files
                            if total_size > 10 * 1024 * 1024:  # Files > 10MB
                                progress = (downloaded_size / total_size) * 100
                                if int(progress) % 10 == 0 and int(progress) > 0:
                                    if ctx:
                                        asyncio.create_task(ctx.info(f"Download progress: {progress:.0f}%"))
            else:
                logger.warning("No content-length header, downloading without progress tracking")
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        downloaded_size += len(chunk)
        
        download_time = time.time() - start_time
        speed = (downloaded_size / 1024 / 1024) / download_time if download_time > 0 else 0
        logger.info(f"Download completed in {download_time:.2f}s ({speed:.2f} MB/s)")
        
        if ctx:
            asyncio.create_task(ctx.info(f"Downloaded {filename} ({downloaded_size / 1024 / 1024:.2f} MB) in {download_time:.2f}s"))
    
    def download_paper(self, doi: str, output_dir: str = None, ctx=None) -> DownloadResult:
        """
        Downloads a paper given its DOI.
        
        Args:
            doi: The DOI of the paper to download
            output_dir: Directory to save the paper (default: downloads folder in project root)
            ctx: FastMCP context for logging
            
        Returns:
            DownloadResult with success status, message, and file path if successful
        """
        total_start_time = time.time()
        logger.info(f"Starting paper download for DOI: {doi}")
        
        try:
            # If no output directory specified, use downloads folder in project root
            if output_dir is None:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                output_dir = os.path.join(project_root, "downloads")
            elif output_dir == "." and not os.access(".", os.W_OK):
                # If current directory is read-only, fallback to downloads folder
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                output_dir = os.path.join(project_root, "downloads")
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Using output directory: {output_dir}")
            
            # Get all available Sci-Hub URLs
            scihub_urls = self.get_scihub_urls()
            
            last_error = None
            
            # Try each Sci-Hub mirror until one works
            for i, scihub_url in enumerate(scihub_urls):
                mirror_start_time = time.time()
                try:
                    paper_url = f"{scihub_url}{doi}"
                    logger.info(f"Trying mirror {i+1}/{len(scihub_urls)}: {scihub_url}")
                    if ctx:
                        asyncio.create_task(ctx.info(f"Trying mirror {i+1}/{len(scihub_urls)}: {scihub_url}"))
                    
                    response = self._request('GET', paper_url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Check if paper was found
                    not_found_indicators = ["not found", "sorry"]
                    if any(indicator in response.text.lower() for indicator in not_found_indicators):
                        logger.warning(f"Paper not found on {scihub_url}")
                        continue
                    
                    # Try to find PDF URL
                    pdf_url = None
                    
                    # Try iframe first
                    pdf_element = soup.select("iframe#pdf")
                    if pdf_element:
                        pdf_url = pdf_element[0].get("src")
                        logger.debug(f"Found PDF URL via iframe: {pdf_url}")
                    else:
                        # Try embed tag
                        pdf_element = soup.select("embed#pdf")
                        if pdf_element:
                            pdf_url = pdf_element[0].get("src")
                            logger.debug(f"Found PDF URL via embed: {pdf_url}")
                    
                    if not pdf_url:
                        logger.warning(f"Could not extract PDF URL from {scihub_url}")
                        continue
                    
                    # Convert to full URL
                    pdf_url = self.convert_to_url(pdf_url, scihub_url)
                    logger.debug(f"Converted PDF URL: {pdf_url}")
                    
                    # Generate filename
                    safe_doi = doi.replace('/', '_').replace('\\', '_')
                    filename = f"{safe_doi}.pdf"
                    file_path = os.path.join(output_dir, filename)
                    
                    # Download the file
                    logger.info(f"Starting file download from {pdf_url}")
                    if ctx:
                        asyncio.create_task(ctx.info(f"Downloading PDF from {scihub_url}..."))
                    
                    self.download_file(pdf_url, file_path, ctx)
                    
                    total_time = time.time() - total_start_time
                    logger.info(f"Successfully downloaded paper in {total_time:.2f}s total")
                    
                    return DownloadResult(
                        success=True,
                        message=f"Successfully downloaded paper: {filename} (from {scihub_url}) in {total_time:.2f}s",
                        file_path=file_path,
                        doi=doi
                    )
                    
                except Exception as e:
                    mirror_time = time.time() - mirror_start_time
                    last_error = str(e)
                    logger.error(f"Failed to download from {scihub_url} after {mirror_time:.2f}s: {e}")
                    continue
            
            # If we get here, all mirrors failed
            total_time = time.time() - total_start_time
            error_msg = f"Failed to download paper with DOI {doi} from {len(scihub_urls)} mirrors after {total_time:.2f}s. Last error: {last_error}"
            logger.error(error_msg)
            
            return DownloadResult(
                success=False,
                message=error_msg,
                doi=doi
            )
                
        except Exception as e:
            total_time = time.time() - total_start_time
            error_msg = f"Error downloading paper with DOI {doi} after {total_time:.2f}s: {str(e)}"
            logger.error(error_msg)
            
            return DownloadResult(
                success=False,
                message=error_msg,
                doi=doi
            )

class PDFExtractor:
    def __init__(self):
        pass
    
    def _extract_text_from_page(self, page) -> str:
        """Extract text from a single PDF page using pikepdf."""
        try:
            extracted_text = ""
            
            # Check if page has content
            if '/Contents' not in page:
                return ""
            
            # Get content streams
            content_streams = page.Contents
            if not isinstance(content_streams, list):
                content_streams = [content_streams]
            
            for stream in content_streams:
                try:
                    # Read the content stream
                    if hasattr(stream, 'read_bytes'):
                        content = stream.read_bytes()
                        
                        # Decode content - try different encodings
                        try:
                            text_content = content.decode('utf-8', errors='ignore')
                        except:
                            try:
                                text_content = content.decode('latin-1', errors='ignore')
                            except:
                                text_content = content.decode('ascii', errors='ignore')
                        
                        # Extract text using regex patterns for PDF text operators
                        # Look for text showing operators: Tj, TJ, ', "
                        patterns = [
                            r'\(([^)]*)\)\s*Tj',  # (text) Tj
                            r'\(([^)]*)\)\s*\'',   # (text) '
                            r'\(([^)]*)\)\s*"',    # (text) "
                            r'\[([^\]]*)\]\s*TJ', # [text] TJ
                        ]
                        
                        for pattern in patterns:
                            matches = re.findall(pattern, text_content)
                            for match in matches:
                                # Clean up extracted text
                                if isinstance(match, str):
                                    clean_text = match.replace('\\(', '(').replace('\\)', ')')
                                    clean_text = clean_text.replace('\\\\', '\\')
                                    clean_text = clean_text.replace('\\n', '\n')
                                    clean_text = clean_text.replace('\\r', '')
                                    clean_text = clean_text.replace('\\t', ' ')
                                    
                                    if clean_text.strip():
                                        extracted_text += clean_text + " "
                
                except Exception:
                    # Skip problematic streams
                    continue
            
            return extracted_text.strip()
            
        except Exception:
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> PDFExtractionResult:
        """
        Extract text content from a PDF file using pikepdf.
        
        Args:
            pdf_path: Path to the PDF file to extract text from
            
        Returns:
            PDFExtractionResult with extracted text or error information
        """
        try:
            # Check if file exists
            if not os.path.exists(pdf_path):
                return PDFExtractionResult(
                    success=False,
                    error=f"PDF file not found: {pdf_path}",
                    file_path=pdf_path
                )
            
            # Check if file is readable
            if not os.access(pdf_path, os.R_OK):
                return PDFExtractionResult(
                    success=False,
                    error=f"PDF file is not readable: {pdf_path}",
                    file_path=pdf_path
                )
            
            # Extract text using pikepdf
            all_text = ""
            page_count = 0
            
            with pikepdf.Pdf.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = self._extract_text_from_page(page)
                    
                    if page_text:
                        all_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    else:
                        # If no text extracted, note the page exists but may be image-based
                        all_text += f"\n--- Page {page_num + 1} ---\n[Page contains no extractable text - may be image-based or use complex formatting]\n"
            
            # If no meaningful text was extracted at all
            if not any(line.strip() and not line.startswith('[Page contains no extractable text') 
                      for line in all_text.split('\n')):
                all_text = f"[PDF contains {page_count} pages but no extractable text. This may be a scanned document requiring OCR or uses complex formatting.]"
            
            return PDFExtractionResult(
                success=True,
                text=all_text.strip(),
                page_count=page_count,
                file_path=pdf_path
            )
            
        except pikepdf.PdfError as e:
            return PDFExtractionResult(
                success=False,
                error=f"PDF parsing error: {str(e)}",
                file_path=pdf_path
            )
        except Exception as e:
            return PDFExtractionResult(
                success=False,
                error=f"Unexpected error extracting PDF text: {str(e)}",
                file_path=pdf_path
            )
    
    def extract_text_simple(self, pdf_path: str) -> str:
        """
        Simple text extraction that returns just the text content or an error message.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content or error description
        """
        result = self.extract_text_from_pdf(pdf_path)
        
        if result.success and result.text:
            return result.text
        else:
            return f"Error extracting PDF text: {result.error}"
    
    def extract_batch(self, pdf_paths: Dict[str, str], output_dir: str) -> Dict[str, str]:
        """Extract text from multiple PDFs."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {}
        for citation_key, pdf_path in pdf_paths.items():
            result = self.extract_text_from_pdf(pdf_path)
            if result.success and result.text:
                text_file = output_path / f"{citation_key}.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(result.text)
                results[citation_key] = str(text_file)
                logger.info(f"Extracted text for {citation_key} ({result.page_count} pages)")
            else:
                logger.error(f"Failed to extract text for {citation_key}: {result.error}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Validate citations in LaTeX manuscripts')
    parser.add_argument('--tex-dir', default='proposal', help='Directory containing LaTeX files')
    parser.add_argument('--bib-file', default='proposal/refs.bib', help='Bibliography file')
    parser.add_argument('--output-dir', default='citation_validation', help='Output directory')
    parser.add_argument('--pdf-dir', help='Directory for PDFs (default: output-dir/pdfs)')
    parser.add_argument('--text-dir', help='Directory for text files (default: output-dir/texts)')
    
    args = parser.parse_args()
    
    # Set default directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    pdf_dir = args.pdf_dir or str(output_dir / 'pdfs')
    text_dir = args.text_dir or str(output_dir / 'texts')
    
    # Step 1: Extract citations from LaTeX files
    print("Step 1: Extracting citations from LaTeX files...")
    extractor = CitationExtractor()
    all_citations = []
    
    tex_dir = Path(args.tex_dir)
    for tex_file in tex_dir.rglob('*.tex'):
        print(f"Processing {tex_file}...")
        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()
        citations = extractor.extract_citations_with_context(content)
        for citation in citations:
            citation['source_file'] = str(tex_file)
        all_citations.extend(citations)
    
    print(f"Found {len(all_citations)} citations")
    
    # Step 2: Parse bibliography
    print("\nStep 2: Parsing bibliography...")
    bib_parser = BibParser(args.bib_file)
    
    # Create citation summary
    citation_summary = defaultdict(list)
    doi_map = {}
    
    for citation in all_citations:
        key = citation['key']
        entry = bib_parser.get_entry(key)
        
        if entry:
            citation['bib_entry'] = entry
            if entry['doi']:
                doi_map[key] = entry['doi']
        else:
            citation['bib_entry'] = None
            print(f"Warning: No bibliography entry found for {key}")
        
        citation_summary[key].append(citation)
    
    # Step 3: Download PDFs using Sci-Hub
    print(f"\nStep 3: Downloading PDFs for {len(doi_map)} citations with DOIs...")
    downloader = PaperDownloader()
    pdf_paths = {}
    
    for citation_key, doi in doi_map.items():
        result = downloader.download_paper(doi, pdf_dir)
        if result.success:
            pdf_paths[citation_key] = result.file_path
            print(f"Downloaded {citation_key}: {result.message}")
        else:
            print(f"Failed to download {citation_key}: {result.message}")
    
    print(f"Successfully downloaded {len(pdf_paths)} PDFs")
    
    # Step 4: Extract text from PDFs
    print("\nStep 4: Extracting text from PDFs...")
    text_extractor = PDFExtractor()
    text_paths = text_extractor.extract_batch(pdf_paths, text_dir)
    
    print(f"Successfully extracted text from {len(text_paths)} PDFs")
    
    # Step 5: Create validation report
    print("\nStep 5: Creating validation report...")
    validation_data = {
        'total_citations': len(all_citations),
        'unique_citations': len(citation_summary),
        'citations_with_doi': len(doi_map),
        'pdfs_downloaded': len(pdf_paths),
        'texts_extracted': len(text_paths),
        'citations': {}
    }
    
    for key, citations in citation_summary.items():
        validation_data['citations'][key] = {
            'occurrences': len(citations),
            'doi': doi_map.get(key, None),
            'pdf_downloaded': key in pdf_paths,
            'text_extracted': key in text_paths,
            'pdf_path': pdf_paths.get(key, None),
            'text_path': text_paths.get(key, None),
            'contexts': [
                {
                    'source_file': c['source_file'],
                    'sentence': c['sentence'],
                    'context': c['context'],
                    'citation_type': c['citation_type']
                }
                for c in citations
            ],
            'bib_entry': citations[0].get('bib_entry', None) if citations else None
        }
    
    # Save validation data
    validation_file = output_dir / 'citation_validation_data.json'
    with open(validation_file, 'w', encoding='utf-8') as f:
        json.dump(validation_data, f, indent=2)
    
    print(f"\nValidation data saved to {validation_file}")
    
    # Create summary report
    summary_file = output_dir / 'validation_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Citation Validation Summary\n")
        f.write("==========================\n\n")
        f.write(f"Total citations found: {validation_data['total_citations']}\n")
        f.write(f"Unique citations: {validation_data['unique_citations']}\n")
        f.write(f"Citations with DOI: {validation_data['citations_with_doi']}\n")
        f.write(f"PDFs downloaded: {validation_data['pdfs_downloaded']}\n")
        f.write(f"Texts extracted: {validation_data['texts_extracted']}\n\n")
        
        f.write("Citations without DOI:\n")
        for key, data in validation_data['citations'].items():
            if not data['doi']:
                f.write(f"  - {key}\n")
        
        f.write("\nCitations with failed PDF download:\n")
        for key, data in validation_data['citations'].items():
            if data['doi'] and not data['pdf_downloaded']:
                f.write(f"  - {key} (DOI: {data['doi']})\n")
    
    print(f"Summary report saved to {summary_file}")
    print("\nValidation preparation complete!")

if __name__ == "__main__":
    main()