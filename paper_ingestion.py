#!/usr/bin/env python3
"""
Paper Ingestion System for ARX2
-------------------------------
Handles discovery, retrieval, and ingestion of academic papers from multiple sources.
Includes intelligent duplicate detection and metadata enrichment.
"""

import os
import json
import time
import uuid
import hashlib
import re
import logging
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor
import requests
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_ingestion.log"),
        logging.StreamHandler()
    ]
)

# Try to import optional dependencies
try:
    import scholarly
    SCHOLARLY_AVAILABLE = True
except ImportError:
    logger.warning("scholarly package not available. Google Scholar integration disabled.")
    SCHOLARLY_AVAILABLE = False

try:
    from semantic_scholar import SemanticScholar
    SEMANTIC_SCHOLAR_AVAILABLE = True
except ImportError:
    logger.warning("semantic-scholar package not available. Semantic Scholar integration disabled.")
    SEMANTIC_SCHOLAR_AVAILABLE = False

# Import internal components
from advanced_ai_analyzer import CONFIG
from advanced_ai_analyzer_paper_processor import PaperProcessor
from advanced_ai_analyzer_knowledge_base import KnowledgeBase

class PaperIngestionSystem:
    """System for ingesting papers from multiple sources with intelligent processing"""
    
    def __init__(self, papers_dir=None, kb=None, processor=None):
        """Initialize the paper ingestion system
        
        Args:
            papers_dir (str, optional): Directory to store downloaded papers
            kb (KnowledgeBase, optional): Knowledge base to add papers to
            processor (PaperProcessor, optional): Paper processor to process papers
        """
        self.papers_dir = papers_dir or CONFIG.get('papers_dir', 'papers')
        os.makedirs(self.papers_dir, exist_ok=True)
        
        # Initialize knowledge base and processor if not provided
        self.kb = kb or KnowledgeBase()
        self.processor = processor or PaperProcessor(papers_dir=self.papers_dir)
        
        # Set up citation enrichment options
        if SEMANTIC_SCHOLAR_AVAILABLE:
            self.ss = SemanticScholar()
        else:
            self.ss = None
            
        # Track paper hashes to avoid duplicates
        self.paper_hashes = self._load_paper_hashes()
        
        # Set up HTTP session for efficient connections
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ARX2 Research System/1.0 (Academic Research Purpose)'
        })
        
        # Set concurrent download limits
        self.max_concurrent_downloads = CONFIG.get('max_concurrent_downloads', 10)
        
        # Local cache for paper metadata
        self.metadata_cache = {}
        
        logger.info(f"Initialized paper ingestion system (papers_dir={self.papers_dir})")
    
    def _load_paper_hashes(self) -> Dict[str, str]:
        """Load paper hashes from disk to avoid reprocessing known papers
        
        Returns:
            dict: Mapping of paper hash to paper ID
        """
        cache_path = os.path.join(self.papers_dir, 'paper_hashes.json')
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except:
                logger.warning(f"Could not load paper hashes from {cache_path}")
        
        # Initialize with existing papers in knowledge base
        paper_hashes = {}
        for paper_id, paper in self.kb.papers.items():
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            
            if title:
                title_hash = hashlib.md5(title.lower().encode()).hexdigest()
                paper_hashes[title_hash] = paper_id
            
            if abstract:
                abstract_hash = hashlib.md5(abstract.lower().encode()).hexdigest()
                paper_hashes[abstract_hash] = paper_id
        
        return paper_hashes
    
    def _save_paper_hashes(self):
        """Save paper hashes to disk"""
        cache_path = os.path.join(self.papers_dir, 'paper_hashes.json')
        try:
            with open(cache_path, 'w') as f:
                json.dump(self.paper_hashes, f)
            logger.debug(f"Saved {len(self.paper_hashes)} paper hashes to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save paper hashes: {e}")
    
    def is_duplicate(self, title: str, abstract: str) -> Optional[str]:
        """Check if a paper with this title or abstract already exists
        
        Args:
            title (str): Paper title
            abstract (str): Paper abstract
            
        Returns:
            str or None: Paper ID if duplicate exists, None otherwise
        """
        if not title and not abstract:
            return None
            
        # Check by title hash
        if title:
            title_hash = hashlib.md5(title.lower().encode()).hexdigest()
            if title_hash in self.paper_hashes:
                return self.paper_hashes[title_hash]
        
        # Check by abstract hash
        if abstract:
            abstract_hash = hashlib.md5(abstract.lower().encode()).hexdigest()
            if abstract_hash in self.paper_hashes:
                return self.paper_hashes[abstract_hash]
        
        # Not a duplicate
        return None
    
    def add_to_paper_hashes(self, paper_id: str, title: str, abstract: str):
        """Add a paper to the hash index
        
        Args:
            paper_id (str): Paper ID
            title (str): Paper title
            abstract (str): Paper abstract
        """
        if title:
            title_hash = hashlib.md5(title.lower().encode()).hexdigest()
            self.paper_hashes[title_hash] = paper_id
        
        if abstract:
            abstract_hash = hashlib.md5(abstract.lower().encode()).hexdigest()
            self.paper_hashes[abstract_hash] = paper_id
    
    async def ingest_from_arxiv(self, query: str, max_results: int = 100) -> List[str]:
        """Ingest papers from ArXiv based on a query
        
        Args:
            query (str): Search query for ArXiv
            max_results (int): Maximum number of papers to ingest
            
        Returns:
            list: Paper IDs of ingested papers
        """
        logger.info(f"Searching ArXiv for query: '{query}' (max_results={max_results})")
        
        # Delegate to the processor for ArXiv search and download
        paper_metadata = self.processor.download_papers(query, max_results=max_results)
        
        # Process the papers
        processed_papers = self.processor.process_papers_batch(paper_metadata)
        
        # Add to knowledge base
        added_count = self.kb.add_papers(processed_papers)
        
        # Update paper hashes for the added papers
        for paper in processed_papers:
            paper_id = paper.get('id')
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            
            if paper_id:
                self.add_to_paper_hashes(paper_id, title, abstract)
        
        # Save paper hashes
        self._save_paper_hashes()
        
        logger.info(f"Ingested {added_count} papers from ArXiv for query: '{query}'")
        
        # Return the paper IDs
        return [paper.get('id') for paper in processed_papers if 'id' in paper]
    
    async def ingest_from_semantic_scholar(self, query: str, max_results: int = 100) -> List[str]:
        """Ingest papers from Semantic Scholar based on a query
        
        Args:
            query (str): Search query for Semantic Scholar
            max_results (int): Maximum number of papers to ingest
            
        Returns:
            list: Paper IDs of ingested papers
        """
        if not SEMANTIC_SCHOLAR_AVAILABLE or not self.ss:
            logger.warning("Semantic Scholar integration not available")
            return []
            
        logger.info(f"Searching Semantic Scholar for query: '{query}' (max_results={max_results})")
        
        try:
            # Search Semantic Scholar API
            search_results = self.ss.search_paper(query, limit=max_results)
            papers_data = []
            
            for paper in search_results:
                # Get basic paper data
                paper_id = paper.get('paperId')
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                
                if not paper_id or not title:
                    continue
                    
                # Check for duplicates
                duplicate_id = self.is_duplicate(title, abstract)
                if duplicate_id:
                    logger.debug(f"Skipping duplicate paper: {title} (duplicate of {duplicate_id})")
                    continue
                
                # Get more detailed paper data if available
                try:
                    detailed_paper = self.ss.get_paper(paper_id)
                    pdf_url = None
                    
                    # Try to get PDF URL from various sources
                    if detailed_paper.get('openAccessPdf'):
                        pdf_url = detailed_paper['openAccessPdf'].get('url')
                    elif detailed_paper.get('externalIds', {}).get('ArXiv'):
                        arxiv_id = detailed_paper['externalIds']['ArXiv']
                        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    
                    if not pdf_url:
                        logger.debug(f"No PDF available for paper: {title}")
                        continue
                        
                    # Generate safe filename
                    safe_id = re.sub(r'[^\w\-\.]', '_', paper_id)
                    filename = f"{safe_id}.pdf"
                    filepath = os.path.join(self.papers_dir, filename)
                    
                    # Only download if not already exists
                    if not os.path.exists(filepath):
                        # Download PDF
                        try:
                            response = self.session.get(pdf_url, stream=True)
                            response.raise_for_status()
                            
                            with open(filepath, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                                    
                            logger.debug(f"Downloaded PDF to {filepath}")
                        except Exception as e:
                            logger.warning(f"Failed to download PDF for {title}: {e}")
                            continue
                    
                    # Prepare paper metadata
                    papers_data.append({
                        'id': paper_id,
                        'filepath': filepath,
                        'title': title,
                        'abstract': abstract,
                        'authors': [author.get('name', '') for author in detailed_paper.get('authors', [])],
                        'published': detailed_paper.get('year', ''),
                        'updated': '',
                        'categories': [field.get('name', '') for field in detailed_paper.get('fieldsOfStudy', [])],
                        'pdf_url': pdf_url,
                        'entry_id': f"semantic_scholar:{paper_id}"
                    })
                    
                    # Allow time between requests to avoid rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Failed to get detailed data for paper {paper_id}: {e}")
            
            # Process papers
            processed_papers = self.processor.process_papers_batch(papers_data)
            
            # Add to knowledge base
            added_count = self.kb.add_papers(processed_papers)
            
            # Update paper hashes
            for paper in processed_papers:
                paper_id = paper.get('id')
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                
                if paper_id:
                    self.add_to_paper_hashes(paper_id, title, abstract)
            
            # Save paper hashes
            self._save_paper_hashes()
            
            logger.info(f"Ingested {added_count} papers from Semantic Scholar for query: '{query}'")
            
            # Return the paper IDs
            return [paper.get('id') for paper in processed_papers if 'id' in paper]
            
        except Exception as e:
            logger.error(f"Error ingesting from Semantic Scholar: {e}")
            return []
    
    async def ingest_from_url(self, url: str) -> Optional[str]:
        """Ingest a single paper from a URL
        
        Args:
            url (str): URL to the paper (PDF)
            
        Returns:
            str or None: Paper ID if successful, None otherwise
        """
        logger.info(f"Ingesting paper from URL: {url}")
        
        try:
            # Parse URL to get potential filename
            parsed_url = urlparse(url)
            path = unquote(parsed_url.path)
            
            # Try to extract meaningful filename or generate one
            if path.endswith('.pdf'):
                filename = os.path.basename(path)
            else:
                # Generate a random filename
                filename = f"{uuid.uuid4().hex}.pdf"
            
            filepath = os.path.join(self.papers_dir, filename)
            
            # Download the PDF
            try:
                response = self.session.get(url, stream=True)
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                logger.debug(f"Downloaded PDF to {filepath}")
            except Exception as e:
                logger.error(f"Failed to download PDF from {url}: {e}")
                return None
            
            # Generate a paper ID
            paper_id = f"url_{uuid.uuid4().hex[:10]}"
            
            # Process the paper
            paper_data = {
                'id': paper_id,
                'filepath': filepath,
                'title': '',  # Will be extracted from PDF
                'abstract': '',  # Will be extracted from PDF
                'authors': [],  # Will be extracted from PDF
                'published': '',
                'updated': datetime.now().isoformat(),
                'categories': [],
                'pdf_url': url,
                'entry_id': f"url:{paper_id}"
            }
            
            processed_paper = self.processor.process_paper(paper_data)
            
            if not processed_paper:
                logger.warning(f"Failed to process paper from URL: {url}")
                return None
                
            # Check if it's a duplicate based on extracted title/abstract
            title = processed_paper.get('title', '')
            abstract = processed_paper.get('abstract', '')
            
            duplicate_id = self.is_duplicate(title, abstract)
            if duplicate_id:
                logger.info(f"Paper from URL is a duplicate of {duplicate_id}")
                # Clean up the downloaded file
                os.remove(filepath)
                return duplicate_id
            
            # Add to knowledge base
            papers_to_add = [processed_paper]
            added_count = self.kb.add_papers(papers_to_add)
            
            if added_count > 0:
                # Update paper hashes
                self.add_to_paper_hashes(paper_id, title, abstract)
                self._save_paper_hashes()
                logger.info(f"Successfully ingested paper from URL: {url}")
                return paper_id
            else:
                logger.warning(f"Failed to add paper from URL to knowledge base: {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error ingesting from URL: {e}")
            return None
    
    async def ingest_from_directory(self, directory: str, recursive: bool = True) -> List[str]:
        """Ingest all PDF files from a directory
        
        Args:
            directory (str): Directory path
            recursive (bool): Whether to search subdirectories
            
        Returns:
            list: Paper IDs of ingested papers
        """
        logger.info(f"Ingesting papers from directory: {directory} (recursive={recursive})")
        
        # Find all PDF files
        pdf_files = []
        
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(directory, file))
        
        logger.info(f"Found {len(pdf_files)} PDF files in directory")
        
        # Process each PDF file
        paper_ids = []
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent_downloads) as executor:
            futures = []
            
            for pdf_file in pdf_files:
                futures.append(executor.submit(self._process_pdf_file, pdf_file))
            
            # Process results as they complete
            for future in tqdm(futures, desc="Processing PDFs"):
                try:
                    paper_id = future.result()
                    if paper_id:
                        paper_ids.append(paper_id)
                except Exception as e:
                    logger.error(f"Error processing PDF file: {e}")
        
        # Save paper hashes
        self._save_paper_hashes()
        
        logger.info(f"Ingested {len(paper_ids)} papers from directory")
        return paper_ids
    
    def _process_pdf_file(self, pdf_file: str) -> Optional[str]:
        """Process a single PDF file
        
        Args:
            pdf_file (str): Path to PDF file
            
        Returns:
            str or None: Paper ID if successful, None otherwise
        """
        try:
            # Generate a unique ID for this paper
            filename = os.path.basename(pdf_file)
            paper_id = f"local_{uuid.uuid4().hex[:10]}"
            
            # Copy to papers directory if needed
            target_path = os.path.join(self.papers_dir, filename)
            
            if pdf_file != target_path and not os.path.exists(target_path):
                # Copy the file
                import shutil
                shutil.copy2(pdf_file, target_path)
            
            # Process the paper
            paper_data = {
                'id': paper_id,
                'filepath': target_path,
                'title': '',  # Will be extracted from PDF
                'abstract': '',  # Will be extracted from PDF
                'authors': [],  # Will be extracted from PDF
                'published': '',
                'updated': datetime.now().isoformat(),
                'categories': [],
                'pdf_url': '',
                'entry_id': f"local:{paper_id}"
            }
            
            processed_paper = self.processor.process_paper(paper_data)
            
            if not processed_paper:
                logger.warning(f"Failed to process paper from file: {pdf_file}")
                return None
                
            # Check if it's a duplicate based on extracted title/abstract
            title = processed_paper.get('title', '')
            abstract = processed_paper.get('abstract', '')
            
            duplicate_id = self.is_duplicate(title, abstract)
            if duplicate_id:
                logger.info(f"Paper from file is a duplicate of {duplicate_id}")
                return duplicate_id
            
            # Add to knowledge base
            papers_to_add = [processed_paper]
            added_count = self.kb.add_papers(papers_to_add)
            
            if added_count > 0:
                # Update paper hashes
                self.add_to_paper_hashes(paper_id, title, abstract)
                logger.debug(f"Successfully ingested paper from file: {pdf_file}")
                return paper_id
            else:
                logger.warning(f"Failed to add paper from file to knowledge base: {pdf_file}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing PDF file {pdf_file}: {e}")
            return None
    
    async def enrich_citations(self, paper_ids: List[str]) -> int:
        """Enrich paper metadata with citation information
        
        Args:
            paper_ids (list): List of paper IDs to enrich
            
        Returns:
            int: Number of papers enriched
        """
        if not SEMANTIC_SCHOLAR_AVAILABLE or not self.ss:
            logger.warning("Semantic Scholar integration not available, skipping citation enrichment")
            return 0
            
        logger.info(f"Enriching citation information for {len(paper_ids)} papers")
        
        enriched_count = 0
        
        for paper_id in tqdm(paper_ids, desc="Enriching citations"):
            try:
                # Check if paper exists in knowledge base
                if paper_id not in self.kb.papers:
                    continue
                    
                paper = self.kb.papers[paper_id]
                
                # Skip if already has citations
                if 'citations' in paper and paper['citations']:
                    continue
                
                # Try to find on Semantic Scholar by title
                title = paper.get('title', '')
                if not title:
                    continue
                    
                # Search for the paper
                search_results = self.ss.search_paper(title, limit=3)
                
                # Find the best match
                best_match = None
                for result in search_results:
                    # Simple string comparison for now - could be improved
                    if result.get('title', '').lower() == title.lower():
                        best_match = result
                        break
                
                if not best_match:
                    continue
                
                # Get detailed paper info
                ss_paper_id = best_match.get('paperId')
                if not ss_paper_id:
                    continue
                    
                detailed_paper = self.ss.get_paper(ss_paper_id)
                
                # Extract citation information
                citations = []
                if 'references' in detailed_paper:
                    for ref in detailed_paper['references']:
                        cited_paper = ref.get('citedPaper', {})
                        cited_id = cited_paper.get('paperId')
                        
                        if cited_id:
                            citations.append(cited_id)
                
                # Add citation information to paper
                if citations:
                    paper['citations'] = citations
                    
                    # Update in knowledge base
                    self.kb.papers[paper_id] = paper
                    
                    enriched_count += 1
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"Error enriching citations for paper {paper_id}: {e}")
        
        logger.info(f"Enriched citation information for {enriched_count} papers")
        
        # Save knowledge base if papers were enriched
        if enriched_count > 0:
            self.kb.save()
        
        return enriched_count

    async def ingest_from_multiple_sources(self, query: str, max_results: int = 100) -> List[str]:
        """Ingest papers from multiple sources based on a query
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of papers to ingest per source
            
        Returns:
            list: Paper IDs of ingested papers
        """
        logger.info(f"Ingesting papers from multiple sources for query: '{query}'")
        
        paper_ids = []
        
        # Ingest from ArXiv
        arxiv_ids = await self.ingest_from_arxiv(query, max_results=max_results)
        paper_ids.extend(arxiv_ids)
        
        # Ingest from Semantic Scholar if available
        if SEMANTIC_SCHOLAR_AVAILABLE:
            ss_ids = await self.ingest_from_semantic_scholar(query, max_results=max_results)
            paper_ids.extend(ss_ids)
        
        logger.info(f"Ingested a total of {len(paper_ids)} papers from all sources")
        
        # Enrich citation information if available
        if paper_ids and SEMANTIC_SCHOLAR_AVAILABLE:
            await self.enrich_citations(paper_ids)
        
        return paper_ids

async def main():
    """Command-line interface for the paper ingestion system"""
    parser = argparse.ArgumentParser(description="ARX2 Paper Ingestion System")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ArXiv subcommand
    arxiv_parser = subparsers.add_parser("arxiv", help="Ingest papers from ArXiv")
    arxiv_parser.add_argument("query", help="Search query for ArXiv")
    arxiv_parser.add_argument("--max", type=int, default=100, help="Maximum number of papers to ingest")
    
    # Semantic Scholar subcommand
    ss_parser = subparsers.add_parser("semantic", help="Ingest papers from Semantic Scholar")
    ss_parser.add_argument("query", help="Search query for Semantic Scholar")
    ss_parser.add_argument("--max", type=int, default=100, help="Maximum number of papers to ingest")
    
    # URL subcommand
    url_parser = subparsers.add_parser("url", help="Ingest a paper from a URL")
    url_parser.add_argument("url", help="URL to the paper (PDF)")
    
    # Directory subcommand
    dir_parser = subparsers.add_parser("directory", help="Ingest all PDF files from a directory")
    dir_parser.add_argument("directory", help="Directory path")
    dir_parser.add_argument("--no-recursive", action="store_true", help="Don't search subdirectories")
    
    # Multiple sources subcommand
    multi_parser = subparsers.add_parser("multi", help="Ingest papers from multiple sources")
    multi_parser.add_argument("query", help="Search query")
    multi_parser.add_argument("--max", type=int, default=100, help="Maximum number of papers to ingest per source")
    
    # Enrichment subcommand
    enrich_parser = subparsers.add_parser("enrich", help="Enrich papers with citation information")
    enrich_parser.add_argument("--limit", type=int, default=100, help="Maximum number of papers to enrich")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize ingestion system
    ingestion_system = PaperIngestionSystem()
    
    # Run command
    if args.command == "arxiv":
        await ingestion_system.ingest_from_arxiv(args.query, max_results=args.max)
    elif args.command == "semantic":
        await ingestion_system.ingest_from_semantic_scholar(args.query, max_results=args.max)
    elif args.command == "url":
        await ingestion_system.ingest_from_url(args.url)
    elif args.command == "directory":
        await ingestion_system.ingest_from_directory(args.directory, recursive=not args.no_recursive)
    elif args.command == "multi":
        await ingestion_system.ingest_from_multiple_sources(args.query, max_results=args.max)
    elif args.command == "enrich":
        # Get paper IDs from knowledge base
        paper_ids = list(ingestion_system.kb.papers.keys())[:args.limit]
        await ingestion_system.enrich_citations(paper_ids)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main()) 