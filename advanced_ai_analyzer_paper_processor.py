import time
import json
import random
import numpy as np
import spacy # Import spaCy
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# Remove CountVectorizer if only using TF-IDF for concepts
# from sklearn.feature_extraction.text import CountVectorizer
import arxiv
import os
from advanced_ai_analyzer import logger, CONFIG
import re
from advanced_ai_analyzer import *
from utils.embedding_manager import EmbeddingManager
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import PyPDF2
import nltk
import torch

# Remove NLTK imports
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# try:
#     from nltk.corpus import stopwords, wordnet
# except ImportError:
#     logger.warning("NLTK corpus not available. Will download necessary resources.")

class PaperProcessor:
    """Enhanced processor for scientific papers with advanced embedding generation"""
    
    def __init__(self, papers_dir=CONFIG['papers_dir']):
        self.papers_dir = papers_dir
        os.makedirs(papers_dir, exist_ok=True)
        
        # Initialize embedding manager with modern transformer models
        self.embedding_manager = EmbeddingManager()
        
        # Initialize NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        
        # Initialize spaCy NLP model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            # Set max text length for spaCy (increase for very long documents)
            self.nlp.max_length = 2500000  # Increased from 1,500,000 characters
            logger.info("Loaded spaCy model for advanced text processing")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}. Trying to download...")
            try:
                # Try to download the model if not available
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                self.nlp.max_length = 2500000
                logger.info("Downloaded and loaded spaCy model")
            except Exception as e2:
                logger.error(f"Could not download spaCy model: {e2}. Text preprocessing will be limited.")
                self.nlp = None
        
        # Initialize the TF-IDF vectorizer for concept extraction
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=CONFIG.get('max_features', 10000),  # Increased from 7500
            ngram_range=(1, 3),  # Allow single words, bigrams, and trigrams
            stop_words='english',
            min_df=1
        )
        
        # Get embedding dimension from config
        self.embedding_dim = CONFIG.get('embedding_size', 1536)  # Increased dimension
        
        # Citation graph analysis
        self.citation_graph = defaultdict(set)
        self.inverse_citation_graph = defaultdict(set)
        
        # Initialize text augmentation parameters
        self.augmentation_probability = CONFIG.get('augmentation_probability', 0.2)
        
        # Initialize code snippet extraction patterns
        self.code_patterns = [
            r'```[\s\S]*?```',  # Markdown code blocks
            r'<pre><code>[\s\S]*?</code></pre>',  # HTML code blocks
            r'def\s+\w+\s*\([^)]*\)\s*:',  # Python function definitions
            r'class\s+\w+\s*(\([^)]*\))?\s*:',  # Python class definitions
            r'for\s+\w+\s+in\s+.+:',  # Python for loops
            r'if\s+.+:',  # Python if statements
            r'while\s+.+:',  # Python while loops
            r'import\s+[\w\s,]+',  # Python imports
            r'from\s+[\w.]+\s+import\s+[\w\s,]+',  # Python from imports
        ]
        self.code_patterns = [re.compile(pattern) for pattern in self.code_patterns]
        
        # Initialize sentence transformer for text embeddings
        try:
            model_name = CONFIG.get('sentence_transformer_model')
            self.sentence_transformer = SentenceTransformer(model_name) if CONFIG.get('use_sentence_transformer') else None
            if self.sentence_transformer:
                logger.info(f"Loaded sentence transformer model: {model_name}")
            else:
                logger.info("Sentence transformer disabled in config")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_transformer = None
        
        # Multi-processing settings
        self.max_workers = CONFIG.get('num_workers', 4)
        
        # New: Initialize section patterns for section extraction
        self.section_patterns = {
            'abstract': [r'abstract', r'summary'],
            'introduction': [r'introduction', r'background', r'overview'],
            'methodology': [r'method', r'approach', r'algorithm', r'implementation', r'framework'],
            'results': [r'result', r'experimental result', r'evaluation', r'performance'],
            'discussion': [r'discussion', r'analysis', r'limitation'],
            'conclusion': [r'conclusion', r'future work', r'future direction'],
            'references': [r'reference', r'bibliography']
        }
        
        # New: Initialize section classifier for paper structure understanding
        try:
            # Use a lightweight model that can classify academic paper sections
            self.section_classifier = None
            if CONFIG.get('use_section_classifier', True):
                # Import only if needed
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                section_model_name = CONFIG.get('section_classifier_model', 'allenai/scibert_scivocab_uncased')
                self.section_classifier_tokenizer = AutoTokenizer.from_pretrained(section_model_name)
                self.section_classifier = AutoModelForSequenceClassification.from_pretrained(
                    section_model_name, 
                    num_labels=len(self.section_patterns),
                    id2label={i: label for i, label in enumerate(self.section_patterns.keys())},
                    label2id={label: i for i, label in enumerate(self.section_patterns.keys())}
                )
                self.section_classifier.eval()  # Set to evaluation mode
                if torch.cuda.is_available():
                    self.section_classifier.to('cuda')
                logger.info(f"Loaded section classifier model: {section_model_name}")
        except Exception as e:
            logger.warning(f"Failed to load section classifier: {e}. Will use rule-based section detection.")
            self.section_classifier = None
        
        logger.info(f"Initialized PaperProcessor with embedding model {self.embedding_manager.embedding_model}")

    def _extract_text_from_pdf(self, filepath):
        """Extract text from PDF file with robust error handling"""
        if not os.path.exists(filepath):
            logger.warning(f"PDF file does not exist: {filepath}")
            return ""
            
        try:
            # Temporarily suppress warnings about unknown widths
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*?unknown widths.*?")
                
                text = ""
                with open(filepath, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    num_pages = len(reader.pages)
                    
                    # Extract text from all pages
                    for page_num in range(num_pages):
                        try:
                            page = reader.pages[page_num]
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n\n"
                        except Exception as e:
                            logger.warning(f"Error extracting text from page {page_num}: {e}")
            
            # Clean the text
            text = self._clean_text(text)
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {filepath}: {e}")
            return ""
    
    def _clean_text(self, text):
        """Clean and normalize text from PDFs"""
        if not text:
            return ""
            
        # Replace excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove special characters but keep meaningful punctuation
        text = re.sub(r'[^\w\s.,;:?!-]', '', text)
        
        # Replace sequences of punctuation with single instances
        text = re.sub(r'[.,;:?!-]+', lambda m: m.group(0)[0], text)
        
        return text.strip()
    
    def _extract_concepts(self, text, top_n=CONFIG['top_n_concepts']):
        """Extract key concepts from text using improved NLP techniques"""
        if not text:
            return []
            
        # Tokenize text
        tokens = nltk.word_tokenize(text.lower())
        
        # Remove stopwords and short tokens
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stopwords and len(token) > 2]
        
        # Count token frequencies
        token_counts = Counter(tokens)
        
        # Extract most common tokens as concepts
        concepts = [concept for concept, count in token_counts.most_common(top_n)]
        
        return concepts
    
    def generate_embedding(self, text):
        """Generate embedding for text using advanced transformer models"""
        # Delegate to embedding manager
        return self.embedding_manager.get_embedding_for_text(text)
    
    def process_paper(self, paper_data, file_path=None):
        """Process a paper, extracting key information.
        
        Args:
            paper_data (dict): Paper metadata
            file_path (str, optional): Path to PDF file to extract text from
            
        Returns:
            dict: Processed paper data with extracted information
        """
        # Initialize a minimal result dictionary to ensure we always return valid data
        result = {
            "paper_id": None,
            "title": "",
            "abstract": "",
            "full_text": "",
            "status": "error",
            "errors": []
        }
        
        # Input validation
        if not isinstance(paper_data, dict):
            error_msg = f"Expected paper_data to be a dictionary, got {type(paper_data)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result
            
        # Ensure paper_id exists
        if "paper_id" not in paper_data:
            error_msg = "Missing paper_id in paper_data"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            return result
            
        # Update basic fields from paper_data
        result["paper_id"] = paper_data.get("paper_id")
        result["title"] = paper_data.get("title", "")
        result["abstract"] = paper_data.get("abstract", "")
        
        # Extract text from PDF if file_path is provided
        text = ""
        if file_path:
            try:
                text = self._extract_text_from_pdf(file_path)
                if not text.strip():
                    error_msg = f"Empty text extracted from {file_path}"
                    logger.warning(error_msg)
                    result["errors"].append(error_msg)
            except Exception as e:
                error_msg = f"Failed to extract text from PDF {file_path}: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
        
        # If we have paper_data with full_text, use that
        if "full_text" in paper_data and paper_data["full_text"]:
            text = paper_data["full_text"]
            
        # Update the result with the extracted text
        result["full_text"] = text
        
        # Process the paper data and text with robust error handling
        try:
            # Generate embeddings
            embeddings = {}
            try:
                if text and self.embedding_manager:
                    embeddings = self.generate_embedding(text)
                    result["embeddings"] = embeddings
            except Exception as e:
                error_msg = f"Failed to generate embeddings: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
            
            # Extract sections
            try:
                sections = self.extract_sections(text)
                result["sections"] = sections
            except Exception as e:
                error_msg = f"Failed to extract sections: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                result["sections"] = []
            
            # Extract concepts
            try:
                concepts = self.extract_concepts(text)
                result["concepts"] = concepts
            except Exception as e:
                error_msg = f"Failed to extract concepts: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                result["concepts"] = []
            
            # Extract code snippets
            try:
                code_snippets = self.extract_code_snippets(text)
                result["code_snippets"] = code_snippets
            except Exception as e:
                error_msg = f"Failed to extract code snippets: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                result["code_snippets"] = []
            
            # Extract references
            try:
                references = self.extract_references(text)
                result["references"] = references
            except Exception as e:
                error_msg = f"Failed to extract references: {str(e)}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                result["references"] = []
            
            # Update status if we got this far without critical errors
            if not result["errors"]:
                result["status"] = "success"
            else:
                # Non-critical errors still allow partial success
                result["status"] = "partial"
                
        except Exception as e:
            error_msg = f"Unexpected error processing paper: {str(e)}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            
        return result
    
    def process_papers_batch(self, papers_metadata):
        """Process a batch of papers in parallel
        
        Args:
            papers_metadata (list): List of paper metadata dictionaries
            
        Returns:
            list: Processed papers with features
        """
        if not papers_metadata:
            return []
            
        processed_papers = []
        
        # Process papers in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_paper = {executor.submit(self.process_paper, paper): paper for paper in papers_metadata}
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_paper), total=len(papers_metadata), desc="Processing papers"):
                paper = future_to_paper[future]
                
                try:
                    processed_paper = future.result()
                    if processed_paper:
                        processed_papers.append(processed_paper)
                except Exception as e:
                    logger.error(f"Error in paper processing task for {paper.get('paper_id', 'unknown')}: {e}")
        
        logger.info(f"Processed {len(processed_papers)} papers out of {len(papers_metadata)} submitted")
        
        # Print embedding cache stats
        cache_stats = self.embedding_manager.get_cache_stats()
        logger.info(f"Embedding cache stats: hit rate {cache_stats['hit_rate']:.1f}%, hits: {cache_stats['cache_hits']}, misses: {cache_stats['cache_misses']}")
        
        return processed_papers
    
    def download_papers(self, query, max_results=100):
        """Download papers from ArXiv based on a query
        
        Args:
            query (str): Search query for ArXiv
            max_results (int): Maximum number of papers to download
            
        Returns:
            list: List of paper metadata dictionaries
        """
        logger.info(f"Searching ArXiv for query: '{query}' (max_results={max_results})")
        
        try:
            # Search ArXiv
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = list(search.results())
            if not results:
                logger.warning(f"No papers found on ArXiv for query: {query}")
                return []
                
            logger.info(f"Found {len(results)} potential papers on ArXiv")
            
            # Download papers in parallel
            papers_metadata = []
            download_count = 0
            
            with ThreadPoolExecutor(max_workers=min(CONFIG.get('max_parallel_downloads', 8), len(results))) as executor:
                # Submit download tasks
                futures = []
                for paper in results:
                    futures.append(executor.submit(self._download_single_paper, paper))
                
                # Process results as they complete
                for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading papers"):
                    try:
                        paper_meta = future.result()
                        if paper_meta:
                            papers_metadata.append(paper_meta)
                            download_count += 1
                    except Exception as e:
                        logger.error(f"Error in paper download task: {e}")
            
            logger.info(f"Successfully downloaded {download_count} papers")
            return papers_metadata
            
        except Exception as e:
            logger.error(f"ArXiv search failed for query '{query}': {e}")
            return []
    
    def _download_single_paper(self, paper):
        """Download a single paper from ArXiv
        
        Args:
            paper: ArXiv paper object
            
        Returns:
            dict: Paper metadata or None on failure
        """
        try:
            # Extract ID
            paper_id_raw = paper.entry_id.split('/abs/')[-1].split('v')[0]
            # Sanitize paper ID for filename
            paper_id_safe = paper_id_raw.replace('/', '_')
            pdf_filename = f"{paper_id_safe}.pdf"
            filepath = os.path.join(self.papers_dir, pdf_filename)
            
            # Check if already exists
            if not os.path.exists(filepath):
                paper.download_pdf(dirpath=self.papers_dir, filename=pdf_filename)
                time.sleep(0.5)  # Be polite to ArXiv API
            
            # Prepare metadata
            meta = {
                'paper_id': paper_id_raw,
                'filepath': filepath,
                'title': paper.title,
                'abstract': paper.summary,
                'authors': [str(a) for a in paper.authors],
                'year': paper.published.isoformat(),
                'updated': paper.updated.isoformat(),
                'categories': paper.categories,
                'url': paper.pdf_url,
                'entry_id': paper.entry_id
            }
            
            return meta
            
        except Exception as e:
            logger.warning(f"Failed to download paper {paper.entry_id}: {e}")
            return None
    
    def get_embedding_cache_stats(self):
        """Get embedding cache statistics"""
        return self.embedding_manager.get_cache_stats()

    def _save_cache(self, cache_data, cache_path):
        try:
             temp_path = cache_path + ".tmp"
             with open(temp_path, 'w') as f:
                  json.dump(cache_data, f, indent=2)
             os.replace(temp_path, cache_path)
             logger.debug(f"Saved cache to {cache_path}")
        except Exception as e:
             logger.error(f"Error saving cache to {cache_path}: {e}")

    def _load_cache(self, cache_path):
        """Load cache data from a JSON file."""
        if not os.path.exists(cache_path):
            logger.warning(f"Cache file not found: {cache_path}. Returning empty cache.")
            return {} # Return empty dict if file doesn't exist
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            logger.debug(f"Loaded cache from {cache_path}")
            return cache_data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from cache file {cache_path}: {e}. Returning empty cache.")
            return {}
        except Exception as e:
            logger.error(f"Error loading cache from {cache_path}: {e}. Returning empty cache.")
            return {}

    # ADD new preprocess_text using spaCy
    def preprocess_text(self, text):
        """Preprocess text using spaCy for lemmatization and stopword removal."""
        if not text:
            logger.debug("Cannot preprocess empty text.")
            # Return minimal valid token list instead of empty list
            return ["placeholder"]
        
        try:
            # Process text with spaCy - consider increasing max_length if needed for long papers
            if self.nlp is None:
                logger.warning("spaCy model not available for preprocessing, using fallback tokenization.")
                # Fallback to basic preprocessing if spaCy is not available
                tokens = nltk.word_tokenize(text.lower())
                tokens = [token for token in tokens 
                         if token not in self.stopwords 
                         and not token.isdigit()
                         and len(token) > CONFIG.get('min_token_length', 2)]
                return tokens if tokens else ["placeholder"]
            
            # Process text with spaCy
            doc = self.nlp(text.lower())

            # Lemmatize, remove stopwords, punctuation, and short tokens
            processed_tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop
                and not token.is_punct
                and not token.is_space
                and len(token.lemma_) > CONFIG.get('min_token_length', 2)
            ]
            
            # Basic check if processing yielded any result
            if not processed_tokens:
                logger.warning("Preprocessing resulted in empty token list, using fallback.")
                # Return token from raw text as fallback
                raw_tokens = text.lower().split()
                filtered_tokens = [t for t in raw_tokens if len(t) > 2 and not t.isdigit()]
                return filtered_tokens if filtered_tokens else ["placeholder"]
            
            return processed_tokens
        except ValueError as e:
            # Handle potential spaCy length limit errors
            if "is greater than the model's maximum attribute" in str(e):
                logger.error(f"Text length ({len(text)}) exceeds spaCy model's max length. Truncating.")
                # Truncate text and use basic tokenization
                truncated_text = text[:self.nlp.max_length-1000] if hasattr(self.nlp, 'max_length') else text[:100000]
                tokens = nltk.word_tokenize(truncated_text.lower())
                tokens = [token for token in tokens 
                        if token not in self.stopwords 
                        and not token.isdigit()
                        and len(token) > CONFIG.get('min_token_length', 2)]
                return tokens if tokens else ["placeholder"]
            else:
                logger.error(f"SpaCy preprocessing failed with ValueError: {e}")
                # Fallback to simple tokenization
                tokens = text.lower().split()
                return tokens if tokens else ["placeholder"]
        except Exception as e:
            logger.error(f"SpaCy preprocessing failed: {e}")
            # Fallback to simple split
            tokens = text.lower().split()
            return tokens if tokens else ["placeholder"]

    def extract_concepts(self, processed_tokens, top_n=CONFIG.get('top_n_concepts', 50)):
        """Extract key concepts using TF-IDF on preprocessed tokens."""
        if not processed_tokens:
            logger.warning("Cannot extract concepts from empty token list.")
            return ["no_concept_extracted"]  # Return a placeholder concept instead of empty list
        
        try:
            # Use a combination of TF-IDF and domain-specific keyword extraction
            # Join tokens back into a single string for TF-IDF
            text_for_tfidf = ' '.join(processed_tokens)
            if not text_for_tfidf.strip():  # Check if joined text is empty
                logger.warning("Empty text for TF-IDF, returning default concepts.")
                return ["no_concept_extracted"]
            
            # Fit TF-IDF on the single processed document
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([text_for_tfidf])
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                scores = tfidf_matrix.toarray().flatten()
                term_scores = list(zip(feature_names, scores))
                
                # Add domain-specific keyword weighting
                weighted_scores = []
                domain_keywords = CONFIG.get('domain_keywords', {
                    'algorithm', 'model', 'method', 'framework', 'neural', 'network',
                    'deep', 'learning', 'transformer', 'attention', 'embedding', 
                    'training', 'inference', 'evaluation', 'loss', 'gradient' 
                })
                keyword_boost = CONFIG.get('keyword_boost', 1.5)
                ngram_boost = CONFIG.get('ngram_boost', 1.2)

                for term, score in term_scores:
                    boost = 1.0
                    if term in domain_keywords:
                        boost *= keyword_boost
                    if ' ' in term:
                        boost *= ngram_boost
                    weighted_scores.append((term, score * boost))

                # Sort by weighted score
                sorted_terms = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

                # Filter out terms with very low scores
                min_score_threshold = CONFIG.get('min_concept_score', 0.01)
                important_terms = [term for term, score in sorted_terms if score > min_score_threshold]

                # Return top N terms, or placeholder if no terms found
                return important_terms[:top_n] if important_terms else ["no_concept_extracted"]
            except Exception as e:
                logger.warning(f"TF-IDF extraction failed: {e}, using fallback method.")
                # Fallback to frequency counting
                counts = Counter(processed_tokens)
                terms = [term for term, _ in counts.most_common(top_n)]
                return terms if terms else ["no_concept_extracted"]
            
        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            # Return a placeholder concept to avoid empty lists
            return ["no_concept_extracted"]

    def generate_embedding(self, text):
        """Generate embedding using Sentence Transformer.

        Args:
            text (str): The text (e.g., title + abstract) to embed.

        Returns:
            np.ndarray: The embedding vector (float32), or a zero vector on failure.
        """
        if self.sentence_transformer is None or not text:
            if self.sentence_transformer is None:
                 logger.warning("Sentence transformer not available. Cannot generate embedding.")
            if not text:
                 logger.debug("Cannot generate embedding for empty text.")
            # Return zero vector of the expected dimension
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            # Encode text - SBERT models generally prefer single strings or list of strings
            # Pass as a list to handle potential batching internally, then flatten
            embedding = self.sentence_transformer.encode([text], convert_to_numpy=True)
            embedding = embedding.flatten() # Get the 1D array

            # Ensure correct dimension (model should handle, but double-check)
            if embedding.shape[0] != self.embedding_dim:
                logger.warning(f"Embedding dim mismatch: {embedding.shape[0]} vs {self.embedding_dim}. Resizing.")
                # Pad or truncate
                if embedding.shape[0] > self.embedding_dim:
                    embedding = embedding[:self.embedding_dim]
            else:
                    embedding = np.pad(embedding, (0, self.embedding_dim - embedding.shape[0]), 'constant')

            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Sentence Transformer encoding failed for text snippet starting with '{text[:50]}...': {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32) # Return zero vector on failure

    # Rename original generate_paper_embedding to avoid confusion, or remove if unused
    # def generate_paper_embedding(self, title, abstract): ... 

    def _extract_important_sentences(self, text, num_sentences=10):
        """Extract the most important sentences from text based on keyword density"""
        try:
            # Split text into sentences
            # Use a more robust sentence splitter if needed
            sentences = re.split(r'(?:[.!?]+\s+)|(?:\n\s*\n)', text) # Split on punctuation+space OR blank lines
            sentences = [s.strip() for s in sentences if len(s.strip()) > 20]  # Filter short sentences
            
            if not sentences:
                return []

            # Extract key terms from the entire text using the updated extract_concepts
            # Requires preprocessed tokens first
            processed_tokens = self.preprocess_text(text)
            if not processed_tokens:
                return sentences[:num_sentences] # Fallback: return first few sentences
               
            key_terms = set(self.extract_concepts(processed_tokens, top_n=30))
            if not key_terms:
                return sentences[:num_sentences] # Fallback if no key terms

            # Score sentences based on presence of key terms
            sentence_scores = []
            for sentence in sentences:
                # Simple check for term presence
                term_count = sum(1 for term in key_terms if term.lower() in sentence.lower())
                # Normalize by sentence length (using token count might be better)
                score = term_count / max(1, len(sentence.split()))
                sentence_scores.append((sentence, score))

            # Sort by score and take top sentences
            top_sentences = [s for s, _ in sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]]
            return top_sentences

        except Exception as e:
            logger.warning(f"Error extracting important sentences: {e}")
            # Fallback: return first few sentences
            return sentences[:num_sentences] if 'sentences' in locals() else []

    def process_papers_batch(self, papers_metadata):
        """Process a batch of papers: extract text, concepts, embeddings."""
        results = []
        if not papers_metadata:
            return results

        logger.info(f"Processing batch of {len(papers_metadata)} papers...")

        # --- Step 1: Extract Text (Parallel I/O) --- 
        texts_to_process = {} # Use dict {paper_id: text}
        with ThreadPoolExecutor(max_workers=CONFIG.get('max_pdf_workers', 4)) as executor:
             futures = {}
             for paper_meta in papers_metadata:
                  filepath = paper_meta.get('filepath')
                  paper_id = paper_meta.get('paper_id')
                  if filepath and paper_id:
                       # Submit text extraction task
                       futures[executor.submit(self._extract_text_from_pdf, filepath)] = paper_id
                  else:
                       logger.warning(f"Skipping paper {paper_id or 'Unknown ID'}: missing filepath.")

             for future in tqdm(futures, desc="Extracting PDF text", total=len(futures)):
                  paper_id = futures[future]
                  try:
                       text = future.result()
                       if text:
                            texts_to_process[paper_id] = text
                       else:
                            # Log warning but don't necessarily skip the paper yet
                            logger.warning(f"Text extraction failed or yielded no content for {paper_id}. Concepts/Embeddings may be based on abstract only.")
                  except Exception as e:
                       logger.error(f"Error getting text extraction result for {paper_id}: {e}")

        if not texts_to_process:
             logger.warning("No text could be extracted from any paper in the batch. Attempting abstract-only processing.")
             # Allow proceeding using abstracts if text extraction fails

        # --- Step 2: Preprocess Text & Extract Concepts (Sequential for now) --- 
        concepts_map = {}
        processed_tokens_map = {}
        logger.info("Preprocessing text and extracting concepts...")
        
        # Map paper_id to original metadata for easy access
        meta_map = {p['paper_id']: p for p in papers_metadata}

        for paper_id, meta in tqdm(meta_map.items(), desc="Preprocessing/Concepts"):
             # Use extracted text if available, otherwise fallback to abstract
             text_to_process = texts_to_process.get(paper_id, meta.get('abstract'))
             
             if not text_to_process:
                  logger.warning(f"No text or abstract available for {paper_id}. Skipping concept extraction.")
                  continue # Skip if no text source

             processed_tokens = self.preprocess_text(text_to_process)
             if processed_tokens:
                  processed_tokens_map[paper_id] = processed_tokens
                  concepts = self.extract_concepts(processed_tokens)
                  concepts_map[paper_id] = concepts
             else:
                  logger.warning(f"Preprocessing failed for {paper_id}")
                  # Skip concept extraction if preprocessing fails

        # --- Step 3: Generate Embeddings (Batch GPU-bound) --- 
        embeddings_map = {}
        if self.sentence_transformer:
            logger.info("Generating embeddings...")
            ids_for_embedding = []
            texts_for_embedding = []
            
            for paper_id, meta in meta_map.items():
                 # Use Title + Abstract for embedding consistency
                 text_to_embed = f"{meta.get('title', '')}. {meta.get('abstract', '')}"
                 if text_to_embed.strip() and len(text_to_embed.split()) > 5:
                      ids_for_embedding.append(paper_id)
                      texts_for_embedding.append(text_to_embed)
            else:
                      logger.warning(f"Skipping embedding for {paper_id}: Title/Abstract too short or missing.")
            
            if texts_for_embedding:
                try:
                    embeddings = self.sentence_transformer.encode(
                         texts_for_embedding, 
                         batch_size=CONFIG.get('embedding_batch_size', 32), 
                         show_progress_bar=True, 
                         convert_to_numpy=True
                    )
                    
                    embeddings = embeddings.astype(np.float32)
                    processed_embeddings = []
                    for emb in embeddings:
                         if emb.shape[0] != self.embedding_dim:
                              # Resize logic (as before)
                              if emb.shape[0] > self.embedding_dim: emb = emb[:self.embedding_dim]
                              else: emb = np.pad(emb, (0, self.embedding_dim - emb.shape[0]), 'constant')
                         processed_embeddings.append(emb)
                    
                    embeddings_map = dict(zip(ids_for_embedding, processed_embeddings))
                    logger.info(f"Generated {len(embeddings_map)} embeddings.")
                except Exception as e:
                     logger.error(f"Batch embedding generation failed: {e}")

        # --- Step 4: Assemble Results --- 
        logger.info("Assembling processed paper data...")
        for paper_meta in papers_metadata:
            paper_id = paper_meta.get('paper_id')
            if not paper_id: continue # Skip if metadata has no ID
            
            # Include paper even if some steps failed, but mark data as potentially incomplete
            embedding = embeddings_map.get(paper_id)
            embedding_list = embedding.tolist() if embedding is not None else None
            extracted_text = texts_to_process.get(paper_id)
            
            processed_paper = {
                'paper_id': paper_id,
                'title': paper_meta.get('title', ''),
                'abstract': paper_meta.get('abstract', ''),
                'authors': paper_meta.get('authors', []),
                'year': paper_meta.get('year', ''),
                'url': paper_meta.get('url', ''),
                'extracted_text': extracted_text,
                'embedding': embedding_list,
                'sections': self.extract_sections(extracted_text),
                'concepts': concepts_map.get(paper_id, []),
                'references': self.extract_references(extracted_text),
                'has_full_text': extracted_text is not None,
                'processed_timestamp': time.time()
            }
            results.append(processed_paper)

        logger.info(f"Finished processing batch. Produced {len(results)} processed paper objects.")
        return results

    # Remove _process_single_paper if it exists
    # def _process_single_paper(self, paper, extract_concepts):
    #     ...
        
    # Remove _create_category_vector if it exists
    # def _create_category_vector(self, categories):
    #    ...

    def extract_code_snippets(self, text):
        """Extract code snippets from paper text
        
        Args:
            text (str): Paper text to extract code from
            
        Returns:
            list: Extracted code snippets
        """
        if not text:
            return []
        
        snippets = []
        
        # Apply all regex patterns to find code
        for pattern in self.code_patterns:
            matches = pattern.findall(text)
            snippets.extend(matches)
        
        # Clean snippets
        cleaned_snippets = []
        for snippet in snippets:
            # Remove markdown code block markers
            snippet = re.sub(r'```\w*\n|```', '', snippet)
            # Remove HTML code markers
            snippet = re.sub(r'<pre><code>|</code></pre>', '', snippet)
            if len(snippet.strip()) > 10:  # Only keep substantial snippets
                cleaned_snippets.append(snippet)
        
        return cleaned_snippets
    
    def generate_code_embeddings(self, code_snippets):
        """Generate embeddings for code snippets
        
        Args:
            code_snippets (list): List of code snippets
            
        Returns:
            numpy.ndarray: Combined code embedding
        """
        if not code_snippets:
            return np.zeros(self.embedding_dim)
        
        # Generate embeddings for each snippet
        embeddings = []
        for snippet in code_snippets:
            embedding = self.embedding_manager.get_embedding_for_text(snippet)
            embeddings.append(embedding)
        
        # Combine embeddings (average)
        combined = np.mean(embeddings, axis=0)
        return combined
    
    def build_citation_graph(self, papers):
        """Build citation graph from papers
        
        Args:
            papers (list): List of paper dictionaries
            
        Returns:
            tuple: Citation graph and inverse citation graph
        """
        # Reset graphs
        self.citation_graph = defaultdict(set)
        self.inverse_citation_graph = defaultdict(set)
        
        # Build graphs
        for paper in papers:
            paper_id = paper.get('paper_id')
            if not paper_id:
                continue
            
            # Extract citations from references
            references = paper.get('references', [])
            
            for ref in references:
                ref_id = ref.get('paper_id')
                if ref_id:
                    # This paper cites ref_id
                    self.citation_graph[paper_id].add(ref_id)
                    # ref_id is cited by paper_id
                    self.inverse_citation_graph[ref_id].add(paper_id)
        
        return self.citation_graph, self.inverse_citation_graph
    
    def augment_text(self, text, augmentation_level=CONFIG.get('augmentation_level', 1)):
        """
        Augment text with synonyms from WordNet.
        
        Args:
            text (str): Text to augment
            augmentation_level (int): Level of augmentation (1-3)
            
        Returns:
            str: Augmented text
        """
        if not text or augmentation_level <= 0:
            return text
        
        try:
            if self.nlp is None:
                logger.warning("spaCy model not available for text augmentation.")
                return text
            
            doc = self.nlp(text)
            augmented_tokens = []
            
            for token in doc:
                # Skip stopwords, punctuation, etc.
                if token.is_stop or token.is_punct or token.is_space or token.is_digit:
                    augmented_tokens.append(token.text)
                    continue
                    
                # For content words, add synonyms based on augmentation level
                if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                    try:
                        # Fix: Use correct path to WordNet synsets
                        synsets = nltk.corpus.wordnet.synsets(token.text)
                        if synsets and random.random() < 0.3:  # Only augment some words
                            # Get synonyms from WordNet
                            synonyms = []
                            for synset in synsets[:augmentation_level]:
                                synonyms.extend([lemma.name() for lemma in synset.lemmas()])
                                
                            # Remove duplicates and original word
                            synonyms = list(set(synonyms))
                            if token.text in synonyms:
                                synonyms.remove(token.text)
                                
                            # Choose a random synonym to replace the original word
                            if synonyms:
                                synonym = random.choice(synonyms)
                                # Replace underscores with spaces (WordNet format)
                                synonym = synonym.replace('_', ' ')
                                augmented_tokens.append(synonym)
                                continue
                    except Exception as e:
                        logger.debug(f"Error getting synonyms for '{token.text}': {e}")
                
                # If no synonym found or not eligible for augmentation, use original token
                augmented_tokens.append(token.text)
                
            # Join tokens back into text
            augmented_text = ' '.join(augmented_tokens)
            return augmented_text
        except Exception as e:
            logger.error(f"Text augmentation failed: {e}")
            return text  # Return original text if augmentation fails

    def extract_references(self, text):
        """
        Extract bibliographic references from paper text.
        
        Args:
            text (str): The full paper text
            
        Returns:
            list: List of extracted references
        """
        if not text:
            logger.warning("Cannot extract references from empty text.")
            return []
        
        # Initialize references list
        references = []
        
        try:
            # First approach: Look for a "References" section
            sections = self.extract_sections(text)
            references_section = None
            
            # Find the references section by looking for common reference section titles
            reference_section_titles = [
                "references", "bibliography", "works cited", "literature cited"
            ]
            
            for section_title, section_content in sections.items():
                if section_title.lower() in reference_section_titles:
                    references_section = section_content
                    break
            
            if references_section:
                # Split the references section into individual entries
                # References are typically separated by newlines or numbered
                reference_entries = re.split(r'\n\s*\n|\[\d+\]|\n\d+\.|\n\s*•', references_section)
                
                # Clean and process each entry
                for entry in reference_entries:
                    entry = entry.strip()
                    if not entry:
                        continue
                    
                    # Skip entries that are too short (likely not references)
                    if len(entry) < 20:
                        continue
                    
                    # Build a structured reference for each entry
                    ref = {
                        "text": entry,
                        "title": None,
                        "authors": None,
                        "year": None
                    }
                    
                    # Try to extract paper title - usually in quotes or italics
                    title_match = re.search(r'[""]([^""]+)[""]|["\'"]([^"\'"]+)["\'""]|[""](.*?)["""]', entry)
                    if title_match:
                        title = next(group for group in title_match.groups() if group is not None)
                        ref["title"] = title.strip()
                    
                    # Try to extract year - usually in parentheses
                    year_match = re.search(r'\((\d{4}[a-z]?)\)|\b((?:19|20)\d{2}[a-z]?)\b', entry)
                    if year_match:
                        year = next(group for group in year_match.groups() if group is not None)
                        ref["year"] = year.strip()
                    
                    # Try to extract authors - typically at the beginning of the reference
                    # This is complex due to various formats, so a simple approach is used
                    if title_match:
                        # Authors are typically before the title
                        title_start = title_match.start()
                        authors_text = entry[:title_start].strip()
                        # Remove punctuation at the end
                        authors_text = re.sub(r'[,.;:]$', '', authors_text)
                        if authors_text:
                            ref["authors"] = authors_text
                    
                    references.append(ref)
            
            # Second approach: If no references section found, try to find citations in the text
            if not references:
                # Look for common citation patterns
                # Harvard style: (Author, Year)
                harvard_citations = re.findall(r'\(([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)*),\s+(\d{4}[a-z]?)\)', text)
                # IEEE style: [number]
                ieee_citations = re.findall(r'\[(\d+)\]', text)
                
                # Process Harvard style citations
                for author, year in harvard_citations:
                    references.append({
                        "text": f"({author}, {year})",
                        "authors": author,
                        "year": year,
                        "title": None
                    })
                
                # Process IEEE style citations (limited information)
                for ref_num in ieee_citations:
                    ref_text = f"[{ref_num}]"
                    # Try to find the actual reference in the text
                    ref_pattern = rf'\[{ref_num}\]\s+([^\[\]]+?)(?=\[\d+\]|\n\n|\Z)'
                    ref_match = re.search(ref_pattern, text)
                    if ref_match:
                        ref_text = ref_match.group(1).strip()
                    
                    references.append({
                        "text": ref_text,
                        "reference_number": ref_num
                    })
            
            # Remove duplicates while preserving order
            unique_refs = []
            seen_texts = set()
            for ref in references:
                ref_text = ref["text"]
                if ref_text not in seen_texts:
                    seen_texts.add(ref_text)
                    unique_refs.append(ref)
            
            return unique_refs
        
        except Exception as e:
            logger.error(f"Reference extraction failed: {e}")
            return []

    def process_paper_with_augmentation(self, paper_metadata):
        """Process a paper with data augmentation
        
        Args:
            paper_metadata (dict): Paper metadata dictionary
            
        Returns:
            dict: Processed paper with augmentation
        """
        # First process the paper normally
        processed_paper = self.process_paper(paper_metadata)
        
        # Apply text augmentation
        if 'abstract' in processed_paper:
            processed_paper['augmented_abstract'] = self.augment_text(processed_paper['abstract'])
        
        # Extract code snippets
        if 'extracted_text' in processed_paper:
            code_snippets = self.extract_code_snippets(processed_paper['extracted_text'])
            processed_paper['code_snippets'] = code_snippets
            
            # Generate code embeddings
            if code_snippets:
                processed_paper['code_embedding'] = self.generate_code_embeddings(code_snippets).tolist()
        
        return processed_paper
    
    def process_papers_batch(self, papers_metadata, with_augmentation=True):
        """Process a batch of papers with improved parallelism
        
        Args:
            papers_metadata (list): List of paper metadata dictionaries
            with_augmentation (bool): Whether to apply data augmentation
            
        Returns:
            list: List of processed papers
        """
        if not papers_metadata:
            return []
        
        # Choose processing function based on augmentation flag
        process_func = self.process_paper_with_augmentation if with_augmentation else self.process_paper
        
        # Process in parallel using ThreadPoolExecutor
        processed_papers = []
        with ThreadPoolExecutor(max_workers=CONFIG.get('processing_max_workers', 8)) as executor:
            future_to_paper = {executor.submit(process_func, paper): paper for paper in papers_metadata}
            
            for future in as_completed(future_to_paper):
                try:
                    processed_paper = future.result()
                    if processed_paper:
                        processed_papers.append(processed_paper)
                except Exception as e:
                    logger.error(f"Error processing paper: {e}")
        
        # Build citation graph for processed papers
        self.build_citation_graph(processed_papers)
        
        # Add citation graph enrichment
        for paper in processed_papers:
            paper_id = paper.get('paper_id')
            if paper_id:
                # Add citation counts
                paper['citation_count'] = len(self.inverse_citation_graph.get(paper_id, set()))
                paper['reference_count'] = len(self.citation_graph.get(paper_id, set()))
                
                # Add citation network
                paper['cited_by'] = list(self.inverse_citation_graph.get(paper_id, set()))
                paper['cites'] = list(self.citation_graph.get(paper_id, set()))
        
        return processed_papers

    # Add new method for section extraction and classification
    def extract_sections(self, text):
        """Extract and classify sections from a scientific paper
        
        Args:
            text (str): Full text of the paper
            
        Returns:
            dict: Dictionary of section name to section text
        """
        if not text:
            return {}
            
        # First try to identify sections by common headings and patterns
        sections = {}
        
        # Split by potential section headings (patterns like "1. Introduction", "II. Methods", etc.)
        section_splits = re.split(r'\n\s*(?:\d+\.|\d+\s+|[IVX]+\.\s+|\[.*?\]\s+)?([A-Z][A-Za-z\s]+)(?:\s*\n|:)', text)
        
        # If we have a clean split with headings
        if len(section_splits) > 1:
            # Process the splits into sections
            for i in range(1, len(section_splits), 2):
                if i < len(section_splits) - 1:
                    heading = section_splits[i].strip().lower()
                    content = section_splits[i+1].strip()
                    
                    # Map the heading to standardized section names
                    for section_name, patterns in self.section_patterns.items():
                        if any(pattern in heading for pattern in patterns):
                            sections[section_name] = content
                            break
                    else:
                        # If no known section pattern matched, use the heading as is
                        sections[heading] = content
        
        # If section detection by headings failed, try paragraph-based approach
        if not sections:
            # Split into paragraphs
            paragraphs = re.split(r'\n\s*\n', text)
            
            if self.section_classifier and len(paragraphs) > 0:
                # Use transformer model to classify paragraphs into sections
                for para in paragraphs:
                    if len(para.strip()) < 50:  # Skip very short paragraphs
                        continue
                        
                    # Classify paragraph
                    with torch.no_grad():
                        inputs = self.section_classifier_tokenizer(
                            para[:512],  # Limit to model's max length
                            return_tensors="pt",
                            truncation=True,
                            padding=True
                        )
                        
                        if torch.cuda.is_available():
                            inputs = {k: v.to('cuda') for k, v in inputs.items()}
                            
                        outputs = self.section_classifier(**inputs)
                        predictions = outputs.logits.argmax(dim=-1)
                        section_name = list(self.section_patterns.keys())[predictions.item()]
                        
                        # Append to section if it exists, otherwise create it
                        if section_name in sections:
                            sections[section_name] += " " + para
                        else:
                            sections[section_name] = para
            else:
                # Fallback to simple heuristics if classifier isn't available
                # Abstract is usually first
                if paragraphs and len(paragraphs) > 0:
                    sections["abstract"] = paragraphs[0]
                    
                # Last portion often contains references
                if paragraphs and len(paragraphs) > 2:
                    sections["references"] = paragraphs[-1]
                    
                # Middle paragraphs are likely methodology and results
                if paragraphs and len(paragraphs) > 3:
                    middle_idx = len(paragraphs) // 2
                    sections["methodology"] = " ".join(paragraphs[1:middle_idx])
                    sections["results"] = " ".join(paragraphs[middle_idx:-1])
        
        return sections
    
    # Add method for handling multiple document formats
    def extract_text_from_document(self, filepath):
        """Extract text from various document formats (PDF, DOCX, HTML, TXT)
        
        Args:
            filepath (str): Path to the document
            
        Returns:
            str: Extracted text
        """
        if not os.path.exists(filepath):
            logger.warning(f"Document file does not exist: {filepath}")
            return ""
            
        # Determine file type based on extension
        file_ext = os.path.splitext(filepath)[1].lower()
        
        try:
            if file_ext == '.pdf':
                return self._extract_text_from_pdf(filepath)
            elif file_ext == '.docx':
                # Use python-docx for DOCX files
                try:
                    import docx
                    doc = docx.Document(filepath)
                    return '\n\n'.join([para.text for para in doc.paragraphs])
                except ImportError:
                    logger.warning("python-docx not installed. Cannot process DOCX files.")
                    return ""
            elif file_ext == '.html' or file_ext == '.htm':
                # Use BeautifulSoup for HTML files
                try:
                    from bs4 import BeautifulSoup
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.extract()
                        # Get text
                        text = soup.get_text()
                        # Break into lines and remove leading/trailing space
                        lines = (line.strip() for line in text.splitlines())
                        # Break multi-headlines into a line each
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        # Drop blank lines
                        text = '\n'.join(chunk for chunk in chunks if chunk)
                        return text
                except ImportError:
                    logger.warning("BeautifulSoup not installed. Cannot process HTML files.")
                    return ""
            elif file_ext == '.txt':
                # Simple text files
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            else:
                logger.warning(f"Unsupported file format: {file_ext}")
                return ""
        except Exception as e:
            logger.error(f"Error extracting text from {filepath}: {e}")
            return ""