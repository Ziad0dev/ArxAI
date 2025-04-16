#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to create experimental balanced datasets for training.
"""

import os
import json
import random
import logging
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_papers(file_path):
    """Load papers from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            papers = json.load(f)
        logger.info(f"Loaded {len(papers)} papers from {file_path}")
        return papers
    except Exception as e:
        logger.error(f"Error loading papers from {file_path}: {e}")
        return []

def analyze_papers(papers):
    """Analyze papers for domain distribution and impact scores."""
    # Count papers by domain
    domain_counts = defaultdict(int)
    domain_names = {}
    impact_scores = []
    
    for paper in papers:
        domain_id = paper.get("domain_id", 0)
        domain_counts[domain_id] += 1
        
        # Store domain name if available
        if "domain_name" in paper:
            domain_names[domain_id] = paper["domain_name"]
        
        # Collect impact scores
        if "impact_score" in paper:
            impact_scores.append(paper["impact_score"])
    
    # Output analysis
    logger.info("Paper domain distribution:")
    for domain_id, count in domain_counts.items():
        domain_name = domain_names.get(domain_id, f"Domain {domain_id}")
        logger.info(f"  - {domain_name} (ID: {domain_id}): {count} papers")
    
    # Impact score statistics if available
    if impact_scores:
        impact_scores.sort()
        logger.info(f"Impact score range: {min(impact_scores):.2f} to {max(impact_scores):.2f}")
        logger.info(f"Impact score average: {sum(impact_scores)/len(impact_scores):.2f}")
    
    return {
        "domain_counts": domain_counts,
        "impact_scores": impact_scores
    }

def create_balanced_domain_dataset(papers, target_size=150):
    """Create a dataset with balanced domain representation."""
    # Group papers by domain
    papers_by_domain = defaultdict(list)
    
    for paper in papers:
        domain_id = paper.get("domain_id", 0)
        papers_by_domain[domain_id].append(paper)
    
    # Calculate how many papers to include from each domain
    num_domains = len(papers_by_domain)
    papers_per_domain = target_size // num_domains
    
    logger.info(f"Creating balanced domain dataset with {papers_per_domain} papers per domain")
    
    # Select papers from each domain
    balanced_papers = []
    for domain_id, domain_papers in papers_by_domain.items():
        # Shuffle papers in this domain
        random.shuffle(domain_papers)
        # Take at most papers_per_domain papers
        selected = domain_papers[:papers_per_domain]
        balanced_papers.extend(selected)
        logger.info(f"  - Selected {len(selected)} papers from domain {domain_id}")
    
    # Shuffle the final dataset
    random.shuffle(balanced_papers)
    
    return balanced_papers

def create_varied_impact_dataset(papers, target_size=150):
    """Create a dataset with varied impact scores."""
    # Filter papers with impact scores
    papers_with_impact = [p for p in papers if "impact_score" in p]
    
    # Sort by impact score
    papers_with_impact.sort(key=lambda p: p.get("impact_score", 0))
    
    # Divide into low, medium, and high impact
    total_papers = len(papers_with_impact)
    third = total_papers // 3
    
    low_impact = papers_with_impact[:third]
    medium_impact = papers_with_impact[third:2*third]
    high_impact = papers_with_impact[2*third:]
    
    # Take equal numbers from each group
    papers_per_group = target_size // 3
    logger.info(f"Creating varied impact dataset with {papers_per_group} papers per impact level")
    
    # Randomly select from each group
    random.shuffle(low_impact)
    random.shuffle(medium_impact)
    random.shuffle(high_impact)
    
    varied_papers = []
    varied_papers.extend(low_impact[:papers_per_group])
    varied_papers.extend(medium_impact[:papers_per_group])
    varied_papers.extend(high_impact[:papers_per_group])
    
    # Shuffle the final dataset
    random.shuffle(varied_papers)
    
    return varied_papers

def create_mixed_sources_dataset(train_papers, val_papers, test_papers=None, kb_papers=None, target_size=200):
    """Create a dataset mixing papers from different sources."""
    all_sources = [
        ("train", train_papers),
        ("val", val_papers)
    ]
    
    if test_papers:
        all_sources.append(("test", test_papers))
    
    if kb_papers:
        all_sources.append(("kb", kb_papers))
    
    # Calculate papers per source
    num_sources = len(all_sources)
    papers_per_source = target_size // num_sources
    
    logger.info(f"Creating mixed sources dataset with {papers_per_source} papers per source")
    
    # Select papers from each source
    mixed_papers = []
    for source_name, source_papers in all_sources:
        # Shuffle papers from this source
        random.shuffle(source_papers)
        # Take at most papers_per_source papers
        selected = source_papers[:papers_per_source]
        mixed_papers.extend(selected)
        logger.info(f"  - Selected {len(selected)} papers from {source_name}")
    
    # Shuffle the final dataset
    random.shuffle(mixed_papers)
    
    return mixed_papers

def save_dataset(papers, file_path):
    """Save papers to a JSON file."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(papers, f, indent=2)
        logger.info(f"Saved {len(papers)} papers to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving papers to {file_path}: {e}")
        return False

def main():
    """Main function to create experimental datasets."""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load existing datasets
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_papers = load_papers(os.path.join(base_dir, "data/papers_train.json"))
    val_papers = load_papers(os.path.join(base_dir, "data/papers_val.json"))
    test_papers = load_papers(os.path.join(base_dir, "data/papers_test.json"))
    
    # Try to load knowledge base data if available
    kb_papers = []
    kb_path = os.path.join(base_dir, "data/knowledge_base.json")
    if os.path.exists(kb_path):
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                kb_data = json.load(f)
            
            if isinstance(kb_data, dict):
                for paper_id, paper_data in kb_data.items():
                    if isinstance(paper_data, dict):
                        paper = {
                            "id": paper_id,
                            "title": paper_data.get("title", ""),
                            "abstract": paper_data.get("abstract", ""),
                            "domain_id": paper_data.get("domain_id", 0),
                            "impact_score": paper_data.get("impact_score", 0.5)
                        }
                        kb_papers.append(paper)
            
            logger.info(f"Loaded {len(kb_papers)} papers from knowledge base")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
    
    # Combine all papers for analysis and selection
    all_papers = train_papers + val_papers + test_papers + kb_papers
    logger.info(f"Combined {len(all_papers)} papers from all sources")
    
    # Analyze papers
    logger.info("Analyzing paper distribution...")
    analysis = analyze_papers(all_papers)
    
    # Create balanced domain dataset
    balanced_papers = create_balanced_domain_dataset(all_papers)
    save_dataset(balanced_papers, os.path.join(base_dir, "data/experiments/balanced_domains.json"))
    
    # Create varied impact dataset
    varied_papers = create_varied_impact_dataset(all_papers)
    save_dataset(varied_papers, os.path.join(base_dir, "data/experiments/varied_impact.json"))
    
    # Create mixed sources dataset
    mixed_papers = create_mixed_sources_dataset(train_papers, val_papers, test_papers, kb_papers)
    save_dataset(mixed_papers, os.path.join(base_dir, "data/experiments/mixed_sources.json"))
    
    # Print training command
    print("\n" + "="*50)
    print("DATASETS CREATED SUCCESSFULLY")
    print("="*50)
    print("\nTo train with the balanced domain dataset, run:")
    print("python train.py --train_data data/experiments/balanced_domains.json --val_data data/papers_val.json --test_data data/papers_test.json --domain_file data/domains/domain_mapping.json --output_dir outputs/balanced_test --batch_size 16 --num_epochs 5 --fp16 --logging_steps 5")
    
    print("\nTo train with the varied impact dataset, run:")
    print("python train.py --train_data data/experiments/varied_impact.json --val_data data/papers_val.json --test_data data/papers_test.json --domain_file data/domains/domain_mapping.json --output_dir outputs/varied_test --batch_size 16 --num_epochs 5 --fp16 --logging_steps 5")
    
    # Display absolute paths to the created files
    print("\nAbsolute paths to created datasets:")
    print(f"1. Balanced domains: {os.path.abspath(os.path.join(base_dir, 'data/experiments/balanced_domains.json'))}")
    print(f"2. Varied impact: {os.path.abspath(os.path.join(base_dir, 'data/experiments/varied_impact.json'))}")
    print(f"3. Mixed sources: {os.path.abspath(os.path.join(base_dir, 'data/experiments/mixed_sources.json'))}")

if __name__ == "__main__":
    main() 