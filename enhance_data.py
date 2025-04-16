#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to enhance training data with better balance, synthetic examples,
and improved test sets for better evaluation.
"""

import os
import sys
import json
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/data_enhancement.log"),
        logging.StreamHandler()
    ]
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

def analyze_dataset(papers):
    """Analyze dataset for imbalances and quality issues."""
    # Check domain distribution
    domain_counts = defaultdict(int)
    
    # Check class distribution for relevance
    relevance_counts = defaultdict(int)
    
    # Check impact score distribution
    impact_scores = []
    
    # Track missing or low-quality data
    missing_abstract = 0
    short_abstract = 0
    missing_title = 0
    duplicate_titles = set()
    titles_seen = set()
    
    for paper in papers:
        # Domain counts
        domain_id = paper.get("domain_id", 0)
        domain_counts[domain_id] += 1
        
        # Relevance class counts
        relevance = paper.get("is_primary_domain", 0)
        relevance_counts[relevance] += 1
        
        # Impact scores
        if "impact_score" in paper:
            impact_scores.append(paper["impact_score"])
        
        # Data quality checks
        abstract = paper.get("abstract", "")
        
        # Handle case where abstract is not a string
        if not isinstance(abstract, str):
            # Try to convert list to string if possible
            if isinstance(abstract, list):
                try:
                    abstract = " ".join(str(item) for item in abstract)
                    # Update the paper with the converted abstract
                    paper["abstract"] = abstract
                except:
                    abstract = ""
            else:
                abstract = ""
        
        if not abstract:
            missing_abstract += 1
        elif len(abstract.split()) < 20:
            short_abstract += 1
            
        title = paper.get("title", "")
        # Handle case where title is not a string
        if not isinstance(title, str):
            if isinstance(title, list):
                try:
                    title = " ".join(str(item) for item in title)
                    # Update the paper with the converted title
                    paper["title"] = title
                except:
                    title = ""
            else:
                title = ""
                
        if not title:
            missing_title += 1
        elif title in titles_seen:
            duplicate_titles.add(title)
        else:
            titles_seen.add(title)
    
    # Output analysis
    logger.info("=== Dataset Analysis ===")
    
    logger.info("Domain distribution:")
    for domain_id, count in sorted(domain_counts.items()):
        percentage = 100 * count / len(papers)
        logger.info(f"  - Domain {domain_id}: {count} papers ({percentage:.1f}%)")
    
    logger.info("Relevance class distribution:")
    for relevance, count in sorted(relevance_counts.items()):
        percentage = 100 * count / len(papers)
        logger.info(f"  - Relevance {relevance}: {count} papers ({percentage:.1f}%)")
    
    if impact_scores:
        impact_scores.sort()
        min_impact = min(impact_scores)
        max_impact = max(impact_scores)
        avg_impact = sum(impact_scores) / len(impact_scores)
        logger.info(f"Impact score range: {min_impact:.2f} to {max_impact:.2f}, avg: {avg_impact:.2f}")
    
    logger.info("Data quality issues:")
    logger.info(f"  - Missing abstracts: {missing_abstract} ({100*missing_abstract/len(papers):.1f}%)")
    logger.info(f"  - Short abstracts: {short_abstract} ({100*short_abstract/len(papers):.1f}%)")
    logger.info(f"  - Missing titles: {missing_title} ({100*missing_title/len(papers):.1f}%)")
    logger.info(f"  - Duplicate titles: {len(duplicate_titles)} ({100*len(duplicate_titles)/len(papers):.1f}%)")
    
    return {
        "domain_counts": domain_counts,
        "relevance_counts": relevance_counts,
        "impact_scores": impact_scores,
        "quality_issues": {
            "missing_abstract": missing_abstract,
            "short_abstract": short_abstract,
            "missing_title": missing_title,
            "duplicate_titles": len(duplicate_titles)
        }
    }

def clean_data(papers):
    """Clean dataset by removing or fixing quality issues."""
    original_count = len(papers)
    
    # Filter out papers with critical problems
    filtered_papers = []
    for paper in papers:
        # Get abstract and handle non-string case
        abstract = paper.get("abstract", "")
        if not isinstance(abstract, str):
            if isinstance(abstract, list):
                try:
                    abstract = " ".join(str(item) for item in abstract)
                    paper["abstract"] = abstract
                except:
                    abstract = ""
            else:
                abstract = ""
                
        # Get title and handle non-string case
        title = paper.get("title", "")
        if not isinstance(title, str):
            if isinstance(title, list):
                try:
                    title = " ".join(str(item) for item in title)
                    paper["title"] = title
                except:
                    title = ""
            else:
                title = ""
                
        # Skip papers with no title or no abstract
        if not title or not abstract:
            continue
        
        # Skip papers with very short abstracts (likely incomplete)
        if len(abstract.split()) < 10:
            continue
        
        # Ensure all papers have required fields
        paper_id = paper.get("id")
        if not paper_id:
            paper["id"] = f"paper_{random.randint(10000, 99999)}"
        
        # Ensure domain_id exists
        if "domain_id" not in paper:
            paper["domain_id"] = 0
        
        # Ensure impact_score exists
        if "impact_score" not in paper:
            paper["impact_score"] = 0.5  # Default to middle value
        
        # Ensure is_primary_domain exists (for relevance)
        if "is_primary_domain" not in paper:
            paper["is_primary_domain"] = 1  # Default to relevant
        
        # Add cleaned paper
        filtered_papers.append(paper)
    
    removed_count = original_count - len(filtered_papers)
    logger.info(f"Removed {removed_count} papers with quality issues ({100*removed_count/original_count:.1f}%)")
    
    return filtered_papers

def balance_domains(papers, target_per_domain=None):
    """Balance papers across domains."""
    # Group papers by domain
    papers_by_domain = defaultdict(list)
    for paper in papers:
        domain_id = paper.get("domain_id", 0)
        papers_by_domain[domain_id].append(paper)
    
    domain_counts = {domain: len(papers) for domain, papers in papers_by_domain.items()}
    logger.info("Original domain distribution:")
    for domain, count in sorted(domain_counts.items()):
        logger.info(f"  - Domain {domain}: {count} papers")
    
    # Determine target count per domain
    if target_per_domain is None:
        min_count = min(domain_counts.values())
        target_per_domain = min(min_count, 1000)  # Cap at 1000 papers per domain
    
    # Balance domains by sampling
    balanced_papers = []
    for domain, domain_papers in papers_by_domain.items():
        # If fewer papers than target, duplicate papers to reach target
        if len(domain_papers) < target_per_domain:
            # Add all existing papers
            sample = domain_papers.copy()
            
            # Duplicate papers randomly until reaching target
            while len(sample) < target_per_domain:
                additional = random.choice(domain_papers)
                # Create a shallow copy and assign a new ID to avoid duplication
                additional_copy = additional.copy()
                additional_copy["id"] = f"{additional.get('id', 'paper')}_dup_{len(sample)}"
                sample.append(additional_copy)
        else:
            # Randomly sample to reach target
            sample = random.sample(domain_papers, target_per_domain)
        
        balanced_papers.extend(sample)
    
    logger.info(f"Created balanced domain distribution with {target_per_domain} papers per domain")
    logger.info(f"New dataset size: {len(balanced_papers)} papers")
    
    return balanced_papers

def balance_relevance_classes(papers, target_ratio=0.5):
    """Balance papers for relevance classification (primary vs non-primary domain).
    
    Args:
        papers: List of paper dictionaries
        target_ratio: Target ratio of relevant papers (1s) in final dataset
                      1.0 means all relevant, 0.0 means all non-relevant
    """
    # Group papers by relevance class
    papers_by_relevance = defaultdict(list)
    for paper in papers:
        relevance = paper.get("is_primary_domain", 1)  # Default to relevant
        papers_by_relevance[relevance].append(paper)
    
    # Count papers in each class
    relevant_count = len(papers_by_relevance.get(1, []))
    non_relevant_count = len(papers_by_relevance.get(0, []))
    
    logger.info("Original relevance distribution:")
    logger.info(f"  - Relevant (1): {relevant_count} papers ({100*relevant_count/(relevant_count+non_relevant_count):.1f}%)")
    logger.info(f"  - Non-relevant (0): {non_relevant_count} papers ({100*non_relevant_count/(relevant_count+non_relevant_count):.1f}%)")
    
    # If no non-relevant papers, create some by changing the relevance flag
    if non_relevant_count == 0:
        logger.info("No non-relevant papers found, creating synthetic examples")
        
        # Take a subset of papers and make them non-relevant
        relevant_papers = papers_by_relevance.get(1, [])
        num_to_convert = int(len(relevant_papers) * (1 - target_ratio))
        
        papers_to_convert = random.sample(relevant_papers, num_to_convert)
        for paper in papers_to_convert:
            # Create a copy to avoid modifying the original
            paper_copy = paper.copy()
            paper_copy["is_primary_domain"] = 0
            paper_copy["id"] = f"{paper.get('id', 'paper')}_nonrel"
            papers_by_relevance[0].append(paper_copy)
        
        non_relevant_count = len(papers_by_relevance[0])
        logger.info(f"Created {non_relevant_count} non-relevant papers")
    
    # Calculate how many of each class we need
    total_count = relevant_count + non_relevant_count
    target_relevant_count = int(total_count * target_ratio)
    target_non_relevant_count = total_count - target_relevant_count
    
    balanced_papers = []
    
    # Adjust relevant papers
    if relevant_count > target_relevant_count:
        # Sample down relevant papers
        balanced_papers.extend(random.sample(papers_by_relevance[1], target_relevant_count))
    else:
        # Use all relevant papers and potentially duplicate some
        balanced_papers.extend(papers_by_relevance[1])
        if relevant_count < target_relevant_count:
            # Duplicate papers randomly until reaching target
            additional_needed = target_relevant_count - relevant_count
            for i in range(additional_needed):
                additional = random.choice(papers_by_relevance[1])
                additional_copy = additional.copy()
                additional_copy["id"] = f"{additional.get('id', 'paper')}_dup_rel_{i}"
                balanced_papers.append(additional_copy)
    
    # Adjust non-relevant papers
    if non_relevant_count > target_non_relevant_count:
        # Sample down non-relevant papers
        balanced_papers.extend(random.sample(papers_by_relevance[0], target_non_relevant_count))
    else:
        # Use all non-relevant papers and potentially duplicate some
        balanced_papers.extend(papers_by_relevance[0])
        if non_relevant_count < target_non_relevant_count:
            # Duplicate papers randomly until reaching target
            additional_needed = target_non_relevant_count - non_relevant_count
            for i in range(additional_needed):
                additional = random.choice(papers_by_relevance[0])
                additional_copy = additional.copy()
                additional_copy["id"] = f"{additional.get('id', 'paper')}_dup_nonrel_{i}"
                balanced_papers.append(additional_copy)
    
    logger.info("New relevance distribution:")
    new_relevant = sum(1 for p in balanced_papers if p.get("is_primary_domain", 1) == 1)
    new_non_relevant = len(balanced_papers) - new_relevant
    logger.info(f"  - Relevant (1): {new_relevant} papers ({100*new_relevant/len(balanced_papers):.1f}%)")
    logger.info(f"  - Non-relevant (0): {new_non_relevant} papers ({100*new_non_relevant/len(balanced_papers):.1f}%)")
    
    return balanced_papers

def normalize_impact_scores(papers):
    """Normalize impact scores to a more standard distribution."""
    # Extract current impact scores
    scores = [paper.get("impact_score", 0.5) for paper in papers]
    
    # Skip if no scores
    if not scores:
        return papers
    
    # Calculate statistics
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    
    if score_range < 0.001:
        logger.warning("Impact scores have almost no variation, applying synthetic distribution")
        # Create a synthetic distribution
        for paper in papers:
            paper["impact_score"] = random.uniform(0.0, 1.0)
        return papers
    
    # Normalize scores to [0, 1] range
    normalized_papers = []
    for paper in papers:
        paper_copy = paper.copy()
        old_score = paper.get("impact_score", 0.5)
        new_score = (old_score - min_score) / score_range
        paper_copy["impact_score"] = new_score
        normalized_papers.append(paper_copy)
    
    logger.info(f"Normalized impact scores from range [{min_score:.3f}, {max_score:.3f}] to [0, 1]")
    
    return normalized_papers

def create_evaluation_sets(papers, eval_set_size=200):
    """Create specialized evaluation sets for testing specific aspects."""
    random.shuffle(papers)
    
    eval_sets = {}
    
    # Domain-focused evaluation set (balanced domains)
    papers_by_domain = defaultdict(list)
    for paper in papers:
        domain_id = paper.get("domain_id", 0)
        papers_by_domain[domain_id].append(paper)
    
    domain_eval_papers = []
    papers_per_domain = eval_set_size // len(papers_by_domain)
    
    for domain, domain_papers in papers_by_domain.items():
        if len(domain_papers) > papers_per_domain:
            domain_eval_papers.extend(random.sample(domain_papers, papers_per_domain))
        else:
            domain_eval_papers.extend(domain_papers)
    
    eval_sets["domain_balanced"] = domain_eval_papers
    
    # Relevance-focused evaluation sets with different ratios
    relevance_ratios = {
        "relevance_balanced_50_50": 0.5,   # 50% relevant, 50% non-relevant
        "relevance_balanced_70_30": 0.7,   # 70% relevant, 30% non-relevant
        "relevance_balanced_90_10": 0.9,   # 90% relevant, 10% non-relevant
        "relevance_balanced_30_70": 0.3,   # 30% relevant, 70% non-relevant
    }
    
    papers_by_relevance = defaultdict(list)
    for paper in papers:
        relevance = paper.get("is_primary_domain", 1)
        papers_by_relevance[relevance].append(paper)
    
    for set_name, ratio in relevance_ratios.items():
        relevant_count = int(eval_set_size * ratio)
        non_relevant_count = eval_set_size - relevant_count
        
        relevance_eval_papers = []
        
        # Add relevant papers
        if len(papers_by_relevance[1]) > relevant_count:
            relevance_eval_papers.extend(random.sample(papers_by_relevance[1], relevant_count))
        else:
            relevance_eval_papers.extend(papers_by_relevance[1])
            logger.warning(f"Not enough relevant papers for {set_name}, using all {len(papers_by_relevance[1])} available")
        
        # Add non-relevant papers
        if len(papers_by_relevance[0]) > non_relevant_count:
            relevance_eval_papers.extend(random.sample(papers_by_relevance[0], non_relevant_count))
        else:
            relevance_eval_papers.extend(papers_by_relevance[0])
            logger.warning(f"Not enough non-relevant papers for {set_name}, using all {len(papers_by_relevance[0])} available")
        
        eval_sets[set_name] = relevance_eval_papers
        logger.info(f"Created {set_name} with {len(relevance_eval_papers)} papers ({ratio*100:.0f}% relevant)")
    
    # Impact score evaluation set (stratified)
    if len(papers) > 0 and "impact_score" in papers[0]:
        # Sort by impact score
        sorted_papers = sorted(papers, key=lambda p: p.get("impact_score", 0))
        
        # Create buckets
        num_buckets = 5
        bucket_size = len(sorted_papers) // num_buckets
        impact_eval_papers = []
        
        for i in range(num_buckets):
            start_idx = i * bucket_size
            end_idx = start_idx + bucket_size
            bucket_papers = sorted_papers[start_idx:end_idx]
            papers_from_bucket = eval_set_size // num_buckets
            
            if len(bucket_papers) > papers_from_bucket:
                impact_eval_papers.extend(random.sample(bucket_papers, papers_from_bucket))
            else:
                impact_eval_papers.extend(bucket_papers)
        
        eval_sets["impact_stratified"] = impact_eval_papers
    
    return eval_sets

def save_dataset(papers, file_path):
    """Save papers to a JSON file."""
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
    """Main function to enhance data."""
    parser = argparse.ArgumentParser(description="Enhance training data for better model performance")
    
    parser.add_argument("--input_dir", type=str, default="data/comprehensive",
                       help="Directory with comprehensive training data")
    parser.add_argument("--output_dir", type=str, default="data/enhanced",
                       help="Output directory for enhanced datasets")
    parser.add_argument("--papers_per_domain", type=int, default=200,
                       help="Target number of papers per domain")
    parser.add_argument("--relevance_ratio", type=float, default=0.7,
                       help="Target ratio of relevant to non-relevant papers (default: 0.7 = 70% relevant)")
    parser.add_argument("--skip_normalization", action="store_true",
                       help="Skip impact score normalization")
    parser.add_argument("--skip_cleaning", action="store_true",
                       help="Skip data cleaning")
    parser.add_argument("--analysis_only", action="store_true",
                       help="Only analyze data without creating new datasets")
    
    args = parser.parse_args()
    
    # Create log directory
    os.makedirs("logs", exist_ok=True)
    
    # Set base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the comprehensive dataset
    train_path = os.path.join(base_dir, args.input_dir, "train.json")
    val_path = os.path.join(base_dir, args.input_dir, "val.json")
    test_path = os.path.join(base_dir, args.input_dir, "test.json")
    
    train_papers = load_papers(train_path)
    val_papers = load_papers(val_path)
    test_papers = load_papers(test_path)
    
    # Analyze datasets
    logger.info("Analyzing training dataset...")
    train_analysis = analyze_dataset(train_papers)
    
    logger.info("\nAnalyzing validation dataset...")
    val_analysis = analyze_dataset(val_papers)
    
    logger.info("\nAnalyzing test dataset...")
    test_analysis = analyze_dataset(test_papers)
    
    if args.analysis_only:
        logger.info("Analysis completed. Exiting as requested.")
        return
    
    # Process the training data
    enhanced_train = train_papers
    
    if not args.skip_cleaning:
        logger.info("Cleaning training data...")
        enhanced_train = clean_data(enhanced_train)
    
    logger.info("Balancing domains...")
    enhanced_train = balance_domains(enhanced_train, args.papers_per_domain)
    
    logger.info(f"Balancing relevance classes to {args.relevance_ratio:.1f} ratio...")
    enhanced_train = balance_relevance_classes(enhanced_train, target_ratio=args.relevance_ratio)
    
    if not args.skip_normalization:
        logger.info("Normalizing impact scores...")
        enhanced_train = normalize_impact_scores(enhanced_train)
    
    # Create specialized evaluation sets
    logger.info("Creating specialized evaluation sets...")
    eval_sets = create_evaluation_sets(val_papers + test_papers)
    
    # Save enhanced datasets
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    save_dataset(enhanced_train, os.path.join(output_dir, "train.json"))
    save_dataset(val_papers, os.path.join(output_dir, "val.json"))  # Keep original validation set
    save_dataset(test_papers, os.path.join(output_dir, "test.json"))  # Keep original test set
    
    # Save specialized evaluation sets
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    for eval_name, eval_papers in eval_sets.items():
        save_dataset(eval_papers, os.path.join(eval_dir, f"{eval_name}.json"))
    
    # Print training commands
    print("\n" + "="*50)
    print("DATA ENHANCEMENT COMPLETE")
    print("="*50)
    print("\nTo train with the enhanced dataset, run:")
    print(f"python train.py --train_data {os.path.join(args.output_dir, 'train.json')} --val_data {os.path.join(args.output_dir, 'val.json')} --test_data {os.path.join(args.output_dir, 'test.json')} --domain_file data/domains/domain_mapping.json --output_dir outputs/enhanced --batch_size 16 --num_epochs 40 --learning_rate 2e-6 --gradient_accumulation_steps 8 --fp16")
    
    print("\nFor specialized evaluation, use:")
    for eval_name in eval_sets.keys():
        print(f"python evaluate.py --model_dir outputs/enhanced --eval_data {os.path.join(args.output_dir, 'eval', f'{eval_name}.json')} --output_file results_{eval_name}.json")

if __name__ == "__main__":
    main() 