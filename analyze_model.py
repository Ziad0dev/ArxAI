#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to analyze model results and understand what the model has learned.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure matplotlib for better plots
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_results(results_file):
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)

def analyze_metrics(results):
    """Analyze the evaluation metrics."""
    metrics = results.get('test_metrics', {})
    
    print("\n===== MODEL PERFORMANCE =====")
    print(f"Overall Loss: {metrics.get('eval/loss', 'N/A'):.4f}")
    print(f"Impact Score Loss: {metrics.get('eval/loss_score', 'N/A'):.4f}")
    print(f"Domain Relevance Loss: {metrics.get('eval/loss_relevance', 'N/A'):.4f}")
    print(f"Mean Absolute Error (Impact): {metrics.get('eval/mae', 'N/A'):.4f}")
    print(f"Domain Classification Accuracy: {metrics.get('eval/accuracy', 'N/A'):.4f}")
    
    # Create a bar chart of the metrics
    metric_keys = ['eval/loss', 'eval/loss_score', 'eval/loss_relevance', 'eval/mae']
    metric_values = [metrics.get(k, 0) for k in metric_keys]
    metric_labels = ['Overall Loss', 'Impact Loss', 'Relevance Loss', 'MAE']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_labels, metric_values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom')
    
    plt.title('Model Performance Metrics')
    plt.ylabel('Value')
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('outputs/analysis', exist_ok=True)
    plt.savefig('outputs/analysis/model_metrics.png')
    print(f"\nMetrics visualization saved to outputs/analysis/model_metrics.png")

def analyze_training_params(results):
    """Analyze the training parameters."""
    args = results.get('args', {})
    
    print("\n===== TRAINING PARAMETERS =====")
    print(f"Model: {args.get('model_name', 'N/A')}")
    print(f"Batch Size: {args.get('batch_size', 'N/A')}")
    print(f"Learning Rate: {args.get('learning_rate', 'N/A')}")
    print(f"Number of Epochs: {args.get('num_epochs', 'N/A')}")
    print(f"Max Sequence Length: {args.get('max_seq_length', 'N/A')}")
    print(f"Model Dimensions: d_model={args.get('d_model', 'N/A')}, n_heads={args.get('n_heads', 'N/A')}, n_layers={args.get('n_layers', 'N/A')}")
    
    # Create a table visualization of important hyperparameters
    param_names = ['Learning Rate', 'Batch Size', 'Epochs', 'Sequence Length', 'Dropout']
    param_values = [
        args.get('learning_rate', 'N/A'),
        args.get('batch_size', 'N/A'),
        args.get('num_epochs', 'N/A'),
        args.get('max_seq_length', 'N/A'),
        args.get('dropout', 'N/A')
    ]
    
    plt.figure(figsize=(10, 6))
    # Create a table plot
    table = plt.table(
        cellText=[[str(v)] for v in param_values],
        rowLabels=param_names,
        colLabels=['Value'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    plt.axis('off')
    plt.title('Training Hyperparameters')
    plt.tight_layout()
    
    plt.savefig('outputs/analysis/training_params.png')
    print(f"Training parameters visualization saved to outputs/analysis/training_params.png")

def visualize_test_papers():
    """Visualize test papers by domain to see what the model was trying to learn."""
    try:
        # Load test papers
        with open('data/papers_test.json', 'r') as f:
            papers = json.load(f)
        
        # Load domain mapping
        with open('data/domains/domain_mapping.json', 'r') as f:
            domain_mapping = json.load(f)
        
        # Create a mapping from domain id to name
        domain_names = {domain['id']: domain['name'] for domain in domain_mapping.get('domains', [])}
        
        # Group papers by domain
        domain_groups = {}
        for paper in papers:
            domain_id = paper.get('domain_id', 0)
            domain_name = domain_names.get(domain_id, f"Unknown ({domain_id})")
            
            if domain_name not in domain_groups:
                domain_groups[domain_name] = []
            
            domain_groups[domain_name].append(paper)
        
        # Print statistics
        print("\n===== TEST PAPERS STATISTICS =====")
        print(f"Total test papers: {len(papers)}")
        
        for domain, papers_list in domain_groups.items():
            impact_scores = [p.get('impact_score', 0) for p in papers_list]
            avg_impact = sum(impact_scores) / len(impact_scores) if impact_scores else 0
            print(f"Domain {domain}: {len(papers_list)} papers, avg impact: {avg_impact:.2f}")
        
        # Create a pie chart of papers by domain
        domain_counts = {domain: len(papers_list) for domain, papers_list in domain_groups.items()}
        plt.figure(figsize=(10, 8))
        plt.pie(
            domain_counts.values(),
            labels=domain_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
        )
        plt.axis('equal')
        plt.title('Test Papers by Domain')
        plt.tight_layout()
        
        plt.savefig('outputs/analysis/papers_by_domain.png')
        print(f"Test papers visualization saved to outputs/analysis/papers_by_domain.png")
        
        # Create a bar chart of average impact by domain
        domain_impacts = {domain: sum(p.get('impact_score', 0) for p in papers_list) / len(papers_list) 
                          if papers_list else 0 
                          for domain, papers_list in domain_groups.items()}
        
        plt.figure(figsize=(10, 6))
        domains = list(domain_impacts.keys())
        impacts = list(domain_impacts.values())
        
        bars = plt.bar(domains, impacts, color='#3498db')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title('Average Impact Score by Domain')
        plt.ylabel('Impact Score')
        plt.ylim(0, max(impacts) * 1.2)  # Add some headroom for labels
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig('outputs/analysis/impact_by_domain.png')
        print(f"Impact score visualization saved to outputs/analysis/impact_by_domain.png")
        
    except Exception as e:
        print(f"Error visualizing test papers: {e}")

def main():
    """Main analysis function."""
    # Set base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    # Load results
    results_file = os.path.join('outputs/large_run/final_results.json')
    if not os.path.exists(results_file):
        print(f"Results file not found at {results_file}")
        return
    
    results = load_results(results_file)
    
    # Analyze metrics
    analyze_metrics(results)
    
    # Analyze training parameters
    analyze_training_params(results)
    
    # Visualize test papers
    visualize_test_papers()
    
    print("\nAnalysis complete. Visualizations saved to outputs/analysis/ directory.")

if __name__ == "__main__":
    main() 