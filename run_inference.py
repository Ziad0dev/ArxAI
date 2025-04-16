#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script for loading and testing the trained Research Transformer model.
"""

import os
import json
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from transformers import AutoTokenizer, AutoModel

# We'll define this function locally instead of importing
def load_domain_mapping(domain_file):
    """Load domain mapping from file."""
    try:
        with open(domain_file, 'r', encoding='utf-8') as f:
            domain_mapping = json.load(f)
        
        logger.info(f"Loaded domain mapping with {len(domain_mapping.get('domains', []))} domains")
        return domain_mapping
    except Exception as e:
        logger.warning(f"Error loading domain mapping from {domain_file}: {e}")
        # Return a default domain mapping if the file can't be loaded
        return {
            "domains": [
                {"id": 0, "name": "Computer Science"},
                {"id": 1, "name": "Physics"},
                {"id": 2, "name": "Mathematics"}
            ]
        }

# Create a simple ResearchTransformer class for inference
class ResearchTransformer(torch.nn.Module):
    """Research Transformer model for inference."""
    def __init__(self, 
                 model_name,
                 d_model=768,
                 n_heads=12,
                 n_layers=6,
                 d_ff=3072,
                 dropout=0.1,
                 num_domains=10):
        super().__init__()
        
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Embedding projection
        self.embedding_projection = torch.nn.Linear(d_model, d_model)
        
        # Domain embedding
        self.domain_embedding = torch.nn.Embedding(num_domains, d_model)
        
        # Task-specific heads
        self.impact_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model // 2, 1)
        )
        
        self.domain_relevance_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model // 2, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, domain_id=None):
        # Get base embeddings
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use CLS token as sequence representation
        sequence_output = base_outputs.last_hidden_state[:, 0, :]
        
        # Apply embedding projection
        embedding = self.embedding_projection(sequence_output)
        
        # Add domain embedding if provided
        if domain_id is not None:
            domain_embed = self.domain_embedding(domain_id)
            embedding = embedding + domain_embed
        
        # Calculate impact score
        impact_score = self.impact_head(embedding)
        
        # Calculate domain relevance
        domain_relevance = self.domain_relevance_head(embedding)
        
        return {
            "embedding": embedding,
            "impact_score": impact_score,
            "domain_relevance": domain_relevance
        }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def load_model(checkpoint_dir):
    """Load the trained model from checkpoint directory."""
    # Determine model path
    if os.path.exists(os.path.join(checkpoint_dir, "best")):
        model_path = os.path.join(checkpoint_dir, "best")
        logger.info(f"Loading best model from {model_path}")
    else:
        # Find the latest checkpoint
        checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
        model_path = os.path.join(checkpoint_dir, latest_checkpoint)
        logger.info(f"Loading latest checkpoint from {model_path}")
    
    # Load model and configuration
    model_pt_path = os.path.join(model_path, "model.pt")
    if not os.path.exists(model_pt_path):
        raise ValueError(f"Model file not found at {model_pt_path}")
    
    # Load training state to get configuration
    training_state_path = os.path.join(model_path, "training_state.json")
    if os.path.exists(training_state_path):
        with open(training_state_path, "r") as f:
            training_state = json.load(f)
            # Extract model configuration from training state
            config = {
                "model_name": "sentence-transformers/all-mpnet-base-v2",  # Default
                "d_model": 768,
                "n_heads": 12,
                "n_layers": 6, 
                "d_ff": 3072,
                "dropout": 0.1
            }
            # Update with values from training state if available
            if "args" in training_state:
                args = training_state["args"]
                for key in config.keys():
                    if key in args:
                        config[key] = args[key]
    else:
        # Use default configuration if training state is not available
        logger.warning(f"Training state not found at {training_state_path}, using default configuration")
        config = {
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 6,
            "d_ff": 3072, 
            "dropout": 0.1
        }
    
    # Load tokenizer
    tokenizer_path = os.path.join(model_path)
    if os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        # Fall back to the original tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    # Load domain mapping (use training_state if available)
    domain_file = os.path.join(os.path.dirname(checkpoint_dir), "..", "data/domains/domain_mapping.json")
    if "domain_file" in config:
        domain_file = config["domain_file"]
    
    domain_mapping = load_domain_mapping(domain_file)
    num_domains = len(domain_mapping["domains"])
    
    # Create model
    model = ResearchTransformer(
        model_name=config["model_name"],
        d_model=config.get("d_model", 768),
        n_heads=config.get("n_heads", 12),
        n_layers=config.get("n_layers", 6),
        d_ff=config.get("d_ff", 3072),
        dropout=config.get("dropout", 0.1),
        num_domains=num_domains
    )
    
    # Load state dict
    model_state_dict = torch.load(model_pt_path, map_location="cpu")
    model.load_state_dict(model_state_dict)
    
    return model, tokenizer, domain_mapping

def run_inference(model, tokenizer, papers, domain_mapping):
    """Run inference on a list of papers."""
    # Move model to available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Store results
    results = []
    embeddings = []
    
    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        text = f"{title} {abstract}"
        
        # Tokenize
        inputs = tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                domain_id=torch.tensor([paper.get("domain_id", 0)]).to(device)
            )
        
        # Extract outputs
        embedding = outputs["embedding"].cpu().numpy()[0]
        impact_score = outputs["impact_score"].cpu().numpy()[0].item()
        domain_relevance = outputs["domain_relevance"].cpu().numpy()[0].item()
        
        # Get domain name
        domain_id = paper.get("domain_id", 0)
        domain_name = "Unknown"
        for domain in domain_mapping["domains"]:
            if domain["id"] == domain_id:
                domain_name = domain["name"]
                break
        
        # Store result
        result = {
            "paper_id": paper.get("id", "unknown"),
            "title": title,
            "predicted_impact_score": impact_score,
            "domain_relevance": domain_relevance,
            "domain_id": domain_id,
            "domain_name": domain_name
        }
        results.append(result)
        embeddings.append(embedding)
    
    return results, np.array(embeddings)

def visualize_embeddings(embeddings, labels, title="Paper Embeddings Visualization", base_dir=None):
    """Visualize paper embeddings using dimensionality reduction."""
    # Reduce dimensionality for visualization
    if len(embeddings) < 3:
        logger.warning("Need at least 3 samples for visualization")
        return
    
    # Use PCA first to reduce to 50 dimensions
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50)
        embeddings_pca = pca.fit_transform(embeddings)
        logger.info(f"PCA explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    else:
        embeddings_pca = embeddings
    
    # Then use t-SNE to reduce to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Get unique domains
    unique_domains = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_domains)))
    
    # Plot each domain with a different color
    for i, domain in enumerate(unique_domains):
        idx = [j for j, label in enumerate(labels) if label == domain]
        plt.scatter(
            embeddings_2d[idx, 0],
            embeddings_2d[idx, 1],
            c=[colors[i]],
            label=domain,
            alpha=0.7
        )
    
    plt.title(title, fontsize=18)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the visualization
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs/visualizations")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "paper_embeddings.png")
    plt.savefig(output_path)
    logger.info(f"Visualization saved to {output_path}")
    
    # Show the plot (commented out for headless environments)
    # plt.show()

def main():
    # Set base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the model
    model_dir = os.path.join(base_dir, "outputs/large_run")
    model, tokenizer, domain_mapping = load_model(model_dir)
    
    # Load test papers
    test_file = os.path.join(base_dir, "data/papers_test.json")
    with open(test_file, "r") as f:
        test_papers = json.load(f)
    
    # Run inference on test papers
    logger.info(f"Running inference on {len(test_papers)} test papers")
    results, embeddings = run_inference(model, tokenizer, test_papers, domain_mapping)
    
    # Save inference results
    output_dir = os.path.join(base_dir, "outputs/inference")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "inference_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Inference results saved to {os.path.join(output_dir, 'inference_results.json')}")
    
    # Visualize embeddings
    domain_labels = [result["domain_name"] for result in results]
    visualize_embeddings(embeddings, domain_labels, base_dir=base_dir)
    
    # Print summary of results
    print("\n===== INFERENCE RESULTS SUMMARY =====")
    print(f"Processed {len(results)} papers")
    
    # Group by domain
    domain_groups = {}
    for result in results:
        domain = result["domain_name"]
        if domain not in domain_groups:
            domain_groups[domain] = []
        domain_groups[domain].append(result)
    
    print("\nDomain distribution:")
    for domain, papers in domain_groups.items():
        avg_impact = sum(p["predicted_impact_score"] for p in papers) / len(papers)
        avg_relevance = sum(p["domain_relevance"] for p in papers) / len(papers)
        print(f"  - {domain}: {len(papers)} papers, avg impact: {avg_impact:.2f}, avg relevance: {avg_relevance:.2f}")
    
    # Print top papers by predicted impact
    print("\nTop 5 papers by predicted impact:")
    top_papers = sorted(results, key=lambda x: x["predicted_impact_score"], reverse=True)[:5]
    for i, paper in enumerate(top_papers, 1):
        print(f"  {i}. {paper['title']} (Impact: {paper['predicted_impact_score']:.2f}, Domain: {paper['domain_name']})")

if __name__ == "__main__":
    main() 