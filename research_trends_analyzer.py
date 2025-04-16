#!/usr/bin/env python3
"""
Research Trends Analyzer for ARX2
-------------------------------
Analyzes papers in the knowledge base to identify trends, emergent topics,
and generate insights.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
import seaborn as sns
import networkx as nx
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import community as community_louvain
from wordcloud import WordCloud

# Import internal components
from advanced_ai_analyzer import CONFIG
from knowledge_graph import EnhancedKnowledgeGraph
from advanced_ai_analyzer_knowledge_base import KnowledgeBase
from utils.embedding_manager import EmbeddingManager

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("research_trends.log"),
        logging.StreamHandler()
    ]
)

class ResearchTrendsAnalyzer:
    """Analyzes research trends from papers in the knowledge base"""
    
    def __init__(self, kb=None, kg=None, output_dir=None):
        """Initialize the research trends analyzer
        
        Args:
            kb (KnowledgeBase, optional): Knowledge base with papers
            kg (EnhancedKnowledgeGraph, optional): Knowledge graph
            output_dir (str, optional): Directory to save visualizations and reports
        """
        self.kb = kb or KnowledgeBase()
        
        # Initialize knowledge graph if not provided
        if kg:
            self.kg = kg
        else:
            self.kg = EnhancedKnowledgeGraph(self.kb)
            # Build graph if not already built
            if not hasattr(self.kg, 'graph') or self.kg.graph.number_of_nodes() == 0:
                self.kg.build_graph_from_knowledge_base()
        
        # Set up output directory
        self.output_dir = output_dir or os.path.join(CONFIG.get('models_dir', '.'), 'research_trends')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager()
        
        # Track results for combined reports
        self.analysis_results = {}
        
        logger.info(f"Initialized research trends analyzer (output_dir={self.output_dir})")
    
    def analyze_temporal_trends(self, timeframe_years=5, min_papers=5):
        """Analyze how research topics evolve over time
        
        Args:
            timeframe_years (int): Number of years to analyze
            min_papers (int): Minimum papers per concept to include
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Analyzing temporal trends over {timeframe_years} years")
        
        # Get papers with publication dates
        papers_with_dates = []
        for paper_id, paper in self.kb.papers.items():
            published = paper.get('published', '')
            if published:
                try:
                    # Convert ISO date to datetime
                    if 'T' in published:  # ISO format with time
                        pub_date = datetime.fromisoformat(published.split('T')[0])
                    elif len(published) == 4:  # Just year
                        pub_date = datetime(int(published), 1, 1)
                    else:  # Try direct parsing
                        pub_date = datetime.fromisoformat(published)
                    
                    papers_with_dates.append((paper_id, paper, pub_date))
                except:
                    continue
        
        if not papers_with_dates:
            logger.warning("No papers with valid publication dates found")
            return {"error": "No papers with valid publication dates found"}
        
        # Sort by publication date
        papers_with_dates.sort(key=lambda x: x[2])
        
        # Define time periods
        now = datetime.now()
        end_date = now
        start_date = end_date - timedelta(days=365*timeframe_years)
        
        # Filter to papers within timeframe
        recent_papers = [p for p in papers_with_dates if start_date <= p[2] <= end_date]
        
        if not recent_papers:
            logger.warning(f"No papers found in the last {timeframe_years} years")
            return {"error": f"No papers found in the last {timeframe_years} years"}
        
        # Create time bins (quarters)
        num_quarters = timeframe_years * 4
        quarters = pd.date_range(start=start_date, end=end_date, periods=num_quarters + 1)
        
        # Count papers and concepts per quarter
        quarter_papers = [[] for _ in range(num_quarters)]
        for paper_id, paper, pub_date in recent_papers:
            for i, (q_start, q_end) in enumerate(zip(quarters[:-1], quarters[1:])):
                if q_start <= pub_date < q_end:
                    quarter_papers[i].append((paper_id, paper))
                    break
        
        # Calculate concept frequency by quarter
        concept_trends = defaultdict(lambda: [0] * num_quarters)
        concept_papers = defaultdict(list)
        
        for q_idx, papers in enumerate(quarter_papers):
            for paper_id, paper in papers:
                for concept in paper.get('concepts', []):
                    concept_trends[concept][q_idx] += 1
                    concept_papers[concept].append(paper_id)
        
        # Filter to concepts with enough papers
        popular_concepts = {c: papers for c, papers in concept_papers.items() if len(papers) >= min_papers}
        
        if not popular_concepts:
            logger.warning(f"No concepts found with at least {min_papers} papers")
            return {"error": f"No concepts found with at least {min_papers} papers"}
        
        # Calculate growth rate for each concept (last quarter vs. first non-zero quarter)
        concept_growth = {}
        for concept, counts in concept_trends.items():
            if concept not in popular_concepts:
                continue
                
            # Find first non-zero quarter
            first_idx = next((i for i, c in enumerate(counts) if c > 0), -1)
            if first_idx >= 0 and first_idx < len(counts) - 1 and counts[first_idx] > 0:
                # Calculate growth rate
                growth_rate = counts[-1] / counts[first_idx] if counts[first_idx] > 0 else 0
                concept_growth[concept] = growth_rate
        
        # Get top growing concepts
        top_growing = sorted(concept_growth.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Plot concept trends for top growing concepts
        plt.figure(figsize=(12, 8))
        
        # Convert quarters to strings for x-axis
        quarter_labels = [f"{q.year} Q{(q.month-1)//3+1}" for q in quarters[:-1]]
        
        # Plot top 10 growing concepts
        for concept, _ in top_growing[:10]:
            plt.plot(quarter_labels, concept_trends[concept], marker='o', linewidth=2, label=concept)
        
        plt.title(f"Concept Trends Over {timeframe_years} Years", fontsize=16)
        plt.xlabel("Quarter", fontsize=12)
        plt.ylabel("Paper Count", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        # Save plot
        trend_plot_path = os.path.join(self.output_dir, f"concept_trends_{timeframe_years}years.png")
        plt.savefig(trend_plot_path, dpi=300)
        plt.close()
        
        # Create growth rate visualization
        plt.figure(figsize=(12, 8))
        concepts, growth_rates = zip(*top_growing[:15])
        y_pos = np.arange(len(concepts))
        
        plt.barh(y_pos, growth_rates, align='center')
        plt.yticks(y_pos, concepts)
        plt.xlabel('Growth Rate')
        plt.title('Top 15 Fastest Growing Concepts')
        plt.tight_layout()
        
        # Save plot
        growth_plot_path = os.path.join(self.output_dir, f"concept_growth_{timeframe_years}years.png")
        plt.savefig(growth_plot_path, dpi=300)
        plt.close()
        
        # Create detailed results
        results = {
            "top_growing_concepts": [{
                "concept": concept,
                "growth_rate": rate,
                "paper_count": len(concept_papers[concept]),
                "trend": concept_trends[concept]
            } for concept, rate in top_growing],
            "time_periods": quarter_labels,
            "plots": {
                "trend_plot": trend_plot_path,
                "growth_plot": growth_plot_path
            }
        }
        
        # Save results to file
        results_path = os.path.join(self.output_dir, f"temporal_trends_{timeframe_years}years.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Store in analysis results
        self.analysis_results["temporal_trends"] = results
        
        logger.info(f"Saved temporal trend analysis to {results_path}")
        return results
    
    def discover_emergent_topics(self, num_topics=10, num_words=10):
        """Discover emergent topics using topic modeling
        
        Args:
            num_topics (int): Number of topics to discover
            num_words (int): Number of words per topic
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Discovering emergent topics (num_topics={num_topics})")
        
        # Get paper texts (title + abstract)
        paper_texts = []
        paper_ids = []
        
        for paper_id, paper in self.kb.papers.items():
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            
            if title or abstract:
                text = f"{title} {abstract}"
                paper_texts.append(text)
                paper_ids.append(paper_id)
        
        if not paper_texts:
            logger.warning("No paper texts found for topic modeling")
            return {"error": "No paper texts found for topic modeling"}
        
        # Create TF-IDF representation
        logger.info("Computing TF-IDF representation")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(paper_texts)
        
        # Get feature names for later interpretation
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # NMF for topic modeling (often better than LDA for short texts)
        logger.info("Running NMF topic modeling")
        nmf_model = NMF(n_components=num_topics, random_state=42, max_iter=1000)
        nmf_topics = nmf_model.fit_transform(tfidf_matrix)
        
        # Get top words for each topic
        nmf_topics_words = []
        for topic_idx, topic in enumerate(nmf_model.components_):
            top_words_idx = topic.argsort()[:-num_words-1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            nmf_topics_words.append(top_words)
        
        # Associate papers with topics
        paper_topics = []
        for i, paper_id in enumerate(paper_ids):
            # Get topic distribution for this paper
            topic_dist = nmf_topics[i]
            # Get top topic
            top_topic = np.argmax(topic_dist)
            # Get topic score
            topic_score = topic_dist[top_topic]
            
            paper_topics.append({
                "paper_id": paper_id,
                "top_topic": int(top_topic),
                "topic_score": float(topic_score)
            })
        
        # Count papers per topic
        topic_counts = Counter([p["top_topic"] for p in paper_topics])
        
        # Visualize topics
        # 1. Topic sizes
        plt.figure(figsize=(12, 6))
        topic_indices = list(range(num_topics))
        topic_sizes = [topic_counts.get(t, 0) for t in topic_indices]
        
        plt.bar(topic_indices, topic_sizes)
        plt.xlabel('Topic Index')
        plt.ylabel('Number of Papers')
        plt.title('Papers per Topic')
        plt.xticks(topic_indices)
        plt.tight_layout()
        
        # Save plot
        topic_sizes_plot = os.path.join(self.output_dir, "topic_sizes.png")
        plt.savefig(topic_sizes_plot, dpi=300)
        plt.close()
        
        # 2. Word clouds for top topics
        top_topic_indices = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for topic_idx, count in top_topic_indices:
            # Create word cloud
            topic_words = nmf_topics_words[topic_idx]
            topic_weights = nmf_model.components_[topic_idx]
            
            # Create dictionary of word: weight for wordcloud
            word_weights = {feature_names[i]: abs(topic_weights[i]) for i in topic_weights.argsort()[:-50-1:-1]}
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100
            ).generate_from_frequencies(word_weights)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Topic {topic_idx} - {count} Papers')
            plt.tight_layout()
            
            # Save plot
            wordcloud_path = os.path.join(self.output_dir, f"topic_{topic_idx}_wordcloud.png")
            plt.savefig(wordcloud_path, dpi=300)
            plt.close()
        
        # 3. Topic similarity heatmap
        topic_similarity = cosine_similarity(nmf_model.components_)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            topic_similarity,
            annot=True,
            cmap="YlGnBu",
            xticklabels=range(num_topics),
            yticklabels=range(num_topics)
        )
        plt.title('Topic Similarity Matrix')
        plt.tight_layout()
        
        # Save plot
        similarity_path = os.path.join(self.output_dir, "topic_similarity.png")
        plt.savefig(similarity_path, dpi=300)
        plt.close()
        
        # Create detailed results
        topic_details = []
        for topic_idx in range(num_topics):
            # Get papers for this topic
            topic_papers = [p for p in paper_topics if p["top_topic"] == topic_idx]
            # Sort by score
            topic_papers.sort(key=lambda x: x["topic_score"], reverse=True)
            
            # Get top papers
            top_papers = []
            for tp in topic_papers[:5]:
                paper_id = tp["paper_id"]
                if paper_id in self.kb.papers:
                    paper = self.kb.papers[paper_id]
                    top_papers.append({
                        "id": paper_id,
                        "title": paper.get("title", ""),
                        "score": tp["topic_score"]
                    })
            
            topic_details.append({
                "topic_id": topic_idx,
                "words": nmf_topics_words[topic_idx],
                "paper_count": topic_counts.get(topic_idx, 0),
                "top_papers": top_papers
            })
        
        # Sort topics by paper count
        topic_details.sort(key=lambda x: x["paper_count"], reverse=True)
        
        results = {
            "topics": topic_details,
            "paper_topics": paper_topics[:100],  # Limit to first 100
            "plots": {
                "topic_sizes": topic_sizes_plot,
                "topic_similarity": similarity_path,
                "topic_wordclouds": [os.path.join(self.output_dir, f"topic_{t[0]}_wordcloud.png") for t in top_topic_indices]
            }
        }
        
        # Save results to file
        results_path = os.path.join(self.output_dir, "emergent_topics.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Store in analysis results
        self.analysis_results["emergent_topics"] = results
        
        logger.info(f"Saved emergent topics analysis to {results_path}")
        return results
    
    def analyze_citation_patterns(self, min_citations=2):
        """Analyze citation patterns between papers
        
        Args:
            min_citations (int): Minimum citations to include a paper
            
        Returns:
            dict: Analysis results
        """
        logger.info("Analyzing citation patterns")
        
        # Check if papers have citation information
        papers_with_citations = [p for p_id, p in self.kb.papers.items() if 'citations' in p and p['citations']]
        
        if not papers_with_citations:
            logger.warning("No citation information found in papers")
            return {"error": "No citation information found in papers"}
        
        # Build citation network
        citation_graph = nx.DiGraph()
        
        # Add nodes for all papers
        for paper_id, paper in self.kb.papers.items():
            citation_graph.add_node(
                paper_id,
                title=paper.get('title', ''),
                year=paper.get('published', '')[:4] if paper.get('published', '') else ''
            )
            
            # Add citation edges
            if 'citations' in paper and paper['citations']:
                for cited_id in paper['citations']:
                    if cited_id in self.kb.papers:  # Only add if cited paper is in KB
                        citation_graph.add_edge(paper_id, cited_id)
        
        # Calculate various centrality metrics
        logger.info("Calculating citation network metrics")
        
        # In-degree (citation count)
        in_degree = dict(citation_graph.in_degree())
        # PageRank (influence metric)
        try:
            pagerank = nx.pagerank(citation_graph)
        except:
            pagerank = {n: 0 for n in citation_graph.nodes()}
        
        # Combine metrics
        paper_metrics = []
        for paper_id in citation_graph.nodes():
            citations = in_degree.get(paper_id, 0)
            influence = pagerank.get(paper_id, 0)
            
            # Only include papers with sufficient citations
            if citations >= min_citations:
                if paper_id in self.kb.papers:
                    paper = self.kb.papers[paper_id]
                    paper_metrics.append({
                        "id": paper_id,
                        "title": paper.get("title", ""),
                        "citations": citations,
                        "influence": influence,
                        "published": paper.get("published", "")
                    })
        
        # Sort by influence
        paper_metrics.sort(key=lambda x: x["influence"], reverse=True)
        
        # Visualize citation network (focusing on top papers)
        top_papers = [p["id"] for p in paper_metrics[:50]]  # Top 50 by influence
        
        if not top_papers:
            logger.warning(f"No papers with at least {min_citations} citations found")
            return {"error": f"No papers with at least {min_citations} citations found"}
        
        # Create subgraph with only top papers and their connections
        top_subgraph = citation_graph.subgraph(top_papers)
        
        plt.figure(figsize=(12, 10))
        
        # Use spring layout for visualization
        pos = nx.spring_layout(top_subgraph, k=0.3, iterations=50)
        
        # Node sizes based on citation count
        node_sizes = [300 * in_degree.get(node, 1) for node in top_subgraph.nodes()]
        
        # Draw nodes, labels, and edges
        nx.draw_networkx_nodes(
            top_subgraph,
            pos,
            node_size=node_sizes,
            node_color=[pagerank.get(n, 0) for n in top_subgraph.nodes()],
            cmap=plt.cm.viridis,
            alpha=0.8
        )
        
        # Add labels for top 10 papers
        top_10_papers = [p["id"] for p in paper_metrics[:10]]
        labels = {node: self.kb.papers[node].get('title', '')[:20] + '...' for node in top_10_papers if node in top_subgraph}
        nx.draw_networkx_labels(top_subgraph, pos, labels=labels, font_size=8, font_color='black')
        
        # Draw edges
        nx.draw_networkx_edges(top_subgraph, pos, alpha=0.3, arrows=True, arrowsize=10)
        
        plt.title("Citation Network of Influential Papers")
        plt.axis('off')
        plt.tight_layout()
        
        # Save plot
        network_path = os.path.join(self.output_dir, "citation_network.png")
        plt.savefig(network_path, dpi=300)
        plt.close()
        
        # Results
        results = {
            "influential_papers": paper_metrics[:20],  # Top 20
            "citation_stats": {
                "total_papers": len(citation_graph.nodes()),
                "total_citations": len(citation_graph.edges()),
                "avg_citations": sum(in_degree.values()) / max(1, len(in_degree))
            },
            "plots": {
                "citation_network": network_path
            }
        }
        
        # Save results to file
        results_path = os.path.join(self.output_dir, "citation_patterns.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Store in analysis results
        self.analysis_results["citation_patterns"] = results
        
        logger.info(f"Saved citation pattern analysis to {results_path}")
        return results
    
    def identify_research_frontiers(self):
        """Identify research frontiers using knowledge graph
        
        Returns:
            dict: Analysis results
        """
        logger.info("Identifying research frontiers")
        
        # Use knowledge graph to get research frontiers
        frontiers = self.kg.get_research_frontiers()
        
        if not frontiers:
            logger.warning("No research frontiers found in knowledge graph")
            return {"error": "No research frontiers found in knowledge graph"}
        
        # Create network visualization of frontiers and related concepts
        plt.figure(figsize=(15, 12))
        
        # Create graph for visualization
        frontier_graph = nx.Graph()
        
        # Add frontier concepts as nodes
        for frontier in frontiers:
            concept = frontier["concept"]
            frontier_graph.add_node(concept, type="frontier")
            
            # Add related concepts
            for related in frontier["related_concepts"][:5]:  # Top 5 related
                rel_concept = related["concept"]
                frontier_graph.add_node(rel_concept, type="related")
                frontier_graph.add_edge(
                    concept,
                    rel_concept,
                    weight=related["weight"],
                    relation=related["relation"]
                )
        
        # Node positions
        pos = nx.spring_layout(frontier_graph, k=0.5, iterations=50)
        
        # Draw nodes with different colors for frontiers vs. related
        frontier_nodes = [n for n, attr in frontier_graph.nodes(data=True) if attr.get('type') == 'frontier']
        related_nodes = [n for n, attr in frontier_graph.nodes(data=True) if attr.get('type') == 'related']
        
        # Draw frontier nodes (larger, red)
        nx.draw_networkx_nodes(
            frontier_graph,
            pos,
            nodelist=frontier_nodes,
            node_size=800,
            node_color='red',
            alpha=0.8
        )
        
        # Draw related nodes (smaller, blue)
        nx.draw_networkx_nodes(
            frontier_graph,
            pos,
            nodelist=related_nodes,
            node_size=300,
            node_color='blue',
            alpha=0.6
        )
        
        # Draw edges with varying width based on weight
        edge_weights = [data.get('weight', 0.1) * 5 for _, _, data in frontier_graph.edges(data=True)]
        nx.draw_networkx_edges(frontier_graph, pos, width=edge_weights, alpha=0.5)
        
        # Add labels
        nx.draw_networkx_labels(frontier_graph, pos, font_size=10)
        
        plt.title("Research Frontiers and Related Concepts")
        plt.axis('off')
        plt.tight_layout()
        
        # Save plot
        frontier_graph_path = os.path.join(self.output_dir, "research_frontiers.png")
        plt.savefig(frontier_graph_path, dpi=300)
        plt.close()
        
        # Create detailed results
        formatted_frontiers = []
        for frontier in frontiers:
            # Get relevant papers
            concept = frontier["concept"]
            papers = frontier.get("papers", [])
            
            # Format for output
            formatted_frontiers.append({
                "concept": concept,
                "cluster": frontier.get("cluster", ""),
                "related_concepts": [r["concept"] for r in frontier.get("related_concepts", [])[:5]],
                "papers": papers
            })
        
        results = {
            "frontiers": formatted_frontiers,
            "plots": {
                "frontier_graph": frontier_graph_path
            }
        }
        
        # Save results to file
        results_path = os.path.join(self.output_dir, "research_frontiers.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Store in analysis results
        self.analysis_results["research_frontiers"] = results
        
        logger.info(f"Saved research frontiers analysis to {results_path}")
        return results
    
    def analyze_concept_relationships(self):
        """Analyze relationships between concepts
        
        Returns:
            dict: Analysis results
        """
        logger.info("Analyzing concept relationships")
        
        # Get concept nodes from knowledge graph
        concept_nodes = [node for node, attr in self.kg.graph.nodes(data=True) if attr.get('type') == 'concept']
        
        if not concept_nodes:
            logger.warning("No concept nodes found in knowledge graph")
            return {"error": "No concept nodes found in knowledge graph"}
        
        # Get concept importance from knowledge graph
        concept_importance = self.kg.get_concept_importance(limit=100)
        important_concepts = [c["concept"] for c in concept_importance]
        
        # Get concept clusters
        concept_clusters = self.kg.get_concept_clusters()
        
        # Create concept relationship visualization using force-directed layout
        plt.figure(figsize=(15, 12))
        
        # Create subgraph with important concepts
        top_concepts = important_concepts[:50]  # Top 50 concepts
        concept_subgraph = self.kg.graph.subgraph(top_concepts)
        
        # Convert to undirected for visualization
        undirected = concept_subgraph.to_undirected()
        
        # Apply community detection
        communities = community_louvain.best_partition(undirected)
        
        # Node positions
        pos = nx.spring_layout(undirected, k=0.3, iterations=50)
        
        # Draw nodes colored by community
        nx.draw_networkx_nodes(
            undirected,
            pos,
            node_size=[undirected.degree(n) * 20 + 100 for n in undirected.nodes()],
            node_color=[communities.get(n, 0) for n in undirected.nodes()],
            cmap=plt.cm.tab20,
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(undirected, pos, alpha=0.2)
        
        # Add labels for top 20 concepts
        labels = {node: node for node in top_concepts[:20]}
        nx.draw_networkx_labels(undirected, pos, labels=labels, font_size=8)
        
        plt.title("Concept Relationship Network (Top 50 Concepts)")
        plt.axis('off')
        plt.tight_layout()
        
        # Save plot
        concept_network_path = os.path.join(self.output_dir, "concept_network.png")
        plt.savefig(concept_network_path, dpi=300)
        plt.close()
        
        # Analyze concept hierarchy and create a dendrogram-like visualization
        # This uses clusters from the knowledge graph
        if concept_clusters:
            plt.figure(figsize=(15, 10))
            
            # Organize data for visualization
            cluster_data = []
            for cluster in concept_clusters:
                cluster_name = cluster["name"]
                concepts = cluster["concepts"]
                size = cluster["size"]
                
                if concepts:
                    cluster_data.append((cluster_name, concepts, size))
            
            # Sort by size
            cluster_data.sort(key=lambda x: x[2], reverse=True)
            
            # Plot as horizontal bars for cluster membership
            y_pos = 0
            yticks = []
            ylabels = []
            
            for cluster_name, concepts, size in cluster_data[:10]:  # Top 10 clusters
                # Add cluster name
                yticks.append(y_pos)
                ylabels.append(f"{cluster_name} ({size})")
                
                # Add concepts in this cluster
                for i, concept in enumerate(concepts[:5]):  # Top 5 concepts per cluster
                    y_pos += 1
                    plt.barh(y_pos, 1, color=plt.cm.tab20(cluster_data.index((cluster_name, concepts, size)) % 20))
                    plt.text(1.05, y_pos, concept, va='center')
                
                y_pos += 2  # Add space between clusters
            
            plt.yticks(yticks, ylabels)
            plt.xlim(0, 2)
            plt.title("Concept Hierarchy (Top 10 Clusters)")
            plt.tight_layout()
            
            # Save plot
            hierarchy_path = os.path.join(self.output_dir, "concept_hierarchy.png")
            plt.savefig(hierarchy_path, dpi=300)
            plt.close()
        else:
            hierarchy_path = None
        
        # Create detailed results
        results = {
            "important_concepts": concept_importance,
            "concept_clusters": concept_clusters,
            "plots": {
                "concept_network": concept_network_path,
                "concept_hierarchy": hierarchy_path
            }
        }
        
        # Save results to file
        results_path = os.path.join(self.output_dir, "concept_relationships.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Store in analysis results
        self.analysis_results["concept_relationships"] = results
        
        logger.info(f"Saved concept relationship analysis to {results_path}")
        return results
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive research trends report
        
        Returns:
            dict: Comprehensive report data
        """
        logger.info("Generating comprehensive research trends report")
        
        # Run all analyses if they haven't been run already
        if "temporal_trends" not in self.analysis_results:
            self.analyze_temporal_trends()
        
        if "emergent_topics" not in self.analysis_results:
            self.discover_emergent_topics()
        
        if "citation_patterns" not in self.analysis_results:
            self.analyze_citation_patterns()
        
        if "research_frontiers" not in self.analysis_results:
            self.identify_research_frontiers()
        
        if "concept_relationships" not in self.analysis_results:
            self.analyze_concept_relationships()
        
        # Combine all results
        comprehensive_report = {
            "summary": {
                "total_papers": len(self.kb.papers),
                "total_concepts": len(self.kb.concept_index),
                "generated_at": datetime.now().isoformat()
            },
            "key_findings": {
                "top_growing_concepts": self.analysis_results.get("temporal_trends", {}).get("top_growing_concepts", [])[:5],
                "emergent_topics": self.analysis_results.get("emergent_topics", {}).get("topics", [])[:5],
                "influential_papers": self.analysis_results.get("citation_patterns", {}).get("influential_papers", [])[:5],
                "research_frontiers": self.analysis_results.get("research_frontiers", {}).get("frontiers", [])[:5]
            },
            "analysis_components": list(self.analysis_results.keys()),
            "visualizations": {}
        }
        
        # Collect all visualizations
        for analysis_type, results in self.analysis_results.items():
            if "plots" in results:
                comprehensive_report["visualizations"][analysis_type] = results["plots"]
        
        # Save comprehensive report
        report_path = os.path.join(self.output_dir, "comprehensive_report.json")
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        logger.info(f"Saved comprehensive research trends report to {report_path}")
        return comprehensive_report

def main():
    """Command-line interface for the research trends analyzer"""
    parser = argparse.ArgumentParser(description="ARX2 Research Trends Analyzer")
    
    # Analysis type
    parser.add_argument("--analysis", type=str, choices=[
        "temporal", "topics", "citations", "frontiers", "relationships", "all"
    ], default="all", help="Type of analysis to perform")
    
    # Output directory
    parser.add_argument("--output", type=str, default=None, help="Output directory for visualizations and reports")
    
    # Temporal trends options
    parser.add_argument("--years", type=int, default=5, help="Number of years for temporal analysis")
    
    # Topic modeling options
    parser.add_argument("--topics", type=int, default=10, help="Number of topics to discover")
    
    # Citation analysis options
    parser.add_argument("--min-citations", type=int, default=2, help="Minimum citations for citation analysis")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ResearchTrendsAnalyzer(output_dir=args.output)
    
    # Run analyses
    if args.analysis == "temporal" or args.analysis == "all":
        analyzer.analyze_temporal_trends(timeframe_years=args.years)
    
    if args.analysis == "topics" or args.analysis == "all":
        analyzer.discover_emergent_topics(num_topics=args.topics)
    
    if args.analysis == "citations" or args.analysis == "all":
        analyzer.analyze_citation_patterns(min_citations=args.min_citations)
    
    if args.analysis == "frontiers" or args.analysis == "all":
        analyzer.identify_research_frontiers()
    
    if args.analysis == "relationships" or args.analysis == "all":
        analyzer.analyze_concept_relationships()
    
    # Generate comprehensive report if all analyses were run
    if args.analysis == "all":
        analyzer.generate_comprehensive_report()

if __name__ == "__main__":
    main() 