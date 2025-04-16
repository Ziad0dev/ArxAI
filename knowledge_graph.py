"""
Knowledge Graph implementation for the AI Research System
"""

from advanced_ai_analyzer import *
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import pickle
import os
from tqdm import tqdm
import community as community_louvain
import scipy.sparse
import json
import logging
import math
import itertools
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity
from utils.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

class EnhancedKnowledgeGraph:
    """Enhanced knowledge graph with hierarchical concept organization and advanced relation detection"""
    
    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base
        self.graph = nx.DiGraph()
        self.concept_embedding_cache = {}
        self.relation_types = {
            'is_a': 0,
            'part_of': 1,
            'similar_to': 2,
            'prerequisite': 3,
            'application': 4,
            'variation_of': 5,
            'contrasts_with': 6,
            'evaluates': 7,
            'introduces': 8,
            'implements': 9,
            'extends': 10
        }
        
        # Mapping of relation ID to label
        self.relation_labels = {v: k for k, v in self.relation_types.items()}
        
        # Initialize embedding manager for concept embeddings
        self.embedding_manager = EmbeddingManager()
        
        # Cache for computed metrics
        self._metrics_cache = {}
        
        # Track hierarchy of concepts
        self.concept_hierarchy = {}
        self.concept_clusters = {}
        
        # Track research frontiers
        self.research_frontiers = []
        
        # Core concept dictionary
        self.core_concepts = {}
        
        # Temporal evolution tracking
        self.concept_evolution = defaultdict(list)
        
    def build_graph_from_knowledge_base(self):
        """Build a directed knowledge graph from the knowledge base papers and concepts"""
        if not self.kb:
            logger.error("No knowledge base provided to build graph")
            return False
            
        logger.info(f"Building knowledge graph from {len(self.kb.papers)} papers and {len(self.kb.concept_index)} concepts")
        
        # Clear existing graph
        self.graph = nx.DiGraph()
        
        # Add all concepts as nodes
        for concept in self.kb.concept_index:
            self.graph.add_node(concept, type='concept', papers=self.kb.concept_index[concept])
        
        # Add papers as nodes
        for paper_id, paper in self.kb.papers.items():
            self.graph.add_node(
                paper_id, 
                type='paper',
                title=paper.get('title', 'Unknown'),
                authors=paper.get('authors', []),
                published=paper.get('published', ''),
                concepts=paper.get('concepts', [])
            )
            
            # Add edges from papers to concepts
            for concept in paper.get('concepts', []):
                if concept in self.graph:
                    self.graph.add_edge(paper_id, concept, type='has_concept', weight=1.0)
        
        # Add citation relationships if available
        if hasattr(self.kb, 'citation_graph'):
            for citing_id, cited_ids in self.kb.citation_graph.items():
                for cited_id in cited_ids:
                    if citing_id in self.graph and cited_id in self.graph:
                        self.graph.add_edge(citing_id, cited_id, type='cites', weight=1.0)
        
        # Add concept relationships from knowledge base
        if hasattr(self.kb, 'concept_relations'):
            for source_concept, relations in self.kb.concept_relations.items():
                for relation in relations:
                    target_concept = relation.get('target')
                    relation_type = relation.get('relation')
                    confidence = relation.get('confidence', 0.5)
                    
                    if source_concept in self.graph and target_concept in self.graph:
                        self.graph.add_edge(
                            source_concept, 
                            target_concept, 
                            type=relation_type, 
                            weight=confidence
                        )
        
        # Extend with detected relationships between concepts
        self._detect_concept_relationships()
        
        # Build concept hierarchy
        self._build_concept_hierarchy()
        
        # Identify key research areas
        self._identify_research_frontiers()
        
        logger.info(f"Built knowledge graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return True
    
    def _get_concept_embedding(self, concept):
        """Get embedding for a concept, with caching"""
        if concept in self.concept_embedding_cache:
            return self.concept_embedding_cache[concept]
            
        # Generate embedding for concept
        embedding = self.embedding_manager.get_embedding_for_text(concept)
        self.concept_embedding_cache[concept] = embedding
        return embedding
    
    def _detect_concept_relationships(self):
        """Detect relationships between concepts using multiple methods:
        1. Co-occurrence in papers
        2. Semantic similarity
        3. Hierarchical patterns (is-a, part-of)
        4. Causal and prerequisite relationships
        """
        logger.info("Detecting concept relationships")
        
        # Get all concepts
        concepts = [node for node, attr in self.graph.nodes(data=True) if attr.get('type') == 'concept']
        
        if not concepts:
            logger.warning("No concepts in graph to detect relationships")
            return
            
        # 1. Co-occurrence analysis
        concept_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        # Count co-occurring concepts in papers
        for paper_id, paper in self.kb.papers.items():
            paper_concepts = paper.get('concepts', [])
            
            # Count pairwise co-occurrences
            for i, concept1 in enumerate(paper_concepts):
                for concept2 in paper_concepts[i+1:]:
                    concept_cooccurrence[concept1][concept2] += 1
                    concept_cooccurrence[concept2][concept1] += 1
        
        # Add high-confidence co-occurrence edges (normalized by concept frequency)
        for concept1, cooccurrences in concept_cooccurrence.items():
            concept1_count = len(self.kb.concept_index.get(concept1, []))
            if concept1_count == 0:
                continue
                
            for concept2, count in cooccurrences.items():
                concept2_count = len(self.kb.concept_index.get(concept2, []))
                if concept2_count == 0:
                    continue
                    
                # Normalize by concept frequency
                pmi = math.log((count * len(self.kb.papers)) / (concept1_count * concept2_count))
                
                # Only add strong relationships (PMI > 0)
                if pmi > 0:
                    confidence = min(1.0, pmi / 5.0)  # Normalize to [0,1]
                    
                    # Add as similar_to relationship
                    self.graph.add_edge(
                        concept1,
                        concept2,
                        type='similar_to',
                        weight=confidence,
                        method='co-occurrence'
                    )
        
        # 2. Semantic similarity using embeddings
        # Process in batches to avoid memory issues
        concept_batches = [concepts[i:i+100] for i in range(0, len(concepts), 100)]
        
        for batch in concept_batches:
            # Get embeddings for this batch
            embeddings = np.array([self._get_concept_embedding(concept) for concept in batch])
            
            # Compute pairwise similarities
            similarities = cosine_similarity(embeddings)
            
            # Add edges for highly similar concepts
            for i, concept1 in enumerate(batch):
                for j, concept2 in enumerate(batch):
                    if i != j and similarities[i, j] > 0.7:  # Threshold for similarity
                        self.graph.add_edge(
                            concept1,
                            concept2,
                            type='similar_to',
                            weight=similarities[i, j],
                            method='semantic'
                        )
        
        # 3. Detect hierarchical patterns (is-a, part-of)
        for concept1 in concepts:
            for concept2 in concepts:
                if concept1 == concept2:
                    continue
                
                # Check for "is-a" relationships
                if f"{concept1} is a {concept2}" in concept1 or f"{concept1} is an {concept2}" in concept1:
                    self.graph.add_edge(
                        concept1,
                        concept2,
                        type='is_a',
                        weight=0.9,
                        method='pattern'
                    )
                
                # Check for "part-of" relationships
                if f"{concept1} of {concept2}" in concept1 or f"{concept1} in {concept2}" in concept1:
                    self.graph.add_edge(
                        concept1,
                        concept2,
                        type='part_of',
                        weight=0.8,
                        method='pattern'
                    )
        
        # 4. Causal and prerequisite relationships based on temporal analysis
        paper_dates = {}
        for paper_id, paper in self.kb.papers.items():
            if 'published' in paper and paper['published']:
                try:
                    paper_dates[paper_id] = paper['published']
                except:
                    continue
        
        # For each concept, find its first mention date
        concept_first_dates = {}
        for concept in concepts:
            paper_ids = self.kb.concept_index.get(concept, [])
            concept_papers_dates = [paper_dates.get(pid) for pid in paper_ids if pid in paper_dates]
            if concept_papers_dates:
                concept_first_dates[concept] = min(concept_papers_dates)
        
        # Analyze temporal patterns to detect prerequisites
        for concept1, date1 in concept_first_dates.items():
            for concept2, date2 in concept_first_dates.items():
                if concept1 == concept2:
                    continue
                
                # If concept1 appeared significantly before concept2
                if date1 < date2:
                    # Check co-occurrence to ensure they're related
                    if concept_cooccurrence[concept1][concept2] > 0:
                        self.graph.add_edge(
                            concept1,
                            concept2,
                            type='prerequisite',
                            weight=0.6,
                            method='temporal'
                        )
        
        logger.info(f"Added {sum(1 for e in self.graph.edges(data=True) if e[2].get('method') in ['co-occurrence', 'semantic', 'pattern', 'temporal'])} relationship edges between concepts")
    
    def _build_concept_hierarchy(self):
        """Build a hierarchical organization of concepts using clustering and graph metrics"""
        logger.info("Building concept hierarchy")
        
        # Get all concepts
        concepts = [node for node, attr in self.graph.nodes(data=True) if attr.get('type') == 'concept']
        
        if not concepts:
            logger.warning("No concepts in graph to build hierarchy")
            return
        
        # 1. Get concept embeddings
        concept_embeddings = {}
        for concept in concepts:
            concept_embeddings[concept] = self._get_concept_embedding(concept)
        
        # Convert to numpy array for clustering
        embedding_matrix = np.array([concept_embeddings[c] for c in concepts])
        
        # 2. Hierarchical clustering
        try:
            # Compute linkage matrix
            Z = linkage(embedding_matrix, method='ward')
            
            # Form flat clusters with automatic threshold determination
            max_clusters = min(20, len(concepts) // 5) if len(concepts) > 20 else 5
            labels = fcluster(Z, max_clusters, criterion='maxclust')
            
            # Store clusters
            clusters = defaultdict(list)
            for concept, label in zip(concepts, labels):
                clusters[label].append(concept)
            
            # Find the most representative concept for each cluster
            for cluster_id, cluster_concepts in clusters.items():
                # Calculate centrality for each concept in the subgraph
                subgraph = self.graph.subgraph(cluster_concepts)
                centrality = nx.eigenvector_centrality_numpy(subgraph.to_undirected(), max_iter=1000, tol=1e-06)
                
                # Sort by centrality
                sorted_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                
                if sorted_concepts:
                    cluster_name = sorted_concepts[0][0]
                    self.concept_clusters[cluster_name] = cluster_concepts
                    
                    # Create hierarchy entries for each concept in cluster
                    for concept in cluster_concepts:
                        self.concept_hierarchy[concept] = {
                            'cluster': cluster_id,
                            'cluster_name': cluster_name,
                            'centrality': centrality.get(concept, 0)
                        }
                        
                        # Add hierarchical edge in graph
                        if concept != cluster_name:
                            self.graph.add_edge(
                                concept,
                                cluster_name,
                                type='part_of',
                                weight=0.7,
                                method='hierarchical'
                            )
            
            logger.info(f"Created {len(clusters)} concept clusters")
            
        except Exception as e:
            logger.error(f"Error in hierarchical clustering: {e}")
    
    def _identify_research_frontiers(self):
        """Identify emerging research frontiers based on multiple indicators:
        1. Recent growth in publications
        2. Citation patterns
        3. Network structure (betweenness centrality)
        4. Topic novelty
        """
        logger.info("Identifying research frontiers")
        
        # Get concepts and papers
        concepts = [node for node, attr in self.graph.nodes(data=True) if attr.get('type') == 'concept']
        papers = self.kb.papers
        
        if not concepts or not papers:
            logger.warning("Not enough data to identify research frontiers")
            return
        
        # Calculate concept growth rate (recent papers / older papers)
        concept_growth = {}
        
        # Get current year and define horizon
        from datetime import datetime
        current_year = datetime.now().year
        recent_horizon = current_year - 2  # Papers from last 2 years
        
        for concept in concepts:
            paper_ids = self.kb.concept_index.get(concept, [])
            
            recent_count = 0
            older_count = 0
            
            for paper_id in paper_ids:
                if paper_id in papers:
                    paper = papers[paper_id]
                    if 'published' in paper and paper['published']:
                        try:
                            # Extract year from ISO format date
                            year = int(paper['published'][:4])
                            if year >= recent_horizon:
                                recent_count += 1
                            else:
                                older_count += 1
                        except:
                            older_count += 1
            
            # Calculate growth rate
            if older_count > 0:
                growth_rate = recent_count / older_count
            else:
                growth_rate = recent_count if recent_count > 0 else 0
                
            concept_growth[concept] = growth_rate
        
        # Calculate betweenness centrality to find concepts bridging different areas
        concept_betweenness = {}
        try:
            # Use only the concept subgraph
            concept_subgraph = self.graph.subgraph(concepts).to_undirected()
            betweenness = nx.betweenness_centrality(concept_subgraph)
            concept_betweenness = betweenness
        except Exception as e:
            logger.error(f"Error calculating betweenness: {e}")
            # Fallback: assign zero betweenness
            concept_betweenness = {concept: 0 for concept in concepts}
        
        # Calculate novelty score based on uniqueness of concept combinations
        concept_novelty = {}
        for concept in concepts:
            paper_ids = self.kb.concept_index.get(concept, [])
            related_concepts = set()
            
            for paper_id in paper_ids:
                if paper_id in papers:
                    related_concepts.update(papers[paper_id].get('concepts', []))
            
            # Remove self from related concepts
            if concept in related_concepts:
                related_concepts.remove(concept)
            
            # Novelty is inversely proportional to number of related concepts
            if related_concepts:
                concept_novelty[concept] = 1 / math.sqrt(len(related_concepts))
            else:
                concept_novelty[concept] = 1.0  # Maximum novelty for isolated concepts
        
        # Combine metrics to calculate frontier score
        frontier_scores = {}
        for concept in concepts:
            growth = concept_growth.get(concept, 0)
            betweenness = concept_betweenness.get(concept, 0)
            novelty = concept_novelty.get(concept, 0)
            
            # Weighted combination
            frontier_score = (0.6 * growth) + (0.25 * betweenness) + (0.15 * novelty)
            frontier_scores[concept] = frontier_score
        
        # Sort concepts by frontier score
        sorted_frontiers = sorted(frontier_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N as research frontiers
        top_n = min(10, len(sorted_frontiers))
        self.research_frontiers = [concept for concept, score in sorted_frontiers[:top_n]]
        
        # Store core concepts (high betweenness)
        self.core_concepts = {concept: betweenness for concept, betweenness in concept_betweenness.items() 
                             if betweenness > 0.01}  # Threshold for significance
        
        logger.info(f"Identified {len(self.research_frontiers)} research frontiers")
        
        # Store the frontier concepts with their scores for later reference
        for concept, score in sorted_frontiers[:top_n]:
            # Add to evolution tracking with current score
            self.concept_evolution[concept].append({
                'time': datetime.now().isoformat(),
                'frontier_score': score,
                'growth': concept_growth.get(concept, 0),
                'betweenness': concept_betweenness.get(concept, 0),
                'novelty': concept_novelty.get(concept, 0)
            })
            
            # Add as node attribute
            if concept in self.graph:
                self.graph.nodes[concept]['frontier_score'] = score
                self.graph.nodes[concept]['is_frontier'] = True
    
    def get_concept_importance(self, limit=None):
        """Get concepts ranked by importance using PageRank
        
        Args:
            limit (int, optional): Limit number of results
            
        Returns:
            list: List of dictionaries with concept and importance score
        """
        # Check if we have a cached result
        if 'concept_importance' in self._metrics_cache:
            ranks = self._metrics_cache['concept_importance']
        else:
            # Calculate PageRank on the concept subgraph
            concept_nodes = [node for node, attr in self.graph.nodes(data=True) if attr.get('type') == 'concept']
            concept_subgraph = self.graph.subgraph(concept_nodes)
            
            try:
                pagerank = nx.pagerank(concept_subgraph)
                ranks = [(concept, score) for concept, score in pagerank.items()]
                ranks.sort(key=lambda x: x[1], reverse=True)
                
                # Cache the result
                self._metrics_cache['concept_importance'] = ranks
            except:
                # Fallback: use degree centrality
                degree = nx.degree_centrality(concept_subgraph)
                ranks = [(concept, score) for concept, score in degree.items()]
                ranks.sort(key=lambda x: x[1], reverse=True)
        
        # Format the results
        result = [{'concept': concept, 'importance': float(score)} for concept, score in ranks]
        
        # Limit if necessary
        if limit is not None:
            result = result[:limit]
            
        return result
    
    def get_related_concepts(self, concept, relation_type=None, min_weight=0.0):
        """Get concepts related to a given concept
        
        Args:
            concept (str): The concept to find relations for
            relation_type (str, optional): Filter by specific relation type
            min_weight (float, optional): Minimum weight threshold
            
        Returns:
            list: List of related concepts with relation information
        """
        if concept not in self.graph:
            return []
            
        related = []
        
        # Get outgoing edges
        for _, target, data in self.graph.out_edges(concept, data=True):
            if data.get('type') != 'has_concept' and self.graph.nodes[target].get('type') == 'concept':
                if relation_type is None or data.get('type') == relation_type:
                    if data.get('weight', 0) >= min_weight:
                        related.append({
                            'concept': target,
                            'relation': data.get('type', 'related'),
                            'weight': data.get('weight', 0.0),
                            'direction': 'outgoing'
                        })
        
        # Get incoming edges
        for source, _, data in self.graph.in_edges(concept, data=True):
            if data.get('type') != 'has_concept' and self.graph.nodes[source].get('type') == 'concept':
                if relation_type is None or data.get('type') == relation_type:
                    if data.get('weight', 0) >= min_weight:
                        related.append({
                            'concept': source,
                            'relation': data.get('type', 'related'),
                            'weight': data.get('weight', 0.0),
                            'direction': 'incoming'
                        })
        
        # Sort by weight
        related.sort(key=lambda x: x['weight'], reverse=True)
        return related
    
    def get_papers_for_concept(self, concept, limit=None):
        """Get papers associated with a concept
        
        Args:
            concept (str): The concept to find papers for
            limit (int, optional): Limit number of results
            
        Returns:
            list: List of paper IDs
        """
        if concept not in self.graph:
            return []
            
        # Get papers directly from knowledge base
        paper_ids = self.kb.concept_index.get(concept, [])
        
        # Sort by recency if dates available
        sorted_papers = []
        for paper_id in paper_ids:
            if paper_id in self.kb.papers:
                paper = self.kb.papers[paper_id]
                published = paper.get('published', '')
                sorted_papers.append((paper_id, published))
        
        # Sort by publication date (descending)
        sorted_papers.sort(key=lambda x: x[1], reverse=True)
        
        # Get paper IDs only
        paper_ids = [paper_id for paper_id, _ in sorted_papers]
        
        # Limit if necessary
        if limit is not None:
            paper_ids = paper_ids[:limit]
            
        return paper_ids
    
    def get_research_frontiers(self, limit=None):
        """Get current research frontiers
        
        Args:
            limit (int, optional): Limit number of results
            
        Returns:
            list: List of frontier concepts with associated information
        """
        frontiers = []
        
        for concept in self.research_frontiers:
            # Get related papers
            paper_ids = self.get_papers_for_concept(concept, limit=5)
            papers = []
            
            for paper_id in paper_ids:
                if paper_id in self.kb.papers:
                    papers.append({
                        'id': paper_id,
                        'title': self.kb.papers[paper_id].get('title', 'Unknown'),
                        'published': self.kb.papers[paper_id].get('published', '')
                    })
            
            # Get related concepts
            related = self.get_related_concepts(concept, min_weight=0.4)
            
            # Get evolution data
            evolution = self.concept_evolution.get(concept, [])
            
            frontiers.append({
                'concept': concept,
                'papers': papers,
                'related_concepts': related,
                'evolution': evolution,
                'cluster': self.concept_hierarchy.get(concept, {}).get('cluster_name', '')
            })
        
        # Limit if necessary
        if limit is not None:
            frontiers = frontiers[:limit]
            
        return frontiers
    
    def get_concept_clusters(self, limit=None):
        """Get concept clusters with key related concepts
        
        Args:
            limit (int, optional): Limit number of clusters
            
        Returns:
            list: List of clusters with related concepts
        """
        clusters = []
        
        for cluster_name, concepts in self.concept_clusters.items():
            # Get top concepts in this cluster by centrality
            top_concepts = sorted(
                [(c, self.concept_hierarchy.get(c, {}).get('centrality', 0)) for c in concepts],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Take top 5 concepts
            top_concepts = [c[0] for c in top_concepts[:5]]
            
            clusters.append({
                'name': cluster_name,
                'concepts': top_concepts,
                'size': len(concepts)
            })
        
        # Sort by cluster size
        clusters.sort(key=lambda x: x['size'], reverse=True)
        
        # Limit if necessary
        if limit is not None:
            clusters = clusters[:limit]
            
        return clusters
    
    def visualize(self, output_path=None, max_nodes=100):
        """Visualize the knowledge graph
        
        Args:
            output_path (str, optional): Path to save the visualization
            max_nodes (int, optional): Maximum number of nodes to include
            
        Returns:
            str: Path to the saved visualization or None if failed
        """
        if not output_path:
            output_path = os.path.join(CONFIG.get('models_dir', '.'), 'knowledge_graph.png')
            
        try:
            # Create a smaller subgraph for visualization
            if len(self.graph) > max_nodes:
                # Focus on frontiers and core concepts
                key_concepts = set(self.research_frontiers + list(self.core_concepts.keys()))
                
                # Limit to max_nodes
                if len(key_concepts) > max_nodes:
                    key_concepts = list(key_concepts)[:max_nodes]
                
                # Create subgraph
                subgraph = self.graph.subgraph(key_concepts)
            else:
                subgraph = self.graph
            
            # Set up layout
            pos = nx.spring_layout(subgraph)
            
            # Set up figure
            plt.figure(figsize=(12, 10))
            
            # Draw nodes
            nx.draw_networkx_nodes(
                subgraph, 
                pos, 
                node_size=[300 if node in self.research_frontiers else 100 for node in subgraph],
                node_color=['red' if node in self.research_frontiers else 'blue' for node in subgraph],
                alpha=0.7
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                subgraph, 
                pos, 
                width=[data.get('weight', 0.5) for _, _, data in subgraph.edges(data=True)],
                alpha=0.5
            )
            
            # Draw labels
            nx.draw_networkx_labels(subgraph, pos, font_size=8)
            
            # Set title and save
            plt.title(f"Knowledge Graph: {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved knowledge graph visualization to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error visualizing knowledge graph: {e}")
            return None
    
    def export_graph(self, output_path=None):
        """Export the knowledge graph to JSON format
        
        Args:
            output_path (str, optional): Path to save the exported graph
            
        Returns:
            str: Path to the saved file or None if failed
        """
        if not output_path:
            output_path = os.path.join(CONFIG.get('models_dir', '.'), 'knowledge_graph.json')
            
        try:
            # Create serializable representation
            graph_data = {
                'nodes': [],
                'edges': [],
                'frontiers': self.research_frontiers,
                'clusters': [{'name': name, 'concepts': concepts[:10]} for name, concepts in self.concept_clusters.items()],
                'metrics': {
                    'nodes': len(self.graph.nodes),
                    'edges': len(self.graph.edges),
                    'frontiers': len(self.research_frontiers),
                    'clusters': len(self.concept_clusters)
                }
            }
            
            # Add nodes
            for node, attrs in self.graph.nodes(data=True):
                node_data = {'id': node}
                node_data.update({k: v for k, v in attrs.items() if isinstance(v, (str, int, float, bool, list, dict))})
                graph_data['nodes'].append(node_data)
            
            # Add edges
            for source, target, attrs in self.graph.edges(data=True):
                edge_data = {
                    'source': source,
                    'target': target
                }
                edge_data.update({k: v for k, v in attrs.items() if isinstance(v, (str, int, float, bool, list, dict))})
                graph_data['edges'].append(edge_data)
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
                
            logger.info(f"Exported knowledge graph to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting knowledge graph: {e}")
            return None
