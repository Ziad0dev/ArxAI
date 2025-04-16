# ARX2: Autonomous AI Research System

ARX2 is a state-of-the-art autonomous research system that continuously learns from academic literature in computing, machine learning, and AI fields. It discovers, analyzes, and synthesizes research papers to generate insights and track emerging trends.

## Features

### Autonomous Research Pipeline
- **Multi-source paper discovery**: Finds papers from arXiv, Semantic Scholar, and local files
- **Advanced document processing**: Extracts text, sections, concepts, and code from various formats
- **Semantic understanding**: Analyzes paper structure and content with transformer models
- **Knowledge graph construction**: Creates a rich network of concepts, papers, and relationships

### Knowledge Synthesis & Insight Generation
- **Research trend analysis**: Tracks emerging topics and concept evolution over time
- **Topic modeling**: Discovers latent themes across the paper corpus
- **Citation analysis**: Identifies influential papers and citation patterns
- **Research frontier detection**: Highlights cutting-edge areas ripe for exploration

### Technical Architecture
- **Modular design**: Separate components for ingestion, processing, knowledge representation, and analysis
- **Vectorized knowledge base**: Efficient similarity search via FAISS indexing
- **Transformer-based embeddings**: High-quality representation of research content
- **Scalable processing**: Supports large paper collections with optimized batch processing

### Evaluation & Reporting
- **Comprehensive metrics**: Measures growth rates, influence, network centrality, and more
- **Interactive visualizations**: Generates graphs, charts, and knowledge maps
- **Exportable insights**: Structured JSON output for further analysis or integration
- **Continuous learning**: Improves understanding as new research is ingested

## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arx2.git
cd arx2
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install optional dependencies for additional features:
```bash
pip install -r requirements_optional.txt
```

### Configuration

Copy the example configuration and modify as needed:
```bash
cp config/config.example.json config/config.json
# Edit config.json to customize settings
```

Key configuration options:
- `papers_dir`: Directory to store downloaded papers
- `models_dir`: Directory to store trained models and embeddings
- `embedding_size`: Dimension of paper/concept embeddings
- `sentence_transformer_model`: Model to use for embeddings

## Usage

### Paper Ingestion

Ingest papers from arXiv:
```bash
python paper_ingestion.py arxiv "machine learning"
```

Ingest papers from Semantic Scholar:
```bash
python paper_ingestion.py semantic "deep learning transformers"
```

Process local PDF files:
```bash
python paper_ingestion.py directory /path/to/papers
```

Ingest from multiple sources with a single command:
```bash
python paper_ingestion.py multi "reinforcement learning"
```

### Knowledge Graph Construction

Build the knowledge graph:
```bash
python run_advanced_mode.py --build-graph
```

Visualize the knowledge graph:
```bash
python knowledge_graph.py --visualize
```

### Research Analysis

Run a comprehensive research trend analysis:
```bash
python research_trends_analyzer.py --analysis all
```

Generate a report on specific research areas:
```bash
python research_trends_analyzer.py --analysis topics --topics 15
```

Analyze temporal trends:
```bash
python research_trends_analyzer.py --analysis temporal --years 3
```

### Training Models

Train a research domain classifier:
```bash
bash run_training.sh
```

Or customize the training with specific parameters:
```bash
python train.py --train_data data/papers_train.json --output_dir checkpoints/custom_run --batch_size 16 --num_epochs 10
```

## Architecture Overview

```
arx2/
├── advanced_ai_analyzer_*.py  # Core components
├── train.py                    # Model training
├── paper_ingestion.py          # Paper discovery and intake
├── knowledge_graph.py          # Knowledge representation
├── research_trends_analyzer.py # Insight generation
├── utils/                      # Shared utilities
├── data/                       # Data storage
├── models/                     # Model storage
└── research_output/            # Generated insights
```

## Extending the System

ARX2 is designed to be modular and extensible:

1. **Add new data sources**:
   - Implement a new method in the `PaperIngestionSystem` class
   - Follow the pattern in `ingest_from_arxiv` or `ingest_from_semantic_scholar`

2. **Add new analysis methods**:
   - Add a new method to `ResearchTrendsAnalyzer`
   - Register it in the command-line interface in `main()`

3. **Customize the knowledge graph**:
   - Extend relationship types in `EnhancedKnowledgeGraph.relation_types`
   - Add new detection methods in `_detect_concept_relationships`

4. **Integrate with other tools**:
   - Export data using `export_graph` method
   - Call other analysis tools using the exported JSON data

## Citation

If you use ARX2 in your research, please cite:

```
@software{arx2_2025,
  author = {ARX2 Team},
  title = {ARX2: Autonomous AI Research System},
  year = {2025},
  url = {https://github.com/yourusername/arx2}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ARX2 builds upon various open-source libraries including PyTorch, HuggingFace Transformers, FAISS, NetworkX, scikit-learn, and more.
- Special thanks to the research community for their valuable feedback and contributions.
