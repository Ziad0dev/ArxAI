# ArX2: Advanced Research Document Analysis System

ArX2 is a comprehensive system for harvesting, processing, and analyzing research papers from academic repositories like arXiv.

## Features

- **Paper Harvesting**: Collect papers from arXiv and other sources based on search queries
- **PDF Processing**: Download and extract content from research paper PDFs
- **Model Training**: Train machine learning models on harvested research papers
- **Analysis**: Extract insights and patterns from research papers

## Quick Start

The easiest way to get started is to use the setup script:

```bash
# Clone the repository
git clone https://github.com/yourusername/arx2.git
cd arx2

# Make the setup script executable
chmod +x setup_training_env.sh

# Run the setup script
./setup_training_env.sh

# Activate the virtual environment
source .venv/bin/activate
```

## Paper Harvesting

Harvest papers from arXiv:

```bash
python -m arx2.examples.harvest_arxiv_papers \
    --output-dir ./harvested_data \
    --query "cat:cs.AI AND \"large language model\"" \
    --max-results 100 \
    --download-pdfs
```

## Model Training

Train a model using harvested papers:

```bash
python -m arx2.examples.train_from_harvested \
    --harvested-dir ./harvested_data \
    --output-dir ./trained_models
```

Or harvest and train in one step:

```bash
python -m arx2.examples.train_from_harvested \
    --output-dir ./trained_models \
    --harvest-new \
    --query "cat:cs.AI AND \"large language model\"" \
    --max-papers 100
```

## Manual Installation

If you prefer to set up the environment manually:

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Directory Structure

- `arx2/`: Main package
  - `harvesting/`: Paper harvesting modules
  - `examples/`: Example scripts
  - `models/`: Model definitions
- `continuous_training/`: Training system components
- `requirements.txt`: Dependencies

## Documentation

See the example directories for detailed README files on specific components:

- [Paper Harvesting README](arx2/harvesting/README.md)
- [Training Models README](arx2/examples/README_TRAINING.md)

## License

[MIT License](LICENSE)
