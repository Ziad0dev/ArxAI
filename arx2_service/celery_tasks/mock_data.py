"""
ARX2 Research API - Mock Data
----------------------------
Mock data for testing and development purposes.
"""

# Sample research frontiers based on the actual research summary
research_frontiers = [
    {'concept': 'learning', 'growth_rate': 0.0, 'citation_importance': 0.0, 'frequency': 654, 'cluster': [], 'papers': ['2108.11510', '2401.02349', '1909.04751', '1708.05866', '2106.10112'], 'importance': 0.17053455019556715},
    {'concept': 'learn', 'growth_rate': 0.0, 'citation_importance': 0.0, 'frequency': 627, 'cluster': [], 'papers': ['2108.11510', '2401.02349', '1909.04751', '1708.05866', '2106.10112'], 'importance': 0.1634941329856584},
    {'concept': 'model', 'growth_rate': 0.0, 'citation_importance': 0.0, 'frequency': 581, 'cluster': [], 'papers': ['2108.11510', '1909.04751', '1708.05866', '2106.10112', '2009.07888'], 'importance': 0.1514993481095176},
    {'concept': 'network', 'growth_rate': 0.0, 'citation_importance': 0.0, 'frequency': 536, 'cluster': [], 'papers': ['2108.11510', '2401.02349', '1909.04751', '1708.05866', '2106.10112'], 'importance': 0.13976531942633638},
    {'concept': 'neural network', 'growth_rate': 0.0, 'citation_importance': 0.0, 'frequency': 479, 'cluster': [], 'papers': ['2108.11510', '2401.02349', '1909.04751', '1708.05866', '2106.10112'], 'importance': 0.12490221642764017},
    {'concept': 'deep', 'growth_rate': 0.0, 'citation_importance': 0.0, 'frequency': 466, 'cluster': [], 'papers': ['2108.11510', '2401.02349', '1909.04751', '1708.05866', '2106.10112'], 'importance': 0.12151238591916558},
    {'concept': 'neural', 'growth_rate': 0.0, 'citation_importance': 0.0, 'frequency': 455, 'cluster': [], 'papers': ['2108.11510', '2401.02349', '1909.04751', '1708.05866', '2106.10112'], 'importance': 0.11864406779661017},
    {'concept': 'base', 'growth_rate': 0.0, 'citation_importance': 0.0, 'frequency': 446, 'cluster': [], 'papers': ['2108.11510', '2401.02349', '1909.04751', '1708.05866', '1609.03348'], 'importance': 0.11629726205997393},
    {'concept': 'datum', 'growth_rate': 0.0, 'citation_importance': 0.0, 'frequency': 405, 'cluster': [], 'papers': ['1909.04751', '2008.02708', '1806.07692', '2109.00525', '2302.06370'], 'importance': 0.10560625814863103},
    {'concept': 'method', 'growth_rate': 0.0, 'citation_importance': 0.0, 'frequency': 392, 'cluster': [], 'papers': ['2108.11510', '2401.02349', '1909.04751', '1708.05866', '2009.07888'], 'importance': 0.10221642764015647}
]

# Sample paper data
paper_samples = [
    {
        "paper_id": "2108.11510",
        "title": "Deep Reinforcement Learning for Neural Network Optimization",
        "authors": ["Smith, John", "Johnson, Robert", "Williams, Sarah"],
        "abstract": "This paper presents a novel approach to optimizing neural networks using deep reinforcement learning techniques...",
        "year": 2021,
        "url": "https://arxiv.org/abs/2108.11510",
        "citations": 42
    },
    {
        "paper_id": "2401.02349",
        "title": "Transfer Learning in Multi-Modal Neural Architectures",
        "authors": ["Chen, Wei", "Garcia, Maria", "Kumar, Raj"],
        "abstract": "We explore transfer learning approaches for multi-modal neural network architectures that can process both visual and textual data...",
        "year": 2024,
        "url": "https://arxiv.org/abs/2401.02349",
        "citations": 18
    },
    {
        "paper_id": "1909.04751",
        "title": "Attention Mechanisms for Deep Learning Models",
        "authors": ["Zhang, Li", "Anderson, Thomas", "Wilson, Emily"],
        "abstract": "This work investigates various attention mechanisms and their impact on the performance of deep learning models...",
        "year": 2019,
        "url": "https://arxiv.org/abs/1909.04751",
        "citations": 312
    },
    {
        "paper_id": "1708.05866",
        "title": "Gradient Descent Optimization for Neural Networks",
        "authors": ["Brown, Michael", "Lee, Jennifer", "Patel, Anooj"],
        "abstract": "We propose enhanced gradient descent optimization techniques specifically designed for training deep neural networks...",
        "year": 2017,
        "url": "https://arxiv.org/abs/1708.05866",
        "citations": 271
    },
    {
        "paper_id": "2106.10112",
        "title": "Regularization Methods for Deep Neural Networks",
        "authors": ["Nguyen, Tran", "Davis, Anna", "Gupta, Rahul"],
        "abstract": "This paper examines various regularization techniques to prevent overfitting in deep neural networks...",
        "year": 2021,
        "url": "https://arxiv.org/abs/2106.10112",
        "citations": 85
    }
] 