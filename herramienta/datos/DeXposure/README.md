# DeXposure: A Dataset and Benchmarks for Inter-protocol Credit Exposure in Decentralized Financial Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository contains the DeXposure dataset and benchmarks for analyzing credit exposure networks in decentralized financial (DeFi) networks. More details are in the following paper:

Wenbin Wu, Kejiang Qian, Alexis Lui, Christopher Jack, Yue Wu, Peter McBurney, Fengxiang He, Bryan Zheng Zhang. DeXposure: A Dataset and Benchmarks for Inter-protocol Credit Exposure in Decentralized Financial Networks. 

- arXiv: [https://arxiv.org/pdf/2511.22314](https://arxiv.org/pdf/2511.22314)  
- Kaggle: [https://www.kaggle.com/datasets/spicycorn/defi-network-dataset-2020-2025](https://www.kaggle.com/datasets/spicycorn/defi-network-dataset-2020-2025)  
- Visualisation: [https://ccaf.io/defi/ecosystem-map/visualisation/graph](https://ccaf.io/defi/ecosystem-map/visualisation/graph)

It is a joint project of the following institutes: 

- Cambridge Centre for Alternative Finance, University of Cambridge
- School of Informatics, University of Edinburgh
- Department of Informatics, King's College London

## Overview

This project introduces the first inter-protocol credit exposure network encompassing thousands of DeFi protocols. The dataset comprises **43.7 million entries** covering **4,300+ protocols**, **602 blockchains**, and **24,300+ unique tokens** from 2020 to 2025.

### Key Features

- **Network Construction**: Build comprehensive credit exposure networks from TVL data
- **Temporal Analysis**: Track network evolution across 283 weekly snapshots
- **Shock Analysis**: Study major market events (Terra collapse, FTX crash)
- **Dynamic Link Prediction**: GNN-based temporal link prediction using ROLAND framework
- **Network Metrics**: Comprehensive network analysis and visualization tools

## Quick Start
### Option 1: Run with Docker (Recommended)
We provide a pre-built Docker image containing all dependencies and environment settings.
#### **Run the full pipeline:**

```bash
docker run --rm -it \
  --memory=15g \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  jiangkkk/dexposure:latest \
  python main.py --all
```

#### **Explanation:**
* `--memory=15g` → allocate sufficient memory for graph analysis and model training
* `-v "$(pwd)/data:/app/data"` → mount your local `data/` directory into the container
* `-v "$(pwd)/output:/app/output"` → results will be saved to your local `output/` directory
* `--all` → run the complete analysis pipeline (`--network-metrics`, `--link-prediction`, `--model-comparison`)

#### **Run a specific module (e.g., only network metrics):**

```bash
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  jiangkkk/dexposure:latest \
  python main.py ---network-metrics
```

### Option 2: Manual Installation (Python)
### Manual Installation (Python)
If you prefer to run locally without Docker:
```bash
# Clone the repository
git clone https://github.com/yourusername/defi-tvl-public.git
cd defi-tvl-public

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r pyproject.toml
```

### Configuration

The repository includes a `config.yaml` file with all necessary configuration. Update API keys if you need to fetch fresh data:

```yaml
# config.yaml
FLIPSIDE_KEYS:
  - 'your-flipside-api-key'
INFURA_KEYS:
  - 'your-infura-api-key'
COINGECKO_KEY: 'your-coingecko-api-key'
```

### Basic Usage

```python
# Import the necessary modules
from src.data_analysis.network_analysis import process_network_data
from src.models.roland import ROLANDGNN
from src.visualization.create_network_plots import generate_altair_plot

# Load and analyze network data
communities, edges = process_network_data('terra')

# Train GNN model for link prediction
# See scripts/train_link_prediction.py for full example
```

## Data

### Network Snapshots

The repository includes temporal network snapshots:

- **`historical-network_week_2020-03-30.json`** (1 GB): Complete dataset with 283 weekly snapshots from 2020-03-23 to 2025-08-18
- **`historical-network_week_2025-07-01.json`** (75 MB): Recent 8 snapshots for quick testing

### Data Structure

Each snapshot contains:
```json
{
  "data": {
    "YYYY-MM-DD": {
      "nodes": [{"id": "...", "size": ..., "composition": {...}}],
      "links": [{"source": "...", "target": "...", "size": ..., "composition": {...}}]
    }
  }
}
```

### Mapping Files

- `data/mapping/id_to_info.json`: Protocol ID to name/info mapping
- `data/mapping/token_to_protocol.json`: Token to protocol mapping
- `data/mapping/rev_map.json`: Reverse mapping

## Reproducing Paper Results

### 1. Network Metrics (Table 1)

```bash
python scripts/generate_network_metrics.py
```

### 2. Market Shock Analysis (Figures for Terra/FTX)

```python
from src.data_analysis.network_analysis import process_network_data, visualize_communities

# Terra analysis
communities_terra, edges_terra = process_network_data('terra')
visualize_communities(communities_terra, 'terra', edges_terra)

# FTX analysis
communities_ftx, edges_ftx = process_network_data('ftx')
visualize_communities(communities_ftx, 'ftx', edges_ftx)
```

### 3. Dynamic Link Prediction (Figure: AUPRC over time)

```bash
python scripts/train_link_prediction.py
```

This trains the ROLAND GNN model on all temporal snapshots and generates AUPRC scores.

### 4. Model Comparison (Table: Baseline Comparison)

```bash
python scripts/compare_link_prediction_models.py
```

Compares ROLAND against 8 baseline methods:
- **Heuristics**: Adamic-Adar, Common Neighbors, Jaccard, Preferential Attachment
- **Embeddings**: Node2Vec + Logistic Regression
- **Static GNNs**: Vanilla GCN, GAT
- **Temporal GNN**: ROLAND

## Jupyter Notebooks

The `notebooks/` directory contains Jupyter notebooks used for advanced analyses in the paper:

### `network.ipynb`
- **Purpose**: Network topology analysis and phase transition detection
- **Key Analyses**:
  - Community detection using Infomap
  - Network density and clustering coefficients over time
  - Ricci curvature and entropy analysis for phase transitions
- **Paper Figures**: Network evolution plots, change point detection

### `plot.ipynb`
- **Purpose**: Market shock analysis and econometric modeling
- **Key Analyses**:
  - Terra and FTX shock analysis
  - Sectoral inflow/outflow dynamics
  - Vector Autoregression (VAR) models
  - Impulse Response Functions (IRF)
- **Paper Figures**: Cross-sectoral value flow plots, IRF plots

### `GAT.ipynb`
- **Purpose**: Graph Attention Network experiments and anomaly detection
- **Key Analyses**:
  - Unsupervised anomaly detection (Isolation Forest, K-means, DBSCAN)
  - PCA and Autoencoder approaches
  - Network metrics and graph analysis
- **Paper Figures**: GAT performance comparisons

### `data_processing.ipynb`
- **Purpose**: Data validation and preprocessing workflows
- **Key Analyses**:
  - Market cap validation (Table: WBTC & ETHENA)
  - Net inflow vs. supply change correlation
  - Data quality checks

### Running Notebooks

```bash
# Dependencies are already installed via pyproject.toml

# Launch Jupyter
jupyter notebook notebooks/

# Or use JupyterLab
jupyter lab
```

**Note**: Some notebooks require data fetching from APIs. Update `config.yaml` with your API keys if needed.

## Project Structure

```
defi-tvl-public/
├── src/
│   ├── data_access/       # Data fetching and database wrappers
│   ├── data_analysis/     # Network analysis and TVL construction
│   ├── visualization/     # Plotting and visualization tools
│   ├── models/            # ROLAND GNN and data loaders
│   └── utils/             # Utility functions and config loading
├── scripts/               # Executable analysis scripts
├── notebooks/             # Jupyter notebooks for paper analyses
├── data/                  # Network snapshots and mappings
└── output/                # Generated results and figures
```

## Models

### ROLAND GNN

The repository implements the ROLAND (Recurrent One-Layer Approximation of Neural Dynamics) framework for temporal link prediction on evolving networks.

**Architecture:**
- 2 MLP layers for preprocessing node features
- 2 GCN/GAT layers for graph convolution
- GRU-based temporal updates
- Hadamard product for link prediction

**Key Files:**
- `src/models/roland.py`: Base ROLAND implementation
- `src/models/defi_roland_gnn.py`: DeFi-specific variant with GAT support
- `src/models/data_loader.py`: Temporal snapshot data loader

## Network Analysis

### Metrics Computed

- Degree centralization and coefficient of variation
- Degree distribution entropy
- Top 10% degree concentration
- Assortativity coefficient
- Average closeness centrality
- Clustering coefficient
- Ollivier-Ricci curvature
- Network entropy

### Visualization

- t-SNE projections with DBSCAN clustering
- Temporal evolution plots
- Cross-sectoral value flow diagrams
- Community structure visualization

## Advanced Usage

### Custom Network Analysis

```python
from src.data_analysis.etl_network import ETL
import json

# Load network data
etl = ETL()
with open('data/historical-network_week_2020-03-30.json') as f:
    data = json.load(f)

# Process specific time period
snapshot = data['data']['2022-05-09']  # Terra collapse date
# Analyze snapshot...
```

### Training Custom GNN Models

```python
from src.models.defi_roland_gnn import DeFiROLANDGNN
from src.models.data_loader import DeFiDataLoader

# Initialize model
model = DeFiROLANDGNN(
    input_dim=1,
    num_nodes=1000,
    dropout=0.1,
    update='gru',
    use_gat=True
)

# Load and prepare data
loader = DeFiDataLoader(nodes_path, edges_path, id_to_info_path)
snapshots = loader.create_temporal_snapshots(nodes_df, edges_df, freq='W')
```

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{wu2025dexposure,
  title={DeXposure: Decentralised Credit Exposure Dataset and Benchmarks},
  author={Wu et al.},
  journal={},
  year={2025},
  publisher={Cambridge Centre for Alternative Finance}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Cambridge Centre for Alternative Finance
- DefiLlama for TVL data
- ROLAND framework authors

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

**Note**: This is research software. While we strive for accuracy, please verify results independently before using in production or making financial decisions.

