# ENTROPHY: Multi-modal User Interaction Data from Live Enterprise Business Workflows

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)


**ENTROPHY** is the first open, click-level record of how finance, legal, and HR workflows are carried out on real enterprise software—modern SaaS, browsers, and legacy desktop apps alike. Nine domain professionals executed 283 authentic workflow runs over five working days, producing 33 hours of screen-level activity across 19 applications.

This repository contains the accompanying code for ENTROPHY that evaluates the zero-shot performance of frontier Large Language Models (LLMs) on workflow classification and segmentation tasks.

---

## 📖 Table of Contents

- [🌟 Key Features](#-key-features)
- [🏗️ Project Structure](#-project-structure)
- [⚡ Quick Start](#-quick-start)
- [🔧 Installation](#-installation)
- [📊 Dataset](#-dataset)
- [⚙️ Configuration](#-configuration)
- [🚀 Usage](#-usage)
  - [Workflow Classification](#workflow-classification)
  - [Workflow Segmentation](#workflow-segmentation)
  - [Similarity Analysis](#similarity-analysis)
  - [Generating Visualizations](#generating-visualizations)
- [🧪 Experiments & Evaluation](#-experiments--evaluation)
<!-- - [📈 Results & Outputs](#-results--outputs) -->
- [📄 License](#-license)
<!-- - [📚 Citation](#-citation) -->

---

## 🌟 Key Features

- **Multi-domain Workflow Analysis**: Process user interaction data from diverse enterprise domains
- **Zero-shot LLM Evaluation**: Test frontier models on workflow understanding without fine-tuning
- **Comprehensive Task Suite**: 
  - **Classification**: Identify workflow types from interaction sequences
  - **Segmentation**: Split concatenated workflows into individual process instances
  - **Similarity Analysis**: Compare workflow patterns using semantic embeddings
- **Multi-provider LLM Support**: OpenAI, Anthropic, Google, Hugging Face models
- **Rich Visualizations**: Generate publication-quality plots and analysis reports
- **Reproducible Research**: Detailed configuration and comprehensive logging

---

## 🏗️ Project Structure

```
entrophy/
├── 📄 README.md                     # Comprehensive project documentation
├── 📄 LICENSE                       # CC BY-NC-SA 4.0 International License
├── 📄 requirements.txt              # Python dependencies with versions
│
├── 📁 src/                          # Source code
│   ├── 📄 __init__.py               # Package initialization
│   ├── 📄 data_processor.py         # Data loading and preprocessing utilities
│   ├── 📄 classification.py         # Workflow classification implementation
│   ├── 📄 segmentation.py           # Workflow segmentation implementation
│   ├── 📄 generate_plots.py         # Visualization and plotting utilities
│   └── 📁 similarity_analysis/     # Similarity analysis modules
│       ├── 📄 similarities.ipynb   # Jupyter notebook for similarity analysis
│       ├── 📄 process_similarity_heatmap.pdf  # Generated heatmap
│       ├── 📄 finance_embeddings.csv         # Pre-computed embeddings
│       ├── 📄 legal_embeddings.csv           # Pre-computed embeddings
│       └── 📄 hr_embeddings.csv              # Pre-computed embeddings
│
├── 📁 configs/                      # Configuration files
│   ├── 📄 classification.yaml      # Classification task configuration
│   └── 📄 segmentation.yaml       # Segmentation task configuration
│
├── 📁 data/                         # Dataset files
│   ├── 📄 process_definitions.json # Workflow process definitions
│   └── 📁 json/                    # Raw interaction data
│       ├── 📄 hr.json              # HR domain workflows (download from Kaggle)
│       ├── 📄 legal.json           # Legal domain workflows (download from Kaggle)
│       └── 📄 finance.json         # Finance domain workflows (download from Kaggle)
│
├── 📁 outputs/                      # Experimental results
└── 📁 figures/                      # Generated visualizations
```

---

## ⚡ Quick Start

Get up and running with ENTROPHY in under 5 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/entrophy.git
cd entrophy

# 2. Set up environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run a quick classification experiment
python src/classification.py --config configs/classification.yaml

# 5. Generate similarity analysis (optional)
cd src/similarity_analysis && jupyter notebook similarities.ipynb
```

---

## 🔧 Installation

### Prerequisites

- **Python**: 3.11 or higher
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ available space
- **GPU**: Optional but recommended for local model inference

### Environment Setup

#### Option 1: Conda (Recommended)

```bash
# Create conda environment
conda create -n entrophy python=3.11
conda activate entrophy

# Install PyTorch with CUDA support (optional)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

#### Option 2: Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Verification

```bash
# Test the installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from src.data_processor import WorkflowDataProcessor; print('✅ Installation successful!')"
```

---

## 📊 Dataset

### Overview

ENTROPHY contains real-world workflow data from three enterprise domains:

| Domain | Workflows | Size  | Description |
|--------|-----------|------|-------------|
| **Finance** | 94 instances | 14MB |  Invoice processing, payments, revenue accounting |
| **Legal** | 89 instances | 9.6MB  | MSA/SOW creation, contract iterations |
| **HR** | 100 instances | 13MB  | Employee onboarding, leave management |



## ⚙️ Configuration

### Configuration Files

ENTROPHY uses YAML configuration files for reproducible experiments:

#### Classification Configuration (`configs/classification.yaml`)


#### Segmentation Configuration (`configs/segmentation.yaml`)

---

## 🚀 Usage

### Workflow Classification

Classify individual workflow instances into predefined categories:

```bash
# Using configuration file
python src/classification.py --config configs/classification.yaml
```

### Workflow Segmentation

Segment concatenated workflows into individual process instances:

```bash
# Using configuration file
python src/segmentation.py --config configs/segmentation.yaml
```

### Similarity Analysis

Analyze workflow similarities using semantic embeddings:

#### Interactive Analysis

```bash
# Open Jupyter notebook
cd src/similarity_analysis
jupyter notebook similarities.ipynb
```

### Generating Visualizations

Create publication-quality plots and analysis reports:

#### Generate All Plots

```bash
python src/generate_plots.py --input_root outputs/ --output_dir figures/
```

#### Available Visualizations

- **Classification Performance**: Accuracy, precision, recall, F1-score
- **Confusion Matrices**: Model performance breakdown by class
- **Segmentation Metrics**: Boundary detection accuracy, edit distance
- **Cross-domain Comparisons**: Performance across HR, Legal, Finance
- **Similarity Heatmaps**: Workflow pattern similarities

---

## 🧪 Experiments & Evaluation

### Evaluation Metrics

#### Classification Metrics

- **Accuracy**: Overall classification correctness
- **Precision/Recall/F1**: Per-class and macro-averaged metrics
- **Confusion Matrix**: Detailed error analysis
- **Support**: Number of instances per class

#### Segmentation Metrics

- **Boundary Precision/Recall**: Accuracy of workflow boundary detection
- **Edit Distance**: Sequence alignment cost between predicted and true segments
- **Tolerance-based Accuracy**: Boundary detection within tolerance window


---

## 📈 Results & Outputs

### Output Structure

```
outputs/
├── finance/
│   ├── classification/
│       ├──gpt-4.1/
│           ├── report.json                     # Classification metrics
│           ├── confusion_matrix.json           # Confusion matrix data
│           ├── detailed_results.json           # Per-instance results
│           └── prompt_and_generated_text.json  # Raw model outputs
│       ├──deepseek-r1/
│           ├── ...
│   └── segmentation/
│       ├──gpt-4.1/
│           ├── segmentation_results.json      # Segmentation metrics
│           ├── prompts_and_responses.json     # Raw model outputs
│           └── visualization_sequence_*.png   # Per-instance visuals
│       ├──deepseek-r1/
│           ├── ...
├── hr/
│   ├── ...
└── legal/
    ├── ...
```

---

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.

**Key Points:**
- ✅ **Attribution**: Credit must be given to the original authors
- ✅ **NonCommercial**: Only non-commercial use is allowed
- ✅ **ShareAlike**: Adaptations must use the same license
- ✅ **Academic Research**: Permitted and encouraged
- ❌ **Commercial Use**: Not permitted without explicit permission

For the complete license text, see [LICENSE](LICENSE) or visit [https://creativecommons.org/licenses/by-nc-sa/4.0/](https://creativecommons.org/licenses/by-nc-sa/4.0/)

<!-- ### Commercial Licensing

For commercial use, please contact [Workfabric](mailto:contact@workfabric.com) for licensing arrangements. -->

---

<!-- ## 📚 Citation

If you use ENTROPHY in your research, please cite our work:

```bibtex
@misc{entrophy2025,
  title={ENTROPHY: Multi-modal User Interaction Data from Live Enterprise Business Workflows},
  author={Workfabric Team},
  year={2025},
  publisher={Workfabric},
  howpublished={\url{https://github.com/workfabric/entrophy}},
  note={Licensed under CC BY-NC-SA 4.0}
}
``` 

--- -->

## 🙏 Acknowledgments

- **Workfabric Team** for dataset collection and curation
- **Domain Experts** who participated in workflow data collection

---

**© 2025 Workfabric. ENTROPHY is licensed under CC BY-NC-SA 4.0 International.**

