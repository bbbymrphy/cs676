# URL Credibility Analyzer with Neural Network

A comprehensive URL credibility analysis system that combines heuristic scoring with neural network predictions powered by Claude AI.

## 🌟 Features

- **Heuristic Analysis**: URL structure, content quality, PageRank, popularity scoring
- **Neural Network**: Deep learning model trained on Claude-labeled data
- **Claude Integration**: AI-powered credibility assessment
- **Streamlit UI**: Interactive web interface  
- **Web Search**: Automated URL discovery and analysis

## 📁 Project Structure

```
project1/
├── app.py                  # Main Streamlit application
├── deliverable1.py         # Core heuristic scoring logic
├── run_app.sh             # Launch script for Streamlit app
├── requirements.txt       # Python dependencies
│
├── nn/                    # Neural Network Package
│   ├── __init__.py
│   ├── feature_extractor.py    # Extract 46+ features from URLs
│   ├── credibility_nn.py        # PyTorch neural network model
│   ├── web_search.py            # Web search & Claude labeling
│   └── dataset_generator.py    # Training dataset creation
│
├── scripts/               # Training & Utility Scripts
│   ├── train_model.py          # Neural network training CLI
│   ├── train_with_api.sh       # Training script with API key
│   ├── search_google.py        # Google search utilities
│   └── sensitivity_analysis.py # Model analysis
│
├── data/                  # Data Files
│   ├── balanced_urls.txt       # Curated training URLs
│   ├── sample_urls.txt         # Sample URLs
│   └── *.csv, *.json          # Generated datasets
│
├── models/                # Trained Models
│   ├── credibility_model.pth   # Neural network weights
│   └── credibility_model_scaler.pkl  # Feature scaler
│
└── docs/                  # Documentation
    ├── QUICKSTART-NN.md         # Quick start guide
    ├── README-NEURAL-NETWORK.md # Complete NN documentation
    └── NEURAL-NETWORK-SUMMARY.md # Architecture overview
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip3 install -r requirements.txt
```

### 2. Train Neural Network (Optional)

```bash
# Quick demo training
./scripts/train_with_api.sh

# Or use pre-selected URLs
./scripts/train_with_api.sh --mode urls --urls-file data/balanced_urls.txt --epochs 50
```

### 3. Run the App

```bash
./run_app.sh
```

The app will open at http://localhost:8501

## 📚 Documentation

- **[docs/QUICKSTART-NN.md](docs/QUICKSTART-NN.md)** - Get started in 5 minutes
- **[docs/README-NEURAL-NETWORK.md](docs/README-NEURAL-NETWORK.md)** - Complete technical documentation
- **[docs/NEURAL-NETWORK-SUMMARY.md](docs/NEURAL-NETWORK-SUMMARY.md)** - Architecture overview

---

**Built with PyTorch, Streamlit, Claude AI, and ❤️**
