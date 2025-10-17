# Neural Network for URL Credibility Prediction

This system uses a neural network trained on Claude-labeled data to predict URL credibility.

## 🏗️ Architecture

### System Components

1. **Feature Extractor** (`feature_extractor.py`)
   - Extracts 40+ features from URLs and webpage content
   - URL features: length, domain structure, TLD, suspicious patterns
   - Content features: readability scores, text quality, structure
   - Combines with existing heuristic scores

2. **Web Search Analyzer** (`web_search.py`)
   - Searches the web using Google Search
   - Fetches webpage content
   - Uses Claude to analyze and label URLs for training

3. **Dataset Generator** (`dataset_generator.py`)
   - Generates training datasets from search queries or URL lists
   - Uses Claude to provide ground truth labels
   - Balances classes for better training

4. **Neural Network Model** (`credibility_nn.py`)
   - Feed-forward architecture: Input → 256 → 128 → 64 → Output
   - 3-class classification: Low, Medium, High credibility
   - Uses PyTorch with batch normalization and dropout

5. **Training Script** (`train_model.py`)
   - End-to-end training pipeline
   - Multiple training modes (demo, search, URLs, dataset)
   - Saves trained model for inference

6. **Streamlit Integration** (`app.py`)
   - Auto-loads trained model
   - Displays NN predictions alongside heuristic scores
   - Color-coded visualization

## 🚀 Quick Start

### Step 1: Train a Model (Demo Mode)

The easiest way to get started is with the demo mode:

```bash
python3 train_model.py
```

This will:
- Search for diverse topics (science, news, controversial claims)
- Analyze ~20-30 URLs using Claude
- Train a neural network
- Save the model as `credibility_model.pth`

### Step 2: Run the Streamlit App

```bash
./run_app.sh
```

The app will automatically detect and load the trained model.

## 📚 Training Modes

### 1. Demo Mode (Recommended for Testing)
```bash
python3 train_model.py --mode demo
```
Uses predefined diverse queries for quick training.

### 2. Custom Search Queries
```bash
python3 train_model.py --mode search \
    --queries "scientific research" "news articles" "blog posts" \
    --urls-per-query 10 \
    --epochs 100
```

### 3. From URL List
Create a file `urls.txt` with one URL per line, then:
```bash
python3 train_model.py --mode urls \
    --urls-file urls.txt \
    --epochs 100
```

### 4. From Existing Dataset
```bash
python3 train_model.py --mode dataset \
    --dataset credibility_training_data.csv \
    --epochs 100
```

## 🔬 How It Works

### 1. Feature Extraction
For each URL, the system extracts:

**URL Features:**
- Length, domain structure, TLD type
- Character patterns (dots, hyphens, numbers)
- Suspicious keywords

**Content Features:**
- Text length, word count, readability scores
- HTML structure (headings, links, images)
- Author/date information presence
- Lexical diversity

**Heuristic Scores:**
- Your existing URL, content, popularity, PageRank scores

### 2. Claude Labeling
Claude analyzes each URL and provides:
- Authority score (0-1)
- Quality score (0-1)
- Evidence score (0-1)
- Objectivity score (0-1)
- Recency score (0-1)
- Overall credibility (low/medium/high)

### 3. Neural Network Training
- Combines all features into input vector
- Trains feed-forward NN with cross-entropy loss
- Uses validation set for early stopping
- Evaluates on held-out test set

### 4. Prediction
- Extracts features from new URL
- Scales features using trained scaler
- Passes through NN to get prediction
- Returns class label and confidence scores

## 📊 Using the Trained Model

### In Streamlit App

1. Start the app: `./run_app.sh`
2. In the sidebar, check "Use Neural Network predictions"
3. Ask Claude about URLs or analyze them directly
4. See both heuristic and NN predictions side-by-side

### Programmatic Usage

```python
from credibility_nn import CredibilityPredictor
from feature_extractor import URLFeatureExtractor
import deliverable1 as d1

# Load model
predictor = CredibilityPredictor()
predictor.load_model("credibility_model.pth")

# Prepare feature extractor
feature_extractor = URLFeatureExtractor()

# Analyze URL
url = "https://example.com/article"
pagerank_scores = d1.compute_pagerank([url])
heuristic_scores = d1.score_url_with_content(url, pagerank_scores)

# Get NN prediction
result = predictor.predict_url(url, feature_extractor, heuristic_scores)

print(f"URL: {result['url']}")
print(f"Prediction: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

## 🎯 Model Performance

Expected performance with demo training (~30 samples):
- **Validation Accuracy**: 60-80%
- **Test Accuracy**: 60-80%

For better performance:
- Train on more data (100+ samples per class)
- Use more diverse search queries
- Fine-tune hyperparameters

## 🔧 Advanced Usage

### Custom Dataset Generation

```python
from dataset_generator import DatasetGenerator

generator = DatasetGenerator()

# Generate from searches
queries = ["scientific papers", "blog posts", "news articles"]
df = generator.generate_from_search_queries(queries, urls_per_query=20)

# Save dataset
generator.save_dataset(df, "my_training_data")

# Balance classes
df_balanced = generator.augment_with_balanced_samples(df, target_per_class=100)
```

### Custom Model Training

```python
from credibility_nn import CredibilityPredictor
import pandas as pd

# Load or create dataset
df = pd.read_csv("credibility_training_data.csv")

# Train model
predictor = CredibilityPredictor()
results = predictor.train(
    df,
    epochs=150,
    batch_size=16,
    learning_rate=0.0005
)

print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
print(f"Test accuracy: {results['test_acc']:.4f}")
```

### Hyperparameter Tuning

Modify `credibility_nn.py`:

```python
# Try different architectures
model = CredibilityNN(
    input_dim=num_features,
    hidden_dims=[512, 256, 128, 64],  # Deeper network
    dropout=0.4  # More regularization
)

# Try different optimizers
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
```

## 📈 Tips for Better Results

### 1. Dataset Quality
- Use diverse search queries
- Include examples from all credibility levels
- Aim for balanced classes (equal samples per class)
- Use at least 100 samples per class

### 2. Feature Engineering
- Add domain-specific features
- Experiment with feature combinations
- Consider feature selection/dimensionality reduction

### 3. Model Architecture
- Start simple, add complexity if needed
- Use cross-validation for hyperparameter tuning
- Monitor for overfitting (train vs. validation gap)

### 4. Training Process
- Use early stopping to prevent overfitting
- Save checkpoints during training
- Track metrics (accuracy, precision, recall)

## 🐛 Troubleshooting

### Model Not Loading in App
```bash
# Check if model file exists
ls -la credibility_model.pth

# Verify model can be loaded
python3 -c "from credibility_nn import CredibilityPredictor; p = CredibilityPredictor(); p.load_model('credibility_model.pth'); print('✅ Model loaded')"
```

### Low Accuracy
- Increase dataset size (more samples)
- Balance classes
- Add more diverse examples
- Try different architectures or hyperparameters

### Training Too Slow
- Reduce `urls_per_query`
- Use smaller dataset for testing
- Reduce number of epochs
- Use GPU if available

### Claude API Errors
- Check API key and credits
- Add rate limiting (increase sleep intervals)
- Use smaller batches

## 📝 Files Generated

After training, you'll have:
- `credibility_model.pth` - Trained neural network weights
- `credibility_model_scaler.pkl` - Feature scaler
- `credibility_training_data_TIMESTAMP.csv` - Training dataset
- `credibility_training_data_TIMESTAMP.json` - Training dataset (JSON)

## 🔮 Future Enhancements

Potential improvements:
1. **Ensemble Methods**: Combine multiple models
2. **Active Learning**: User feedback to improve model
3. **Domain Adaptation**: Specialize for specific topics
4. **Explainability**: SHAP/LIME for feature importance
5. **Real-time Learning**: Continuously update model
6. **Multi-task Learning**: Predict credibility + other attributes

## 📄 License

Same license as the main project.

## 🤝 Contributing

To contribute improvements:
1. Test with diverse URLs
2. Share training datasets
3. Report issues and results
4. Suggest new features

---

**Happy training! 🚀**
