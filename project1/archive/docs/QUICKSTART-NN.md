# Neural Network Quick Start Guide

Get your URL credibility neural network up and running in 3 steps!

## ⚡ Quick Start (5 minutes)

### Step 1: Install Dependencies

```bash
pip3 install torch scikit-learn textstat nltk googlesearch-python
```

### Step 2: Train a Demo Model

```bash
python3 train_model.py
```

This will:
- Search for ~20-30 diverse URLs
- Label them using Claude
- Train a neural network
- Save model as `credibility_model.pth`

Expected output:
```
====================================================================
URL Credibility Neural Network - Training Pipeline
====================================================================

📊 STEP 1: Generating Training Dataset
--------------------------------------------------------------------
Searching for: scientific research climate change
Found 3 URLs
...

🧠 STEP 2: Training Neural Network
--------------------------------------------------------------------
📊 Data prepared:
   Training samples: 21
   Validation samples: 4
   Test samples: 5
   Features: 51

🚀 Starting training for 50 epochs...
Epoch [10/50] - Loss: 0.8234, Val Acc: 0.7500
...

✅ TRAINING COMPLETE!
💾 Model saved to: credibility_model.pth
```

### Step 3: Run the App

```bash
./run_app.sh
```

In the app:
1. Check "Auto-analyze URLs in responses" ✅
2. Check "Use Neural Network predictions" ✅
3. Ask Claude: "What are some credible sources for climate science?"
4. See both heuristic and NN predictions!

## 📊 Understanding the Results

The app will show a table with:

| Column | Description |
|--------|-------------|
| **URL** | The analyzed URL |
| **Heuristic** | Original heuristic credibility score (0-1) |
| **NN Prediction** | Neural network classification (low/medium/high) |
| **NN Confidence** | How confident the model is (0-1) |
| **NN Score** | Probability of being "high credibility" (0-1) |

The table is color-coded:
- 🟢 Green = High credibility
- 🟡 Yellow = Medium credibility
- 🔴 Red = Low credibility

## 🎯 Training Your Own Model

### Option 1: Custom Search Queries

```bash
python3 train_model.py --mode search \
    --queries "scientific papers" "government data" "news articles" "blog posts" \
    --urls-per-query 15 \
    --epochs 100
```

### Option 2: From Your Own URLs

Create `my_urls.txt`:
```
https://www.nature.com/articles/s41586-021-03302-y
https://en.wikipedia.org/wiki/Climate_change
https://www.cdc.gov/coronavirus/2019-ncov/
https://some-blog.com/opinion-piece
https://sketchy-site.com/clickbait
```

Train:
```bash
python3 train_model.py --mode urls --urls-file my_urls.txt --epochs 100
```

### Option 3: Use Existing Dataset

```bash
# First, generate and save a dataset
python3 -c "
from dataset_generator import DatasetGenerator
generator = DatasetGenerator()
queries = ['health research', 'tech news', 'conspiracy theories']
df = generator.generate_from_search_queries(queries, 10)
generator.save_dataset(df, 'my_dataset')
"

# Then train on it
python3 train_model.py --mode dataset --dataset my_dataset.csv --epochs 100
```

## 🔍 Example Usage in App

### Example 1: Analyzing URLs from Claude

**You ask:** "Find me 5 sources about COVID-19 vaccines"

**Claude responds with URLs, and the app automatically analyzes them:**

| URL | Heuristic | NN Prediction | NN Confidence | NN Score |
|-----|-----------|---------------|---------------|----------|
| https://www.cdc.gov/vaccines | 0.85 | high | 0.92 | 0.89 |
| https://www.who.int/vaccines | 0.82 | high | 0.88 | 0.87 |
| https://en.wikipedia.org/wiki/Vaccine | 0.75 | medium | 0.71 | 0.68 |
| https://naturalnews.com/vaccines | 0.35 | low | 0.85 | 0.12 |
| https://blog.example.com/my-opinion | 0.42 | low | 0.79 | 0.23 |

### Example 2: Direct URL Analysis

Go to "URL Analysis" tab and enter: `https://www.nature.com/articles/some-paper`

The app will show:
- Heuristic scores (URL, content, popularity, PageRank)
- Neural network prediction
- Claude's detailed analysis
- Combined credibility assessment

## 🛠️ Troubleshooting

### "Could not load neural network model"
- Make sure you've trained a model first: `python3 train_model.py`
- Check that `credibility_model.pth` exists in the project directory

### "Not enough data for training"
- Increase `--urls-per-query` (try 15-20)
- Add more queries
- Make sure you have internet connection for searches

### "Low accuracy (~50%)"
This is normal with small datasets! To improve:
- Train on more data (100+ samples per class)
- Use more diverse queries
- Run more epochs (150-200)

### Claude API errors
- Check your API key in `run_app.sh`
- Verify you have API credits
- Add delays between requests (training includes rate limiting)

## 📈 Expected Performance

With demo training (~30 samples):
- **Training**: Quick (1-2 minutes)
- **Accuracy**: 60-80%
- **Good for**: Testing and proof-of-concept

With production training (300+ samples):
- **Training**: Longer (10-30 minutes)
- **Accuracy**: 80-90%+
- **Good for**: Real-world use

## 🚀 Next Steps

1. **Collect More Data**: The more diverse training data, the better
2. **Fine-tune**: Experiment with different architectures and hyperparameters
3. **Specialize**: Train domain-specific models (health, tech, news, etc.)
4. **Integrate**: Use the model in your own applications

## 💡 Tips for Best Results

1. **Balanced Dataset**: Equal samples of low/medium/high credibility
2. **Diverse Sources**: Mix of domains, topics, and content types
3. **Quality Labels**: Claude provides good labels, but verify edge cases
4. **Regular Updates**: Retrain periodically with new data
5. **Ensemble**: Combine NN predictions with heuristics for best results

## 📚 More Information

- Full documentation: [README-NEURAL-NETWORK.md](README-NEURAL-NETWORK.md)
- Main README: [README.md](README.md)
- Architecture details: Check the module docstrings

---

**Questions? Issues? Check the full README or open an issue!**
