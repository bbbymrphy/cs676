# Running the Streamlit App Locally

Quick guide to test the URL Credibility Analyzer with Claude AI on your local machine.

## Prerequisites

- Python 3.8+
- Anthropic API key (you already have one in `key.py`)

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/bobbymurphy/Documents/pace/cs676/project1
pip install -r requirements.txt
```

### 2. Set Up API Key

**Option A: Export Environment Variable**
```bash
export ANTHROPIC_API_KEY='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'
```

**Option B: Use .env file (cleaner)**
```bash
# Copy the example
cp .env.example .env

# Edit .env and add your key
# Then install python-dotenv
pip install python-dotenv
```

Then modify the top of `app.py` to load from .env:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Testing the Features

### Test 1: Chat with Claude
1. Go to "🤖 Claude Chat" tab
2. Ask: "What makes a URL credible?"
3. Verify Claude responds

### Test 2: Single URL Analysis
1. Go to "📊 URL Analysis" tab
2. Try: `https://www.nasa.gov/`
3. Check "Enhance with Claude Analysis"
4. Click "Analyze URL"
5. Verify you see scores and Claude's analysis

### Test 3: Batch Analysis
1. Go to "📈 Batch Analysis" tab
2. Enter these URLs (one per line):
   ```
   https://www.nasa.gov/
   https://www.bbc.com/news
   https://www.reddit.com/
   ```
3. Click "Analyze Batch"
4. Verify table appears with scores
5. Download CSV to test export

## Troubleshooting

### "ANTHROPIC_API_KEY not found"
Make sure you've exported the variable in the same terminal where you run streamlit:
```bash
export ANTHROPIC_API_KEY='your-key-here'
streamlit run app.py
```

### Module Not Found Errors
```bash
pip install -r requirements.txt --upgrade
```

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

## Ready for Hugging Face?

Once you've tested locally and everything works:
1. See [README-hf.md](README-hf.md) for deployment instructions
2. Your app is already configured to work on HF Spaces
3. Just upload the files and add your API key as a secret

## File Checklist for Deployment

✅ `app.py` - Main application
✅ `deliverable1.py` - Scoring logic
✅ `requirements.txt` - Dependencies
✅ `.gitignore` - Excludes sensitive files

❌ `key.py` - DO NOT upload (use HF secrets instead)
❌ `.env` - DO NOT upload (use HF secrets instead)

## Next Steps

1. Test all three tabs
2. Try different URLs (gov, edu, social media, news)
3. Experiment with Claude model settings in sidebar
4. When ready, deploy to Hugging Face!
