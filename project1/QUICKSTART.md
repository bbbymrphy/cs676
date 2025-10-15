# Quick Start Guide

## Run Locally (Testing)

### Option 1: Use the Launch Script (Easiest)
```bash
cd /Users/bobbymurphy/Documents/pace/cs676/project1
./run_app.sh
```

### Option 2: Manual Launch
```bash
cd /Users/bobbymurphy/Documents/pace/cs676/project1
export ANTHROPIC_API_KEY='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'
streamlit run app.py
```

The app will open at: **http://localhost:8501**

## What You'll See

Three tabs:
1. **🤖 Claude Chat** - Talk to Claude about URL credibility
2. **📊 URL Analysis** - Analyze a single URL with AI insights
3. **📈 Batch Analysis** - Analyze multiple URLs at once

## Test URLs

Try these to see how the scoring works:

**High Credibility:**
- https://www.nasa.gov/
- https://www.harvard.edu/
- https://www.bbc.com/news

**Medium Credibility:**
- https://www.cnn.com/
- https://www.techcrunch.com/

**Low Credibility:**
- https://www.reddit.com/
- https://twitter.com/

## Deploy to Hugging Face

When you're ready to deploy:
1. Read [README-hf.md](README-hf.md)
2. Create a Streamlit Space on Hugging Face
3. Upload: `app.py`, `deliverable1.py`, `requirements.txt`
4. Add `ANTHROPIC_API_KEY` as a secret
5. Done!

## Files Overview

**For Local Testing:**
- `app.py` - Main Streamlit app
- `deliverable1.py` - URL scoring logic
- `requirements.txt` - Dependencies
- `run_app.sh` - Launch script
- `key.py` - Your API key (local only)

**For Deployment:**
- `app.py` ✅
- `deliverable1.py` ✅
- `requirements.txt` ✅
- **Don't upload:** `key.py`, `.env`

## Troubleshooting

**"Module not found"**
```bash
pip3 install -r requirements.txt
```

**"Port already in use"**
```bash
streamlit run app.py --server.port 8502
```

**"API key not found"**
Make sure you exported the variable in the same terminal:
```bash
export ANTHROPIC_API_KEY='your-key'
streamlit run app.py
```

## Next Steps

1. ✅ Test the app locally
2. ✅ Try all three tabs
3. ✅ Verify Claude responses work
4. ⏭️ Deploy to Hugging Face (when ready)

**Questions?** See [README-local.md](README-local.md) for detailed instructions.
