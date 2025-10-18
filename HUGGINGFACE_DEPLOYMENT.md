# Hugging Face Deployment Guide

This guide will help you deploy the URL Credibility Analyzer to Hugging Face Spaces.

## Prerequisites

1. **Hugging Face Account**: Create an account at [huggingface.co](https://huggingface.co)
2. **Anthropic API Key**: Get your API key from [console.anthropic.com](https://console.anthropic.com)
3. **Git**: Ensure git is installed on your system

## Deployment Steps

### Step 1: Create a New Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Configure your Space:
   - **Name**: Choose a name (e.g., `url-credibility-analyzer`)
   - **License**: MIT
   - **SDK**: Select **Streamlit**
   - **Hardware**: CPU basic (free tier) is sufficient
   - **Visibility**: Public or Private (your choice)
4. Click "Create Space"

### Step 2: Prepare Your Repository

Navigate to your project directory:

```bash
cd /Users/bobbymurphy/Documents/pace/cs676/project1
```

### Step 3: Rename README for Hugging Face

The Hugging Face README needs special metadata at the top. Replace the current README:

```bash
# Backup original README
mv README.md README_LOCAL.md

# Use the Hugging Face README
mv README_HF.md README.md
```

Or manually add the following header to your README.md:

```yaml
---
title: URL Credibility Analyzer
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
license: mit
---
```

### Step 4: Initialize Git (if not already done)

```bash
# Check if this is already a git repository
git status

# If not a git repo, initialize it
git init
git add .
git commit -m "Initial commit for Hugging Face deployment"
```

### Step 5: Add Hugging Face Remote

Replace `YOUR_USERNAME` and `YOUR_SPACE_NAME` with your actual values:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

Example:
```bash
git remote add hf https://huggingface.co/spaces/johndoe/url-credibility-analyzer
```

### Step 6: Configure Git LFS (for model files)

Hugging Face uses Git LFS for large files. If you have trained models:

```bash
# Install git-lfs if not already installed
# macOS: brew install git-lfs
# Ubuntu: sudo apt-get install git-lfs

git lfs install
git lfs track "*.pth"
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### Step 7: Push to Hugging Face

```bash
# Push your code to Hugging Face
git push hf main

# If your branch is named differently (e.g., 'master'), use:
# git push hf master:main
```

You may be prompted for credentials:
- **Username**: Your Hugging Face username
- **Password**: Your Hugging Face **access token** (not your account password)

To create an access token:
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with "write" permissions
3. Copy and use it as the password

### Step 8: Configure Environment Variables

1. Go to your Space on Hugging Face
2. Click on "Settings" tab
3. Scroll to "Variables and secrets"
4. Add a new secret:
   - **Name**: `ANTHROPIC_API_KEY`
   - **Value**: Your Anthropic API key (starts with `sk-ant-api...`)
5. Click "Save"

### Step 9: Wait for Build

1. Go to the "App" tab of your Space
2. You'll see build logs as Hugging Face installs dependencies
3. This may take 3-5 minutes for the first build
4. Once complete, your app will be live!

## Verification

Your Space should be accessible at:
```
https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
```

Test the following features:
- Chat with Claude in the first tab
- Analyze a single URL in the second tab
- Run batch analysis in the third tab
- Check if Neural Network predictions work (if you uploaded model files)

## Troubleshooting

### Build Fails

**Check build logs** in the Space's "App" tab:

1. **Missing dependencies**: Verify all packages are in `requirements.txt`
2. **Import errors**: Ensure all Python files are committed
3. **Port issues**: Streamlit should use port 7860 (configured in `.streamlit/config.toml`)

### API Key Not Working

1. Verify the secret name is exactly `ANTHROPIC_API_KEY`
2. Check the API key is valid at [console.anthropic.com](https://console.anthropic.com)
3. Restart the Space (Settings > Factory reboot)

### Neural Network Not Loading

1. Ensure model files are committed:
   ```bash
   git add models/credibility_model.pth models/credibility_model_scaler.pkl
   git commit -m "Add trained model files"
   git push hf main
   ```
2. Check `.gitignore` isn't excluding model files
3. Verify Git LFS is tracking `.pth` and `.pkl` files

### App is Slow

1. Consider upgrading to a better hardware tier (Settings > Hardware)
2. CPU basic is free but may be slower for neural network inference
3. Optimize batch analysis for large URL sets

## Updating Your Space

To update your deployed app:

```bash
# Make your changes
git add .
git commit -m "Description of changes"
git push hf main
```

The Space will automatically rebuild and redeploy.

## Optional: Add Model Files

If you have a trained neural network model and want to include it:

```bash
# Ensure Git LFS is tracking model files
git lfs track "*.pth"
git lfs track "*.pkl"

# Add and commit model files
git add models/credibility_model.pth
git add models/credibility_model_scaler.pkl
git commit -m "Add trained neural network model"
git push hf main
```

## File Checklist

Ensure these files exist in your repository:

- ✅ `app.py` - Main Streamlit application
- ✅ `requirements.txt` - Python dependencies
- ✅ `packages.txt` - System dependencies (libgomp1)
- ✅ `README.md` - With Hugging Face metadata header
- ✅ `.streamlit/config.toml` - Streamlit configuration
- ✅ `deliverable1.py` - Core scoring logic
- ✅ `nn/` directory - Neural network code
- ✅ (Optional) `models/*.pth` - Trained model files

## Support

- **Hugging Face Docs**: [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community**: [huggingface.co/spaces/community](https://discuss.huggingface.co)

## Security Notes

- Never commit your `.env` file or API keys to the repository
- Always use Hugging Face Secrets for sensitive credentials
- The `.gitignore` file prevents committing `.env` files
- API keys in Secrets are encrypted and not visible in logs

---

**Your app should now be live on Hugging Face! 🚀**
