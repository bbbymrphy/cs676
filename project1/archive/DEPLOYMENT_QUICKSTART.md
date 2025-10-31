# Hugging Face Deployment - Quick Start

## 🚀 Fast Track Deployment (5 minutes)

### Option 1: Automated Script (Recommended)

```bash
cd /Users/bobbymurphy/Documents/pace/cs676/project1
./deploy_to_hf.sh
```

The script will guide you through the entire process.

### Option 2: Manual Deployment

1. **Create Space on Hugging Face**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose Streamlit SDK
   - Note your username and space name

2. **Prepare README** (choose one method)
   ```bash
   # Method A: Use the prepared HF README
   cp README_HF.md README.md

   # Method B: Add this header to your existing README.md
   # (at the very top of the file)
   ```
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

3. **Setup Git and Push**
   ```bash
   cd /Users/bobbymurphy/Documents/pace/cs676/project1

   # Initialize git (if needed)
   git init

   # Add all files
   git add .
   git commit -m "Deploy to Hugging Face"

   # Add HF remote (replace YOUR_USERNAME and YOUR_SPACE)
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE

   # Push to HF
   git push hf main
   ```

   If your branch is named `master`:
   ```bash
   git push hf master:main
   ```

4. **Configure API Key**
   - Go to your Space settings: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE/settings`
   - Under "Variables and secrets"
   - Add new secret:
     - Name: `ANTHROPIC_API_KEY`
     - Value: Your API key from [console.anthropic.com](https://console.anthropic.com)

5. **Done!**
   - Your app will be at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE`
   - Build takes 3-5 minutes

## 📋 Files Created for Deployment

All necessary files have been created in your project:

- ✅ `packages.txt` - System dependencies (libgomp1 for PyTorch)
- ✅ `.streamlit/config.toml` - Streamlit configuration for HF
- ✅ `README_HF.md` - README with HF metadata
- ✅ `deploy_to_hf.sh` - Automated deployment script
- ✅ `HUGGINGFACE_DEPLOYMENT.md` - Detailed deployment guide
- ✅ `.gitignore` - Updated to allow model files

Existing files (already compatible):
- ✅ `requirements.txt` - Python dependencies
- ✅ `app.py` - Main application
- ✅ All other project files

## 🔑 Get Your Anthropic API Key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Sign in or create account
3. Navigate to API Keys
4. Create a new key
5. Copy the key (starts with `sk-ant-api...`)

## 🆘 Troubleshooting

**Build fails?**
- Check logs in the Space's "App" tab
- Verify all files are committed
- Check requirements.txt syntax

**API errors?**
- Verify `ANTHROPIC_API_KEY` secret is set correctly
- Restart space: Settings > Factory reboot

**Can't push to HF?**
- Use access token (not password): [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Ensure you created the Space first
- Check remote URL: `git remote -v`

## 📚 Full Documentation

For detailed instructions, see [HUGGINGFACE_DEPLOYMENT.md](HUGGINGFACE_DEPLOYMENT.md)

---

**Ready to deploy? Run `./deploy_to_hf.sh` now!**
