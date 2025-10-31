#!/bin/bash

# Hugging Face Deployment Helper Script
# This script helps prepare and deploy the URL Credibility Analyzer to Hugging Face Spaces

set -e  # Exit on error

echo "=========================================="
echo "Hugging Face Deployment Helper"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo -e "${RED}Error: app.py not found. Please run this script from the project1 directory.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Project directory verified${NC}"

# Ask for Hugging Face username and space name
echo ""
read -p "Enter your Hugging Face username: " HF_USERNAME
read -p "Enter your Space name (e.g., url-credibility-analyzer): " HF_SPACE

# Verify inputs
if [ -z "$HF_USERNAME" ] || [ -z "$HF_SPACE" ]; then
    echo -e "${RED}Error: Username and Space name are required.${NC}"
    exit 1
fi

HF_REPO="https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE"
echo ""
echo -e "${YELLOW}Target repository: $HF_REPO${NC}"
echo ""

# Backup original README if it exists and doesn't have HF metadata
echo "Checking README.md..."
if [ -f "README.md" ]; then
    if ! grep -q "^---$" README.md; then
        echo -e "${YELLOW}Backing up current README.md to README_LOCAL.md${NC}"
        cp README.md README_LOCAL.md
    fi
fi

# Use the Hugging Face README
if [ -f "README_HF.md" ]; then
    echo -e "${GREEN}✓ Using README_HF.md as README.md${NC}"
    cp README_HF.md README.md
else
    echo -e "${RED}Warning: README_HF.md not found. README.md may not have proper HF metadata.${NC}"
fi

# Check for required files
echo ""
echo "Checking required files..."
REQUIRED_FILES=("app.py" "deliverable1.py" "requirements.txt" "packages.txt" ".streamlit/config.toml")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ $file${NC}"
    else
        echo -e "${RED}✗ $file${NC}"
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo -e "${RED}Error: Missing required files. Please ensure all files are present.${NC}"
    exit 1
fi

# Check if git is initialized
echo ""
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
else
    echo -e "${GREEN}✓ Git repository already initialized${NC}"
fi

# Check for Git LFS
echo ""
echo "Checking Git LFS..."
if command -v git-lfs &> /dev/null; then
    echo -e "${GREEN}✓ Git LFS is installed${NC}"

    # Initialize Git LFS
    git lfs install

    # Track model files if they exist
    if [ -f "models/credibility_model.pth" ] || [ -f "models/credibility_model_scaler.pkl" ]; then
        echo "Setting up LFS tracking for model files..."
        git lfs track "*.pth"
        git lfs track "*.pkl"
        git add .gitattributes
        echo -e "${GREEN}✓ Git LFS tracking configured for model files${NC}"
    fi
else
    echo -e "${YELLOW}Warning: Git LFS not installed. Model files may not upload correctly.${NC}"
    echo "Install with: brew install git-lfs (macOS) or apt-get install git-lfs (Linux)"
    echo ""
    read -p "Continue without Git LFS? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        exit 1
    fi
fi

# Stage all files
echo ""
echo "Staging files for commit..."
git add .

# Check git status
echo ""
echo "Git status:"
git status --short

# Commit
echo ""
read -p "Enter commit message (or press Enter for default): " COMMIT_MSG
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Prepare for Hugging Face deployment"
fi

git commit -m "$COMMIT_MSG" || echo -e "${YELLOW}No changes to commit${NC}"

# Add Hugging Face remote
echo ""
echo "Configuring Hugging Face remote..."
if git remote | grep -q "^hf$"; then
    echo -e "${YELLOW}Remote 'hf' already exists. Removing and re-adding...${NC}"
    git remote remove hf
fi

git remote add hf "$HF_REPO"
echo -e "${GREEN}✓ Added remote: $HF_REPO${NC}"

# Show final instructions
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Create your Space on Hugging Face:"
echo "   - Go to: https://huggingface.co/spaces"
echo "   - Click 'Create new Space'"
echo "   - Name: $HF_SPACE"
echo "   - SDK: Streamlit"
echo "   - Visibility: Public or Private"
echo ""
echo "2. Push to Hugging Face:"
echo "   git push hf main"
echo ""
echo "   (Use 'git push hf master:main' if your branch is 'master')"
echo ""
echo "3. Configure API Key:"
echo "   - Go to: $HF_REPO/settings"
echo "   - Add Secret: ANTHROPIC_API_KEY"
echo "   - Value: Your Anthropic API key"
echo ""
echo "4. Your app will be live at:"
echo "   https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE"
echo ""
echo -e "${YELLOW}For detailed instructions, see: HUGGINGFACE_DEPLOYMENT.md${NC}"
echo ""

# Ask if user wants to push now
read -p "Push to Hugging Face now? (y/n): " PUSH_NOW
if [ "$PUSH_NOW" = "y" ]; then
    echo ""
    echo "Pushing to Hugging Face..."
    echo -e "${YELLOW}You will be prompted for your HF username and access token.${NC}"
    echo -e "${YELLOW}Token (not password) can be created at: https://huggingface.co/settings/tokens${NC}"
    echo ""

    # Try to push
    if git push hf main; then
        echo ""
        echo -e "${GREEN}=========================================="
        echo "Success! Your app is deploying!"
        echo "==========================================${NC}"
        echo ""
        echo "Visit your Space: $HF_REPO"
        echo ""
        echo "Don't forget to add your ANTHROPIC_API_KEY in Settings!"
    else
        echo ""
        echo -e "${YELLOW}Push failed. Try manually with: git push hf main${NC}"
        echo ""
        echo "If your local branch is named 'master', use:"
        echo "git push hf master:main"
    fi
else
    echo ""
    echo "When you're ready, push with:"
    echo "  git push hf main"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
