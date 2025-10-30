#!/bin/bash

# Fresh Deployment Script
# Cleans up and deploys to new Hugging Face Space

set -e

echo "=========================================="
echo "Fresh Hugging Face Deployment"
echo "=========================================="
echo ""

cd /Users/bobbymurphy/Documents/pace/cs676/project1

# Get new Space info
echo "Enter your NEW Hugging Face Space URL"
echo "Example: https://huggingface.co/spaces/bbbymrphy/url-analyzer-new"
read -p "New Space URL: " NEW_SPACE_URL

# Extract space name for display
SPACE_NAME=$(echo $NEW_SPACE_URL | sed 's/.*spaces\///')

echo ""
echo "Target Space: $SPACE_NAME"
echo ""

# Remove old hf remote if it exists
if git remote | grep -q "^hf$"; then
    echo "Removing old 'hf' remote..."
    git remote remove hf
    echo "✓ Old remote removed"
fi

# Add new remote
echo "Adding new remote..."
git remote add hf $NEW_SPACE_URL
echo "✓ New remote added"

# Prepare README
echo ""
echo "Preparing README for Hugging Face..."
if [ -f "README.md" ]; then
    cp README.md README_BACKUP.md
    echo "✓ Backed up current README to README_BACKUP.md"
fi
cp README_HF.md README.md
echo "✓ Using Hugging Face README"

# Stage new files
echo ""
echo "Staging deployment files..."
git add packages.txt
git add .streamlit/config.toml
git add README.md
git add deploy_to_hf.sh
git add HUGGINGFACE_DEPLOYMENT.md
git add DEPLOYMENT_QUICKSTART.md
git add UPDATE_EXISTING_SPACE.md

# Commit
echo ""
git status --short
echo ""
read -p "Commit these changes? (y/n): " COMMIT
if [ "$COMMIT" = "y" ]; then
    git commit -m "Add Hugging Face deployment configuration" || echo "No changes to commit"
    echo "✓ Changes committed"
fi

# Show what will be pushed
echo ""
echo "Ready to push to: $NEW_SPACE_URL"
echo ""
read -p "Push now? (y/n): " PUSH

if [ "$PUSH" = "y" ]; then
    echo ""
    echo "Pushing to Hugging Face..."
    echo "You'll be prompted for credentials:"
    echo "  Username: bbbymrphy"
    echo "  Password: Your HF access token (from https://huggingface.co/settings/tokens)"
    echo ""

    git push hf main || git push hf master:main

    echo ""
    echo "=========================================="
    echo "✓ Deployment Complete!"
    echo "=========================================="
    echo ""
    echo "Your Space: $NEW_SPACE_URL"
    echo ""
    echo "Next steps:"
    echo "1. Go to: $NEW_SPACE_URL/settings"
    echo "2. Add secret: ANTHROPIC_API_KEY"
    echo "3. Wait for build (3-5 minutes)"
    echo "4. Your app will be live!"
    echo ""
else
    echo ""
    echo "When ready, push with: git push hf main"
fi
