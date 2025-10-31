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

# URL Credibility Analyzer with Neural Network

A comprehensive URL credibility analysis system that combines heuristic scoring with neural network predictions powered by Claude AI.

## 🌟 Features

- **Heuristic Analysis**: URL structure, content quality, PageRank, popularity scoring
- **Neural Network**: Deep learning model trained on Claude-labeled data
- **Claude Integration**: AI-powered credibility assessment and chat
- **Interactive UI**: Three modes - Chat with Claude, Single URL Analysis, Batch Analysis
- **Real-time Analysis**: Automatically analyzes URLs mentioned in conversations

## 🚀 How to Use

### Configuration

1. **Set API Key**: Add your Anthropic API key in the Hugging Face Space secrets:
   - Go to Settings > Variables and secrets
   - Add a new secret: `ANTHROPIC_API_KEY` with your API key

### Chat Mode

1. Navigate to the "Claude Chat" tab
2. Ask Claude questions about credibility or provide text with URLs
3. Claude will automatically analyze any URLs in the responses
4. Enable/disable Neural Network predictions in the sidebar

### Single URL Analysis

1. Go to the "URL Analysis" tab
2. Enter a URL to analyze
3. Get detailed credibility scores and Claude's interpretation
4. View metrics: Combined Score, URL Score, Content Score, Popularity

### Batch Analysis

1. Navigate to "Batch Analysis" tab
2. Paste multiple URLs (one per line)
3. Get comprehensive analysis with:
   - Summary statistics
   - Sortable results table
   - CSV export
   - Claude's batch summary

## 📊 Credibility Metrics

- **URL Score**: Domain authority, HTTPS, suspicious patterns
- **Content Score**: Text quality, readability, sentiment analysis
- **Popularity Score**: Domain ranking and recognition
- **PageRank**: Network-based credibility
- **NN Prediction**: Neural network confidence scores (if model available)

## 🎯 Settings

- **Claude Model**: Select AI model (Haiku for speed)
- **Max Tokens**: Control response length (100-4000)
- **Temperature**: Adjust response creativity (0.0-1.0)
- **Auto-analyze URLs**: Toggle automatic URL detection
- **Neural Network**: Enable/disable NN predictions

## 🧠 Neural Network

The optional neural network analyzes 46+ features including:
- URL structure and security indicators
- Content metrics (sentiment, readability, formality)
- PageRank and popularity signals
- Advertisement detection
- Domain characteristics

Note: Neural network predictions require a trained model file.

## 🔒 Privacy & Security

- URL content is fetched server-side
- No user data is stored permanently
- API keys are securely managed through Hugging Face Secrets
- All communications use HTTPS

## 💡 Tips

- Higher scores (>0.7) indicate more credible sources
- Medium scores (0.4-0.7) warrant additional verification
- Low scores (<0.4) suggest caution
- Check multiple metrics for comprehensive assessment
- Use Claude's analysis for contextual insights

## 📝 Technical Details

Built with:
- **Streamlit**: Interactive web interface
- **PyTorch**: Neural network implementation
- **Claude AI**: Natural language processing
- **NetworkX**: PageRank computation
- **BeautifulSoup**: Web scraping
- **Pandas**: Data analysis

## 🤝 Contributing

This project was developed as part of CS676. For issues or improvements, please contact the development team.

---

**Built with PyTorch, Streamlit, Claude AI, and ❤️**
