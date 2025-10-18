# URL Credibility Analyzer with Claude AI

An intelligent URL credibility analysis tool that combines heuristic scoring with Claude AI insights.

## Features

- **🤖 Interactive Chat**: Conversational interface with Claude for discussing URL credibility
- **📊 Single URL Analysis**: Detailed credibility scoring for individual URLs
- **📈 Batch Processing**: Analyze multiple URLs simultaneously with comprehensive reports
- **🎯 Multi-Factor Scoring**: Evaluates URLs based on:
  - Domain reputation (.gov, .edu, etc.)
  - Content quality and length
  - Ad presence and spam indicators
  - Popularity metrics
  - PageRank algorithm

## How It Works

The app uses a comprehensive scoring system that analyzes:

1. **URL Structure**: Domain extensions, path complexity, suspicious patterns
2. **Content Analysis**: Word count, spam keywords, formatting quality
3. **Ad Detection**: Identifies and penalizes excessive advertising
4. **Popularity**: Bonus for well-known credible sources
5. **PageRank**: Network-based credibility scoring

Claude AI enhances the analysis by providing:
- Human-readable interpretations of scores
- Contextual credibility assessments
- Actionable recommendations
- Pattern recognition across batches

## Usage

### Single URL Analysis
1. Go to the "📊 URL Analysis" tab
2. Enter a URL
3. Click "Analyze URL"
4. Review scores and Claude's insights

### Batch Analysis
1. Go to the "📈 Batch Analysis" tab
2. Enter multiple URLs (one per line)
3. Click "Analyze Batch"
4. Review results table and download CSV

### Chat with Claude
1. Go to the "🤖 Claude Chat" tab
2. Ask questions about URL credibility
3. Get AI-powered insights and advice

## Configuration

Use the sidebar to adjust:
- **Claude Model**: Choose between Sonnet, Opus, or Haiku
- **Max Tokens**: Control response length
- **Temperature**: Adjust creativity vs. precision

## API Key Required

This app requires an Anthropic API key. Get yours at: https://console.anthropic.com/

Set it as the `ANTHROPIC_API_KEY` environment variable or add it in the Hugging Face Space secrets.

## Technical Details

Built with:
- **Streamlit**: Web application framework
- **Anthropic Claude**: Advanced language model
- **NetworkX**: PageRank algorithm
- **BeautifulSoup**: Web scraping and content analysis
- **Pandas**: Data processing and export

## Credibility Scoring Breakdown

- **Combined Score** (0-1): Weighted average of all factors
  - 40% URL structure score
  - 30% Content quality score
  - 20% Popularity bonus
  - 10% PageRank score

Higher scores indicate greater credibility.

## Limitations

- Content analysis requires accessible web pages
- Some sites may block scraping
- Scores are heuristic-based, not definitive truth
- Always verify critical information from multiple sources

## Privacy & Data

- URLs are analyzed in real-time
- Results cached locally in `url_results_db.json`
- No data shared with third parties
- API calls sent to Anthropic (see their privacy policy)

## License

See LICENSE file for details.

## Credits

Developed for CS676 Project - Pace University
