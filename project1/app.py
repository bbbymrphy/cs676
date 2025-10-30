import streamlit as st
import os
from anthropic import Anthropic
import deliverable1 as d1
import json
import pandas as pd
import re
from nn.feature_extractor import URLFeatureExtractor
from nn.credibility_nn import CredibilityPredictor

# Initialize Anthropic client
def get_claude_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("⚠️ ANTHROPIC_API_KEY not found in environment variables!")
        st.stop()
    return Anthropic(api_key=api_key)

# Extract URLs from text
def extract_urls(text):
    """Extract all URLs from text using regex"""
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    urls = re.findall(url_pattern, text)
    # Clean up trailing punctuation
    cleaned_urls = []
    for url in urls:
        # Remove trailing punctuation that's likely not part of the URL
        url = re.sub(r'[.,;:!?)]+$', '', url)
        cleaned_urls.append(url)
    return list(set(cleaned_urls))  # Remove duplicates

# Load neural network model
@st.cache_resource
def load_nn_model():
    """Load the trained neural network model"""
    try:
        predictor = CredibilityPredictor()
        if os.path.exists("models/credibility_model.pth"):
            predictor.load_model("models/credibility_model.pth")
            return predictor, URLFeatureExtractor()
        return None, URLFeatureExtractor()
    except Exception as e:
        st.warning(f"Could not load neural network model: {str(e)}")
        return None, URLFeatureExtractor()

# Analyze URLs and return results
def analyze_urls(urls, model, max_tokens, temperature, use_nn=False):
    """Analyze a list of URLs and return credibility scores"""
    if not urls:
        return None

    try:
        # Load NN model if requested
        nn_predictor, feature_extractor = load_nn_model() if use_nn else (None, None)

        # Compute PageRank for all URLs
        pagerank_scores = d1.compute_pagerank(urls)

        # Analyze each URL
        results = []
        for url in urls:
            try:
                result = d1.score_url_with_content(url, pagerank_scores)

                row = {
                    "URL": url,
                    "Heuristic": result['combined_score'],
                    "URL Score": result['url_score'],
                    "Content": result['text_score'],
                    "Popularity": result['popularity_score'],
                    "PageRank": result['pagerank_score'],
                }

                # Add NN prediction if available
                if nn_predictor and use_nn:
                    try:
                        nn_result = nn_predictor.predict_url(url, feature_extractor, result)
                        row["NN Prediction"] = nn_result['predicted_label']
                        row["NN Confidence"] = nn_result['confidence']
                        row["NN Score"] = nn_result['probabilities']['high']
                    except Exception as e:
                        row["NN Prediction"] = "error"
                        row["NN Confidence"] = 0.0
                        row["NN Score"] = 0.0

                results.append(row)
            except Exception as e:
                results.append({
                    "URL": url,
                    "Heuristic": 0.0,
                    "URL Score": 0.0,
                    "Content": 0.0,
                    "Popularity": 0.0,
                    "PageRank": 0.0,
                    "Error": str(e)
                })

        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"URL analysis error: {str(e)}")
        return None

# Main app
def main():
    st.set_page_config(
        page_title="URL Credibility Analyzer with Claude",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 URL Credibility Analyzer with Claude")
    st.markdown("Analyze URL credibility using heuristics and Claude AI")

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        model = st.selectbox(
            "Claude Model",
            ["claude-3-haiku-20240307"],
            index=0
        )
        st.info("💡 Using Claude 3 Haiku - fastest and most cost-effective model")
        max_tokens = st.slider("Max Tokens", 100, 4000, 1024)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

        st.divider()
        st.header("🔍 URL Analysis")
        auto_analyze_urls = st.checkbox("Auto-analyze URLs in responses", value=True,
                                        help="Automatically analyze credibility of URLs mentioned in Claude's responses")

        # Check if NN model is available
        nn_predictor, _ = load_nn_model()
        if nn_predictor:
            use_nn = st.checkbox("Use Neural Network predictions", value=True,
                                help="Use trained neural network for enhanced credibility predictions")
            st.success("✅ Neural Network model loaded")
        else:
            use_nn = False
            st.info("ℹ️ Train a model using train_model.py to enable NN predictions")

        st.divider()
        st.markdown("### About")
        st.markdown("This app combines URL credibility scoring with Claude AI for intelligent analysis.")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🤖 Claude Chat", "📊 URL Analysis", "📈 Batch Analysis"])

    # Tab 1: Claude Chat
    with tab1:
        st.header("Chat with Claude")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show URL analysis for assistant messages if available
                if message["role"] == "assistant" and "url_analysis" in message and message["url_analysis"] is not None:
                    st.divider()
                    st.subheader("📊 URL Credibility Analysis")
                    df = message["url_analysis"]

                    # Choose gradient column based on available columns
                    if "NN Score" in df.columns:
                        gradient_col = "NN Score"
                    elif "Heuristic" in df.columns:
                        gradient_col = "Heuristic"
                    else:
                        gradient_col = "Combined"

                    # Display as a styled dataframe
                    st.dataframe(
                        df.style.background_gradient(subset=[gradient_col], cmap='RdYlGn', vmin=0, vmax=1),
                        use_container_width=True,
                        hide_index=True
                    )

        # Chat input
        if prompt := st.chat_input("Ask Claude anything ! I'm really smart :)"):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get Claude response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        client = get_claude_client()
                        response = client.messages.create(
                            model=model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            messages=[{"role": msg["role"], "content": msg["content"]}
                                     for msg in st.session_state.messages]
                        )

                        assistant_message = response.content[0].text
                        st.markdown(assistant_message)

                        # Extract and analyze URLs if enabled
                        url_analysis_df = None
                        if auto_analyze_urls:
                            urls = extract_urls(assistant_message)
                            if urls:
                                with st.spinner(f"Analyzing {len(urls)} URL(s)..."):
                                    url_analysis_df = analyze_urls(urls, model, max_tokens, temperature, use_nn=use_nn)

                                if url_analysis_df is not None and not url_analysis_df.empty:
                                    st.divider()
                                    st.subheader("📊 URL Credibility Analysis")

                                    # Choose column for gradient based on what's available
                                    gradient_col = "NN Score" if use_nn and "NN Score" in url_analysis_df.columns else "Heuristic"

                                    st.dataframe(
                                        url_analysis_df.style.background_gradient(subset=[gradient_col], cmap='RdYlGn', vmin=0, vmax=1),
                                        use_container_width=True,
                                        hide_index=True
                                    )

                        # Store message with URL analysis
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": assistant_message,
                            "url_analysis": url_analysis_df
                        })

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Tab 2: Single URL Analysis
    with tab2:
        st.header("Analyze Single URL")

        col1, col2 = st.columns([2, 1])

        with col1:
            url_input = st.text_input(
                "Enter URL to analyze",
                placeholder="https://example.com/article"
            )

        with col2:
            use_claude = st.checkbox("Enhance with Claude Analysis", value=True)

        if st.button("Analyze URL", type="primary"):
            if not url_input:
                st.warning("Please enter a URL")
            else:
                with st.spinner("Analyzing URL..."):
                    try:
                        # Compute PageRank with single URL
                        pagerank_scores = d1.compute_pagerank([url_input])

                        # Get credibility scores
                        result = d1.score_url_with_content(url_input, pagerank_scores)

                        # Display results
                        st.subheader("📊 Credibility Scores")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Combined Score", f"{result['combined_score']:.2f}")
                        with col2:
                            st.metric("URL Score", f"{result['url_score']:.2f}")
                        with col3:
                            st.metric("Content Score", f"{result['text_score']:.2f}")
                        with col4:
                            st.metric("Popularity", f"{result['popularity_score']:.2f}")

                        # Show detailed breakdown
                        st.subheader("📋 Detailed Breakdown")
                        df = pd.DataFrame([result])
                        st.dataframe(df, use_container_width=True)

                        # Claude analysis
                        if use_claude:
                            st.subheader("🤖 Claude's Analysis")

                            with st.spinner("Getting Claude's insights..."):
                                try:
                                    client = get_claude_client()

                                    analysis_prompt = f"""Analyze this URL credibility assessment:

URL: {url_input}

Scores:
- Combined Score: {result['combined_score']:.2f}
- URL Score: {result['url_score']:.2f}
- Content Score: {result['text_score']:.2f}
- Popularity Score: {result['popularity_score']:.2f}
- PageRank Score: {result['pagerank_score']:.2f}
- Ad Count: {result['ad_count']}

Please provide:
1. An interpretation of these scores
2. Potential credibility concerns
3. Recommendations for using this source
4. Overall credibility assessment (High/Medium/Low)

Be concise and actionable."""

                                    response = client.messages.create(
                                        model=model,
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                        messages=[{"role": "user", "content": analysis_prompt}]
                                    )

                                    st.markdown(response.content[0].text)

                                except Exception as e:
                                    st.error(f"Claude analysis error: {str(e)}")

                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")

    # Tab 3: Batch Analysis
    with tab3:
        st.header("Batch URL Analysis")

        st.markdown("Enter multiple URLs (one per line) for batch analysis:")

        urls_text = st.text_area(
            "URLs to analyze",
            height=200,
            placeholder="https://example1.com\nhttps://example2.com\nhttps://example3.com"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            analyze_batch = st.button("Analyze Batch", type="primary")
        with col2:
            use_claude_batch = st.checkbox("Generate Claude Summary", value=True)

        if analyze_batch:
            if not urls_text.strip():
                st.warning("Please enter at least one URL")
            else:
                # Parse URLs
                urls = [url.strip() for url in urls_text.split('\n') if url.strip()]

                if len(urls) > 50:
                    st.warning("⚠️ Large batch detected. This may take a while...")

                with st.spinner(f"Analyzing {len(urls)} URLs..."):
                    try:
                        # Compute PageRank for all URLs
                        pagerank_scores = d1.compute_pagerank(urls)

                        # Analyze each URL
                        results = []
                        progress_bar = st.progress(0)

                        for idx, url in enumerate(urls):
                            try:
                                result = d1.score_url_with_content(url, pagerank_scores)
                                results.append({
                                    "URL": url,
                                    **result
                                })
                            except Exception as e:
                                st.warning(f"Failed to analyze {url}: {str(e)}")

                            progress_bar.progress((idx + 1) / len(urls))

                        # Display results
                        if results:
                            st.subheader(f"📊 Results for {len(results)} URLs")

                            df = pd.DataFrame(results)

                            # Sort by combined score
                            df = df.sort_values("combined_score", ascending=False)

                            # Display summary stats
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Avg Combined Score", f"{df['combined_score'].mean():.2f}")
                            with col2:
                                st.metric("High Credibility", f"{(df['combined_score'] > 0.7).sum()}")
                            with col3:
                                st.metric("Medium Credibility", f"{((df['combined_score'] >= 0.4) & (df['combined_score'] <= 0.7)).sum()}")
                            with col4:
                                st.metric("Low Credibility", f"{(df['combined_score'] < 0.4).sum()}")

                            # Display table
                            st.dataframe(
                                df.style.background_gradient(subset=['combined_score'], cmap='RdYlGn'),
                                use_container_width=True,
                                height=400
                            )

                            # Download button
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="url_credibility_results.csv",
                                mime="text/csv"
                            )

                            # Claude summary
                            if use_claude_batch:
                                st.subheader("🤖 Claude's Batch Summary")

                                with st.spinner("Generating summary..."):
                                    try:
                                        client = get_claude_client()

                                        summary_data = df.head(10).to_dict('records')

                                        summary_prompt = f"""Analyze this batch of URL credibility assessments:

Total URLs analyzed: {len(results)}
Average combined score: {df['combined_score'].mean():.2f}

Top 10 URLs by credibility:
{json.dumps(summary_data, indent=2)}

Provide:
1. Overall credibility trends
2. Notable high-credibility sources
3. Red flags or concerns
4. General recommendations

Be concise and actionable."""

                                        response = client.messages.create(
                                            model=model,
                                            max_tokens=max_tokens,
                                            temperature=temperature,
                                            messages=[{"role": "user", "content": summary_prompt}]
                                        )

                                        st.markdown(response.content[0].text)

                                    except Exception as e:
                                        st.error(f"Claude summary error: {str(e)}")

                    except Exception as e:
                        st.error(f"Batch analysis error: {str(e)}")

if __name__ == "__main__":
    main()
