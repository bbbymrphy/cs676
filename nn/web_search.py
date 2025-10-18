"""
Web Search Integration for URL Credibility Analysis
Uses Claude to search the web and analyze search results for credibility assessment
"""

import os
from anthropic import Anthropic
from googlesearch import search as google_search
import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Optional
import time
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

class WebSearchAnalyzer:
    """Search the web and analyze results using Claude"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Anthropic API key"""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        self.client = Anthropic(api_key=self.api_key)

    def search_web(self, query: str, num_results: int = 10) -> List[str]:
        """Search using Claude to suggest URLs or use Google search fallback"""

        # Try Google search first
        try:
            results = []
            for url in google_search(query, num_results=num_results, sleep_interval=2):
                results.append(url)
                if len(results) >= num_results:
                    break

            if results:
                return results
        except Exception as e:
            print(f"Google search failed: {e}")

        # Fallback: Ask Claude to suggest URLs
        print(f"Using Claude to suggest URLs for: {query}")
        try:
            prompt = f"""For the search query "{query}", suggest {num_results} real, credible URLs that would appear in search results.
Include a mix of different credibility levels if relevant to the query.

Return ONLY the URLs, one per line, no explanations or numbering.
Example format:
https://example.com/article1
https://example.org/page2"""

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text
            # Extract URLs from response
            import re
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            urls = re.findall(url_pattern, text)
            urls = [re.sub(r'[.,;:!?)]+$', '', url) for url in urls]

            return urls[:num_results]

        except Exception as e:
            print(f"Claude URL suggestion failed: {e}")
            return []

    def fetch_page_content(self, url: str, max_length: int = 5000) -> Dict[str, str]:
        """Fetch webpage content and metadata"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get title
            title = soup.find('title')
            title_text = title.get_text() if title else ""

            # Get meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ""

            # Get main text content
            text = soup.get_text(separator=' ', strip=True)
            text = text[:max_length]  # Limit length

            return {
                'url': url,
                'title': title_text,
                'description': description,
                'content': text,
                'success': True
            }

        except Exception as e:
            return {
                'url': url,
                'title': '',
                'description': '',
                'content': '',
                'success': False,
                'error': str(e)
            }

    def analyze_url_with_claude(self, url: str, content: str, model: str = "claude-3-haiku-20240307") -> Dict:
        """Use Claude to analyze URL credibility based on content"""
        try:
            prompt = f"""Analyze the credibility of this webpage:

URL: {url}

Content preview:
{content[:2000]}

Evaluate the following aspects and provide a score from 0 to 1 for each:

1. **Source Authority** (0-1): Is this from a reputable, authoritative source?
2. **Content Quality** (0-1): Is the content well-written, professional, and informative?
3. **Evidence & Citations** (0-1): Does it provide evidence, sources, or citations?
4. **Objectivity** (0-1): Is the content balanced and objective, or biased/promotional?
5. **Recency** (0-1): Does the content appear current and up-to-date?

Respond in JSON format:
{{
    "authority_score": 0.0-1.0,
    "quality_score": 0.0-1.0,
    "evidence_score": 0.0-1.0,
    "objectivity_score": 0.0-1.0,
    "recency_score": 0.0-1.0,
    "overall_credibility": 0.0-1.0,
    "reasoning": "Brief explanation of the scores",
    "credibility_label": "high|medium|low"
}}"""

            response = self.client.messages.create(
                model=model,
                max_tokens=1024,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse Claude's response
            response_text = response.content[0].text

            # Try to extract JSON from response
            try:
                # Find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    analysis = json.loads(json_str)
                else:
                    # Fallback parsing
                    analysis = self._parse_fallback(response_text)
            except json.JSONDecodeError:
                analysis = self._parse_fallback(response_text)

            analysis['url'] = url
            return analysis

        except Exception as e:
            return {
                'url': url,
                'authority_score': 0.5,
                'quality_score': 0.5,
                'evidence_score': 0.5,
                'objectivity_score': 0.5,
                'recency_score': 0.5,
                'overall_credibility': 0.5,
                'reasoning': f"Error during analysis: {str(e)}",
                'credibility_label': 'medium'
            }

    def _parse_fallback(self, text: str) -> Dict:
        """Fallback parsing if JSON extraction fails"""
        # Simple heuristic: look for credibility indicators in text
        text_lower = text.lower()

        if any(word in text_lower for word in ['high credibility', 'very credible', 'trustworthy']):
            credibility = 0.8
            label = 'high'
        elif any(word in text_lower for word in ['low credibility', 'not credible', 'suspicious']):
            credibility = 0.3
            label = 'low'
        else:
            credibility = 0.5
            label = 'medium'

        return {
            'authority_score': credibility,
            'quality_score': credibility,
            'evidence_score': credibility,
            'objectivity_score': credibility,
            'recency_score': credibility,
            'overall_credibility': credibility,
            'reasoning': text[:200],
            'credibility_label': label
        }

    def search_and_analyze(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search web and analyze all results with Claude"""
        print(f"Searching for: {query}")

        # Get search results
        urls = self.search_web(query, num_results)
        print(f"Found {len(urls)} URLs")

        results = []
        for i, url in enumerate(urls, 1):
            print(f"Analyzing {i}/{len(urls)}: {url}")

            # Fetch content
            page_data = self.fetch_page_content(url)

            if page_data['success']:
                # Analyze with Claude
                analysis = self.analyze_url_with_claude(
                    url,
                    f"Title: {page_data['title']}\n\n{page_data['content']}"
                )
                analysis['title'] = page_data['title']
                results.append(analysis)
            else:
                print(f"  Failed to fetch: {page_data.get('error', 'Unknown error')}")

            # Rate limiting
            time.sleep(1)

        return results

    def create_training_label(self, url: str, content: str) -> Dict:
        """
        Use Claude to create a training label for a URL
        This is specifically for building training datasets
        """
        analysis = self.analyze_url_with_claude(url, content)

        # Convert credibility label to binary/multi-class
        label_map = {'high': 2, 'medium': 1, 'low': 0}
        numeric_label = label_map.get(analysis['credibility_label'], 1)

        return {
            'url': url,
            'credibility_score': analysis['overall_credibility'],
            'credibility_class': numeric_label,
            'credibility_label': analysis['credibility_label'],
            'claude_scores': {
                'authority': analysis['authority_score'],
                'quality': analysis['quality_score'],
                'evidence': analysis['evidence_score'],
                'objectivity': analysis['objectivity_score'],
                'recency': analysis['recency_score'],
            },
            'reasoning': analysis['reasoning']
        }


if __name__ == "__main__":
    # Test the web search analyzer
    analyzer = WebSearchAnalyzer()

    # Test search and analysis
    query = "climate change evidence"
    results = analyzer.search_and_analyze(query, num_results=3)

    print(f"\n\n=== Results for '{query}' ===")
    for result in results:
        print(f"\nURL: {result['url']}")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Credibility: {result['credibility_label']} ({result['overall_credibility']:.2f})")
        print(f"Reasoning: {result['reasoning'][:150]}...")
