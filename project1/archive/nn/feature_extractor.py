"""
Feature Extractor for URL Credibility Analysis
Extracts comprehensive features from URLs and webpage content for neural network training
"""

import re
import numpy as np
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import textstat
from datetime import datetime
import nltk
from collections import Counter

# Download required NLTK data
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass

from nltk.tokenize import word_tokenize, sent_tokenize

# Try to load stopwords, use fallback if unavailable
try:
    from nltk.corpus import stopwords
    STOPWORDS_AVAILABLE = True
except:
    STOPWORDS_AVAILABLE = False


class URLFeatureExtractor:
    """Extract features from URLs and webpage content"""

    def __init__(self):
        if STOPWORDS_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'])
        else:
            # Fallback stopwords
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                                  'is', 'it', 'of', 'as', 'be', 'by', 'from', 'with', 'this', 'that'])

    def extract_url_features(self, url):
        """Extract features from URL structure"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()

        features = {
            # Basic URL features
            'url_length': len(url),
            'domain_length': len(domain),
            'path_length': len(path),
            'has_https': 1 if parsed.scheme == 'https' else 0,

            # Character-based features
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_underscores': url.count('_'),
            'num_slashes': url.count('/'),
            'num_questions': url.count('?'),
            'num_equals': url.count('='),
            'num_ampersands': url.count('&'),
            'num_digits': sum(c.isdigit() for c in url),

            # Domain features
            'has_ip': 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0,
            'num_subdomains': domain.count('.') - 1 if '.' in domain else 0,

            # TLD features
            'tld_com': 1 if domain.endswith('.com') else 0,
            'tld_org': 1 if domain.endswith('.org') else 0,
            'tld_edu': 1 if domain.endswith('.edu') else 0,
            'tld_gov': 1 if domain.endswith('.gov') else 0,
            'tld_net': 1 if domain.endswith('.net') else 0,

            # Suspicious patterns
            'has_at_symbol': 1 if '@' in url else 0,
            'double_slash_in_path': 1 if '//' in path else 0,
            'suspicious_keywords': self._count_suspicious_keywords(url),
        }

        return features

    def extract_content_features(self, url, timeout=10):
        """Extract features from webpage content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=timeout, verify=False)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Get text content
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator=' ', strip=True)

            # Title and meta
            title = soup.find('title')
            title_text = title.get_text() if title else ""

            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ""

            # Extract features
            features = {
                # Text quality features
                'text_length': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(sent_tokenize(text)) if text else 0,
                'avg_word_length': np.mean([len(word) for word in text.split()]) if text else 0,
                'avg_sentence_length': len(text.split()) / max(len(sent_tokenize(text)), 1) if text else 0,

                # Readability scores
                'flesch_reading_ease': textstat.flesch_reading_ease(text) if text else 0,
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text) if text else 0,
                'gunning_fog': textstat.gunning_fog(text) if text else 0,

                # HTML structure features
                'num_links': len(soup.find_all('a')),
                'num_images': len(soup.find_all('img')),
                'num_forms': len(soup.find_all('form')),
                'has_title': 1 if title_text else 0,
                'title_length': len(title_text),
                'has_description': 1 if description else 0,

                # Content quality indicators
                'num_headings': len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
                'num_paragraphs': len(soup.find_all('p')),
                'num_lists': len(soup.find_all(['ul', 'ol'])),

                # Ad/spam indicators
                'num_iframes': len(soup.find_all('iframe')),
                'num_scripts': len(soup.find_all('script')),
                'num_external_links': self._count_external_links(soup, urlparse(url).netloc),

                # Author/date indicators
                'has_author': self._has_author_info(soup),
                'has_date': self._has_date_info(soup),
            }

            # Lexical diversity
            words = word_tokenize(text.lower()) if text else []
            unique_words = set(words)
            features['lexical_diversity'] = len(unique_words) / len(words) if words else 0

            # Content keywords
            content_words = [w for w in words if w.isalpha() and w not in self.stop_words]
            features['content_word_ratio'] = len(content_words) / len(words) if words else 0

            return features

        except Exception as e:
            # Return default features on error
            return self._default_content_features()

    def _count_suspicious_keywords(self, url):
        """Count suspicious keywords in URL"""
        suspicious = ['click', 'free', 'bonus', 'win', 'prize', 'download',
                     'urgent', 'limited', 'offer', 'deal', 'discount']
        url_lower = url.lower()
        return sum(1 for keyword in suspicious if keyword in url_lower)

    def _count_external_links(self, soup, domain):
        """Count links pointing to external domains"""
        count = 0
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http') and domain not in href:
                count += 1
        return count

    def _has_author_info(self, soup):
        """Check if page has author information"""
        author_indicators = [
            soup.find('meta', attrs={'name': 'author'}),
            soup.find(class_=re.compile('author', re.I)),
            soup.find('span', class_=re.compile('author', re.I)),
        ]
        return 1 if any(author_indicators) else 0

    def _has_date_info(self, soup):
        """Check if page has publication date"""
        date_indicators = [
            soup.find('time'),
            soup.find('meta', attrs={'property': 'article:published_time'}),
            soup.find(class_=re.compile('date|published', re.I)),
        ]
        return 1 if any(date_indicators) else 0

    def _default_content_features(self):
        """Return default features when content extraction fails"""
        return {
            'text_length': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'avg_sentence_length': 0,
            'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0, 'gunning_fog': 0,
            'num_links': 0, 'num_images': 0, 'num_forms': 0,
            'has_title': 0, 'title_length': 0, 'has_description': 0,
            'num_headings': 0, 'num_paragraphs': 0, 'num_lists': 0,
            'num_iframes': 0, 'num_scripts': 0, 'num_external_links': 0,
            'has_author': 0, 'has_date': 0, 'lexical_diversity': 0,
            'content_word_ratio': 0
        }

    def extract_all_features(self, url, existing_scores=None):
        """Extract all features from a URL"""
        # URL features
        url_features = self.extract_url_features(url)

        # Content features
        content_features = self.extract_content_features(url)

        # Combine all features
        all_features = {**url_features, **content_features}

        # Add existing heuristic scores if provided
        if existing_scores:
            all_features['heuristic_url_score'] = existing_scores.get('url_score', 0)
            all_features['heuristic_content_score'] = existing_scores.get('text_score', 0)
            all_features['heuristic_popularity_score'] = existing_scores.get('popularity_score', 0)
            all_features['heuristic_pagerank_score'] = existing_scores.get('pagerank_score', 0)
            all_features['heuristic_combined_score'] = existing_scores.get('combined_score', 0)

        return all_features

    def get_feature_vector(self, features):
        """Convert feature dictionary to numpy array"""
        # Ensure consistent ordering
        feature_names = sorted(features.keys())
        return np.array([features[name] for name in feature_names]), feature_names


if __name__ == "__main__":
    # Test the feature extractor
    extractor = URLFeatureExtractor()

    test_urls = [
        "https://www.bbc.com/news",
        "https://en.wikipedia.org/wiki/Machine_learning",
        "http://suspicious-site123.com/free-prize-click-here"
    ]

    for url in test_urls:
        print(f"\nAnalyzing: {url}")
        features = extractor.extract_all_features(url)
        print(f"Total features: {len(features)}")
        print(f"Sample features: {dict(list(features.items())[:5])}")
