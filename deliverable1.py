import re
import json
import os
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import pandas as pd
import networkx as nx
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


# === URL heuristic scoring ===
DOMAIN_WEIGHTS = {
    ".gov": 1.0, ".edu": 0.95, ".org": 0.85, ".com": 0.8,
    ".net": 0.7, ".info": 0.6, ".xyz": 0.4, ".biz": 0.3,
}
SOCIAL_SITES = ["facebook.com", "twitter.com", "x.com", "instagram.com",
                "tiktok.com", "reddit.com", "linkedin.com", "snapchat.com", "pinterest.com"]
CREDIBLE_SITES = ["nytimes.com", "bbc.com", "nature.com", "nasa.gov",
                  "who.int", "mit.edu", "harvard.edu", "whitehouse.gov"]

# === Popular sites (can expand with Tranco/Alexa data) ===
POPULAR_SITES = {
    "google.com": 1.0,
    "youtube.com": 1.0,
    "wikipedia.org": 0.95,
    "twitter.com": 0.9,
    "bbc.com": 0.9,
    "nytimes.com": 0.9,
    "cnn.com": 0.85,
    "reddit.com": 0.8,
    "instagram.com": 0.8,
    "linkedin.com": 0.8,
}


def score_url(url: str) -> float:
    score = 0.5
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path

    # Domain extension
    for ext, weight in DOMAIN_WEIGHTS.items():
        if domain.endswith(ext):
            score += (weight - 0.5)
            break

    # Credible domains
    if any(domain.endswith(site) for site in CREDIBLE_SITES):
        score += 0.2

    # Social sites penalty
    if any(social in domain for social in SOCIAL_SITES):
        score -= 0.3

    # Complexity penalty
    path_len = len(path.split("/"))
    query_len = len(parsed.query.split("&")) if parsed.query else 0
    if (path_len + query_len) > 10:
        score -= 0.2
    elif (path_len + query_len) > 5:
        score -= 0.1

    # Suspicious domains
    if len(re.findall(r"[-\d]", domain)) > 5:
        score -= 0.2

    return max(0.0, min(1.0, score))


# === Popularity Bonus ===
def popularity_bonus(domain: str) -> float:
    domain = domain.lower()
    for site, bonus in POPULAR_SITES.items():
        if domain.endswith(site):
            return bonus * 0.2
    return 0.0


# === Content Extraction ===
def fetch_page_text(url: str):
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return "", 0

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    # Detect ads
    ad_tags = soup.find_all(["iframe", "ins"])
    ad_tags += [t for t in soup.find_all("div") if "ad" in (t.get("class") or []) or "ad" in (t.get("id") or "")]
    ad_tags += [t for t in soup.find_all("span") if "sponsored" in (t.get("class") or [])]

    ad_count = len(ad_tags)
    text = " ".join(soup.stripped_strings)
    return text, ad_count


# === Storage ===
DB_FILE = "url_results_db.json"

def load_storage():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            return json.load(f)
    return {}

def save_storage(storage):
    with open(DB_FILE, "w") as f:
        json.dump(storage, f, indent=2)


# === Credibility Functions ===
def evaluate_text_credibility(text: str, ad_count: int) -> float:
    if not text:
        return 0.2

    score = 0.5
    word_count = len(text.split())

    if word_count > 500:
        score += 0.2
    elif word_count < 50:
        score -= 0.2

    if re.search(r"BUY NOW|CLICK HERE|FREE", text, re.IGNORECASE):
        score -= 0.3

    if sum(1 for w in text.split() if w.isupper()) > 20:
        score -= 0.2

    if ad_count > 10:
        score -= 0.3
    elif ad_count > 3:
        score -= 0.15
    elif ad_count > 0:
        score -= 0.05

    return max(0.0, min(1.0, score))


# === Compute PageRank across all URLs ===
def compute_pagerank(urls):
    G = nx.DiGraph()
    for i, u1 in enumerate(urls):
        for j, u2 in enumerate(urls):
            if i != j:
                G.add_edge(u1, u2)
    pr = nx.pagerank(G, alpha=0.85)
    # Normalize 0–1
    min_pr, max_pr = min(pr.values()), max(pr.values())
    return {u: (score - min_pr) / (max_pr - min_pr + 1e-9) for u, score in pr.items()}


# === Combined scoring with PageRank ===
def score_url_with_content(url: str, pagerank_scores=None) -> dict:
    storage = load_storage()

    if url in storage:
        return storage[url]

    page_text, ad_count = fetch_page_text(url)
    parsed = urlparse(url)
    domain = parsed.netloc

    url_score = score_url(url)
    pop_score = popularity_bonus(domain)
    text_score = evaluate_text_credibility(page_text, ad_count)
    pr_score = pagerank_scores.get(url, 0.5) if pagerank_scores else 0.5

    combined_score = (
        0.4 * url_score +
        0.3 * text_score +
        0.2 * pop_score +
        0.1 * pr_score
    )

    results = {
        "url_score": url_score,
        "text_score": text_score,
        "popularity_score": pop_score,
        "pagerank_score": pr_score,
        "ad_count": ad_count,
        "combined_score": combined_score
    }

    storage[url] = results
    save_storage(storage)

    return results



if __name__ == "__main__":
    test_urls = list(pd.read_csv('urls_from_tranco.csv')['url'])
    test_urls = [
        # High credibility .gov / .edu
        "https://www.nasa.gov/mission_pages/station/main/index.html",
        "https://www.harvard.edu/research/article?id=123",
        "https://www.whitehouse.gov/briefing-room/",
        
        # Well-known .org sites
        "https://www.wikipedia.org/",
        "https://www.who.int/news-room/fact-sheets/detail/coronavirus-disease-(covid-19)",
        
        # News outlets
        "https://www.bbc.com/news/world-us-canada-68888712",
        "https://www.cnn.com/2025/09/19/tech/apple-iphone-ai-update/index.html",
        "https://www.nytimes.com/2025/09/19/business/markets/dow-jones.html",
        
        # Social media (should get lower score)
        "https://www.reddit.com/r/conspiracy/",
        "https://twitter.com/nytimes/status/1234567890",
        "https://www.instagram.com/p/abcd1234/",
        
        # Commercial .com with ads
        "https://www.buzzfeed.com/quiz/which-pizza-are-you",
        "https://www.cnn.com/",
        
        # Clickbait / spammy patterns
        "https://best-health-tips-now.biz/buy-now-free!!!",
        "https://amazing-deals123.xyz/CLICK-HERE/totally-free",
        
        # Tech sites
        "https://github.com/openai/gpt-5",
        "https://arxiv.org/abs/2405.12345",
        "https://www.techcrunch.com/2025/09/19/startup-funding-round/",
        
        # Corporate sites
        "https://www.apple.com/iphone/",
        "https://www.microsoft.com/en-us/security",
        
        # Random blog (lower credibility)
        "https://myrandomblogexample.net/2025/09/19/thoughts-on-ai/",
        "https://johns-cooking-blog.info/recipe-of-the-day",

        # Health information (some credible, some not)
        "https://www.cdc.gov/coronavirus/2019-ncov/index.html",
        "https://www.nih.gov/news-events",
        "https://www.mayoclinic.org/diseases-conditions/coronavirus/symptoms-causes/syc-20479963",
        "https://www.who.int/news-room/fact-sheets/detail/mental-health-strengthening-our-response",

        
        "https://www.webmd.com/cold-and-flu/default.htm",
        "https://www.healthline.com/nutrition/10-benefits-of-exercise",
        "https://www.medicalnewstoday.com/articles/322268",

        
        "https://www.naturalnews.com/",
        "https://www.drweil.com/health-wellness/balanced-living/healthy-living-tips/",
        "https://draxe.com/nutrition/benefits-of-collagen/",

      
        "https://miraclecures-4u.biz/buy-now-free!!!",
        "https://superhealth123.xyz/ultimate-detox-cleanse",
        "https://best-diet-pills-now.info/lose-20lbs-fast"

    ]

    # Compute PageRank once for all URLs

    test_urls = pd.read_csv('back_pain_urls.csv')['URL'].tolist()
    print(test_urls)
    pagerank_scores = compute_pagerank(test_urls)



    for u in test_urls:
        try:
            result = score_url_with_content(u, pagerank_scores=pagerank_scores)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(e)
