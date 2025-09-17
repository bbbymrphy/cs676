import argparse
from urllib.parse import urlparse

def score_url(url: str) -> dict:
    parsed = urlparse(url)
    score = 100
    breakdown = {}

    # Check HTTPS
    if parsed.scheme == "https":
        score += 5
        breakdown["https"] = +5
    else:
        score -= 10
        breakdown["https"] = -10
    
    # Domain length and TLD
    domain = parsed.netloc
    tld = domain.split(".")[-1]
    if len(domain) > 30:
        score -= 10
        breakdown["domain_length"] = -10
    if tld in ["xyz", "top", "info"]:
        score -= 20
        breakdown["tld_penalty"] = -20
    
    # Subdomains
    subdomains = domain.split(".")[:-2]
    if len(subdomains) > 2:
        score -= 10
        breakdown["subdomains"] = -10
    
    # Path segments
    path_segments = [p for p in parsed.path.split("/") if p]
    if len(path_segments) > 5:
        score -= 5
        breakdown["path_depth"] = -5
    
    # Query params
    if parsed.query:
        num_params = parsed.query.count("&") + 1
        if num_params > 3:
            score -= 10
            breakdown["query_params"] = -10
    
    # Characters
    if any(c in url for c in ["%", "=", "_"]):
        score -= 5
        breakdown["special_chars"] = -5
    
    return {
        "url": url,
        "score": max(0, min(100, score)),
        "breakdown": breakdown
    }

def main():
    parser = argparse.ArgumentParser(description="URL Quality Scorer")
    parser.add_argument("urls", nargs="+", help="One or more URLs to score")
    args = parser.parse_args()

    for url in args.urls:
        result = score_url(url)
        print(f"\nURL: {result['url']}")
        print(f"Score: {result['score']}/100")
        print("Breakdown:", result["breakdown"])

if __name__ == "__main__":
    main()
