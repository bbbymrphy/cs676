import requests
import csv
import random
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# ---- Configuration ----

TRANCO_TOP1M_URL = "https://tranco-list.eu/top-1m.csv.zip"
SEED_SAMPLE_SIZE = 100      # how many domains to sample from Tranco
LINKS_PER_DOMAIN = 100        # how many links to extract per domain
TARGET_URLS = 10000        # target total URLs
USER_AGENT = "Mozilla/5.0 (compatible; URLCollector/1.0)"

# ---- Utility Functions ----

def download_and_extract_tranco(seed_csv_path="top-1m.csv", unzip_dir="."):
    """Download the Tranco Top1M CSV (zipped), unzip, return list of domains."""
    import zipfile
    import io
    
    r = requests.get(TRANCO_TOP1M_URL, stream=True)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    # The zip should contain a CSV, often named something like "top-1m.csv"
    csv_names = [name for name in z.namelist() if name.lower().endswith(".csv")]
    if not csv_names:
        raise RuntimeError("No CSV found in Tranco zip")
    csv_name = csv_names[0]
    print(f"Extracting {csv_name} from Tranco zip …")
    with z.open(csv_name) as f:
        lines = (line.decode("utf-8").strip() for line in f)
        # CSV has rank,domain
        domains = []
        for line in lines:
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            rank, domain = parts[0], parts[1]
            domains.append(domain)
    print(f"Got {len(domains)} domains from Tranco")
    return domains

def gather_urls_from_seeds(seed_domains, links_per_domain=10, max_urls=50):
    """
    For each domain in seed_domains:
        - fetch homepage
        - grab up to links_per_domain unique 'a' tag hrefs
    until we collect max_urls total.
    """
    collected = set()
    headers = {"User-Agent": USER_AGENT}
    
    for domain in seed_domains:
        if len(collected) >= max_urls:
            break
        url = "http://" + domain  # try HTTP (optionally you could try HTTPS)
        try:
            resp = requests.get(url, timeout=5, headers=headers)
            if resp.status_code != 200 or 'text/html' not in resp.headers.get("Content-Type", ""):
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            links = [a.get("href") for a in soup.find_all("a", href=True)]
            # process links
            count = 0
            for link in links:
                if count >= links_per_domain:
                    break
                full = urljoin(resp.url, link)
                if full.startswith("http"):
                    if full not in collected:
                        collected.add(full)
                        count += 1
                        if len(collected) >= max_urls:
                            break
        except Exception as e:
            # Could log errors if desired
            # print(f"Error fetching domain {domain}: {e}")
            pass
        # be polite
        
    
    return list(collected)

def save_urls_to_csv(urls, filename="urls_from_tranco.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url"])
        for u in urls:
            writer.writerow([u])
    print(f"Saved {len(urls)} URLs to {filename}")

# ---- Main Logic ----

if __name__ == "__main__":
    # Step 1: get Tranco domains
    domains = download_and_extract_tranco()
    
    # Step 2: sample seeds
    random.seed(42)
    seed_domains = random.sample(domains, min(SEED_SAMPLE_SIZE, len(domains)))
    print(f"Using {len(seed_domains)} seed domains")
    
    # Step 3: gather URLs
    urls = gather_urls_from_seeds(seed_domains, links_per_domain=LINKS_PER_DOMAIN, max_urls=TARGET_URLS)
    print(f"Collected {len(urls)} URLs in total")
    
    # Step 4: save to CSV
    save_urls_to_csv(urls)
