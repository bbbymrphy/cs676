# Project1 — URL Credibility Heuristics

Summary
- Heuristic pipeline to score URL credibility by combining domain heuristics, site popularity, page content signals, and a simple PageRank over the input URL set.
- Primary script: `deliverable1.py`.
- Designed for experimentation, sensitivity analysis, and small-scale crawling with local JSON caching.

Repository layout (project1/)
- deliverable1.py        — Main scoring pipeline and helpers (domain heuristics, content extraction, PageRank, combined scoring).
- urls_from_tranco.csv  — Input CSV of candidate URLs (runtime; column name: `url`).
- url_results_db.json   — Local JSON cache written by `deliverable1.py`.
- sensitivity_analysis.py (recommended) — Script to run sensitivity experiments on heuristic parameters and weights (generate CSV + PNG results). (See "Sensitivity analysis" section.)
- README.md             — This file.

Requirements
- Python 3.8+
- Required packages:
  - requests
  - beautifulsoup4
  - pandas
  - networkx
  - matplotlib, numpy (for sensitivity analysis / plotting)
- Install on macOS:
  - python3 -m pip install requests beautifulsoup4 pandas networkx
  - For sensitivity analysis (optional): python3 -m pip install matplotlib numpy

Quick start — run scoring on the CSV (from project directory)
- Basic run (uses hard-coded test URLs in the script if executed as __main__):
  - python3 deliverable1.py
- To run scoring programmatically, import functions and call:
  - from deliverable1 import score_url_with_content, compute_pagerank
  - pagerank = compute_pagerank(url_list)
  - result = score_url_with_content(url, pagerank_scores=pagerank)

Design & high-level flow
1. Read input URL list (CSV or a supplied list).
2. Compute PageRank across the given URL set (simple fully-connected directed graph by default).
3. For each URL:
   - Compute domain-only heuristic score (score_url).
   - Fetch page HTML and extract visible text and crude ad count (fetch_page_text).
   - Score textual credibility (evaluate_text_credibility).
   - Compute small popularity bonus from a curated POPULAR_SITES map.
   - Combine url_score, text_score, popularity_score and pagerank_score into final combined_score.
4. Cache per-URL results in `url_results_db.json` to avoid re-fetching.

Key constants (in deliverable1.py)
- DOMAIN_WEIGHTS: weight bias by domain suffix (.gov, .edu, .org, .com, etc.). Affects score_url.
- SOCIAL_SITES: substrings used to penalize common social platforms.
- CREDIBLE_SITES: curated list of known credible domains (small boost).
- POPULAR_SITES: curated popularity multipliers used to compute popularity bonus.
- DB_FILE: `url_results_db.json` — local cache file.

Function reference (behavioral description)

- score_url(url: str) -> float
  - Input: URL string.
  - Output: float in [0.0, 1.0].
  - Heuristics:
    - Base = 0.5
    - Add (DOMAIN_WEIGHTS[ext] - 0.5) if domain endswith a known suffix (first match).
    - +0.2 for any domain in CREDIBLE_SITES.
    - -0.3 if domain contains any SOCIAL_SITES substring.
    - Penalize complex URLs (path segment + query param count):
      - >10 → -0.2, >5 → -0.1.
    - Penalize suspicious domains with many digits/dashes (>5 → -0.2).
    - Clamp to [0, 1].
  - Use: fast approximate domain-only credibility when page fetch is unavailable.

- popularity_bonus(domain: str) -> float
  - Input: domain string.
  - Output: small bonus (0.0 to 0.2 typical) computed as POPULAR_SITES[site] * 0.2 when domain endswith site.
  - Use: modest positive signal for widely-known sites.

- fetch_page_text(url: str) -> (text: str, ad_count: int)
  - Fetch page with requests (User-Agent: Mozilla/5.0), timeout=10s.
  - Parse with BeautifulSoup, remove script/style/noscript.
  - Extract visible text via `soup.stripped_strings`.
  - Heuristic ad detection:
    - Count `<iframe>`, `<ins>` tags.
    - Count `<div>` elements with class/id containing "ad".
    - Count `<span>` elements with class containing "sponsored".
  - Returns (text, ad_count). On error returns ("", 0).
  - Limitations: brittle for JS-rendered content and obfuscated ad markup.

- evaluate_text_credibility(text: str, ad_count: int) -> float
  - Input: page text and ad_count.
  - Output: float in [0.0, 1.0].
  - Heuristics:
    - Empty text => 0.2.
    - Base = 0.5.
    - Word count: >500 => +0.2; <50 => -0.2.
    - Clickbait tokens (BUY NOW, CLICK HERE, FREE) => -0.3.
    - Excess uppercase words (>20) => -0.2.
    - Ad penalties: ad_count >10 => -0.3; >3 => -0.15; >0 => -0.05.
    - Clamp to [0,1].
  - Use: content-level credibility signal.

- compute_pagerank(urls: list) -> dict
  - Builds a directed graph connecting each URL to every other URL (complete graph) and runs networkx.pagerank (alpha=0.85).
  - Returns per-URL normalized PageRank in [0,1].
  - Note: current graph is synthetic (every URL links to every other). Replace with actual link graph if available.

- score_url_with_content(url: str, pagerank_scores=None) -> dict
  - Orchestrates the pipeline:
    - Loads local cache (`url_results_db.json`) and returns cached result if present.
    - Calls fetch_page_text, score_url, popularity_bonus, evaluate_text_credibility, and uses pagerank_scores (or defaults to 0.5).
    - Combined score formula (current):
      - combined_score = 0.4*url_score + 0.3*text_score + 0.2*pop_score + 0.1*pr_score
    - Saves results to cache and returns a dictionary:
      - { url_score, text_score, popularity_score, pagerank_score, ad_count, combined_score }

Storage and caching
- `url_results_db.json` is used to persist results across runs.
- load_storage() returns {} when file missing.
- save_storage() writes pretty JSON (indent=2).
- Current implementation: no TTL, no corruption handling; consider adding try/except for json decoding errors and TTL metadata.

Running sensitivity experiments (recommended)
- A sensitivity analysis script (example: `sensitivity_analysis.py`) can be used to systematically perturb:
  - DOMAIN_WEIGHTS global scaling and per-TLD perturbations.
  - Combination-weight grid for url:text:pop contributions.
  - Ablation studies (zeroing each component).
  - Text/ad penalty scaling.
- Expected outputs:
  - CSV summary tables (e.g., domain_weight_scaling.csv, per_tld_perturb.csv, weight_grid_results.csv).
  - PNG plots (spearman heatmaps and sensitivity bar charts).
- Example commands (macOS / terminal):
  - python3 sensitivity_analysis.py --csv urls_from_tranco.csv --output results --max 500
  - To fetch missing pages for accurate text scores (slower / network):
    - python3 sensitivity_analysis.py --fetch
- Required extra libs for plotting: matplotlib, numpy.

Testing recommendations
- Unit tests (pytest) for:
  - score_url for different domain forms (subdomains, ports, numeric-heavy domains).
  - evaluate_text_credibility with synthetic texts (empty, short, long, clickbait tokens, uppercase heavy).
  - popularity_bonus and POPULAR_SITES matching.
- Integration tests:
  - Mock requests.get to return deterministic HTML fixtures (news article, ad-heavy blog, SPA with empty HTML).
  - Verify caching behavior and JSON corruption fallback.
- Performance:
  - Parallelize fetching with asyncio + aiohttp or ThreadPoolExecutor for large URL sets.
  - Respect robots.txt and implement polite crawling (rate limit, domain-level backoff).

Limitations & known caveats
- Heuristic and English-centric: clickbait tokens and uppercase heuristics are tuned for English content.
- Ad detection is fragile and misses JS-injected or obfuscated ads.
- Page fetch failures return empty text and bias text_score low.
- Popularity and TLD biases favor established institutions — document and disclose this when deploying.
- PageRank currently assumes a complete link graph; replace with real link structure for meaningful PR signals.

Suggested improvements (next steps)
- Normalize domains (strip ports, remove leading "www.") before matching.
- Add TTL and metadata to cached entries; handle JSON decode errors gracefully.
- Externalize configuration (DOMAIN_WEIGHTS, POPULAR_SITES, CREDIBLE_SITES) to JSON/YAML for easy tuning.
- Replace brittle ad detection with DOM-ratio heuristics or use a headless browser for JS-rendered pages (Playwright/Selenium) with caution.
- Replace or augment heuristics with a supervised model trained on labeled credible vs non-credible pages.
- Add logging, CI tests, and recorded HTTP fixtures for stable integration tests.

Example output (per-URL)
```json
{
  "url_score": 0.72,
  "text_score": 0.45,
  "popularity_score": 0.18,
  "pagerank_score": 0.34,
  "ad_count": 2,
  "combined_score": 0.532
}
```

Contact / credits
- This project and heuristics are intended for research/education and experimentation. Use in production requires robust validation, auditing of biases, and appropriate crawling policies.
