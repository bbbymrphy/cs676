import os
import argparse
import json
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import deliverable1 as d1

# --- Utilities ---
def spearman_rank_corr(a, b):
    # a, b are 1-D numpy arrays
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    if np.all(ra == ra[0]) or np.all(rb == rb[0]):
        return np.nan
    return np.corrcoef(ra, rb)[0, 1]

def load_urls(csv_path, max_urls=None):
    df = pd.read_csv(csv_path)
    urls = list(df['url'].dropna().astype(str))
    if max_urls:
        urls = urls[:max_urls]
    return urls

def baseline_components(urls, storage, fetch_missing=False):
    rows = {}
    for u in urls:
        if u in storage:
            res = storage[u]
            url_score = res.get('url_score', d1.score_url(u))
            text_score = res.get('text_score', None)
            pop_score = res.get('popularity_score', None)
            ad_count = res.get('ad_count', None)
        else:
            url_score = d1.score_url(u)
            pop_score = None
            text_score = None
            ad_count = None
        parsed = urlparse(u)
        domain = parsed.netloc
        if pop_score is None:
            pop_score = d1.popularity_bonus(domain)
        if text_score is None:
            if fetch_missing:
                text, ad_count = d1.fetch_page_text(u)
                text_score = d1.evaluate_text_credibility(text, ad_count)
            else:
                # fallback synthetic: empty text -> low score
                text_score = 0.2
                ad_count = 0
        rows[u] = {
            'url_score': float(url_score),
            'text_score': float(text_score),
            'pop_score': float(pop_score),
            'ad_count': int(ad_count) if ad_count is not None else 0
        }
    return pd.DataFrame.from_dict(rows, orient='index')

def combined_score_from_components(df, w_url=0.5, w_text=0.3, w_pop=0.2):
    return w_url * df['url_score'] + w_text * df['text_score'] + w_pop * df['pop_score']

# --- Sensitivity experiments ---
def experiment_scale_domain_weights(df, storage, output_dir):
    base_scores = combined_score_from_components(df)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    results = []
    for s in scales:
        # copy and scale
        orig = d1.DOMAIN_WEIGHTS.copy()
        d1.DOMAIN_WEIGHTS = {k: float(v) * s for k, v in orig.items()}
        sc = []
        for u in df.index:
            sc.append(d1.score_url(u))
        sc = np.array(sc)
        pop = df['pop_score'].values
        text = df['text_score'].values
        combined = 0.5 * sc + 0.3 * text + 0.2 * pop
        rho = spearman_rank_corr(base_scores.values, combined)
        mad = np.mean(np.abs(base_scores.values - combined))
        results.append({'scale': s, 'spearman': rho, 'mean_abs_delta': mad})
        # restore
        d1.DOMAIN_WEIGHTS = orig
    pd.DataFrame(results).to_csv(os.path.join(output_dir, 'domain_weight_scaling.csv'), index=False)
    # plot
    dfres = pd.DataFrame(results)
    plt.figure()
    plt.plot(dfres['scale'], dfres['spearman'], marker='o')
    plt.title('Spearman vs DOMAIN_WEIGHTS scale')
    plt.xlabel('scale factor')
    plt.ylabel('Spearman rank correlation')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'domain_weight_scaling_spearman.png'))
    plt.close()

def experiment_per_tld_perturb(df, storage, output_dir):
    base_scores = combined_score_from_components(df)
    results = []
    for ext in list(d1.DOMAIN_WEIGHTS.keys()):
        orig_val = d1.DOMAIN_WEIGHTS[ext]
        for delta in [-0.4, -0.2, 0.0, 0.2, 0.4]:
            d1.DOMAIN_WEIGHTS[ext] = float(max(0.0, orig_val + delta))
            sc = [d1.score_url(u) for u in df.index]
            combined = 0.5 * np.array(sc) + 0.3 * df['text_score'].values + 0.2 * df['pop_score'].values
            rho = spearman_rank_corr(base_scores.values, combined)
            mad = np.mean(np.abs(base_scores.values - combined))
            results.append({'ext': ext, 'delta': delta, 'spearman': rho, 'mean_abs_delta': mad})
        d1.DOMAIN_WEIGHTS[ext] = orig_val
    pd.DataFrame(results).to_csv(os.path.join(output_dir, 'per_tld_perturb.csv'), index=False)
    # bar chart average sensitivity per ext
    summary = pd.DataFrame(results).groupby('ext')['mean_abs_delta'].mean().sort_values(ascending=False)
    plt.figure(figsize=(8,4))
    summary.plot(kind='bar')
    plt.title('Average absolute combined-score change by TLD perturbation')
    plt.ylabel('mean abs delta')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_tld_sensitivity.png'))
    plt.close()

def experiment_weight_grid(df, output_dir, steps=11):
    base_scores = combined_score_from_components(df)
    results = []
    # generate weight triples on simplex
    ws = np.linspace(0.0, 1.0, steps)
    for w_url in ws:
        for w_text in ws:
            w_pop = 1.0 - w_url - w_text
            if w_pop < 0 or w_pop > 1:
                continue
            combined = combined_score_from_components(df, w_url=w_url, w_text=w_text, w_pop=w_pop)
            rho = spearman_rank_corr(base_scores.values, combined.values)
            mad = np.mean(np.abs(base_scores.values - combined.values))
            results.append({'w_url': w_url, 'w_text': w_text, 'w_pop': w_pop, 'spearman': rho, 'mean_abs_delta': mad})
    out = pd.DataFrame(results)
    out.to_csv(os.path.join(output_dir, 'weight_grid_results.csv'), index=False)
    # heatmap for spearman
    pivot = out.pivot_table(index='w_url', columns='w_text', values='spearman')
    plt.figure(figsize=(8,6))
    plt.imshow(pivot.values, origin='lower', aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Spearman')
    plt.title('Spearman across weight grid (w_url index, w_text columns)')
    plt.xlabel('w_text')
    plt.ylabel('w_url')
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=[f"{v:.2f}" for v in pivot.columns], rotation=45)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=[f"{v:.2f}" for v in pivot.index])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_grid_spearman.png'))
    plt.close()

def experiment_ablation(df, output_dir):
    base_scores = combined_score_from_components(df)
    variants = {
        'no_url': (0.0, 0.5, 0.5),
        'no_text': (0.7, 0.0, 0.3),
        'no_pop': (0.6, 0.4, 0.0),
        'url_only': (1.0, 0.0, 0.0),
        'text_only': (0.0, 1.0, 0.0),
        'pop_only': (0.0, 0.0, 1.0),
    }
    rows = []
    for name, (wu, wt, wp) in variants.items():
        combined = combined_score_from_components(df, w_url=wu, w_text=wt, w_pop=wp)
        rho = spearman_rank_corr(base_scores.values, combined.values)
        mad = np.mean(np.abs(base_scores.values - combined.values))
        rows.append({'variant': name, 'w_url': wu, 'w_text': wt, 'w_pop': wp, 'spearman': rho, 'mean_abs_delta': mad})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'ablation_results.csv'), index=False)

def experiment_text_ad_sensitivity(df, output_dir):
    # scale ad penalties by factor and scale clickbait penalty
    base_scores = combined_score_from_components(df)
    ad_scales = [0.5, 1.0, 1.5, 2.0]
    click_scales = [0.5, 1.0, 1.5, 2.0]
    rows = []
    orig_eval = d1.evaluate_text_credibility
    for a_s in ad_scales:
        for c_s in click_scales:
            # monkeypatch evaluate_text_credibility with scaled penalties
            def eval_scaled(text, ad_count, a_s=a_s, c_s=c_s):
                # replicate original logic but scale ad penalties and clickbait penalty
                if not text:
                    return 0.2
                score = 0.5
                wc = len(text.split())
                if wc > 500:
                    score += 0.2
                elif wc < 50:
                    score -= 0.2
                if any(tok.lower() in text.lower() for tok in ["buy now", "click here", "free"]):
                    score -= 0.3 * c_s
                if sum(1 for w in text.split() if w.isupper()) > 20:
                    score -= 0.2
                if ad_count > 10:
                    score -= 0.3 * a_s
                elif ad_count > 3:
                    score -= 0.15 * a_s
                elif ad_count > 0:
                    score -= 0.05 * a_s
                return float(max(0.0, min(1.0, score)))
            d1.evaluate_text_credibility = eval_scaled
            # recompute text_score for dataset using stored ad_count if available
            text_scores = []
            for u in df.index:
                # try storage-based ad_count if in storage
                # fallback: use existing df ad_count
                adc = df.loc[u, 'ad_count'] if 'ad_count' in df.columns else 0
                # we don't have raw text for all URLs; use saved text_score as proxy
                # we reconstruct approximate text_score by calling eval_scaled on placeholder text:
                # if previous text_score <= 0.25 assume short or empty, else assume long text
                prev_ts = df.loc[u, 'text_score']
                if prev_ts <= 0.25:
                    text = ""
                elif prev_ts >= 0.7:
                    text = "word " * 600
                else:
                    text = "word " * 200
                text_scores.append(eval_scaled(text, int(adc)))
            combined = 0.5 * df['url_score'].values + 0.3 * np.array(text_scores) + 0.2 * df['pop_score'].values
            rho = spearman_rank_corr(base_scores.values, combined)
            mad = np.mean(np.abs(base_scores.values - combined))
            rows.append({'ad_scale': a_s, 'click_scale': c_s, 'spearman': rho, 'mean_abs_delta': mad})
    # restore
    d1.evaluate_text_credibility = orig_eval
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'text_ad_sensitivity.csv'), index=False)

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description='Sensitivity analysis for deliverable1 heuristics')
    parser.add_argument('--csv', default='urls_from_tranco.csv', help='CSV with column "url"')
    parser.add_argument('--output', default='results', help='output directory')
    parser.add_argument('--max', type=int, default=200, help='max urls to analyze')
    parser.add_argument('--fetch', action='store_true', help='fetch pages for missing text scores (slower)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    urls = load_urls(args.csv, max_urls=args.max)
    storage = d1.load_storage()
    df = baseline_components(urls, storage, fetch_missing=args.fetch)
    # ensure ad_count column exists
    if 'ad_count' not in df.columns:
        df['ad_count'] = 0
    # save baseline table
    df.to_csv(os.path.join(args.output, 'baseline_components.csv'))
    # run experiments
    experiment_scale_domain_weights(df, storage, args.output)
    experiment_per_tld_perturb(df, storage, args.output)
    experiment_weight_grid(df, args.output)
    experiment_ablation(df, args.output)
    experiment_text_ad_sensitivity(df, args.output)
    print(f'Results saved to {args.output}')

if __name__ == '__main__':
    main()