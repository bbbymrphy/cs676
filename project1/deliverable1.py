from __future__ import annotations
from urllib.parse import urlsplit, parse_qs, unquote
import ipaddress
import math
import re
from typing import Dict, Any

TRUSTED_TLDS = {
    "edu", "gov", "mil"
}
COMMON_TLDS = {
    "com", "org", "net", "io", "co", "ai", "app", "dev"
}
# ⚠️ Heuristics only; adjust for your use case to avoid unfair bias.
HIGH_ABUSE_TLDS = {
    "xyz", "top", "click", "work", "review", "country", "gq", "ml", "cf",
    "tk", "fit", "loan", "men", "date", "stream", "download", "racing",
    "bid", "party", "win", "link"
}

COMMON_SUBDOMAINS = {"www", "m", "blog", "shop", "store", "docs", "support"}
TRACKING_PARAM_PREFIXES = ("utm_",)
TRACKING_PARAMS = {"gclid", "fbclid", "mc_eid"}

LONG_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]{20,}")
PCT_ENCODE_RE = re.compile(r"%[0-9A-Fa-f]{2}")

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    counts = Counter(s)
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in counts.values())

def is_ip(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
        return True
    except Exception:
        return False

def tld_bucket(hostname: str) -> str:
    if not hostname or "." not in hostname:
        return "unknown"
    labels = hostname.lower().split(".")
    tld = labels[-1]
    sld = ".".join(labels[-2:]) if len(labels) >= 2 else tld
    # handle a couple common 2-level public suffixes if you want
    if tld in TRUSTED_TLDS or sld.endswith(".gov"):
        return "trusted"
    if tld in COMMON_TLDS:
        return "common"
    if tld in HIGH_ABUSE_TLDS:
        return "high_abuse"
    return "other"

def looks_random(token: str) -> bool:
    if len(token) < 10:
        return False
    ent = shannon_entropy(token)
    digits = sum(ch.isdigit() for ch in token) / len(token)
    # high entropy and lots of digits often



