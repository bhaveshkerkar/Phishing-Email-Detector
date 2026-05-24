"""
Feature Extractor
Takes raw email input and extracts all features the ML model needs.
"""

import re


# --- Known URL shortener domains ---
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "buff.ly", "rebrand.ly", "short.io", "tiny.cc", "is.gd",
}

# --- Urgent/threatening words ---
URGENT_WORDS = [
    "urgent", "immediately", "action required", "verify now",
    "account suspended", "locked", "terminated", "expired",
    "click here", "confirm your", "update your", "limited time",
    "warning", "alert", "final notice", "last chance",
]

# --- Prize/scam words ---
PRIZE_WORDS = [
    "winner", "won", "prize", "congratulations", "lucky",
    "free gift", "reward", "selected", "claim now", "exclusive offer",
]

# --- Threat words ---
THREAT_WORDS = [
    "suspended", "locked", "arrest", "legal action", "warrant",
    "penalty", "immediately", "urgent", "terminated", "blocked",
]

# --- Financial bait words ---
FINANCIAL_WORDS = [
    "bank", "account", "payment", "billing", "paypal",
    "tax", "refund", "invoice", "wire transfer", "credit card",
]


def extract_urls(text):
    """Extract all URLs from a block of text."""
    pattern = r'https?://[^\s\'"<>]+'
    return re.findall(pattern, text)


def is_ip_url(url):
    """Check if a URL uses a raw IP address instead of a domain name."""
    pattern = r'https?://(\d{1,3}\.){3}\d{1,3}'
    return bool(re.match(pattern, url))


def is_url_shortener(url):
    """Check if URL uses a known shortener service."""
    domain = re.sub(r'https?://', '', url).split('/')[0].lower()
    return domain in SHORTENER_DOMAINS


def has_domain_mismatch(sender, body):
    """
    Check if the sender domain doesn't appear in the email body links.
    Example: sender is paypal.com but link goes to paypa1-xyz.com
    """
    if not sender or "@" not in sender:
        return False
    sender_domain = sender.split("@")[-1].lower()
    urls = extract_urls(body)
    if not urls:
        return False
    for url in urls:
        link_domain = re.sub(r'https?://', '', url).split('/')[0].lower()
        if sender_domain not in link_domain:
            return True
    return False


def count_word_matches(text, word_list):
    """Count how many words/phrases from a list appear in the text."""
    text_lower = text.lower()
    return sum(1 for word in word_list if word in text_lower)


def extract_features(subject: str, sender: str, body: str, url: str = "") -> dict:
    """
    Main function — given raw email fields, returns a feature dict
    that matches what the ML model was trained on.
    """
    full_text = f"{subject} {body}"
    all_urls  = extract_urls(body)
    if url:
        all_urls.append(url)

    features = {
        # Length-based
        "subject_length": len(subject),
        "text_length":    len(body),

        # Punctuation / style signals
        "num_exclamations": full_text.count("!"),
        "num_caps_words":   sum(1 for w in full_text.split() if w.isupper() and len(w) > 2),

        # Keyword signals
        "num_urgent_words":    count_word_matches(full_text, URGENT_WORDS),
        "has_prize_words":     int(count_word_matches(full_text, PRIZE_WORDS) > 0),
        "has_threat_words":    int(count_word_matches(full_text, THREAT_WORDS) > 0),
        "has_financial_words": int(count_word_matches(full_text, FINANCIAL_WORDS) > 0),

        # URL signals
        "has_url":       int(bool(all_urls)),
        "num_links":     len(all_urls),
        "has_ip_url":    int(any(is_ip_url(u) for u in all_urls)),
        "url_shortener": int(any(is_url_shortener(u) for u in all_urls)),

        # Sender signals
        "sender_domain_mismatch": int(has_domain_mismatch(sender, body)),

        # Misc
        "has_attachment":   0,   # set manually if needed
        "html_obfuscation": int("&#" in body or "%2" in body or "&#x" in body),
    }

    return features


def get_red_flags(subject: str, sender: str, body: str, url: str = "") -> list:
    """
    Returns a human-readable list of red flags found in the email.
    Used to show the user WHY something was flagged.
    """
    flags = []
    full_text = f"{subject} {body}"
    all_urls  = extract_urls(body)
    if url:
        all_urls.append(url)

    if full_text.count("!") >= 2:
        flags.append("Excessive exclamation marks detected")

    caps = sum(1 for w in full_text.split() if w.isupper() and len(w) > 2)
    if caps >= 3:
        flags.append(f"{caps} ALL-CAPS words found — common pressure tactic")

    urgent_hits = [w for w in URGENT_WORDS if w in full_text.lower()]
    if urgent_hits:
        flags.append(f"Urgency/threat language: {', '.join(urgent_hits[:3])}")

    prize_hits = [w for w in PRIZE_WORDS if w in full_text.lower()]
    if prize_hits:
        flags.append(f"Prize/scam bait words: {', '.join(prize_hits[:3])}")

    if any(is_ip_url(u) for u in all_urls):
        flags.append("URL uses raw IP address instead of a domain name")

    if any(is_url_shortener(u) for u in all_urls):
        flags.append("URL uses a shortener service to hide destination")

    if has_domain_mismatch(sender, body):
        flags.append("Sender domain does not match links in email body")

    if "&#" in body or "%2" in body:
        flags.append("HTML character obfuscation detected in body")

    for u in all_urls:
        if not u.startswith("https"):
            flags.append(f"Insecure HTTP link found: {u[:60]}")
            break

    return flags