"""
Phishing Email Dataset Generator
Generates a labeled synthetic dataset for training/testing the phishing detector.
Labels: 0 = Legitimate, 1 = Phishing
"""

import csv
import random

random.seed(42)

LEGIT_SUBJECTS = [
    "Your monthly account statement is ready",
    "Meeting notes from today's standup",
    "Welcome to our newsletter",
    "Your order has been shipped",
    "Project update – Q2 milestones",
    "Invoice #4821 attached",
    "Your subscription has been renewed",
    "Weekly digest: Top stories in tech",
    "GitHub: Pull request review requested",
    "Your password was changed successfully",
]

LEGIT_SENDERS = [
    "noreply@github.com",
    "billing@stripe.com",
    "hello@notion.so",
    "info@amazon.in",
    "no-reply@google.com",
    "hr@company.com",
    "support@shopify.com",
    "newsletter@medium.com",
    "notifications@slack.com",
]

LEGIT_BODIES = [
    "Hi there,\n\nYour monthly statement is now available in your account dashboard. Please log in to view it.\n\nBest regards,\nThe Billing Team",
    "Hey team,\n\nHere are the notes from today's standup. We discussed the new feature rollout and sprint planning.\n\nCheers,\nMike",
    "Dear Customer,\n\nYour order #ORD-982341 has been shipped and is expected to arrive by Thursday.\n\nThanks,\nAmazon",
    "Hi,\n\nA pull request has been submitted for your review on the 'feature/auth-improvements' branch.\n\nGitHub Notifications",
    "Dear valued customer,\n\nYour annual subscription has been successfully renewed. No action is required.\n\nThank you,\nSupport Team",
]

LEGIT_URLS = [
    "https://github.com/notifications",
    "https://stripe.com/billing",
    "https://notion.so/dashboard",
    "https://amazon.in/orders",
    "https://accounts.google.com",
    "",
    "",
]

PHISH_SUBJECTS = [
    "URGENT: Your account has been compromised!",
    "Action Required: Verify your identity NOW",
    "You've won a $1000 Amazon Gift Card!!!",
    "Your PayPal account is suspended",
    "Security Alert: Unusual login detected",
    "Claim your prize before it expires",
    "ALERT: Suspicious activity on your account",
    "Immediate action required – account locked",
    "Your Netflix subscription has been declined",
    "IRS: You owe back taxes – respond immediately",
]

PHISH_SENDERS = [
    "security@paypa1-support.com",
    "noreply@amazon-security-alert.net",
    "admin@bank0fsupport.xyz",
    "verify@netflix-billing-update.com",
    "support@apple-id-verify.ru",
    "alerts@irs-taxrefund.info",
    "no-reply@faceb00k-security.com",
    "prize@lucky-winner-claim.biz",
]

PHISH_BODIES = [
    "DEAR USER,\n\nYour account has been SUSPENDED due to suspicious activity!! Click the link below IMMEDIATELY to verify your identity or your account will be PERMANENTLY DELETED within 24 hours.\n\nClick here: http://paypa1-secure-login.xyz/verify?token=abc123",
    "Congratulations!!! You have been selected as our LUCKY WINNER of $1,000,000 USD! To claim your prize, provide your full name, address, and bank details.\n\nClaim now: http://prize-winner-claim.biz/claim?id=999",
    "We have noticed UNUSUAL SIGN-IN ACTIVITY on your account. Your account will be locked in 2 hours unless you verify immediately.\n\nVerify here: http://google-security-alert.net/login",
    "IRS NOTICE: You have an overdue tax payment of $3,400. Failure to pay within 48 hours will result in legal action.\n\nPay securely: http://irs-taxrefund.info/pay",
    "Your Apple ID has been used to sign in from Russia. If this was not you, CLICK HERE: http://apple-id-verify.ru/secure?session=abc",
]

PHISH_URLS = [
    "http://paypa1-secure-login.xyz/verify?token=abc123",
    "http://prize-winner-claim.biz/claim?id=999",
    "http://google-security-alert.net/login",
    "http://irs-taxrefund.info/pay",
    "http://apple-id-verify.ru/secure?session=abc",
    "http://bank0fsupport.xyz/verify",
    "http://amazon-security-alert.net/account/suspend?token=xyz987",
]


def generate_sample(label):
    if label == 0:
        subject = random.choice(LEGIT_SUBJECTS)
        sender = random.choice(LEGIT_SENDERS)
        body = random.choice(LEGIT_BODIES)
        url = random.choice(LEGIT_URLS)
        num_exclamations = random.randint(0, 1)
        num_urgent_words = 0
        sender_domain_mismatch = 0
        has_ip_url = 0
        url_shortener = 0
        html_obfuscation = 0
        num_links = random.randint(0, 2)
        has_attachment = random.choice([0, 0, 0, 1])
    else:
        subject = random.choice(PHISH_SUBJECTS)
        sender = random.choice(PHISH_SENDERS)
        body = random.choice(PHISH_BODIES)
        url = random.choice(PHISH_URLS)
        num_exclamations = random.randint(2, 8)
        num_urgent_words = random.randint(1, 5)
        sender_domain_mismatch = random.choice([0, 1, 1])
        has_ip_url = random.choice([0, 0, 1])
        url_shortener = random.choice([0, 1])
        html_obfuscation = random.choice([0, 1])
        num_links = random.randint(1, 5)
        has_attachment = random.choice([0, 1])

    full_text = subject + " " + body
    num_caps_words = sum(1 for w in full_text.split() if w.isupper() and len(w) > 2)

    return {
        "subject": subject,
        "sender": sender,
        "body": body[:300],
        "url": url,
        "label": label,
        "subject_length": len(subject),
        "text_length": len(body),
        "num_exclamations": num_exclamations,
        "num_caps_words": num_caps_words,
        "num_urgent_words": num_urgent_words,
        "has_attachment": has_attachment,
        "has_url": int(bool(url)),
        "url_shortener": url_shortener,
        "has_ip_url": has_ip_url,
        "sender_domain_mismatch": sender_domain_mismatch,
        "html_obfuscation": html_obfuscation,
        "num_links": num_links,
        "has_prize_words": int(any(w in full_text.lower() for w in ["prize", "winner", "congratulations", "free", "gift", "lucky"])),
        "has_threat_words": int(any(w in full_text.lower() for w in ["suspended", "locked", "arrest", "legal", "immediately", "urgent", "warning"])),
        "has_financial_words": int(any(w in full_text.lower() for w in ["bank", "account", "payment", "billing", "paypal", "tax", "refund"])),
    }


def generate_dataset(n=2000, output_path="phishing_dataset.csv"):
    samples = [generate_sample(0) for _ in range(n // 2)]
    samples += [generate_sample(1) for _ in range(n // 2)]
    random.shuffle(samples)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)

    print(f"✅ Dataset generated: {output_path} ({n} samples)")


if __name__ == "__main__":
    generate_dataset(2000, "phishing_dataset.csv")