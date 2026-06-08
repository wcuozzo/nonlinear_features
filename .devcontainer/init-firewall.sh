#!/bin/bash
set -euo pipefail

# ============================================================
# Network firewall for Claude Code devcontainer
# Based on Anthropic's official reference, extended with
# PyPI and HuggingFace for mech interp research workflows.
# ============================================================

# Preserve Docker's internal DNS rules before flushing
DOCKER_DNS_RULES=$(iptables-save 2>/dev/null | grep "127.0.0.11" || true)

# Flush existing rules
iptables -F
iptables -X
iptables -t nat -F
iptables -t nat -X

# Restore Docker DNS rules (required for container DNS resolution)
if [ -n "$DOCKER_DNS_RULES" ]; then
    echo "$DOCKER_DNS_RULES" | while IFS= read -r rule; do
        if [[ "$rule" == -A* ]]; then
            iptables -t nat ${rule} 2>/dev/null || true
        fi
    done
fi

# Allow DNS (needed before we can resolve anything)
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

# Allow SSH (for git operations)
iptables -A OUTPUT -p tcp --dport 22 -j ACCEPT

# Create ipset for allowed domains
ipset destroy allowed-domains 2>/dev/null || true
ipset create allowed-domains hash:ip

# ---- GitHub dynamic IPs ----
echo "Fetching GitHub IP ranges..."
GITHUB_META=$(curl -s https://api.github.com/meta 2>/dev/null || echo "{}")
for key in web git api; do
    echo "$GITHUB_META" | jq -r ".${key}[]? // empty" 2>/dev/null | while read -r cidr; do
        if [[ "$cidr" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(/[0-9]+)?$ ]]; then
            ipset add allowed-domains "$cidr" 2>/dev/null || true
        fi
    done
done

# ---- Resolve and whitelist domains ----
ALLOWED_DOMAINS=(
    # Anthropic (Claude Code API)
    "api.anthropic.com"
    "claude.ai"
    "statsig.anthropic.com"
    "statsig.com"
    "sentry.io"

    # npm registry (Claude Code updates)
    "registry.npmjs.org"

    # GitHub (git operations)
    "github.com"
    "api.github.com"

    # PyPI (pip install for TransformerLens, SAELens, torch, etc.)
    "pypi.org"
    "files.pythonhosted.org"

    # HuggingFace (model downloads)
    "huggingface.co"
    "cdn-lfs.huggingface.co"
    "cdn-lfs-us-1.huggingface.co"

    # VS Code extensions marketplace
    "marketplace.visualstudio.com"
    "vscode.blob.core.windows.net"
    "update.code.visualstudio.com"
    "*.gallery.vsassets.io"
)

echo "Resolving whitelisted domains..."
for domain in "${ALLOWED_DOMAINS[@]}"; do
    # Skip wildcard entries (handled by the actual subdomain resolution)
    if [[ "$domain" == \** ]]; then
        continue
    fi

    ips=$(dig +short +noall +answer A "$domain" 2>/dev/null | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' || true)
    if [ -z "$ips" ]; then
        echo "  WARN: No IPs resolved for $domain"
        continue
    fi
    for ip in $ips; do
        ipset add allowed-domains "$ip" 2>/dev/null || true
    done
    echo "  OK: $domain"
done

# Also resolve gallery.vsassets.io (covers *.gallery.vsassets.io)
for subdomain in "anthropic.gallery.vsassets.io" "ms-python.gallery.vsassets.io" "ms-toolsai.gallery.vsassets.io"; do
    ips=$(dig +short +noall +answer A "$subdomain" 2>/dev/null | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' || true)
    for ip in ${ips:-}; do
        ipset add allowed-domains "$ip" 2>/dev/null || true
    done
done

# ---- Apply firewall rules ----

# Default policies: DROP everything
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP

# Allow loopback (localhost)
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established/related connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow DNS
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

# Allow SSH (for git)
iptables -A OUTPUT -p tcp --dport 22 -j ACCEPT

# Allow whitelisted IPs (HTTP/HTTPS)
iptables -A OUTPUT -m set --match-set allowed-domains dst -p tcp --dport 80 -j ACCEPT
iptables -A OUTPUT -m set --match-set allowed-domains dst -p tcp --dport 443 -j ACCEPT

# Reject everything else with immediate feedback (not silent DROP)
iptables -A OUTPUT -j REJECT --reject-with icmp-port-unreachable

# ---- Verify ----
echo ""
echo "=== Firewall verification ==="

# Should FAIL (blocked)
if curl -sf --max-time 3 https://example.com > /dev/null 2>&1; then
    echo "FAIL: example.com should be blocked but is reachable!"
    exit 1
else
    echo "OK: example.com correctly blocked"
fi

# Should SUCCEED (allowed)
if curl -sf --max-time 5 https://api.github.com > /dev/null 2>&1; then
    echo "OK: api.github.com correctly reachable"
else
    echo "WARN: api.github.com not reachable (may be transient)"
fi

echo ""
echo "Firewall initialized. Allowed: Anthropic API, GitHub, PyPI, HuggingFace, npm."
echo "Everything else is blocked."
