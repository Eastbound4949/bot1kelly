#!/bin/bash
# setup.sh — Run this ONCE to set up everything locally.
# Usage: bash setup.sh
#
# What it does:
# 1. Creates the full folder structure
# 2. Installs all Python dependencies
# 3. Checks for git and helps you set up GitHub
# 4. Guides you through getting your API keys

set -e  # Stop on any error

GREEN='\033[0;32m'
AMBER='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "════════════════════════════════════════════"
echo " Crypto ML Trading Bot — Full Setup"
echo "════════════════════════════════════════════"
echo ""

# ── Step 1: Check Python version ──────────────────────────────
echo -e "${BLUE}[1/6] Checking Python...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
major=$(echo $python_version | cut -d. -f1)
minor=$(echo $python_version | cut -d. -f2)

if [ "$major" -lt 3 ] || [ "$minor" -lt 9 ]; then
    echo "Python 3.9+ required. You have $python_version"
    echo "Download from: https://python.org"
    exit 1
fi
echo "  Python $python_version — OK"

# ── Step 2: Create folder structure ───────────────────────────
echo -e "${BLUE}[2/6] Creating project structure...${NC}"
mkdir -p tests
mkdir -p .github/workflows

# Create __init__.py for tests
touch tests/__init__.py

echo "  Folders created."

# ── Step 3: Install dependencies ──────────────────────────────
echo -e "${BLUE}[3/6] Installing Python packages...${NC}"
pip3 install -r requirements.txt --quiet
echo "  All packages installed."

# ── Step 4: Verify imports work ───────────────────────────────
echo -e "${BLUE}[4/6] Verifying imports...${NC}"
python3 -c "
import pandas, numpy, sklearn, ta, apscheduler, binance, requests
print('  All imports OK.')
"

# ── Step 5: Run tests ─────────────────────────────────────────
echo -e "${BLUE}[5/6] Running tests...${NC}"
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20
echo ""

# ── Step 6: Git setup ─────────────────────────────────────────
echo -e "${BLUE}[6/6] Git setup...${NC}"

if ! command -v git &> /dev/null; then
    echo "  Git not found. Install from: https://git-scm.com"
else
    if [ ! -d ".git" ]; then
        git init
        git add .
        git commit -m "Initial commit: crypto ML trading bot"
        echo "  Git repository initialised."
        echo ""
        echo -e "${AMBER}  Next: create a private GitHub repo and push:${NC}"
        echo "    1. Go to github.com → New repository"
        echo "    2. Name it: crypto-trading-bot (private)"
        echo "    3. Run these commands:"
        echo "       git remote add origin https://github.com/YOUR_USERNAME/crypto-trading-bot.git"
        echo "       git push -u origin main"
    else
        echo "  Git already initialised."
    fi
fi

# ── Summary ───────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo -e "${GREEN} Setup complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo ""
echo "Your next steps:"
echo ""
echo "  1. Fill in your API keys:"
echo "     nano config.py"
echo "     (Binance read-only key + Telegram bot token)"
echo ""
echo "  2. Train the model:"
echo "     python3 model_trainer.py"
echo ""
echo "  3. Test locally:"
echo "     python3 bot.py"
echo "     (Watch for Telegram messages — you should get one within seconds)"
echo ""
echo "  4. Push to GitHub → Railway auto-deploys:"
echo "     git add . && git commit -m 'update' && git push"
echo ""
echo "  5. Check performance anytime:"
echo "     python3 performance_monitor.py"
echo ""
echo "  Weekly: you'll get a Telegram report every Sunday."
echo "  When all targets are met, you'll get a promotion alert."
echo ""
