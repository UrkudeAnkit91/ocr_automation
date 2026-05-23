"""Push repo to GitHub via API (bypasses git's libcurl issue)."""

import base64
import json
import os
import requests
from pathlib import Path

TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # replace with your token"
OWNER = "UrkudeAnkit91"
REPO = "ocr_automation"
BASE = "https://api.github.com"
ROOT = Path(__file__).resolve().parent

headers = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github.v3+json"}

GITIGNORE_PATTERNS = [
    "__pycache__", "*.pyc", ".env", "*.log",
    "*.db", "*.db-shm", "*.db-wal",
    "node_modules", "dist", ".angular", "_old",
    ".git", ".gitignore",
]

def should_ignore(p: Path) -> bool:
    for pat in GITIGNORE_PATTERNS:
        if pat.startswith("*") and p.name.endswith(pat[1:]):
            return True
        if pat == p.name or pat in p.parts:
            return True
    return False

def get_all_files():
    files = []
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and not should_ignore(p):
            rel = p.relative_to(ROOT).as_posix()
            files.append((rel, p))
    return files

# 1. Create blobs
files = get_all_files()
print(f"Pushing {len(files)} files...")

TEXT_EXT = {".py",".ts",".html",".css",".js",".json",".md",".yml",".yaml",".conf",".txt",".ps1",".gitignore",".editorconfig",".prettierrc"}

blobs = []
for rel, path in files:
    is_text = path.suffix in TEXT_EXT or path.suffix == "" or path.name == ".gitignore"
    with open(path, "rb") as f:
        raw = f.read()
    if is_text:
        # Decode as utf-8, ignore errors
        content = raw.decode("utf-8", errors="replace")
        encoding = "utf-8"
    else:
        content = base64.b64encode(raw).decode("ascii")
        encoding = "base64"
    r = requests.post(f"{BASE}/repos/{OWNER}/{REPO}/git/blobs",
                      headers=headers,
                      json={"content": content, "encoding": encoding})
    if r.status_code not in (200, 201):
        print(f"  Blob fail {rel}: {r.status_code} {r.text[:100]}")
        continue
    blobs.append({"path": rel, "mode": "100644", "type": "blob", "sha": r.json()["sha"]})
    print(f"  blob {rel}")

# 2. Create tree
r = requests.post(f"{BASE}/repos/{OWNER}/{REPO}/git/trees",
                  headers=headers,
                  json={"tree": blobs})
if r.status_code not in (200, 201):
    print(f"Tree fail: {r.status_code} {r.text[:200]}")
    exit(1)
tree_sha = r.json()["sha"]
print(f"Tree: {tree_sha}")

# 3. Create commit
r = requests.post(f"{BASE}/repos/{OWNER}/{REPO}/git/commits",
                  headers=headers,
                  json={
                      "message": "Initial commit: OCR Automation with FastAPI backend + Angular dashboard",
                      "tree": tree_sha,
                      "parents": [],
                  })
if r.status_code not in (200, 201):
    print(f"Commit fail: {r.status_code} {r.text[:200]}")
    exit(1)
commit_sha = r.json()["sha"]
print(f"Commit: {commit_sha}")

# 4. Update ref
r = requests.patch(f"{BASE}/repos/{OWNER}/{REPO}/git/refs/heads/master",
                   headers=headers,
                   json={"sha": commit_sha, "force": True})
if r.status_code not in (200, 201):
    # Try creating the ref instead
    r = requests.post(f"{BASE}/repos/{OWNER}/{REPO}/git/refs",
                      headers=headers,
                      json={"ref": "refs/heads/master", "sha": commit_sha})
if r.status_code in (200, 201):
    print(f"Pushed! https://github.com/{OWNER}/{REPO}")
else:
    print(f"Ref fail: {r.status_code} {r.text[:200]}")
