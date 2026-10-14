"""
Fix the missing pip install line in README.md.
"""
from pathlib import Path

p = Path("README.md")
t = p.read_text(encoding="utf-8")
t = t.replace(
    "Python 3.10+. Install:\n\n\n## Reproduce",
    "Python 3.10+. Install:\n\n```\npip install -r requirements.txt\n```\n\n## Reproduce"
)
p.write_text(t, encoding="utf-8")
print("Fixed.")
