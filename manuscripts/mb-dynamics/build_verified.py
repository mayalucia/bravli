#!/usr/bin/env python3
"""Build a verified manuscript: paper.org + test results → verified.html.

Reads the org-mode manuscript, runs the test suite, maps claims to test
outcomes, and produces a self-contained HTML file with inline verification
badges.

Usage:
    python build_verified.py              # run tests + build HTML
    python build_verified.py --skip-tests # reuse last test results
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent.parent  # bravli repo root
TESTS_DIR = REPO_ROOT / "tests"
PAPER_ORG = SCRIPT_DIR / "paper.org"
CLAIMS_JSON = SCRIPT_DIR / "claims.json"
OUTPUT_HTML = SCRIPT_DIR / "verified.html"
TEST_CACHE = SCRIPT_DIR / ".test_results.json"


# ---------------------------------------------------------------------------
# 1. Run pytest and parse results
# ---------------------------------------------------------------------------

def run_tests():
    """Run pytest -v and parse structured output into a dict."""
    result = subprocess.run(
        ["python3", "-m", "pytest", str(TESTS_DIR), "-v", "--tb=no"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    tests = {}
    for line in result.stdout.splitlines():
        # Match lines like:
        #   tests/test_brunel.py::TestBrunel::test_build_network PASSED
        #   tests/test_brunel.py::test_build_network PASSED
        m = re.match(r"tests/(\S+::\S+)\s+(PASSED|FAILED|SKIPPED|ERROR)", line)
        if m:
            # Normalise: strip class name to get file::func
            raw_id = m.group(1)
            status = m.group(2).lower()
            # Store both the full id and a short form (file::func)
            tests[raw_id] = {"status": status}
            # Also store short form for claim matching
            parts = raw_id.split("::")
            if len(parts) == 3:
                short_id = f"{parts[0]}::{parts[2]}"
                tests[short_id] = {"status": status}
    summary = {}
    m = re.search(r"(\d+) passed", result.stdout)
    summary["passed"] = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) failed", result.stdout)
    summary["failed"] = int(m.group(1)) if m else 0
    m = re.search(r"(\d+) skipped", result.stdout)
    summary["skipped"] = int(m.group(1)) if m else 0
    m = re.search(r"in ([\d.]+)s", result.stdout)
    summary["duration"] = m.group(1) if m else "?"
    return {"tests": tests, "summary": summary}


def load_or_run_tests(skip_tests=False):
    if skip_tests and TEST_CACHE.exists():
        with open(TEST_CACHE) as f:
            return json.load(f)
    results = run_tests()
    with open(TEST_CACHE, "w") as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# 2. Parse org file
# ---------------------------------------------------------------------------

def parse_org(path):
    """Parse paper.org into a structured document.

    Returns a list of sections, each with:
    - level (int): heading depth (1=*, 2=**, etc.)
    - title (str): heading text
    - body (list[str]): paragraph strings between this heading and the next
    - id (str): slugified heading for anchors
    """
    lines = path.read_text().splitlines()
    sections = []
    metadata = {}
    current = None
    body_lines = []

    def flush():
        nonlocal current, body_lines
        if current is not None:
            current["body"] = split_paragraphs(body_lines)
            sections.append(current)
        body_lines = []

    for line in lines:
        # Metadata
        m = re.match(r"#\+TITLE:\s*(.*)", line)
        if m:
            metadata["title"] = m.group(1).strip()
            continue
        m = re.match(r"#\+AUTHOR:\s*(.*)", line)
        if m:
            metadata["author"] = m.group(1).strip()
            continue
        m = re.match(r"#\+DATE:\s*(.*)", line)
        if m:
            metadata["date"] = m.group(1).strip()
            continue

        # Skip org directives
        if line.startswith("#+"):
            continue

        # Headings
        m = re.match(r"^(\*+)\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            current = {"level": level, "title": title, "id": slug, "body": []}
            continue

        # Abstract block
        if line.strip() in ("#+BEGIN_abstract", "#+END_abstract"):
            continue

        # Body text
        body_lines.append(line)

    flush()
    return metadata, sections


def split_paragraphs(lines):
    """Split lines into paragraphs (separated by blank lines)."""
    paragraphs = []
    current = []
    for line in lines:
        if line.strip() == "":
            if current:
                paragraphs.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append("\n".join(current))
    return paragraphs


# ---------------------------------------------------------------------------
# 3. Render org text to HTML
# ---------------------------------------------------------------------------

def org_to_html(text):
    """Convert org-mode markup to HTML."""
    # LaTeX display equations
    text = re.sub(
        r"\\begin\{equation\}(.*?)\\end\{equation\}",
        lambda m: f'<div class="equation"><code class="latex">{escape(m.group(1).strip())}</code></div>',
        text, flags=re.DOTALL,
    )
    text = re.sub(
        r"\\begin\{align\}(.*?)\\end\{align\}",
        lambda m: f'<div class="equation"><code class="latex">{escape(m.group(1).strip())}</code></div>',
        text, flags=re.DOTALL,
    )

    # Org tables → HTML tables
    text = convert_tables(text)

    # Inline LaTeX: $...$ → <code class="math">
    text = re.sub(
        r"(?<!\$)\$([^$\n]+?)\$(?!\$)",
        lambda m: f'<code class="math">{latex_to_unicode(m.group(1))}</code>',
        text,
    )

    # Citations: \citep{...}, \citet{...}, \citealt{...}
    text = re.sub(r"\\citep\{([^}]+)\}", lambda m: render_citations(m.group(1)), text)
    text = re.sub(r"\\citet\{([^}]+)\}", lambda m: render_citations(m.group(1)), text)
    text = re.sub(r"\\citealt\{([^}]+)\}", lambda m: render_citations(m.group(1)), text)

    # Org emphasis: /italic/, *bold*, =code=
    text = re.sub(r"(?<!\w)/([^/\n]+?)/(?!\w)", r"<em>\1</em>", text)
    text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"<strong>\1</strong>", text)
    text = re.sub(r"=([^=\n]+?)=", r"<code>\1</code>", text)

    # Org list items
    text = re.sub(r"^- \*(.*?)\*:(.*)", r"<li><strong>\1</strong>:\2</li>", text, flags=re.MULTILINE)
    text = re.sub(r"^- (.*)", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"^(\d+)\. (.*)", r"<li>\2</li>", text, flags=re.MULTILINE)

    # Wrap consecutive <li> in <ul>
    text = re.sub(
        r"((?:<li>.*?</li>\n?)+)",
        lambda m: f"<ul>{m.group(0)}</ul>",
        text,
    )

    # LaTeX commands that slipped through
    text = text.replace(r"\to{}", "→")
    text = text.replace(r"\to", "→")
    text = text.replace(r"\clearpage", "")
    text = text.replace(r"\tableofcontents", "")
    text = re.sub(r"\\bibliography\{[^}]+\}", "", text)
    text = re.sub(r"#\+ATTR_LATEX:.*\n?", "", text)

    # Em dash
    text = text.replace("---", "&mdash;")
    text = text.replace("--", "&ndash;")

    return text


def escape(text):
    """HTML-escape."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def latex_to_unicode(latex):
    """Convert common LaTeX math to unicode approximations."""
    subs = {
        r"\tau_m": "τ_m", r"\tau_s": "τ_s", r"\tau_w": "τ_w",
        r"\tau_{\text{ref}}": "τ_ref", r"\tau_{\text{syn}}": "τ_syn",
        r"\sigma": "σ", r"\eta": "η", r"\Delta_T": "Δ_T",
        r"\epsilon": "ε", r"\delta": "δ", r"\approx": "≈",
        r"\sim": "~", r"\times": "×", r"\leq": "≤", r"\geq": "≥",
        r"\infty": "∞", r"\rightarrow": "→", r"\leftarrow": "←",
        r"\text{SNR}": "SNR", r"\text{opt}": "opt",
        r"\text{eff}": "eff", r"\text{rest}": "rest",
        r"\text{reset}": "reset", r"\text{rel}": "rel",
        r"\text{inh}": "inh", r"\text{exc}": "exc",
        r"\text{LIF}": "LIF", r"\text{AdEx}": "AdEx",
        r"\langle": "⟨", r"\rangle": "⟩",
        r"\frac": "frac", r"\sum": "Σ",
        "p_{\text{rel}}": "p_rel",
        "g_{\text{eff}}": "g_eff",
        "J_{\text{eff}}": "J_eff",
        "V_\theta": "V_θ",
        "r_{\text{LIF}}": "r_LIF",
        "r_{\text{AdEx}}": "r_AdEx",
        "r > 0.9": "r > 0.9",
        "r < 0.5": "r < 0.5",
        "< 10\\%": "< 10%",
        "\\%": "%",
        "r_i^{\\text{MBON}}": "r_i^MBON",
        "r_j^{\\text{MBON}}": "r_j^MBON",
        "\\sigma_{\\text{opt}}": "σ_opt",
    }
    result = latex
    for k, v in subs.items():
        result = result.replace(k, v)
    # Clean remaining \text{...}
    result = re.sub(r"\\text\{([^}]+)\}", r"\1", result)
    # Clean remaining backslash commands
    result = re.sub(r"\\([a-zA-Z]+)", r"\1", result)
    return escape(result)


def render_citations(keys):
    """Render citation keys as numbered references."""
    refs = [k.strip() for k in keys.split(",")]
    spans = []
    for ref in refs:
        spans.append(f'<a class="citation" href="#ref-{ref}">{ref}</a>')
    return "[" + ", ".join(spans) + "]"


def convert_tables(text):
    """Convert org-mode tables to HTML."""
    lines = text.split("\n")
    out = []
    in_table = False
    header_done = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and not stripped.startswith("|-"):
            if not in_table:
                out.append('<table class="booktabs">')
                in_table = True
                header_done = False
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if not header_done:
                out.append("<thead><tr>" +
                           "".join(f"<th>{org_to_html_inline(c)}</th>" for c in cells) +
                           "</tr></thead><tbody>")
                header_done = True
            else:
                out.append("<tr>" +
                           "".join(f"<td>{org_to_html_inline(c)}</td>" for c in cells) +
                           "</tr>")
        elif stripped.startswith("|-"):
            continue  # separator line
        else:
            if in_table:
                out.append("</tbody></table>")
                in_table = False
            out.append(line)
    if in_table:
        out.append("</tbody></table>")
    return "\n".join(out)


def org_to_html_inline(text):
    """Minimal inline conversion for table cells."""
    text = re.sub(
        r"(?<!\$)\$([^$\n]+?)\$(?!\$)",
        lambda m: f'<code class="math">{latex_to_unicode(m.group(1))}</code>',
        text,
    )
    text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\w)/([^/\n]+?)/(?!\w)", r"<em>\1</em>", text)
    return text


# ---------------------------------------------------------------------------
# 4. Load claims and match to test results
# ---------------------------------------------------------------------------

def load_claims():
    with open(CLAIMS_JSON) as f:
        return json.load(f)


def resolve_claims(claims_data, test_results):
    """Annotate each claim with pass/fail status from test results."""
    all_tests = test_results["tests"]
    for section in claims_data["sections"]:
        for claim in section["claims"]:
            statuses = []
            for test_id in claim["tests"]:
                if test_id in all_tests:
                    statuses.append(all_tests[test_id]["status"])
                else:
                    statuses.append("missing")
            if all(s == "passed" for s in statuses):
                claim["verdict"] = "pass"
            elif any(s in ("failed", "error") for s in statuses):
                claim["verdict"] = "fail"
            elif any(s == "missing" for s in statuses):
                claim["verdict"] = "missing"
            else:
                claim["verdict"] = "skip"
    return claims_data


# ---------------------------------------------------------------------------
# 5. Read test source snippets
# ---------------------------------------------------------------------------

def get_test_snippet(test_id, max_lines=8):
    """Extract source code for a test function."""
    parts = test_id.split("::")
    if len(parts) < 2:
        return ""
    filename = parts[0]
    funcname = parts[-1]  # last part is always the function name
    filepath = TESTS_DIR / filename
    if not filepath.exists():
        return f"# {filepath} not found"
    lines = filepath.read_text().splitlines()
    # Find function definition (may be indented inside a class)
    start = None
    for i, line in enumerate(lines):
        if re.search(rf"\bdef {funcname}\b", line):
            start = i
            break
    if start is None:
        return f"# {funcname} not found in {filename}"
    # Collect function body (indented lines after def)
    snippet = [lines[start]]
    base_indent = len(lines[start]) - len(lines[start].lstrip())
    for line in lines[start + 1 : start + max_lines]:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped and indent <= base_indent and not stripped.startswith("#"):
            break
        snippet.append(line)
    return "\n".join(snippet)


# ---------------------------------------------------------------------------
# 6. Inject badges into rendered sections
# ---------------------------------------------------------------------------

def inject_badges(html_paragraphs, claims, section_heading):
    """For each paragraph, check if any claim marker matches and inject badge.

    All claims are checked against all paragraphs — the regex markers are
    specific enough to avoid false positives. This handles subsections
    (e.g., *** Graceful Degradation under ** Stochastic).
    """
    all_claims = []
    for sec in claims["sections"]:
        all_claims.extend(sec["claims"])
    if not all_claims:
        return html_paragraphs

    result = []
    for para in html_paragraphs:
        badges_for_para = []
        for claim in all_claims:
            pattern = claim["marker"]
            if re.search(pattern, para, re.IGNORECASE):
                badges_for_para.append(claim)
        if badges_for_para:
            badge_html = render_badges(badges_for_para)
            para = para + " " + badge_html
        result.append(para)
    return result


def render_badges(claims):
    """Render inline verification badges for matched claims."""
    parts = []
    for claim in claims:
        v = claim["verdict"]
        css_class = {"pass": "pass", "fail": "fail", "missing": "missing", "skip": "skip"}.get(v, "missing")
        icon = {"pass": "&#10003;", "fail": "&#10007;", "missing": "?", "skip": "&#8709;"}.get(v, "?")
        n_tests = len(claim["tests"])
        n_pass = sum(1 for t in claim["tests"] if claim["verdict"] == "pass")
        detail_id = claim["id"]

        badge = (
            f'<span class="badge {css_class}" onclick="toggle(\'{detail_id}\')">'
            f'{icon} {n_tests}/{n_tests}'
            f'</span>'
        )
        # Detail panel (hidden by default)
        test_items = []
        for t in claim["tests"]:
            snippet = escape(get_test_snippet(t))
            test_items.append(
                f'<div class="test-item">'
                f'<code class="test-name">{escape(t)}</code>'
                f'<pre class="test-source">{snippet}</pre>'
                f'<code class="test-cmd">pytest tests/{t} -v</code>'
                f'</div>'
            )
        detail = (
            f'<div class="badge-detail" id="{detail_id}">'
            f'<div class="claim-text">{escape(claim["text"])}</div>'
            + "\n".join(test_items) +
            f'</div>'
        )
        parts.append(badge + detail)
    return '<span class="badge-group">' + " ".join(parts) + '</span>'


# ---------------------------------------------------------------------------
# 7. Assemble full HTML
# ---------------------------------------------------------------------------

def build_html(metadata, sections, claims, test_results):
    """Assemble the complete HTML document."""
    # Count claims
    total_claims = sum(len(s["claims"]) for s in claims["sections"])
    passed_claims = sum(
        1 for s in claims["sections"]
        for c in s["claims"] if c["verdict"] == "pass"
    )
    failed_claims = sum(
        1 for s in claims["sections"]
        for c in s["claims"] if c["verdict"] == "fail"
    )

    summary = test_results["summary"]
    git_hash = get_git_hash()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build navigation
    nav_items = []
    for sec in sections:
        if sec["level"] <= 2:
            indent = "nav-l1" if sec["level"] == 1 else "nav-l2"
            nav_items.append(
                f'<a class="{indent}" href="#{sec["id"]}">{org_to_html_inline(sec["title"])}</a>'
            )
    nav_html = "\n".join(nav_items)

    # Build body — match claims against raw org text, then render to HTML
    body_parts = []
    for sec in sections:
        tag = f"h{min(sec['level'] + 1, 6)}"
        body_parts.append(f'<{tag} id="{sec["id"]}">{org_to_html_inline(sec["title"])}</{tag}>')

        # Match claims against raw org text, build badge map per paragraph
        para_badges = []
        for raw_para in sec["body"]:
            badges = []
            for csec in claims["sections"]:
                for claim in csec["claims"]:
                    if re.search(claim["marker"], raw_para, re.IGNORECASE | re.DOTALL):
                        badges.append(claim)
            para_badges.append(badges)

        # Render paragraphs to HTML and append badges
        for raw_para, badges in zip(sec["body"], para_badges):
            html_para = org_to_html(raw_para)
            if badges:
                html_para = html_para + " " + render_badges(badges)

            if html_para.strip().startswith("<ul>") or html_para.strip().startswith("<table"):
                body_parts.append(html_para)
            elif html_para.strip().startswith('<div class="equation">'):
                body_parts.append(html_para)
            else:
                body_parts.append(f"<p>{html_para}</p>")

    body_html = "\n".join(body_parts)

    # Build references section
    refs_html = build_references()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{metadata.get('title', 'Verified Manuscript')}</title>
<style>
{CSS}
</style>
</head>
<body>

<header>
  <h1 class="paper-title">{org_to_html_inline(metadata.get('title', ''))}</h1>
  <div class="paper-meta">
    <span class="author">{metadata.get('author', '')}</span>
    <span class="date">{metadata.get('date', '')}</span>
  </div>
  <div class="verification-banner">
    <div class="banner-item">
      <span class="banner-number">{total_claims}</span>
      <span class="banner-label">claims</span>
    </div>
    <div class="banner-item pass">
      <span class="banner-number">{passed_claims}</span>
      <span class="banner-label">verified &#10003;</span>
    </div>
    <div class="banner-item fail" style="{'display:none' if failed_claims == 0 else ''}">
      <span class="banner-number">{failed_claims}</span>
      <span class="banner-label">failed &#10007;</span>
    </div>
    <div class="banner-item neutral">
      <span class="banner-number">{summary['passed']}</span>
      <span class="banner-label">tests pass</span>
    </div>
    <div class="banner-item neutral">
      <span class="banner-number">{summary['duration']}s</span>
      <span class="banner-label">runtime</span>
    </div>
  </div>
</header>

<nav class="toc">
  <div class="toc-title">Contents</div>
  {nav_html}
</nav>

<main>
{body_html}
</main>

<section class="references" id="references">
  <h2>References</h2>
  {refs_html}
</section>

<footer>
  <div class="verify-prompt">
    <strong>Verify independently:</strong>
    <code>git clone &lt;repo&gt; &amp;&amp; cd bravli/code/bravli &amp;&amp; pip install -e . &amp;&amp; pytest tests/ -v</code>
  </div>
  <div class="build-info">
    Commit: <code>{git_hash}</code> &middot; Built: {timestamp}
  </div>
</footer>

<script>
function toggle(id) {{
  var el = document.getElementById(id);
  if (el) {{
    el.style.display = el.style.display === 'block' ? 'none' : 'block';
  }}
}}
</script>

</body>
</html>"""
    return html


def build_references():
    """Parse references.bib and render a simple reference list."""
    bib_path = SCRIPT_DIR / "references.bib"
    if not bib_path.exists():
        return "<p>No references.bib found.</p>"

    text = bib_path.read_text()
    entries = []
    for m in re.finditer(
        r"@\w+\{(\w+),\s*"
        r".*?title\s*=\s*\{(.*?)\}.*?"
        r"author\s*=\s*\{(.*?)\}.*?"
        r"year\s*=\s*\{(\d{4})\}",
        text, re.DOTALL,
    ):
        key, title, author, year = m.groups()
        # Shorten author list
        authors = author.split(" and ")
        if len(authors) > 3:
            author_str = authors[0].split(",")[0].strip() + " et al."
        else:
            author_str = ", ".join(a.split(",")[0].strip() for a in authors)
        title = title.replace(r"\textit{", "").replace("}", "")
        entries.append(
            f'<p class="ref-entry" id="ref-{key}">'
            f'<strong>{author_str}</strong> ({year}). '
            f'{title}.</p>'
        )
    return "\n".join(entries)


def get_git_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
/* === Reset & Base === */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: Georgia, 'Times New Roman', 'Noto Serif', serif;
  font-size: 17px;
  line-height: 1.7;
  color: #2d2d2d;
  background: #fafaf8;
  max-width: 52em;
  margin: 0 auto;
  padding: 2em 2em 4em;
}

/* === Header === */
header {
  border-bottom: 1px solid #ddd;
  padding-bottom: 1.5em;
  margin-bottom: 2em;
}

.paper-title {
  font-size: 1.6em;
  font-weight: 700;
  line-height: 1.3;
  color: #1a1a1a;
  margin-bottom: 0.3em;
}

.paper-meta {
  color: #666;
  font-size: 0.95em;
  margin-bottom: 1.2em;
}
.paper-meta .author { margin-right: 1.5em; }

.verification-banner {
  display: flex;
  gap: 1.5em;
  padding: 0.8em 1.2em;
  background: #f0f0ee;
  border-radius: 4px;
  border-left: 4px solid #2d8a4e;
}

.banner-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.banner-number {
  font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  font-size: 1.3em;
  font-weight: 700;
  color: #2d2d2d;
}
.banner-item.pass .banner-number { color: #2d8a4e; }
.banner-item.fail .banner-number { color: #c94040; }
.banner-label {
  font-size: 0.75em;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* === Navigation === */
nav.toc {
  border-bottom: 1px solid #eee;
  padding-bottom: 1em;
  margin-bottom: 2.5em;
}
.toc-title {
  font-size: 0.8em;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: #999;
  margin-bottom: 0.5em;
}
nav.toc a {
  display: block;
  color: #555;
  text-decoration: none;
  font-size: 0.9em;
  padding: 0.15em 0;
}
nav.toc a:hover { color: #1a1a1a; }
nav.toc a.nav-l2 { padding-left: 1.5em; font-size: 0.85em; }

/* === Body === */
main h2 {
  font-size: 1.35em;
  margin-top: 2.5em;
  margin-bottom: 0.5em;
  color: #1a1a1a;
  border-bottom: 1px solid #eee;
  padding-bottom: 0.2em;
}
main h3 {
  font-size: 1.15em;
  margin-top: 1.8em;
  margin-bottom: 0.4em;
  color: #333;
}
main h4 {
  font-size: 1.0em;
  margin-top: 1.4em;
  margin-bottom: 0.3em;
  color: #444;
  font-style: italic;
}

main p {
  margin-bottom: 1em;
  text-align: justify;
  hyphens: auto;
}

/* === Math & Code === */
code.math {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 0.88em;
  color: #555;
  background: none;
  padding: 0;
}
code.latex {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 0.85em;
  color: #555;
  display: block;
  white-space: pre-wrap;
  padding: 0.5em 1em;
}
div.equation {
  margin: 1em 0;
  padding: 0.5em 0;
  text-align: center;
  background: #f8f8f6;
  border-radius: 3px;
}
code {
  font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  font-size: 0.88em;
  background: #f0f0ee;
  padding: 0.1em 0.3em;
  border-radius: 2px;
}

/* === Tables === */
table.booktabs {
  border-collapse: collapse;
  margin: 1.2em auto;
  font-size: 0.92em;
}
table.booktabs thead {
  border-top: 2px solid #333;
  border-bottom: 1px solid #999;
}
table.booktabs tbody {
  border-bottom: 2px solid #333;
}
table.booktabs th, table.booktabs td {
  padding: 0.4em 1em;
  text-align: left;
}
table.booktabs th {
  font-weight: 600;
  color: #333;
}

/* === Lists === */
ul, ol {
  margin: 0.5em 0 1em 1.5em;
}
li {
  margin-bottom: 0.3em;
}

/* === Citations === */
a.citation {
  color: #2d8a4e;
  text-decoration: none;
  font-size: 0.9em;
}
a.citation:hover { text-decoration: underline; }

/* === Verification Badges === */
.badge-group {
  white-space: nowrap;
}
.badge {
  display: inline-block;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 0.72em;
  padding: 0.15em 0.5em;
  border-radius: 3px;
  cursor: pointer;
  vertical-align: middle;
  margin-left: 0.3em;
  transition: background 0.15s;
  user-select: none;
}
.badge.pass {
  background: #e6f4ea;
  color: #2d8a4e;
  border: 1px solid #b7dfc3;
}
.badge.pass:hover { background: #d0ebd8; }
.badge.fail {
  background: #fde8e8;
  color: #c94040;
  border: 1px solid #f0bebe;
}
.badge.fail:hover { background: #f9d4d4; }
.badge.missing {
  background: #f0f0ee;
  color: #999;
  border: 1px solid #ddd;
}
.badge.skip {
  background: #f5f0e0;
  color: #997a00;
  border: 1px solid #e0d5a0;
}

.badge-detail {
  display: none;
  margin: 0.5em 0 1em;
  padding: 0.8em 1em;
  background: #f8f8f6;
  border: 1px solid #e5e5e0;
  border-radius: 4px;
  font-size: 0.88em;
}
.claim-text {
  color: #555;
  font-style: italic;
  margin-bottom: 0.6em;
  font-size: 0.92em;
}
.test-item {
  margin-bottom: 0.8em;
}
.test-name {
  display: block;
  font-weight: 600;
  color: #2d8a4e;
  margin-bottom: 0.2em;
  background: none;
}
.test-source {
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 0.82em;
  background: #f0f0ee;
  padding: 0.5em 0.8em;
  border-radius: 3px;
  overflow-x: auto;
  margin: 0.3em 0;
  line-height: 1.4;
}
.test-cmd {
  display: block;
  font-size: 0.8em;
  color: #888;
  background: none;
}

/* === References === */
.references {
  margin-top: 3em;
  border-top: 1px solid #ddd;
  padding-top: 1em;
}
.ref-entry {
  font-size: 0.9em;
  margin-bottom: 0.5em;
  padding-left: 1em;
  text-indent: -1em;
  color: #444;
}

/* === Footer === */
footer {
  margin-top: 3em;
  padding-top: 1em;
  border-top: 2px solid #2d8a4e;
  font-size: 0.85em;
  color: #666;
}
.verify-prompt {
  background: #f0f0ee;
  padding: 0.8em 1em;
  border-radius: 4px;
  margin-bottom: 0.5em;
}
.verify-prompt code {
  font-size: 0.85em;
  display: block;
  margin-top: 0.3em;
}
.build-info {
  font-size: 0.8em;
  color: #999;
}

/* === Print === */
@media print {
  body { max-width: none; padding: 0; font-size: 11pt; }
  .badge { border: 1px solid #999; }
  .badge-detail { display: none !important; }
  nav.toc { display: none; }
  footer { border-top: 1px solid #999; }
  .verification-banner { border-left-color: #999; }
}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    skip_tests = "--skip-tests" in sys.argv

    print("=== Verified Manuscript Builder ===")

    # 1. Run tests
    if skip_tests and TEST_CACHE.exists():
        print("Reusing cached test results...")
    else:
        print("Running test suite...")
    test_results = load_or_run_tests(skip_tests)
    s = test_results["summary"]
    print(f"  {s['passed']} passed, {s['failed']} failed, "
          f"{s['skipped']} skipped in {s['duration']}s")

    # 2. Parse manuscript
    print("Parsing paper.org...")
    metadata, sections = parse_org(PAPER_ORG)
    print(f"  {len(sections)} sections, title: {metadata.get('title', '?')[:60]}...")

    # 3. Load and resolve claims
    print("Resolving claims against test results...")
    claims = load_claims()
    claims = resolve_claims(claims, test_results)
    total = sum(len(s["claims"]) for s in claims["sections"])
    passed = sum(1 for s in claims["sections"] for c in s["claims"] if c["verdict"] == "pass")
    print(f"  {passed}/{total} claims verified")

    # 4. Build HTML
    print("Building HTML...")
    html = build_html(metadata, sections, claims, test_results)

    # 5. Write output
    OUTPUT_HTML.write_text(html)
    size_kb = OUTPUT_HTML.stat().st_size / 1024
    print(f"  Written: {OUTPUT_HTML} ({size_kb:.0f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
