"""Regenerate the README screenshots in docs/images/ (offline, reproducible).

    pip install -e ".[all]" ansi2html nbconvert playwright
    python scripts/make_screenshots.py

Every terminal screenshot is the *real* output of the command shown in its title bar,
run with the offline provider so no API key is needed.
"""

from __future__ import annotations

import html
import os
import subprocess
import sys
from pathlib import Path

from ansi2html import Ansi2HTMLConverter
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
ENV = {**os.environ, "FORCE_COLOR": "1", "PY_COLORS": "1", "LLM_PROVIDER": "offline", "COLUMNS": "120"}

TERMINAL_CSS = """
body{margin:0;padding:28px;background:#e9edf2;font-family:-apple-system,Segoe UI,sans-serif}
.win{display:inline-block;min-width:860px;max-width:1180px;border-radius:10px;overflow:hidden;
     box-shadow:0 12px 32px rgba(15,23,42,.28);background:#0f172a}
.bar{background:#1e293b;padding:10px 14px;display:flex;align-items:center;gap:8px}
.dot{width:12px;height:12px;border-radius:50%}
.title{color:#94a3b8;font-size:13px;margin-left:10px;font-family:ui-monospace,Consolas,monospace}
pre{margin:0;padding:18px 22px;color:#e2e8f0;font:14px/1.55 ui-monospace,'Cascadia Code',Consolas,monospace;
    white-space:pre-wrap;word-break:break-word}
.prompt{color:#22c55e}
"""


def run(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, cwd=ROOT, env=ENV, capture_output=True, text=True)
    return (result.stdout + result.stderr).rstrip()


def terminal_html(command: str, output: str, title: str) -> str:
    conv = Ansi2HTMLConverter(inline=True, dark_bg=True)
    body = conv.convert(output, full=False)
    return f"""<html><head><meta charset="utf-8"><style>{TERMINAL_CSS}</style></head><body>
<div class="win"><div class="bar"><span class="dot" style="background:#ef4444"></span>
<span class="dot" style="background:#f59e0b"></span><span class="dot" style="background:#22c55e"></span>
<span class="title">{html.escape(title)}</span></div>
<pre><span class="prompt">$</span> {html.escape(command)}\n{body}</pre></div></body></html>"""


def shoot(page, html_doc: str, name: str, selector: str = ".win") -> None:
    page.set_content(html_doc, wait_until="networkidle")
    page.locator(selector).first.screenshot(path=str(OUT / name))
    print("  ✓", name)


ARCHITECTURE = """<html><head><meta charset="utf-8"><style>
body{margin:0;padding:28px;background:#e9edf2;font-family:-apple-system,Segoe UI,sans-serif}
.card{display:inline-block;background:#fff;border-radius:12px;padding:22px 26px;box-shadow:0 12px 32px rgba(15,23,42,.18)}
h2{margin:0 0 4px;font-size:20px;color:#0f172a}p{margin:0 0 14px;color:#475569;font-size:14px}
text{font-family:-apple-system,Segoe UI,sans-serif} .mono text, text.mono{font-family:ui-monospace,Consolas,monospace}
</style></head><body><div class="card">
<h2>The tool-calling loop: what happens when you ask "What is 3 + 2?"</h2>
<p>The model never runs code. It asks for a tool by name; your code runs it and sends the result back, linked by the call id.</p>
<svg width="1080" height="400" viewBox="0 0 1080 400" xmlns="http://www.w3.org/2000/svg">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker></defs>
<rect x="10" y="150" width="170" height="80" rx="12" fill="#e0f2fe" stroke="#0284c7" stroke-width="2"/>
<text x="95" y="183" text-anchor="middle" font-size="15" font-weight="700" fill="#075985">HumanMessage</text>
<text x="95" y="207" text-anchor="middle" font-size="13" fill="#0c4a6e">"What is 3 + 2?"</text>
<rect x="250" y="130" width="230" height="120" rx="12" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
<text x="365" y="164" text-anchor="middle" font-size="16" font-weight="700" fill="#92400e">LLM + bind_tools()</text>
<text x="365" y="190" text-anchor="middle" font-size="13" fill="#78350f">Gemini · OpenAI · offline</text>
<text x="365" y="214" text-anchor="middle" font-size="12.5" fill="#78350f">reads tool names, docstrings,</text>
<text x="365" y="232" text-anchor="middle" font-size="12.5" fill="#78350f">argument schemas</text>
<rect x="590" y="20" width="250" height="100" rx="12" fill="#ffedd5" stroke="#ea580c" stroke-width="2"/>
<text x="715" y="50" text-anchor="middle" font-size="15" font-weight="700" fill="#9a3412">AIMessage.tool_calls</text>
<text x="715" y="76" text-anchor="middle" class="mono" font-size="12.5" fill="#7c2d12">name: "add"</text>
<text x="715" y="94" text-anchor="middle" class="mono" font-size="12.5" fill="#7c2d12">args: {a: 3, b: 2}</text>
<text x="715" y="112" text-anchor="middle" class="mono" font-size="12.5" fill="#7c2d12">id:   "call_abc"</text>
<rect x="880" y="140" width="190" height="100" rx="12" fill="#f3e8ff" stroke="#9333ea" stroke-width="2"/>
<text x="975" y="170" text-anchor="middle" font-size="15" font-weight="700" fill="#6b21a8">Your code</text>
<text x="975" y="196" text-anchor="middle" class="mono" font-size="12" fill="#581c87">tool_map["add"]</text>
<text x="975" y="214" text-anchor="middle" class="mono" font-size="12" fill="#581c87">.invoke(args) → 5</text>
<rect x="590" y="260" width="250" height="90" rx="12" fill="#ede9fe" stroke="#7c3aed" stroke-width="2"/>
<text x="715" y="290" text-anchor="middle" font-size="15" font-weight="700" fill="#5b21b6">ToolMessage</text>
<text x="715" y="314" text-anchor="middle" class="mono" font-size="12.5" fill="#4c1d95">content: "5"</text>
<text x="715" y="332" text-anchor="middle" class="mono" font-size="12.5" fill="#4c1d95">tool_call_id: "call_abc"</text>
<rect x="250" y="300" width="230" height="80" rx="12" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
<text x="365" y="332" text-anchor="middle" font-size="15" font-weight="700" fill="#166534">Final AIMessage</text>
<text x="365" y="356" text-anchor="middle" font-size="13" fill="#14532d">"3 + 2 = 5"</text>
<line x1="180" y1="190" x2="246" y2="190" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<path d="M480,160 C530,140 545,80 586,72" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<text x="488" y="112" text-anchor="end" font-size="12" fill="#334155">① asks for a tool</text>
<path d="M840,80 C900,90 960,110 970,136" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<text x="852" y="58" font-size="12" fill="#334155">② look up by name</text>
<path d="M970,240 C960,290 900,305 844,305" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<text x="985" y="292" font-size="12" fill="#334155">③ wrap result</text>
<path d="M590,300 C550,290 520,250 484,232" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#a)"/>
<text x="492" y="332" font-size="12" fill="#334155">④ send back,</text>
<text x="492" y="348" font-size="12" fill="#334155">invoke again</text>
<line x1="365" y1="250" x2="365" y2="296" stroke="#16a34a" stroke-width="2" marker-end="url(#a)"/>
<text x="250" y="278" font-size="12" fill="#166534">⑤ no tool calls</text>
<text x="10" y="320" font-size="12.5" fill="#475569">Loop ①–④ repeats for</text>
<text x="10" y="338" font-size="12.5" fill="#475569">chained steps. ToolCallingAgent</text>
<text x="10" y="356" font-size="12.5" fill="#475569">codes it by hand;</text>
<text x="10" y="374" font-size="12.5" fill="#475569">create_agent automates it.</text>
</svg></div></body></html>"""


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.getenv("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
        shoot(page, ARCHITECTURE, "architecture.png", ".card")

        py = sys.executable
        shots = [
            ("manual-step-by-step.png", "The lab's lesson: one tool call, step by step",
             f"{py} examples/manual_loop_step_by_step.py offline", "python examples/manual_loop_step_by_step.py offline"),
            ("trace-lab-examples.png", "The lab's ToolCallingAgent: words in, tool calls out",
             'tool-agent "one plus 2" -p offline -e manual -t && tool-agent "one - 2" -p offline -e manual -t && tool-agent "three times two" -p offline -e manual -t', None),
            ("trace-chained-parallel.png", "Beyond the lab: chained and parallel tool calls",
             'tool-agent "Add 3 and 2, then multiply by 4" -p offline -t && tool-agent "What is 3+2 and 10*4?" -p offline -t && tool-agent "Divide 10 by 0" -p offline -t', None),
            ("tip-agent-exercises.png", "Exercises 1-3: calculate_tip tool, round trip, TipAgent",
             f"{py} examples/tip_agent_exercises.py offline", "python examples/tip_agent_exercises.py offline"),
            ("compare-engines.png", "Hand-written loop vs LangChain create_agent",
             f"{py} examples/compare_engines.py offline", "python examples/compare_engines.py offline"),
            ("pytest-results.png", "Test suite", f"{py} -m pytest -v --color=yes -p no:cacheprovider", "pytest -v"),
            ("pytest-coverage.png", "Coverage",
             f"{py} -m pytest -q --color=yes -p no:cacheprovider --cov=tool_agent --cov-report=term-missing",
             "pytest --cov=tool_agent --cov-report=term-missing"),
        ]
        for name, title, cmd, shown in shots:
            shoot(page, terminal_html(shown or cmd, run(cmd), title), name)

        nb_html = run(f"{py} -m jupyter nbconvert --to html --stdout notebooks/walkthrough.ipynb 2>/dev/null")
        page.set_viewport_size({"width": 1100, "height": 900})
        page.set_content(nb_html, wait_until="networkidle")
        page.screenshot(path=str(OUT / "notebook-walkthrough.png"), full_page=True)
        print("  ✓ notebook-walkthrough.png")
        browser.close()


if __name__ == "__main__":
    main()
