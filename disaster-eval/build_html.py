import re
from pathlib import Path

import markdown

R = Path(__file__).parent.parent / "reports" / "disasters"
ORDER = [("README", "Genel bakış"), ("01-deprem", "Deprem"), ("02-sel", "Sel"), ("03-heyelan", "Heyelan"), ("04-hortum", "Hortum"),
         ("05-dolu", "Dolu"), ("06-asiri-sicak", "Aşırı sıcak"), ("07-yangin", "Yangın")]

sections, nav = [], []
for slug, label in ORDER:
    md = (R / f"{slug}.md").read_text()
    md = re.sub(r"\]\((0\d-[a-z-]+)\.md\)", lambda m: f"](#{m.group(1)})", md)
    html = markdown.markdown(md, extensions=["tables"])
    html = html.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
    html = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" />\s*<em>([^<]*)</em></p>',
                  r'<figure><img alt="\1" src="\2" loading="lazy"><figcaption>\3</figcaption></figure>', html)
    sid = "genel" if slug == "README" else slug
    sections.append(f'<section id="{sid}" class="rep">{html}</section>')
    nav.append(f'<a href="#{sid}" data-t="{sid}">{label}</a>')

page = f"""<title>Afet Görüntü Analizi</title>
<style>
/* layout: sticky tab rail across the top, one report visible at a time, reading column ~72ch with wide figures */
:root {{
  --bg: #f6f7f5; --panel: #ffffff; --fg: #1d2421; --muted: #5d6a64; --line: #d9dfdb; --accent: #b2491d; --accent-soft: #f3e2d9; --fig-bg: #ffffff; --fig-fg: #4d5652;
  --display: "Fraunces", Georgia, "Times New Roman", serif; --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif; --mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #141917; --panel: #1b2220; --fg: #e4e9e6; --muted: #9aa8a1; --line: #2e3834; --accent: #f08a5d; --accent-soft: #3a2a22; color-scheme: dark }} }}
:root[data-theme="dark"] {{ --bg: #141917; --panel: #1b2220; --fg: #e4e9e6; --muted: #9aa8a1; --line: #2e3834; --accent: #f08a5d; --accent-soft: #3a2a22; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font: 16px/1.6 var(--body); }}
header {{ padding-block: 28px 8px; padding-inline: 16px; max-width: 1100px; margin: 0 auto; }}
header h1 {{ font-family: var(--display); font-weight: 600; font-size: clamp(1.7rem, 4vw, 2.5rem); margin: 0; text-wrap: balance; }}
header p {{ color: var(--muted); margin: .3rem 0 0; max-width: 70ch; }}
nav {{ position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg); border-bottom: 1px solid var(--line); }}
nav .in {{ max-width: 1100px; margin: 0 auto; padding-inline: 16px; display: flex; gap: 4px; overflow-x: auto; }}
nav a {{ padding: 10px 12px; color: var(--muted); text-decoration: none; white-space: nowrap; border-bottom: 3px solid transparent; font-weight: 600; font-size: .95rem; }}
nav a.on {{ color: var(--fg); border-bottom-color: var(--accent); }}
nav a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: -2px; }}
main {{ max-width: 1100px; margin: 0 auto; padding-inline: 16px; padding-block: 8px 64px; }}
.rep {{ display: none; }} .rep.on {{ display: block; }}
.rep h1 {{ font-family: var(--display); font-weight: 600; font-size: 1.85rem; line-height: 1.2; text-wrap: balance; margin: 1.4rem 0 .6rem; }}
.rep h2 {{ font-size: 1.15rem; letter-spacing: .02em; margin: 2rem 0 .6rem; padding-top: .8rem; border-top: 1px solid var(--line); }}
.rep p, .rep li {{ max-width: 76ch; }}
.rep blockquote {{ margin: 1rem 0; padding: .7rem 1rem; background: var(--accent-soft); border-radius: 6px; max-width: 76ch; }}
.rep blockquote p {{ margin: 0; }}
.rep ul {{ padding-left: 1.2rem; }}
.rep h2 + ul li::marker {{ color: var(--accent); }}
.tw {{ overflow-x: auto; margin: .8rem 0 1.2rem; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: .9rem; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font-weight: 700; color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }}
tr:last-child td {{ border-bottom: 0; }}
figure {{ margin: 1rem 0 1.6rem; background: var(--fig-bg); border: 1px solid var(--line); border-radius: 6px; padding: 8px; }}
figure img {{ display: block; width: 100%; height: auto; }}
figcaption {{ color: var(--fig-fg); font-size: .85rem; padding: 6px 4px 0; }}
code {{ font-family: var(--mono); font-size: .85em; }}
a {{ color: var(--accent); }}
@media (max-width: 600px) {{ body {{ font-size: 15px; }} .rep h1 {{ font-size: 1.5rem; }} }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono&display=swap">
<header><h1>Afet türlerine göre uydu ve hava görüntüsü analizi</h1>
<p>CENG391 — deprem, sel, heyelan, hortum, dolu, aşırı sıcak ve yangın için açık verilerle yapılan testler: yöntemler, metrikler, grafikler ve rastgele vakalar.</p></header>
<nav><div class="in">{''.join(nav)}</div></nav>
<main>{''.join(sections)}</main>
<script>
const tabs = [...document.querySelectorAll("nav a")], reps = [...document.querySelectorAll(".rep")];
function show(id) {{
  if (!reps.some(r => r.id === id)) id = "genel";
  reps.forEach(r => r.classList.toggle("on", r.id === id));
  tabs.forEach(t => t.classList.toggle("on", t.dataset.t === id));
}}
document.addEventListener("click", e => {{
  const a = e.target.closest('a[href^="#"]'); if (!a) return;
  e.preventDefault(); const id = a.getAttribute("href").slice(1); show(id); history.replaceState(null, "", "#" + id); window.scrollTo(0, 0);
}});
show(location.hash.slice(1) || "genel");
</script>
"""
(R / "index.html").write_text(page)
print("ok", len(page))
