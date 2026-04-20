"""
Export network_embedding_slides.py to PDF via Playwright.

Usage:
    # 1. Export HTML (uses the marimo server python from uv cache):
    /path/to/marimo_python -m marimo export html network_embedding_slides.py -o /tmp/network_embedding.html
    # 2. Generate PDF:
    python3 export_pdf.py

Hidden cell IDs (define-only cells that create blank pages):
  tqBE → cell-Xref   (data loading)
  NvKN → cell-SFPL   (d_slider)
  WAot → cell-Kclp   (k_slider)
  Acww → cell-nWHF   (biased_rw function)
  RrNs → cell-iLit   (slider defs)
  Hbol → cell-Hbol   (CSS theme)
  hHZo → cell-MJUe   (CSS inject)
"""
import asyncio
import os
from playwright.async_api import async_playwright

HIDDEN_CELLS = ['cell-Xref', 'cell-SFPL', 'cell-Kclp', 'cell-nWHF', 'cell-iLit',
                'cell-Hbol', 'cell-MJUe']

HTML_PATH = "/tmp/network_embedding.html"
PDF_PATH = "/tmp/network_embedding.pdf"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"file://{HTML_PATH}", wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(6000)

        # ── 1. Remove the "Static marimo notebook" banner
        await page.evaluate("""() => {
            for (const el of document.querySelectorAll('div')) {
                const rect = el.getBoundingClientRect();
                if (rect.top <= 5 && rect.height >= 30 && rect.height <= 70 &&
                    (el.textContent || '').includes('Static marimo notebook') &&
                    el.querySelectorAll('[id^="cell-"]').length === 0) {
                    el.style.display = 'none'; break;
                }
            }
        }""")

        # ── 2. Hide define-only cells (they create blank pages)
        await page.evaluate("""() => {
            ['cell-Xref', 'cell-SFPL', 'cell-Kclp', 'cell-nWHF', 'cell-iLit',
             'cell-Hbol', 'cell-MJUe'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.style.display = 'none';
            });
        }""")

        # ── 3. Unlock marimo's overflow-hidden container so all cells print
        #       In print mode, marimo uses print:relative but overflow-hidden clips tall content.
        await page.add_style_tag(content="""
            @media print {
                /* Remove overflow clipping so all cells render */
                .overflow-hidden { overflow: visible !important; }
                * { overflow-anchor: none; }
                /* Prevent page breaks inside cells */
                [id^="cell-"] { page-break-inside: avoid; break-inside: avoid; }
            }
        """)

        await page.wait_for_timeout(500)

        await page.pdf(
            path=PDF_PATH,
            format="A4",
            landscape=True,
            print_background=True,
            margin={"top": "12mm", "bottom": "12mm", "left": "12mm", "right": "12mm"},
        )
        await browser.close()
        size = os.path.getsize(PDF_PATH)
        print(f"PDF exported: {size/1024:.0f} KB  →  {PDF_PATH}")


asyncio.run(main())
