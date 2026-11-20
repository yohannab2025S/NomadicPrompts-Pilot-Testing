"""Builds part4/Part4_Report.docx from the SAME content as build_report.py (single source), so it can be uploaded to Google Drive
(Drive converts .docx to a Google Doc and keeps the charts and tables). Run from the repo root: python part4/build_docx.py"""
import re, html, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from build_report import S                      # flowables; importing also rebuilds the PDF (harmless)
from reportlab.platypus import Paragraph, Table, Image, KeepTogether, PageBreak
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_BREAK
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()
sec = doc.sections[0]; sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for a in ("left_margin", "right_margin"): setattr(sec, a, Inches(0.75))
sec.top_margin = sec.bottom_margin = Inches(0.7)
st = doc.styles
st["Normal"].font.name = "Arial"; st["Normal"].font.size = Pt(10); st["Normal"].element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
st["Normal"].paragraph_format.space_after = Pt(6)
for name, size, before in (("Title", 20, 0), ("Heading 1", 15, 14), ("Heading 2", 12, 10)):
    s = st[name]; s.font.name = "Arial"; s.font.size = Pt(size); s.font.bold = True; s.font.color.rgb = RGBColor(0x1F, 0x2A, 0x44)
    s.paragraph_format.space_before = Pt(before); s.paragraph_format.space_after = Pt(6); s.paragraph_format.keep_with_next = True
    s.element.rPr.rFonts.set(qn("w:ascii"), "Arial"); s.element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")

TOK = re.compile(r"(<[^>]+>)")
def add_markup(par, text, size=None, base_bold=False):
    bold, ital, mono, color = base_bold, False, False, None
    stack = []
    for tok in TOK.split(text):
        if not tok: continue
        if tok.startswith("<"):
            tag = tok.strip("<>/ ").split()[0].lower()
            closing = tok.startswith("</")
            if tag in ("b", "i", "font"):
                if not closing:
                    stack.append((bold, ital, mono, color))
                    if tag == "b": bold = True
                    if tag == "i": ital = True
                    if tag == "font":
                        if "Courier" in tok: mono = True
                        m = re.search(r'color="#([0-9A-Fa-f]{6})"', tok)
                        if m: color = m.group(1)
                elif stack: bold, ital, mono, color = stack.pop()
            continue
        r = par.add_run(html.unescape(tok)); r.bold = bold; r.italic = ital
        if mono: r.font.name = "Courier New"
        if color: r.font.color.rgb = RGBColor.from_string(color.upper())
        if size: r.font.size = Pt(size)

def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hexcolor); tcPr.append(shd)

def add_table(t):
    rows = t._cellvalues; ncol = len(rows[0]); widths = list(t._colWidths); tot = sum(widths); target = 7.0 * 72
    widths = [w * target / tot for w in widths]
    tb = doc.add_table(rows=len(rows), cols=ncol); tb.style = "Table Grid"; tb.autofit = False
    for j, gc in enumerate(tb._tbl.tblGrid.gridCol_lst): gc.w = Pt(widths[j])
    fs = 7.5
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tb.cell(i, j); c.width = Pt(widths[j]); c.text = ""
            p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
            while isinstance(val, (list, tuple)) and len(val) == 1: val = val[0]     # reportlab wraps cell content in a tuple
            txt = val.text if isinstance(val, Paragraph) else (" ".join(x.text if isinstance(x, Paragraph) else str(x) for x in val) if isinstance(val, (list, tuple)) else str(val))
            add_markup(p, txt, size=fs, base_bold=(i == 0))
            if i == 0: shade(c, "DDE6F0")
            elif i % 2 == 0: shade(c, "F5F5F5")
        if i == 0:
            trPr = tb.rows[0]._tr.get_or_add_trPr(); h = OxmlElement("w:tblHeader"); h.set(qn("w:val"), "true"); trPr.append(h)
    for r in tb.rows:                                     # keep rows whole on a page
        trPr = r._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trPr.append(cs)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

def walk(items):
    for f in items:
        if isinstance(f, KeepTogether): walk(f._content)
        elif isinstance(f, PageBreak): doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        elif isinstance(f, Table): add_table(f)
        elif isinstance(f, Image):
            w_in = min(f.drawWidth / 72.0, 7.0)
            doc.add_picture(f.filename, width=Inches(w_in)); doc.paragraphs[-1].paragraph_format.keep_with_next = True
        elif isinstance(f, Paragraph):
            name = f.style.name
            if name == "T": par = doc.add_paragraph(style="Title"); add_markup(par, f.text)
            elif name == "H1": par = doc.add_heading(level=1); add_markup(par, f.text)
            elif name == "H2": par = doc.add_heading(level=2); add_markup(par, f.text)
            else:
                par = doc.add_paragraph()
                add_markup(par, f.text, size=9 if name == "CAP" else None)
                if name == "CAP":
                    par.paragraph_format.space_after = Pt(10)
                    for r in par.runs: r.font.color.rgb = r.font.color.rgb or RGBColor(0x33, 0x33, 0x33)
walk(S)
doc.core_properties.title = "Nomadic Prompts Part 4: Analysis, Charts, and Results"; doc.core_properties.author = ""
doc.save("part4/Part4_Report.docx"); print("docx saved")
