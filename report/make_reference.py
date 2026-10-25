#!/usr/bin/env python3
"""Write report/reference.docx: pandoc's default reference document with
print-oriented styles (11 pt body text, tighter paragraph spacing, 9.5 pt
tables and captions).  pandoc takes fonts, sizes and spacing from it.

Usage: make_reference.py <pandoc> <out.docx>
"""
import re
import subprocess
import sys
import zipfile
from io import BytesIO

pandoc, out = sys.argv[1], sys.argv[2]
src = subprocess.run([pandoc, "--print-default-data-file", "reference.docx"],
                     capture_output=True, check=True).stdout


def set_style(xml, style_id, ppr=None, rpr=None):
    """Replace the pPr / rPr of one style (adding them if missing)."""
    m = re.search(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % style_id, xml, re.S)
    body = m.group(0)
    for tag, new in (("pPr", ppr), ("rPr", rpr)):
        if new is None:
            continue
        block = f"<w:{tag}>{new}</w:{tag}>"
        if f"<w:{tag}>" in body:
            body = re.sub(rf"<w:{tag}>.*?</w:{tag}>", block, body, count=1, flags=re.S)
        else:
            body = body.replace("</w:style>", block + "</w:style>")
    return xml[:m.start()] + body + xml[m.end():]


def styles(xml):
    xml = xml.replace('<w:sz w:val="24" />\n        <w:szCs w:val="24" />',
                      '<w:sz w:val="22" />\n        <w:szCs w:val="22" />', 1)
    xml = set_style(xml, "BodyText", ppr='<w:spacing w:before="0" w:after="120" />')
    xml = set_style(xml, "Compact", ppr='<w:spacing w:before="20" w:after="20" />',
                    rpr='<w:sz w:val="19" /><w:szCs w:val="19" />')
    xml = set_style(xml, "ImageCaption", ppr='<w:spacing w:before="40" w:after="200" />',
                    rpr='<w:i /><w:sz w:val="19" /><w:szCs w:val="19" />')
    xml = set_style(xml, "TableCaption", ppr='<w:keepNext /><w:spacing w:before="160" w:after="60" />',
                    rpr='<w:i /><w:sz w:val="19" /><w:szCs w:val="19" />')
    return xml


zin = zipfile.ZipFile(BytesIO(src))
buf = BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/styles.xml":
            data = styles(data.decode("utf-8")).encode("utf-8")
        zout.writestr(item, data)
with open(out, "wb") as f:
    f.write(buf.getvalue())
