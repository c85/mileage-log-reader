"""Build the three page-limited PDF reports from the repository Markdown.

Run with the Codex bundled Python runtime (ReportLab and pypdf required).
This script does not import or change the frozen reader.
"""
import argparse
import html
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
WIDTH = letter[0] - 108
INK = colors.HexColor('#000000')
GRID = colors.HexColor('#D9D9D9')
NAVY = colors.HexColor('#243C50')


def register_fonts():
    folder = Path('/System/Library/Fonts/Supplemental')
    for name, filename in [('ReportArial','Arial.ttf'), ('ReportArialBold','Arial Bold.ttf'),
                           ('ReportArialItalic','Arial Italic.ttf'), ('ReportArialBoldItalic','Arial Bold Italic.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(folder / filename)))
    pdfmetrics.registerFontFamily('ReportArial', normal='ReportArial', bold='ReportArialBold',
                                  italic='ReportArialItalic', boldItalic='ReportArialBoldItalic')


def inline(value):
    value = value.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
    value = value.replace('\u2192',' to ')
    value = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', value)
    value = html.escape(value, quote=False)
    value = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', value)
    value = re.sub(r'`([^`]+)`', r'<font face="ReportArial">\1</font>', value)
    return value


class PipelineDiagram(Flowable):
    def __init__(self):
        super().__init__()
        self.width = WIDTH
        self.height = 158

    def draw(self):
        canvas = self.canv
        node_width, node_height = 111, 28
        positions = [0, 131, 262, 393]
        def box(x, y, text, width=node_width):
            canvas.setStrokeColor(GRID)
            canvas.setFillColor(colors.white)
            canvas.roundRect(x, y, width, node_height, 3, stroke=1, fill=1)
            canvas.setFillColor(INK)
            canvas.setFont('ReportArial', 9)
            canvas.drawCentredString(x + width/2, y + 10, text)
        def arrow(x1,y1,x2,y2):
            canvas.setStrokeColor(INK)
            canvas.setLineWidth(.7)
            canvas.line(x1,y1,x2,y2)
            if x1==x2:
                sign=1 if y2>y1 else -1
                canvas.line(x2,y2,x2-3,y2-sign*5)
                canvas.line(x2,y2,x2+3,y2-sign*5)
            else:
                sign=1 if x2>x1 else -1
                canvas.line(x2,y2,x2-sign*5,y2-3)
                canvas.line(x2,y2,x2-sign*5,y2+3)
        for x,text in zip(positions,['Photo or scan','Registration','Cell extraction','Normalization']):
            box(x,125,text)
        for x,text in zip(positions,['Post or review','Validation','Field assembly','Classification']):
            box(x,72,text)
        for first,second in zip(positions,positions[1:]):
            arrow(first+node_width,139,second,139)
            arrow(second,86,first+node_width,86)
        arrow(positions[-1]+node_width/2,125,positions[-1]+node_width/2,100)
        canvas.line(node_width/2,72,node_width/2,51)
        canvas.line(node_width/2,51,393,51)
        arrow(130,51,130,38)
        arrow(393,51,393,38)
        box(34,10,'Auto-post candidate',192)
        box(274,10,'Clerk review and audit',230)
        canvas.setFont('ReportArial',7.7)
        canvas.drawCentredString(130,56,'Checks and confidence pass')
        canvas.drawCentredString(393,56,'Failure or uncertain evidence')


def styles(font_size):
    return {
        'body': ParagraphStyle('Body',fontName='ReportArial',fontSize=font_size,leading=font_size*1.29,
                               textColor=INK,spaceAfter=6),
        'title': ParagraphStyle('Title',fontName='ReportArialBold',fontSize=21,leading=24,
                                textColor=INK,spaceAfter=7),
        'meta': ParagraphStyle('Meta',fontName='ReportArial',fontSize=9,leading=12,
                               textColor=INK,spaceAfter=12),
        'heading': ParagraphStyle('Heading',fontName='ReportArialBold',fontSize=11.4,leading=14,
                                  textColor=INK,spaceBefore=8,spaceAfter=5,keepWithNext=True),
        'cell': ParagraphStyle('Cell',fontName='ReportArial',fontSize=9.2,leading=11.6,
                               textColor=INK,spaceAfter=0),
        'header': ParagraphStyle('Header',fontName='ReportArialBold',fontSize=9.2,leading=11.6,
                                 textColor=colors.white,spaceAfter=0),
    }


def table_widths(kind, headers):
    if kind=='results_summary':
        if len(headers)==3: return [WIDTH*.52,WIDTH*.24,WIDTH*.24]
        if headers[0].startswith('Generated'): return [WIDTH*.37,WIDTH*.20,WIDTH*.22,WIDTH*.21]
        return [WIDTH*.37,WIDTH*.25,WIDTH*.19,WIDTH*.19]
    if kind=='business_note': return [WIDTH*.22,WIDTH*.16,WIDTH*.25,WIDTH*.37]
    return [WIDTH*.33,WIDTH*.67]


def markdown_blocks(source):
    lines=source.splitlines()
    index=0
    while index<len(lines):
        line=lines[index].strip()
        if not line:
            index+=1;continue
        if line.startswith('```'):
            contents=[];index+=1
            while index<len(lines) and not lines[index].strip().startswith('```'):
                contents.append(lines[index]);index+=1
            yield ('diagram' if 'mermaid' in line else 'code','\n'.join(contents))
            index+=1;continue
        if line.startswith('# '):
            yield 'title',line[2:];index+=1;continue
        if line.startswith('## '):
            yield 'heading',line[3:];index+=1;continue
        if line.startswith('|'):
            rows=[]
            while index<len(lines) and lines[index].strip().startswith('|'):
                cells=[v.strip() for v in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',v) for v in cells):rows.append(cells)
                index+=1
            yield 'table',rows;continue
        paragraph=[line];index+=1
        while index<len(lines) and lines[index].strip() and not lines[index].lstrip().startswith(('#','|','```')):
            paragraph.append(lines[index].strip());index+=1
        yield 'body',' '.join(paragraph)


def build(kind, font_size, output):
    style=styles(font_size)
    if kind=='results_summary':
        style['body'].spaceAfter=4
    source=(ROOT/'docs'/f'{kind}.md').read_text()
    story=[]
    for block,value in markdown_blocks(source):
        if block=='heading' and kind=='results_summary' and value=='Extraction and failure modes':
            story.append(PageBreak())
        if block=='title':
            # A long approach title is deliberately shortened without changing its subject.
            title='Mileage claim reader approach' if kind=='approach' else value
            story.extend([Paragraph(inline(title),style['title']),
                          Paragraph('Team 1 | ISM 6642 | October 7 2026',style['meta'])])
        elif block=='heading':story.append(Paragraph(inline(value),style['heading']))
        elif block=='diagram':story.extend([PipelineDiagram(),Spacer(1,6)])
        elif block=='table':
            numeric=ParagraphStyle('NumericCell',parent=style['cell'],alignment=TA_RIGHT)
            data=[[Paragraph(inline(v),style['header'] if row_number==0 else
                             (numeric if kind!='approach' and (column>0 or kind=='business_note') else style['cell']))
                   for column,v in enumerate(row)] for row_number,row in enumerate(value)]
            table=Table(data,colWidths=table_widths(kind,value[0]),repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),NAVY),('GRID',(0,0),(-1,-1),.4,GRID),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F3F5F7')]),
                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),
                ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),4 if kind=='results_summary' else 5),
                ('BOTTOMPADDING',(0,0),(-1,-1),4 if kind=='results_summary' else 5),
            ]))
            story.extend([table,Spacer(1,8)])
        elif block=='code':
            for line in value.splitlines():story.append(Paragraph(inline(line),style['body']))
        else:
            paragraph=Paragraph(inline(value),style['body'])
            story.append(KeepTogether([paragraph]) if kind=='approach' else paragraph)
    label={'approach':'Approach','results_summary':'Results summary','business_note':'Business note'}[kind]
    def page(canvas,doc):
        canvas.setFillColor(INK)
        canvas.setFont('ReportArial',8)
        if doc.page>1:canvas.drawString(54,letter[1]-28,'Form ML-7 | '+label)
        canvas.drawString(54,28,'Team 1 | '+label)
        canvas.drawRightString(letter[0]-54,28,f'Page {doc.page}')
    document=SimpleDocTemplate(str(output),pagesize=letter,leftMargin=54,rightMargin=54,
        topMargin=48,bottomMargin=46,title=label+' for the mileage claim reader',author='Team 1',
        subject='ISM 6642 final project deliverable')
    document.build(story,onFirstPage=page,onLaterPages=page)
    pages=len(PdfReader(output).pages)
    print(f'{kind}: {pages} pages at {font_size} point body type')
    return pages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/pdf')
    parser.add_argument('--font-size',type=float,default=10.5)
    args=parser.parse_args()
    register_fonts()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    expected={'approach':{3,4},'results_summary':{2},'business_note':{1}}
    for kind in ['approach','results_summary','business_note']:
        count=build(kind,args.font_size,args.output_dir/f'{kind}.pdf')
        if count not in expected[kind]:
            raise SystemExit(f'{kind} has {count} pages; expected {sorted(expected[kind])}. Adjust content/layout and render again.')


if __name__=='__main__':main()
