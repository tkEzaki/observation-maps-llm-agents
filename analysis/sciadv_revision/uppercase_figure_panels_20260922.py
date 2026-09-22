"""Change only panel glyphs in the current vector PDFs; never replot data."""
from pathlib import Path
import re, json, io, hashlib, shutil
import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, ArrayObject, FloatObject, ByteStringObject
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figures/sciadv_caps_20260922'
OUT.mkdir(exist_ok=True)
fontdir=Path(reportlab.__file__).parent/'fonts'
# Vera is the parent face of DejaVu Sans; its Latin capital glyphs match.
pdfmetrics.registerFont(TTFont('PanelBold',str(fontdir/'VeraBd.ttf')))
pdfmetrics.registerFont(TTFont('PanelRegular',str(fontdir/'Vera.ttf')))
# Frozen conversion inputs avoid any dependency on unpublished manuscript files.
paths=[ROOT/x['source'] for x in json.loads((OUT/'panel_caps_manifest.json').read_text(encoding='utf-8'))]
assert len(paths)==len(set(paths)) and len(paths)>40
audit=[]
for path in paths:
    with pdfplumber.open(path) as doc:
        page=doc.pages[0]
        candidates=[]
        for w in page.extract_words(extra_attrs=['fontname','size']):
            if re.fullmatch('[a-z]',w['text']) and 'Bold' in w['fontname'] and w['size']>=7.9:
                ch=next(c for c in page.chars if c['text']==w['text'] and abs(c['x0']-w['x0'])<.01 and abs(c['top']-w['top'])<.01)
                candidates.append(ch)
        # Two small legend keys in S7 explicitly refer to panels D and E.
        if path.name=='figS07_sparse_finite_peer.pdf':
            candidates += [c for c in page.chars if c['text'] in 'de' and 'Bold' in c['fontname'] and c['x0']<14 and c['size']<6]
        width,height=page.width,page.height
    reader=PdfReader(path)
    stream=ContentStream(reader.pages[0].get_contents(),reader)
    font=None; size=None; used=[]; ctm=(1,0,0,1,0,0); stack=[]; lm=(1,0,0,1,0,0)
    def mult(a,b):
        return (a[0]*b[0]+a[2]*b[1],a[1]*b[0]+a[3]*b[1],a[0]*b[2]+a[2]*b[3],a[1]*b[2]+a[3]*b[3],a[0]*b[4]+a[2]*b[5]+a[4],a[1]*b[4]+a[3]*b[5]+a[5])
    for i,(args,op) in enumerate(stream.operations):
        if op==b'q': stack.append(ctm)
        if op==b'Q': ctm=stack.pop()
        if op==b'cm': ctm=mult(ctm,tuple(float(x) for x in args))
        if op==b'BT': lm=(1,0,0,1,0,0)
        if op==b'Tm': lm=tuple(float(x) for x in args)
        if op==b'Td': lm=mult(lm,(1,0,0,1,float(args[0]),float(args[1])))
        if op==b'Tf': font=str(args[0]); size=float(args[1])
        if op not in (b'Tj',b'TJ'): continue
        arr=list(args[0]) if op==b'TJ' else [args[0]]
        if not arr or not isinstance(arr[0],str): continue
        s=arr[0]
        if not re.match(r'^[a-z](?:\s|$)',s): continue
        fobj=reader.pages[0]['/Resources']['/Font'][font].get_object()
        if 'Bold' not in str(fobj.get('/BaseFont','')): continue
        pos=mult(ctm,lm)
        matches=[c for c in candidates if c not in used and c['text']==s[0] and abs(c['size']-size)<.01 and abs(c['matrix'][4]-pos[4])<.01 and abs(c['matrix'][5]-pos[5])<.01]
        if not matches: continue
        ch=matches[0]; used.append(ch)
        raw=s.original_bytes
        step=2 if fobj['/Subtype']=='/Type0' else 1
        arr[0]=ByteStringObject(raw[step:])
        arr.insert(0,FloatObject(-1000*(ch['x1']-ch['x0'])/size))
        stream.operations[i]=([ArrayObject(arr)],b'TJ')
    assert len(used)==len(candidates),(path,len(used),len(candidates))
    reader.pages[0].replace_contents(stream)
    buf=io.BytesIO(); cv=canvas.Canvas(buf,pagesize=(width,height))
    for c in used:
        cv.setFont('PanelBold',c['size'])
        color=c['non_stroking_color']
        if isinstance(color,(int,float)):cv.setFillGray(color)
        elif len(color)==3:cv.setFillColorRGB(*color)
        # The text matrix records the original baseline in PDF coordinates.
        cv.drawString(c['matrix'][4],c['matrix'][5],c['text'].upper())
    cv.save(); buf.seek(0)
    if used:
        reader.pages[0].merge_page(PdfReader(buf).pages[0])
    writer=PdfWriter(); writer.add_page(reader.pages[0])
    dest=OUT/path.name
    with dest.open('wb') as f: writer.write(f)
    with pdfplumber.open(dest) as d:
        pg=d.pages[0]
        assert abs(pg.width-width)<.001 and abs(pg.height-height)<.001
        for c in used:
            assert any(q['text']==c['text'].upper() and abs(q['x0']-c['x0'])<.02 for q in pg.chars),(path,c)
    item={'source':str(path.relative_to(ROOT)).replace('\\','/'),'output':str(dest.relative_to(ROOT)).replace('\\','/'),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'width':width,'height':height,'labels':[{'from':c['text'],'to':c['text'].upper(),'x0':c['x0'],'top':c['top'],'size':c['size']} for c in used]}
    audit.append(item)
    print(path.name,len(used),flush=True)
(OUT/'panel_caps_manifest.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print('DONE',len(audit),sum(len(x['labels']) for x in audit),flush=True)
