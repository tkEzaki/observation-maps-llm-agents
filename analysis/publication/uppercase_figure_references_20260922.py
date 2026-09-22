"""Uppercase seven artwork-internal panel references, preserving text advances."""
from pathlib import Path
import json,re,hashlib,io
import pdfplumber
from pypdf import PdfReader,PdfWriter
from pypdf.generic import ContentStream,ArrayObject,FloatObject,ByteStringObject
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab
for n,f in [('RefRegular','Vera.ttf'),('RefBold','VeraBd.ttf')]:pdfmetrics.registerFont(TTFont(n,str(Path(reportlab.__file__).parent/'fonts'/f)))
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'figures/panel_conversion'
manifest=json.loads((OUT/'panel_caps_manifest.json').read_text())
names=['figS02a_observation_maps.pdf','figS02b_serialization_metadata.pdf','figS07_sparse_finite_peer.pdf','figS19a_claude_trajectories_moments.pdf','figS19b_claude_trajectories_centers.pdf','figS19c_claude_trajectories_intervals.pdf','figS11a_moments_surrogate_development.pdf']
def mult(a,b):return (a[0]*b[0]+a[2]*b[1],a[1]*b[0]+a[3]*b[1],a[0]*b[2]+a[2]*b[3],a[1]*b[2]+a[3]*b[3],a[0]*b[4]+a[2]*b[5]+a[4],a[1]*b[4]+a[3]*b[5]+a[5])
for name in names:
 p=OUT/name
 item=next(x for x in manifest if Path(x['output']).name==name)
 with pdfplumber.open(ROOT/item['source']) as pdf:
  pg=pdf.pages[0]; words=pg.extract_words(); targets=[]
  for i,w in enumerate(words):
   prev=words[i-1]['text'].strip('(') if i else ''
   if (prev=='panel' and re.match(r'^[a-z](?:\)|;|$)',w['text'])) or (w['text']=='f' and i>=2 and words[i-2]['text']=='lines' and prev=='in') or (w['text']=='4b' and prev=='Fig.'):
    letter='b' if w['text']=='4b' else w['text'][0]
    c=next(c for c in pg.chars if c['text']==letter and c['x0']>=w['x0']-.01 and c['x1']<=w['x1']+.01 and abs(c['top']-w['top'])<.1)
    targets.append(c)
 assert len(targets)==1,(name,targets)
 with pdfplumber.open(p) as pdf:
  c=targets[0]
  if any(q['text']==c['text'].upper() and abs(q['matrix'][4]-c['matrix'][4])<.02 and abs(q['matrix'][5]-c['matrix'][5])<.02 for q in pdf.pages[0].chars):
   if not any(x.get('kind')=='internal_reference' for x in item['labels']):item['labels'].append({'from':c['text'],'to':c['text'].upper(),'x0':c['x0'],'top':c['top'],'size':c['size'],'kind':'internal_reference'})
   item['output_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
   continue
 reader=PdfReader(p); stream=ContentStream(reader.pages[0].get_contents(),reader)
 ctm=(1,0,0,1,0,0); lm=ctm; tm=ctm; stack=[]; font=None; size=0; used=[]
 def widths(fo):
  if fo['/Subtype']=='/Type0':
   desc=fo['/DescendantFonts'][0].get_object(); ws=desc['/W']; d={}; j=0
   while j<len(ws):
    lo=int(ws[j]); v=ws[j+1];j+=2
    if isinstance(v,list):d.update({lo+k:float(x) for k,x in enumerate(v)})
    else:
     hi=int(v);wid=float(ws[j]);j+=1;d.update({k:wid for k in range(lo,hi+1)})
   return d,desc
  return {int(fo['/FirstChar'])+k:float(x) for k,x in enumerate(fo['/Widths'])},fo
 for oi,(args,op) in enumerate(stream.operations):
  if op==b'q':stack.append(ctm)
  if op==b'Q':ctm=stack.pop()
  if op==b'cm':ctm=mult(ctm,tuple(map(float,args)))
  if op==b'BT':lm=tm=(1,0,0,1,0,0)
  if op==b'Tm':lm=tm=tuple(map(float,args))
  if op==b'Td':lm=mult(lm,(1,0,0,1,float(args[0]),float(args[1])));tm=lm
  if op==b'Tf':font=str(args[0]);size=float(args[1])
  if op not in (b'Tj',b'TJ'):continue
  fo=reader.pages[0]['/Resources']['/Font'][font].get_object(); ws,desc=widths(fo)
  arr=list(args[0]) if op==b'TJ' else [args[0]]; new=[]; changed=False
  for token in arr:
   if isinstance(token,bytes):
    step=2 if fo['/Subtype']=='/Type0' else 1
    wid=sum(ws.get(int.from_bytes(token[j:j+step],'big'),1000) for j in range(0,len(token),step))
    tm=mult(tm,(1,0,0,1,wid*size/1000,0));new.append(token);continue
   if not isinstance(token,str):
    tm=mult(tm,(1,0,0,1,-float(token)*size/1000,0));new.append(token);continue
   raw=token.original_bytes;step=2 if fo['/Subtype']=='/Type0' else 1; chunks=[];last=0
   for k,ch in enumerate(token):
    pos=mult(ctm,tm); wid=ws.get(ord(ch),1000)
    matches=[c for c in targets if c not in used and ch==c['text'] and abs(c['matrix'][4]-pos[4])<.02 and abs(c['matrix'][5]-pos[5])<.02]
    if matches:
     c=matches[0];used.append(c)
     chunks.append(ByteStringObject(raw[last:k*step]))
     chunks.append(FloatObject(-wid));last=(k+1)*step;changed=True
    tm=mult(tm,(1,0,0,1,wid*size/1000,0))
   if chunks:new.extend(chunks);new.append(ByteStringObject(raw[last:]))
   else:new.append(token)
  if changed:stream.operations[oi]=([ArrayObject(new)],b'TJ')
 assert len(used)==1,(name,len(used))
 reader.pages[0].replace_contents(stream)
 buf=io.BytesIO();cv=canvas.Canvas(buf,pagesize=(float(reader.pages[0].mediabox.width),float(reader.pages[0].mediabox.height)))
 for c in used:
  cv.setFont('RefBold' if 'Bold' in c['fontname'] else 'RefRegular',c['size'])
  color=c['non_stroking_color']
  if isinstance(color,(int,float)):cv.setFillGray(color)
  elif len(color)==3:cv.setFillColorRGB(*color)
  cv.drawString(c['matrix'][4],c['matrix'][5],c['text'].upper())
 cv.save();buf.seek(0);reader.pages[0].merge_page(PdfReader(buf).pages[0])
 writer=PdfWriter();writer.add_page(reader.pages[0])
 with p.open('wb') as f:writer.write(f)
 item=next(x for x in manifest if Path(x['output']).name==name)
 item['labels'] += [{'from':c['text'],'to':c['text'].upper(),'x0':c['x0'],'top':c['top'],'size':c['size'],'kind':'internal_reference'} for c in used]
 item['output_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
 print(name,used[0]['text'],flush=True)
(OUT/'panel_caps_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
