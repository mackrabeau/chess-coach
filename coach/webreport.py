"""Building blocks for the interactive report (20_report.py): one self-contained HTML file.

No external libraries: boards are drawn by a small inline script from python-chess's piece SVGs;
every line is precomputed in Python (FEN after each ply), so the page needs no chess logic beyond
"click two squares" for puzzles. Light and dark themes follow the system setting.
"""
from __future__ import annotations

import html
import json

import chess
import chess.svg

TAG_TEXT = {"solid": "solid", "lead": "lead", "style": "style (not costing points)",
            "strength": "strength", "info": "info", "exploratory": "exploratory"}


def esc(x) -> str:
    return html.escape(str(x))


def line_data(fen: str, ucis) -> dict:
    """SAN + FEN after each ply of a line (stops at the first illegal move)."""
    b = chess.Board(fen)
    sans, fens, moves = [], [], []
    for u in list(ucis or []):
        try:
            mv = chess.Move.from_uci(u)
        except ValueError:
            break
        if mv not in b.legal_moves:
            break
        num = f"{b.fullmove_number}." if b.turn else (f"{b.fullmove_number}..." if not sans else "")
        sans.append(num + b.san(mv))
        b.push(mv)
        fens.append(b.fen())
        moves.append(u)
    return {"sans": sans, "fens": fens, "ucis": moves}


class Report:
    """Collects tabs, sections and the example/puzzle data, then writes one HTML file."""

    def __init__(self, title: str, home: str | None = None):
        self.title = title
        self.home = home                         # relative link to data/index.html
        self.tabs: list[tuple[str, str, list[str]]] = []   # (id, label, html parts)
        self.examples: list[dict] = []

    def tab(self, tid: str, label: str) -> list[str]:
        parts: list[str] = []
        self.tabs.append((tid, label, parts))
        return parts

    # ------------------------------------------------------------ components
    def example(self, fen: str, best=None, played=None, note: str = "", puzzle: bool = False,
                labels=("best line", "your move")) -> str:
        """Interactive board: step through the best line and the played line; puzzle = find the move."""
        b = chess.Board(fen)
        ex = {"fen": fen, "orient": "w" if b.turn else "b", "note": note, "puzzle": puzzle,
              "lines": [{"name": labels[0], **line_data(fen, best)}]}
        if played:
            ex["lines"].append({"name": labels[1], **line_data(fen, played)})
        self.examples.append(ex)
        return f"<div class='ex' data-ex='{len(self.examples) - 1}'></div>"

    @staticmethod
    def finding(title: str, claim: str, tag: str, evidence: str = "", examples: str = "",
                todo: str = "", elo=None, elo_note: str = "", link: str | None = None, open_: bool = True) -> str:
        gain = (f"<span class='gain' title='{esc(elo_note)}'>≈ +{elo:.0f} Elo</span>" if elo else "")
        more = f" <a class='more' href='{esc(link)}'>details →</a>" if link else ""
        return (f"<details class='finding'{' open' if open_ else ''}><summary><span class='tag {esc(tag)}'>"
                f"{esc(TAG_TEXT.get(tag, tag))}</span> {esc(title)} {gain}</summary><p class='claim'>{claim}{more}</p>"
                + (f"<div class='evidence'>{evidence}</div>" if evidence else "")
                + (f"<div class='exs'>{examples}</div>" if examples else "")
                + (f"<p class='todo'><b>What to do:</b> {todo}</p>" if todo else "") + "</details>")

    @staticmethod
    def table(header, rows, cls: str = "") -> str:
        h = "".join(f"<th>{esc(c)}</th>" for c in header)
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
        return f"<div class='tw'><table class='{cls}'><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>"

    @staticmethod
    def level_chip(rel: float | None, text: str | None = None, scale: float = 300) -> str:
        """Red (below your level) -> green (above), for a relative Elo-like value."""
        if rel is None or rel != rel:
            return "<span class='chip'>–</span>"
        t = max(-1.0, min(1.0, rel / scale))
        hue = 60 + 60 * t                     # 0 = red, 60 = yellow, 120 = green
        return (f"<span class='chip' style='background:hsl({hue:.0f} 65% 45%)'>"
                f"{esc(text if text is not None else f'{rel:+.0f}')}</span>")

    # ------------------------------------------------------------ output
    def html(self) -> str:
        pieces = "".join(f"<symbol id='p-{s}' viewBox='0 0 45 45'>{svg}</symbol>"
                         for s, svg in chess.svg.PIECES.items())
        nav = "".join(f"<button data-tab='{tid}'>{esc(label)}</button>" for tid, label, _ in self.tabs)
        panes = "".join(f"<section id='{tid}' class='pane'>{''.join(parts)}</section>"
                        for tid, _, parts in self.tabs)
        data = json.dumps(self.examples).replace("</", "<\\/")
        return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
                f"<title>{esc(self.title)}</title><style>{CSS}</style></head><body>"
                f"<svg width='0' height='0' style='position:absolute'><defs>{pieces}</defs></svg>"
                f"<header><h1>{esc(self.title)}"
                + (f" <a class='home' href='{esc(self.home)}'>all players &amp; comparisons</a>" if self.home else "")
                + f"</h1><nav>{nav}</nav></header><main>{panes}</main>"
                f"<script>const EX={data};{JS}</script></body></html>")

    def write(self, path) -> None:
        path.write_text(self.html(), encoding="utf-8")


CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1f;--muted:#6b6b70;--card:#fff;--line:#e4e1da;--accent:#2f6fde;
--light:#ecdcb9;--dark:#b58863;--hl:rgba(255,215,0,.55);--best:#2a9d55;--bad:#d1453b;
--link:#1a5fd0;--link-visited:#6b3fb0}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17181b;--fg:#ececec;
--muted:#9a9aa2;--card:#212227;--line:#33343a;--accent:#7aa7ff;--link:#8ab4ff;--link-visited:#c3a6ff}}
a{color:var(--link)}a:visited{color:var(--link-visited)}a:hover{text-decoration-thickness:2px}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px}
h1{font-size:20px;margin:0 0 6px}nav{display:flex;gap:6px;flex-wrap:wrap}
nav button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:999px;
padding:4px 12px;cursor:pointer;font:inherit}nav button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
main{max-width:1100px;margin:0 auto;padding:16px}.pane{display:none}.pane.on{display:block}
h2{font-size:18px;margin:22px 0 8px}h3{font-size:16px;margin:16px 0 6px}
.muted{color:var(--muted)}.finding{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:10px 14px;margin:10px 0}.finding summary{cursor:pointer;font-weight:600}
.home{font-size:13px;font-weight:400;margin-left:10px}.pgnrow{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:8px 12px;margin:8px 0}.copy{border:1px solid var(--line);background:var(--card);color:var(--fg);
border-radius:6px;padding:1px 8px;cursor:pointer;font:inherit}
.claim{margin:8px 0}.gain{float:right;font-weight:600;color:var(--accent)}.more{font-size:13px}.todo{margin:8px 0 2px}
.tag{font-size:12px;font-weight:600;border-radius:6px;padding:1px 7px;margin-right:6px;color:#fff;background:#888}
.tag.solid{background:#c0392b}.tag.lead{background:#d68910}.tag.style{background:#7f8c8d}
.tag.strength{background:#1e8449}.tag.info{background:#2e86c1}.tag.exploratory{background:#8e44ad}
.tw{overflow-x:auto}table{border-collapse:collapse;margin:6px 0;font-size:14px}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}
.chip{display:inline-block;min-width:44px;text-align:center;color:#fff;border-radius:6px;padding:0 6px;
background:#888;font-variant-numeric:tabular-nums}
.exs{display:flex;flex-wrap:wrap;gap:14px;margin-top:8px}.ex{width:300px;max-width:100%}
.ex svg.board{width:100%;height:auto;display:block;border-radius:4px}
.ex .ctl{display:flex;gap:4px;align-items:center;margin-top:4px;flex-wrap:wrap}
.ex button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;
padding:1px 8px;cursor:pointer;font:inherit}.ex button.on{border-color:var(--accent);color:var(--accent)}
.ex .moves{font-size:13px;min-height:2.6em}.ex .note{font-size:13px;color:var(--muted)}
.ex .res{font-weight:600}.co{font-size:9px;fill:#5a4630;pointer-events:none}.ok{color:var(--best)}.no{color:var(--bad)}
.kpis{display:flex;flex-wrap:wrap;gap:12px}.kpi{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:10px 14px;min-width:160px}.kpi b{display:block;font-size:22px}
@media (max-width:600px){main{padding:12px}.ex{width:100%}}
"""

JS = r"""
const SQ=45, FILES='abcdefgh';
function sqXY(sq,orient){const f=FILES.indexOf(sq[0]),r=+sq[1]-1;
 return orient==='w'?[f*SQ,(7-r)*SQ]:[(7-f)*SQ,r*SQ];}
function boardSVG(fen,orient,hl,click){
 const ns='http://www.w3.org/2000/svg',s=document.createElementNS(ns,'svg');
 s.setAttribute('viewBox','0 0 360 360');s.setAttribute('class','board');
 const css=getComputedStyle(document.documentElement);
 for(let r=0;r<8;r++)for(let f=0;f<8;f++){const sq=FILES[f]+(r+1),[x,y]=sqXY(sq,orient);
  const e=document.createElementNS(ns,'rect');e.setAttribute('x',x);e.setAttribute('y',y);
  e.setAttribute('width',SQ);e.setAttribute('height',SQ);
  e.setAttribute('fill',(f+r)%2?css.getPropertyValue('--light'):css.getPropertyValue('--dark'));
  if(hl&&hl.includes(sq)){e.setAttribute('fill',css.getPropertyValue('--hl'));}
  e.dataset.sq=sq;if(click)e.addEventListener('click',()=>click(sq));s.appendChild(e);}
 const rows=fen.split(' ')[0].split('/');
 rows.forEach((row,i)=>{let f=0;for(const c of row){if(/\d/.test(c)){f+=+c;continue;}
  const sq=FILES[f]+(8-i),[x,y]=sqXY(sq,orient),u=document.createElementNS(ns,'use');
  u.setAttribute('href','#p-'+c);u.setAttribute('x',x);u.setAttribute('y',y);
  u.setAttribute('width',SQ);u.setAttribute('height',SQ);u.style.pointerEvents='none';s.appendChild(u);f++;}});
 for(let i=0;i<8;i++){const fl=orient==='w'?FILES[i]:FILES[7-i],rk=orient==='w'?8-i:i+1;
  const a=document.createElementNS(ns,'text');a.setAttribute('x',i*SQ+SQ-3);a.setAttribute('y',357);
  a.setAttribute('class','co');a.setAttribute('text-anchor','end');a.textContent=fl;s.appendChild(a);
  const b=document.createElementNS(ns,'text');b.setAttribute('x',2);b.setAttribute('y',i*SQ+10);
  b.setAttribute('class','co');b.textContent=rk;s.appendChild(b);}
 return s;}
function mount(el){const ex=EX[+el.dataset.ex];let li=0,ply=0,pick=null,solved=!ex.puzzle;
 const wrap=document.createElement('div'),ctl=document.createElement('div'),mv=document.createElement('div'),
  note=document.createElement('div'),res=document.createElement('div');
 ctl.className='ctl';mv.className='moves';note.className='note';res.className='res';
 function hl(){const L=ex.lines[li];if(ply===0)return pick?[pick]:[];const u=L.ucis[ply-1];return [u.slice(0,2),u.slice(2,4)];}
 function draw(){const L=ex.lines[li];const fen=ply===0?ex.fen:L.fens[ply-1];
  wrap.replaceChildren(boardSVG(fen,ex.orient,hl(),!solved?onClick:null));
  if(!solved){mv.textContent='Find the move.';}
  else{mv.innerHTML='<b>'+L.name+':</b> '+L.sans.map((s,i)=>i<ply?(i===ply-1?'<u>'+s+'</u>':s):'<span class="muted">'+s+'</span>').join(' ');}
  [...ctl.querySelectorAll('button[data-li]')].forEach(b=>b.classList.toggle('on',+b.dataset.li===li));}
 function onClick(sq){if(!pick){pick=sq;draw();return;}const u=pick+sq;pick=null;const want=ex.lines[0].ucis[0]||'';
  if(u===want.slice(0,4)){res.textContent='✓ correct';res.className='res ok';solved=true;li=0;ply=1;buildCtl();}
  else{res.textContent='✗ not this one, try again';res.className='res no';}draw();}
 function buildCtl(){ctl.replaceChildren();
  if(!solved){const r=document.createElement('button');r.textContent='show solution';
   r.onclick=()=>{solved=true;li=0;ply=0;res.textContent='';buildCtl();draw();};ctl.appendChild(r);return;}
  const mk=(t,f)=>{const b=document.createElement('button');b.textContent=t;b.onclick=f;ctl.appendChild(b);};
  mk('⏮',()=>{ply=0;draw();});mk('◀',()=>{ply=Math.max(0,ply-1);draw();});
  mk('▶',()=>{ply=Math.min(ex.lines[li].fens.length,ply+1);draw();});
  ex.lines.forEach((L,i)=>{const b=document.createElement('button');b.textContent=L.name;b.dataset.li=i;
   b.onclick=()=>{li=i;ply=0;draw();};ctl.appendChild(b);});}
 note.innerHTML=ex.note||'';buildCtl();el.append(wrap,ctl,res,mv,note);draw();}
document.querySelectorAll('.ex').forEach(mount);
const btns=document.querySelectorAll('nav button'),panes=document.querySelectorAll('.pane');
function show(id){btns.forEach(b=>b.classList.toggle('on',b.dataset.tab===id));
 panes.forEach(p=>p.classList.toggle('on',p.id===id));try{localStorage.setItem('tab',id)}catch(e){}}
btns.forEach(b=>b.onclick=()=>show(b.dataset.tab));
document.querySelectorAll('button.copy').forEach(b=>b.onclick=async()=>{
 const t=document.getElementById(b.dataset.src).value;
 try{await navigator.clipboard.writeText(t);}catch(e){const a=document.createElement('textarea');a.value=t;
  document.body.appendChild(a);a.select();document.execCommand('copy');a.remove();}
 b.textContent='copied \u2713';setTimeout(()=>b.textContent='Copy PGN',1500);});
let start=null;try{start=localStorage.getItem('tab')}catch(e){}
show(start&&document.getElementById(start)?start:btns[0].dataset.tab);
"""
