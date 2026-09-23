#!/usr/bin/env python3
"""Small read-only gallery for timestamped hero screenshots."""

from __future__ import annotations

import datetime as dt
import html
import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit


IMAGE_DIR = Path(os.environ.get("HERO_IMAGE_DIR", "/usr/local/webserver/images"))
LISTEN_HOST = os.environ.get("HERO_GALLERY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("HERO_GALLERY_PORT", "8765"))
FILENAME_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})-hero\.(?P<ext>jpe?g|png|webp)$",
    re.IGNORECASE,
)


def parse_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        # datetime-local values have no timezone; filenames are server local time.
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def list_images(start: dt.datetime | None, end: dt.datetime | None) -> list[dict]:
    result = []
    try:
        entries = IMAGE_DIR.iterdir()
    except OSError:
        return result
    for path in entries:
        match = FILENAME_RE.match(path.name)
        if not match or not path.is_file():
            continue
        timestamp = dt.datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S")
        if start and timestamp < start or end and timestamp > end:
            continue
        result.append({
            "name": path.name,
            "timestamp": timestamp.isoformat(sep=" "),
            "url": "/hero/image/" + quote(path.name),
            "size": path.stat().st_size,
        })
    result.sort(key=lambda item: item["timestamp"], reverse=True)
    return result


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hero 图片检索</title>
<style>
:root{color-scheme:dark}body{margin:0;background:#101522;color:#e8edf7;font:14px system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1400px;margin:0 auto;padding:24px}.toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:end;background:#182033;padding:16px;border-radius:12px;position:sticky;top:0;z-index:2}
label{display:grid;gap:6px;color:#aebbd3}input,button{font:inherit;border:1px solid #3c4b68;border-radius:7px;padding:9px 11px;background:#111827;color:#f2f5fb}button{background:#2563eb;border:0;cursor:pointer}button.secondary{background:#334155}.summary{color:#9fb0cb;margin:16px 2px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}.card{background:#182033;border-radius:10px;overflow:hidden;border:1px solid #273551}.card img{display:block;width:100%;height:420px;object-fit:contain;background:#080c14;cursor:zoom-in}.meta{padding:10px 12px}.name{font-size:13px;word-break:break-all}.time{color:#9fb0cb;margin-top:4px}.empty{padding:50px;text-align:center;color:#9fb0cb}
</style></head><body><main><h1>Hero 图片检索</h1>
<form class="toolbar" id="form"><label>开始时间<input id="start" type="datetime-local"></label><label>结束时间<input id="end" type="datetime-local"></label><button>筛选</button><button type="button" class="secondary" id="last48">最近 48 小时</button><button type="button" class="secondary" id="clear">清空</button></form>
<div class="summary" id="summary">正在加载…</div><section class="grid" id="grid"></section></main>
<script>
const $=id=>document.getElementById(id), pad=n=>String(n).padStart(2,'0');
function localInput(d){return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`}
function set48(){let e=new Date(),s=new Date(e-48*3600e3);$('start').value=localInput(s);$('end').value=localInput(e)}
async function load(){let p=new URLSearchParams(); if($('start').value)p.set('start',$('start').value); if($('end').value)p.set('end',$('end').value); $('summary').textContent='查询中…'; try{let r=await fetch('/hero/api/images?'+p); if(!r.ok)throw Error('HTTP '+r.status); let d=await r.json(); $('summary').textContent=`共 ${d.count} 张（按服务器文件名时间筛选）`; $('grid').innerHTML=d.images.map(x=>`<article class="card"><a href="${x.url}" target="_blank"><img loading="lazy" src="${x.url}" alt="${x.name}"></a><div class="meta"><div class="name">${x.name}</div><div class="time">${x.timestamp}</div></div></article>`).join('')||'<div class="empty">没有匹配图片</div>'}catch(e){$('summary').textContent='查询失败：'+e.message;$('grid').innerHTML=''} }
$('form').onsubmit=e=>{e.preventDefault();load()};$('last48').onclick=()=>{set48();load()};$('clear').onclick=()=>{$('start').value='';$('end').value='';load()};set48();load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "HeroGallery/1.0"

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path in ("/hero/", "/hero"):
            return self.send_bytes(PAGE.encode(), "text/html; charset=utf-8")
        if parsed.path == "/hero/api/images":
            query = parse_qs(parsed.query)
            start, end = parse_time(query.get("start", [None])[0]), parse_time(query.get("end", [None])[0])
            if query.get("start") and start is None or query.get("end") and end is None:
                return self.send_bytes(b'{"error":"invalid datetime"}', "application/json", HTTPStatus.BAD_REQUEST)
            if start and end and start > end:
                return self.send_bytes(b'{"error":"start must not be after end"}', "application/json", HTTPStatus.BAD_REQUEST)
            images = list_images(start, end)
            return self.send_bytes(json.dumps({"count": len(images), "images": images}, ensure_ascii=False).encode(), "application/json")
        if parsed.path.startswith("/hero/image/"):
            name = unquote(parsed.path[len("/hero/image/"):])
            match = FILENAME_RE.match(name)
            path = IMAGE_DIR / name
            if not match or path.parent != IMAGE_DIR or not path.is_file():
                return self.send_bytes(b"Not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
            try:
                return self.send_bytes(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            except OSError:
                return self.send_bytes(b"Read failed", "text/plain; charset=utf-8", HTTPStatus.INTERNAL_SERVER_ERROR)
        self.send_bytes(b"Not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:
        return


if __name__ == "__main__":
    ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler).serve_forever()
