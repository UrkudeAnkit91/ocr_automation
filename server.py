"""
OCR Automation — FastAPI backend
Provides REST API (CRUD + OCR trigger) + WebSocket log streaming.
Run: python server.py
"""

import asyncio
import json
import logging
import sqlite3
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import ocr_automation as core

log = logging.getLogger("server")
DB_PATH = BASE_DIR / "records.db"
FIELD_NAMES = [
    "first_name","last_name","email","ssn","phone","bank_name",
    "account_no","loan_amount","address","city","state","zip",
    "dob","licence_no","licence_state","ip"
]

# ── Database ──

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT (datetime('now','localtime')),
                updated_at TEXT DEFAULT (datetime('now','localtime')),
                first_name TEXT, last_name TEXT, email TEXT, ssn TEXT,
                phone TEXT, bank_name TEXT, account_no TEXT, loan_amount TEXT,
                address TEXT, city TEXT, state TEXT, zip TEXT,
                dob TEXT, licence_no TEXT, licence_state TEXT, ip TEXT,
                raw_ocr TEXT
            )
        """)
        db.commit()

def row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)

# ── WebSocket log ──

log_clients: set[WebSocket] = set()
log_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

class WSLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        try:
            asyncio.get_running_loop().call_soon_threadsafe(
                log_queue.put_nowait, (record.levelname, msg))
        except RuntimeError:
            pass

_handler = WSLogHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger("ocr").addHandler(_handler)

async def broadcast_logs():
    while True:
        level, msg = await log_queue.get()
        dead = set()
        for ws in log_clients:
            try:
                await ws.send_json({"level": level, "message": msg})
            except Exception:
                dead.add(ws)
        log_clients -= dead

# ── Lifecycle ──

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    # Monkey-patch: auto-save after EVERY fill (API + mouse-click triggers)
    _orig_fill = core.do_ocr_fill
    def _patched_fill(cfg, region=None):
        _orig_fill(cfg, region)
        _save_after_fill()
    core.do_ocr_fill = _patched_fill

    asyncio.create_task(broadcast_logs())
    cfg = _load_safe()
    if cfg:
        core.start_mouse_listener(cfg)
        core.keyboard.add_hotkey(cfg.get("cancel_shortcut","esc"), core.cancel_op)
        core.keyboard.add_hotkey(cfg.get("learn_shortcut","ctrl+shift+c"), core.learn_correction)
        threading.Thread(target=core.warmup_ollama, args=(cfg,), daemon=True).start()
        log.info("OCR listeners started")
    yield
    if core._mouse_listener:
        core._mouse_listener.stop()

app = FastAPI(title="OCR Automation", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:4200","http://127.0.0.1:4200"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def _load_safe():
    try:
        return core.load_config()
    except SystemExit:
        return None

def _save_after_fill():
    """Called after OCR fill to auto-save record."""
    if not core._last_filled_values or len(core._last_filled_values) < 16:
        return
    vals = core._last_filled_values[:16]
    with get_db() as db:
        cur = db.execute(f"""
            INSERT INTO records ({','.join(FIELD_NAMES)}, raw_ocr)
            VALUES ({','.join(['?']*16)}, ?)
        """, [*vals, (core._last_ocr_text or "")[:500]])
        db.commit()
        record_id = cur.lastrowid
    log.info("Auto-saved record #%d with %d fields", record_id, sum(1 for v in vals if v))

# ── CRUD API ──

@app.get("/api/records")
async def list_records(search: Optional[str] = Query(None), limit: int = 100, offset: int = 0):
    with get_db() as db:
        if search:
            like = f"%{search}%"
            rows = db.execute(f"""
                SELECT id, created_at, first_name, last_name, email, ssn, phone, bank_name, state
                FROM records
                WHERE first_name LIKE ? OR last_name LIKE ? OR email LIKE ? OR ssn LIKE ? OR phone LIKE ?
                ORDER BY id DESC LIMIT ? OFFSET ?
            """, [like]*5 + [limit, offset]).fetchall()
            total = db.execute("SELECT COUNT(*) FROM records WHERE first_name LIKE ? OR last_name LIKE ? OR email LIKE ? OR ssn LIKE ? OR phone LIKE ?", [like]*5).fetchone()[0]
        else:
            rows = db.execute(f"""
                SELECT id, created_at, first_name, last_name, email, ssn, phone, bank_name, state
                FROM records ORDER BY id DESC LIMIT ? OFFSET ?
            """, [limit, offset]).fetchall()
            total = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    return {"records": [dict(r) for r in rows], "total": total, "limit": limit, "offset": offset}

@app.get("/api/records/{record_id}")
async def get_record(record_id: int):
    with get_db() as db:
        row = db.execute("SELECT * FROM records WHERE id=?", [record_id]).fetchone()
    if not row:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return row_to_dict(row)

@app.post("/api/records")
async def create_record(data: dict):
    vals = [data.get(f, "") for f in FIELD_NAMES]
    raw = data.get("raw_ocr", "")
    with get_db() as db:
        cur = db.execute(f"""
            INSERT INTO records ({','.join(FIELD_NAMES)}, raw_ocr)
            VALUES ({','.join(['?']*16)}, ?)
        """, [*vals, raw])
        db.commit()
        rid = cur.lastrowid
        row = db.execute("SELECT * FROM records WHERE id=?", [rid]).fetchone()
    log.info("Record #%d created", rid)
    return row_to_dict(row)

@app.put("/api/records/{record_id}")
async def update_record(record_id: int, data: dict):
    sets = ", ".join(f"{k}=?" for k in FIELD_NAMES if k in data)
    vals = [data[k] for k in FIELD_NAMES if k in data]
    if not sets:
        return JSONResponse({"error": "No fields to update"}, status_code=400)
    sets += ", updated_at=datetime('now','localtime')"
    with get_db() as db:
        db.execute(f"UPDATE records SET {sets} WHERE id=?", [*vals, record_id])
        db.commit()
        row = db.execute("SELECT * FROM records WHERE id=?", [record_id]).fetchone()
    if not row:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return row_to_dict(row)

@app.delete("/api/records/{record_id}")
async def delete_record(record_id: int):
    with get_db() as db:
        db.execute("DELETE FROM records WHERE id=?", [record_id])
        db.commit()
    log.info("Record #%d deleted", record_id)
    return {"status": "deleted"}

# ── OCR endpoints ──

@app.get("/api/status")
async def get_status():
    fields = core._last_filled_values[:] if core._last_filled_values else []
    return {
        "busy": core._busy,
        "last_ocr": (core._last_ocr_text or "")[:300],
        "last_fields": fields,
        "memory_count": len(core._brain.memory) if hasattr(core,"_brain") else 0,
        "neural_trained": core._brain.total_trained if hasattr(core,"_brain") else 0,
    }

@app.get("/api/config")
async def get_config():
    cfg = _load_safe()
    return cfg if cfg else JSONResponse({"error":"Config not found"},404)

@app.post("/api/trigger")
async def trigger_ocr():
    cfg = _load_safe()
    if not cfg:
        return JSONResponse({"error":"Config not found"},404)
    threading.Thread(target=_ocr_and_save, args=(cfg,), daemon=True).start()
    return {"status":"triggered"}

def _ocr_and_save(cfg: dict):
    core.do_ocr_fill(cfg)  # patch auto-saves via _patched_fill

@app.post("/api/learn")
async def trigger_learn():
    threading.Thread(target=core.learn_correction, daemon=True).start()
    return {"status":"learning"}

# ── WebSocket log ──

@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    await ws.accept()
    log_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        log_clients.discard(ws)

# ── Entry ──

if __name__ == "__main__":
    print("=" * 46)
    print("  OCR Automation Server")
    print("  http://localhost:8000")
    print("=" * 46)
    print()
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
