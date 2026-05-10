#!/usr/bin/env python3
"""
start.py — Wildlife & Fire Detection System Dashboard
═══════════════════════════════════════════════════════
Kills any stale server, launches app.py, and renders a live
terminal dashboard showing all events with zero duplicates.

Usage:
    python start.py
    python start.py --port 9000
"""

import subprocess
import sys
import os
import re
import signal
import time
import threading
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque, OrderedDict

# ── Resolve paths ──────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
APP_PY    = BASE_DIR / "app.py"
VENV_PY   = BASE_DIR / ".venv" / "bin" / "python"
PYTHON    = str(VENV_PY) if VENV_PY.exists() else sys.executable

# ── Parse CLI args ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Start the detection system with live dashboard")
parser.add_argument("--port", type=int, default=9000, help="Flask server port (default: 9000)")
ARGS = parser.parse_args()
APP_PORT = ARGS.port

# ── Import rich ────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("[ERROR] 'rich' not found. Install it: pip install rich")
    sys.exit(1)

console = Console()

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED DASHBOARD STATE
# ══════════════════════════════════════════════════════════════════════════════
class DashboardState:
    def __init__(self):
        self.lock = threading.Lock()

        # System
        self.server_running  = False
        self.server_url      = ""
        self.model_loaded    = False
        self.model_classes   = []
        self.sms_enabled     = False
        self.start_time      = datetime.now()
        self.error_msg       = ""

        # Twilio config (shown in status panel)
        self.twilio_from     = ""
        self.twilio_to       = ""
        self.fire_cd         = ""
        self.animal_cd       = ""
        self.confirm_frames  = ""
        self.batch_win       = ""

        # Detection events — deduplicated by (type, labels)
        # key: (str, tuple) → {first_ts, last_ts, source, count, sms_sent}
        self.events: OrderedDict = OrderedDict()

        # SMS log — each dict: {time, type, labels, sid}
        self.sms_log = deque(maxlen=50)

        # Filtered log lines (no spam, deduplicated)
        self.log_lines   = deque(maxlen=40)
        self._last_log   = ""

        # Live detection active flags (cleared when gone from frame)
        self.fire_active   = False
        self.animal_active = False

ds = DashboardState()

# ══════════════════════════════════════════════════════════════════════════════
#  LOG PARSER
# ══════════════════════════════════════════════════════════════════════════════
# Lines that are noisy/spammy and should never appear in the log panel
_LOG_SKIP = re.compile(
    r"GET /status|GET /video_feed|127\.0\.0\.1|SMS-DBG|"
    r"^\s*$|INACTIVE"
)

PATTERNS = {
    "server_url":      re.compile(r"Running on http://(?!0\.0\.0\.0)(\S+)"),
    "model_loaded":    re.compile(r"Model loaded.*Classes: (.+)"),
    "sms_ready":       re.compile(r"Overall ready\s*:\s*(.+)"),
    "twilio_from":     re.compile(r"FROM number\s*:\s*[✅\u2713ok\s]+(\+\d+)"),
    "twilio_to":       re.compile(r"TO number\s*:\s*[✅\u2713ok\s]+(\+\d+)"),
    "fire_cd":         re.compile(r"Fire cooldown\s*:\s*(\S+)"),
    "animal_cd":       re.compile(r"Animal cooldown\s*:\s*(\S+)"),
    "confirm_frames":  re.compile(r"Confirm frames\s*:\s*(\S+)"),
    "batch_win":       re.compile(r"Animal batch win\s*:\s*(\S+)"),
    "fire_det":        re.compile(r"\[INFO\] FIRE! (\S+) \((\d+\.\d+)\)"),
    "animal_det":      re.compile(r"\[INFO\] ANIMAL! (\S+) \((\d+\.\d+)\)"),
    "sms_dispatch":    re.compile(r"\[SMS\] Dispatching (.+?) alert.*labels=\[(.+?)\] source=(\S+)"),
    "sms_sent":        re.compile(r"\[INFO\] \[(.+?)\] SMS sent\. SID: (\S+) \| Labels: (.+)"),
    "perm_err":        re.compile(r"Permission denied"),
    "port_err":        re.compile(r"Address already in use"),
}


def _upsert_event(etype: str, labels: tuple, source: str, ts: str, sms_sent: bool = False):
    """Insert or update a detection event (deduplication by type+labels)."""
    key = (etype, tuple(sorted(str(l) for l in labels)))
    if key in ds.events:
        ev = ds.events[key]
        ev["count"]  += 1
        ev["last_ts"] = ts
        if sms_sent:
            ev["sms_sent"] = True
        ds.events.move_to_end(key)          # bring to top (most recent)
    else:
        ds.events[key] = {
            "first_ts": ts,
            "last_ts":  ts,
            "source":   source,
            "count":    1,
            "sms_sent": sms_sent,
        }
        ds.events.move_to_end(key)


def _add_log(ts: str, line: str):
    """Append a log line only if not identical to the previous one."""
    entry = f"[{ts}] {line}"
    if entry != ds._last_log:
        ds.log_lines.append(entry)
        ds._last_log = entry


def parse_line(raw: str):
    line = raw.rstrip("\r\n")
    if not line:
        return
    ts = datetime.now().strftime("%H:%M:%S")

    with ds.lock:
        # ── Server URL ────────────────────────────────────────────────────
        m = PATTERNS["server_url"].search(line)
        if m:
            ds.server_url    = m.group(1)
            ds.server_running = True

        # ── Model loaded ──────────────────────────────────────────────────
        m = PATTERNS["model_loaded"].search(line)
        if m:
            ds.model_loaded = True
            ds.model_classes = re.findall(r"'([^']+)'", m.group(1))

        # ── SMS ready ─────────────────────────────────────────────────────
        m = PATTERNS["sms_ready"].search(line)
        if m:
            ds.sms_enabled = "ENABLED" in m.group(1)

        # ── Twilio config values ──────────────────────────────────────────
        for attr, key in [
            ("twilio_from", "twilio_from"), ("twilio_to", "twilio_to"),
            ("fire_cd", "fire_cd"), ("animal_cd", "animal_cd"),
            ("confirm_frames", "confirm_frames"), ("batch_win", "batch_win"),
        ]:
            m = PATTERNS[key].search(line)
            if m:
                setattr(ds, attr, m.group(1).strip())

        # ── Fire detected ─────────────────────────────────────────────────
        m = PATTERNS["fire_det"].search(line)
        if m:
            label, conf = m.group(1), float(m.group(2))
            ds.fire_active = True
            _upsert_event("fire", (label,), "camera", ts)

        # ── Animal detected ───────────────────────────────────────────────
        m = PATTERNS["animal_det"].search(line)
        if m:
            label, conf = m.group(1), float(m.group(2))
            ds.animal_active = True
            _upsert_event("animal", (label,), "camera", ts)

        # ── SMS dispatched ────────────────────────────────────────────────
        m = PATTERNS["sms_dispatch"].search(line)
        if m:
            atype, labels_raw, source = m.group(1), m.group(2), m.group(3)
            labels = tuple(
                l.strip().strip("'\" ") for l in labels_raw.split(",")
            )
            _upsert_event(atype, labels, source, ts, sms_sent=True)

        # ── SMS sent confirmation ─────────────────────────────────────────
        m = PATTERNS["sms_sent"].search(line)
        if m:
            atype, sid, labels = m.group(1), m.group(2), m.group(3)
            # Avoid duplicate SMS log entries
            if not ds.sms_log or ds.sms_log[-1].get("sid") != sid:
                ds.sms_log.append({
                    "time":   ts,
                    "type":   atype,
                    "labels": labels,
                    "sid":    sid,
                })

        # ── Errors ───────────────────────────────────────────────────────
        if PATTERNS["perm_err"].search(line):
            ds.error_msg = "Permission denied on port — try a different port"
        if PATTERNS["port_err"].search(line):
            ds.error_msg = f"Port {APP_PORT} already in use — start.py will retry"

        # ── Filtered log output ───────────────────────────────────────────
        if not _LOG_SKIP.search(line):
            _add_log(ts, line)


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def _status_icon(ok: bool, label_ok: str, label_no: str) -> Text:
    t = Text()
    if ok:
        t.append("✅ ", style="bold green")
        t.append(label_ok, style="green")
    else:
        t.append("❌ ", style="bold red")
        t.append(label_no, style="red")
    return t


def build_dashboard() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header",  size=3),
        Layout(name="body"),
        Layout(name="footer",  size=3),
    )
    layout["body"].split_row(
        Layout(name="left",  ratio=3),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="status",  size=14),
        Layout(name="events"),
    )
    layout["right"].split_column(
        Layout(name="sms",  ratio=2),
        Layout(name="log",  ratio=3),
    )

    with ds.lock:
        now_str  = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        uptime   = str(datetime.now() - ds.start_time).split(".")[0]
        live_str = ""
        if ds.fire_active:
            live_str += "  🔥 FIRE LIVE"
        if ds.animal_active:
            live_str += "  🐾 WILDLIFE LIVE"

        # ── HEADER ────────────────────────────────────────────────────────
        hdr = Text(justify="center")
        hdr.append(
            "🔥  WILDLIFE & FIRE ANOMALY DETECTION SYSTEM  🐾",
            style="bold white on dark_red",
        )
        if live_str:
            hdr.append(live_str, style="bold yellow on dark_red")
        layout["header"].update(Panel(hdr, border_style="bright_red"))

        # ── STATUS PANEL ──────────────────────────────────────────────────
        st = Table.grid(padding=(0, 2))
        st.add_column(style="bold cyan", no_wrap=True, width=16)
        st.add_column(min_width=30)

        st.add_row("Server",
            _status_icon(ds.server_running,
                         f"Running  http://{ds.server_url}", "Not started yet"))
        st.add_row("Model",
            _status_icon(ds.model_loaded, "YOLO11 loaded", "Loading…"))
        if ds.model_classes:
            st.add_row("Classes", Text(", ".join(ds.model_classes), style="dim"))
        st.add_row("SMS / Twilio",
            _status_icon(ds.sms_enabled, "ENABLED", "DISABLED"))
        if ds.twilio_from:
            st.add_row("Phone  From→To",
                Text(f"{ds.twilio_from}  →  {ds.twilio_to}", style="dim"))
        if ds.fire_cd:
            st.add_row("Fire cooldown",   Text(ds.fire_cd, style="dim"))
        if ds.animal_cd:
            st.add_row("Animal cooldown", Text(ds.animal_cd, style="dim"))
        if ds.confirm_frames:
            st.add_row("Confirm frames",  Text(ds.confirm_frames, style="dim"))
        if ds.batch_win:
            st.add_row("Animal batch",    Text(ds.batch_win, style="dim"))
        st.add_row("Uptime",  Text(uptime, style="bold"))
        if ds.error_msg:
            st.add_row("⚠  ERROR", Text(ds.error_msg, style="bold red"))

        layout["status"].update(
            Panel(st, title="[bold cyan]SYSTEM STATUS[/]", border_style="cyan")
        )

        # ── EVENTS TABLE ──────────────────────────────────────────────────
        ev_tbl = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        ev_tbl.add_column("First",  width=8,  no_wrap=True)
        ev_tbl.add_column("Last",   width=8,  no_wrap=True)
        ev_tbl.add_column("Type",   width=12, no_wrap=True)
        ev_tbl.add_column("Labels", min_width=22)
        ev_tbl.add_column("Src",    width=6,  no_wrap=True)
        ev_tbl.add_column("SMS",    width=4,  no_wrap=True)
        ev_tbl.add_column("#",      width=5,  no_wrap=True, justify="right")

        events_list = list(ds.events.items())[-25:]
        for (etype, labels), ev in reversed(events_list):
            if etype == "fire":
                t_txt = Text("🔥 FIRE",    style="bold red")
            elif etype == "animal":
                t_txt = Text("🐾 WILDLIFE", style="bold yellow")
            else:
                t_txt = Text("🔥🐾 BOTH",   style="bold orange1")

            sms_icon = Text("✅", style="green") if ev["sms_sent"] else Text("⏳", style="dim")
            label_str = ", ".join(l.upper() for l in labels)
            count_str = Text(str(ev["count"]), style="bold" if ev["count"] > 1 else "")

            ev_tbl.add_row(
                ev["first_ts"], ev["last_ts"],
                t_txt, label_str,
                ev["source"][:6], sms_icon, count_str,
            )

        if not ds.events:
            ev_tbl.add_row(
                "—", "—", Text("—", style="dim"),
                Text("No detections yet", style="dim italic"),
                "—", "—", "0",
            )

        layout["events"].update(
            Panel(ev_tbl, title="[bold magenta]DETECTION EVENTS[/]", border_style="magenta")
        )

        # ── SMS LOG ───────────────────────────────────────────────────────
        sms_tbl = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold green",
            expand=True,
        )
        sms_tbl.add_column("Time",   width=8,  no_wrap=True)
        sms_tbl.add_column("Type",   width=12, no_wrap=True)
        sms_tbl.add_column("Labels", min_width=18)
        sms_tbl.add_column("SID",    min_width=18, no_wrap=True)

        sms_list = list(ds.sms_log)
        for alert in reversed(sms_list):
            sms_tbl.add_row(
                alert["time"],
                Text(alert["type"], style="bold green"),
                alert["labels"],
                Text(alert["sid"][:22], style="dim"),
            )
        if not sms_list:
            sms_tbl.add_row("—", "—", Text("No SMS sent yet", style="dim italic"), "—")

        layout["sms"].update(
            Panel(sms_tbl, title="[bold green]SMS ALERTS SENT[/]", border_style="green")
        )

        # ── LOG PANEL ─────────────────────────────────────────────────────
        log_text = Text()
        for ll in list(ds.log_lines)[-20:]:
            if "ERROR" in ll or "error" in ll:
                log_text.append(ll + "\n", style="bold red")
            elif "WARN" in ll:
                log_text.append(ll + "\n", style="yellow")
            elif "INFO" in ll or "SMS" in ll:
                log_text.append(ll + "\n", style="white")
            else:
                log_text.append(ll + "\n", style="dim")

        layout["log"].update(
            Panel(log_text, title="[bold white]SYSTEM LOG[/]", border_style="white")
        )

        # ── FOOTER ────────────────────────────────────────────────────────
        total_events = len(ds.events)
        total_sms    = len(sms_list)
        ftr = Text(justify="center")
        ftr.append(
            f"Ctrl+C to stop  │  Port: {APP_PORT}  │  "
            f"Events: {total_events}  │  SMS Sent: {total_sms}  │  {now_str}",
            style="dim",
        )
        layout["footer"].update(Panel(ftr, border_style="dim"))

    return layout


# ══════════════════════════════════════════════════════════════════════════════
#  PORT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def kill_port(port: int):
    """Kill any process currently listening on the given port."""
    try:
        result = subprocess.run(
            ["fuser", f"{port}/tcp"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                console.print(f"[yellow]⚠  Killed stale process PID {pid} on port {port}[/]")
                time.sleep(0.5)
            except (ValueError, ProcessLookupError):
                pass
    except FileNotFoundError:
        pass   # fuser not available on this system


# ══════════════════════════════════════════════════════════════════════════════
#  SUBPROCESS READER THREAD
# ══════════════════════════════════════════════════════════════════════════════
_proc: subprocess.Popen | None = None


def reader_thread(proc: subprocess.Popen):
    """Read app.py stdout/stderr line by line and feed into the parser."""
    for raw in proc.stdout:
        parse_line(raw)
    with ds.lock:
        ds.server_running = False
        _add_log(datetime.now().strftime("%H:%M:%S"), "⚠  app.py process ended")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    global _proc

    console.print(Panel.fit(
        "[bold white]🔥🐾  Wildlife & Fire Detection System  🐾🔥[/]\n"
        f"[dim]Starting app.py on port {APP_PORT}...[/]",
        border_style="bright_red",
    ))

    # Kill anything on the port first
    kill_port(APP_PORT)
    time.sleep(0.3)

    if not APP_PY.exists():
        console.print(f"[bold red]ERROR: {APP_PY} not found.[/]")
        sys.exit(1)

    env = os.environ.copy()
    env["FLASK_PORT"] = str(APP_PORT)  # in case app.py reads this

    # Launch app.py
    _proc = subprocess.Popen(
        [PYTHON, str(APP_PY)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(BASE_DIR),
        env=env,
    )

    # Start reader thread
    t = threading.Thread(target=reader_thread, args=(_proc,), daemon=True)
    t.start()

    def shutdown(sig=None, frame=None):
        console.print("\n[yellow]Stopping server…[/]")
        if _proc and _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _proc.kill()
        console.print("[bold green]✅ System stopped cleanly.[/]")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Live dashboard loop
    with Live(
        build_dashboard(),
        console=console,
        refresh_per_second=2,
        screen=True,
    ) as live:
        while True:
            if _proc.poll() is not None:
                # Process died — show final state for 3s then exit
                live.update(build_dashboard())
                time.sleep(3)
                break
            live.update(build_dashboard())
            time.sleep(0.5)

    shutdown()


if __name__ == "__main__":
    main()
