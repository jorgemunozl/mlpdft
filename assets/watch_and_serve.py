#!/usr/bin/env python3
"""
Typst Live Preview Server
Watches a typst document, compiles it to PDF, and serves it over HTTP
with live auto-reload for remote viewing (e.g. over local Ethernet/LAN).
"""

import argparse
import http.server
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from typing import Set

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Typst Live: {doc_name}</title>
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    body, html {{
      width: 100%;
      height: 100%;
      overflow: hidden;
      background-color: #121214;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      color: #e4e4e7;
    }}
    #topbar {{
      height: 42px;
      background: #18181b;
      border-bottom: 1px solid #27272a;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      font-size: 13px;
      user-select: none;
    }}
    .left-section {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .title-badge {{
      font-weight: 600;
      color: #f4f4f5;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .title-badge svg {{
      color: #06b6d4;
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 9px;
      border-radius: 9999px;
      background: #27272a;
      font-size: 12px;
      color: #a1a1aa;
    }}
    .dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      transition: background 0.2s ease;
    }}
    .dot.connected {{
      background: #10b981;
      box-shadow: 0 0 6px rgba(16, 185, 129, 0.6);
    }}
    .dot.updating {{
      background: #f59e0b;
      box-shadow: 0 0 6px rgba(245, 158, 11, 0.6);
    }}
    .dot.disconnected {{
      background: #ef4444;
      box-shadow: 0 0 6px rgba(239, 68, 68, 0.6);
    }}
    .right-section {{
      display: flex;
      align-items: center;
      gap: 14px;
      font-size: 12px;
      color: #71717a;
    }}
    .btn {{
      background: #27272a;
      color: #d4d4d8;
      border: 1px solid #3f3f46;
      padding: 4px 10px;
      border-radius: 6px;
      cursor: pointer;
      text-decoration: none;
      font-size: 12px;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }}
    .btn:hover {{
      background: #3f3f46;
      color: #ffffff;
      border-color: #52525b;
    }}
    #viewer-container {{
      width: 100%;
      height: calc(100% - 42px);
      position: relative;
    }}
    iframe {{
      width: 100%;
      height: 100%;
      border: none;
      background: #525659;
    }}
  </style>
</head>
<body>
  <div id="topbar">
    <div class="left-section">
      <div class="title-badge">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
        {doc_name}
      </div>
      <div class="status-pill">
        <span id="statusDot" class="dot connected"></span>
        <span id="statusText">Connected</span>
      </div>
    </div>
    <div class="right-section">
      <span id="lastUpdated">Updated: Just now</span>
      <button class="btn" onclick="forceReload()">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 4v6h-6"></path><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
        Refresh (R)
      </button>
      <a class="btn" href="/pdf" target="_blank">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
        Direct PDF
      </a>
      <button class="btn" onclick="toggleFullscreen()">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path></svg>
        Fullscreen
      </button>
    </div>
  </div>
  <div id="viewer-container">
    <iframe id="pdfFrame" src="/pdf"></iframe>
  </div>

  <script>
    const iframe = document.getElementById('pdfFrame');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const lastUpdated = document.getElementById('lastUpdated');

    function updatePdf() {{
      statusDot.className = 'dot updating';
      statusText.textContent = 'Updating...';
      const url = '/pdf?v=' + Date.now();
      try {{
        iframe.contentWindow.location.replace(url);
      }} catch (e) {{
        iframe.src = url;
      }}
      setTimeout(() => {{
        statusDot.className = 'dot connected';
        statusText.textContent = 'Live Reload Active';
        lastUpdated.textContent = 'Updated: ' + new Date().toLocaleTimeString();
      }}, 300);
    }}

    function forceReload() {{
      updatePdf();
    }}

    function toggleFullscreen() {{
      if (!document.fullscreenElement) {{
        document.documentElement.requestFullscreen();
      }} else {{
        if (document.exitFullscreen) {{
          document.exitFullscreen();
        }}
      }}
    }}

    window.addEventListener('keydown', (e) => {{
      if (e.key === 'r' || e.key === 'R') {{
        forceReload();
      }}
    }});

    function setupSSE() {{
      const evtSource = new EventSource('/events');
      evtSource.onopen = () => {{
        statusDot.className = 'dot connected';
        statusText.textContent = 'Live Reload Active';
      }};
      evtSource.onmessage = (e) => {{
        if (e.data === 'reload') {{
          updatePdf();
        }}
      }};
      evtSource.onerror = () => {{
        statusDot.className = 'dot disconnected';
        statusText.textContent = 'Reconnecting...';
      }};
    }}

    setupSSE();
  </script>
</body>
</html>
"""

subscribers: Set[queue.Queue] = set()
subscribers_lock = threading.Lock()
pdf_file_path = ""
doc_filename = ""

class PDFRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress routine log messages to keep terminal clean
        pass

    def do_GET(self):
        global pdf_file_path, doc_filename
        path = self.path.split('?')[0]

        if path == '/' or path == '/index.html':
            content = HTML_TEMPLATE.format(doc_name=doc_filename).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(content)
            return

        elif path == '/pdf':
            if not os.path.exists(pdf_file_path):
                self.send_error(404, "PDF has not been compiled yet")
                return

            try:
                with open(pdf_file_path, 'rb') as f:
                    pdf_data = f.read()
            except Exception as e:
                self.send_error(500, f"Error reading PDF: {e}")
                return

            self.send_response(200)
            self.send_header('Content-Type', 'application/pdf')
            self.send_header('Content-Length', str(len(pdf_data)))
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.send_header('Accept-Ranges', 'bytes')
            self.end_headers()
            self.wfile.write(pdf_data)
            return

        elif path == '/events':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            q = queue.Queue()
            with subscribers_lock:
                subscribers.add(q)

            # Send initial handshake ping
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    msg = q.get()
                    self.wfile.write(f"data: {msg}\n\n".encode('utf-8'))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with subscribers_lock:
                    subscribers.discard(q)
            return

        else:
            self.send_error(404, "Not Found")


def get_local_ips():
    """Retrieve non-loopback IPv4 addresses."""
    ips = []
    try:
        output = subprocess.check_output(["ip", "-4", "-br", "addr"], text=True)
        for line in output.strip().splitlines():
            parts = line.split()
            if len(parts) >= 3 and not parts[0].startswith('lo'):
                for addr in parts[2:]:
                    ip = addr.split('/')[0]
                    ips.append((parts[0], ip))
    except Exception:
        # Fallback
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ips.append(("default", s.getsockname()[0]))
        except Exception:
            pass
        finally:
            s.close()
    return ips


def sync_available_images(doc_dir):
    """Scans doc_dir for existing image files and updates available_images.json."""
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.svg', '.gif', '.pdf'}
    manifest_path = os.path.join(doc_dir, "available_images.json")
    found_images = []
    for root, _, files in os.walk(doc_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS and f != os.path.basename(manifest_path):
                rel_path = os.path.relpath(os.path.join(root, f), doc_dir)
                found_images.append(rel_path)
    
    found_images.sort()
    new_data = json.dumps(found_images, indent=2) + "\n"
    old_data = ""
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as fp:
                old_data = fp.read()
        except Exception:
            pass
    if new_data != old_data:
        try:
            with open(manifest_path, 'w') as fp:
                fp.write(new_data)
        except Exception:
            pass


def file_monitor(target_file, doc_dir):
    last_mtime = 0.0
    if os.path.exists(target_file):
        last_mtime = os.path.getmtime(target_file)

    sync_counter = 0
    while True:
        time.sleep(0.15)
        sync_counter += 1
        # Check for new/removed images once every ~1.5s
        if sync_counter % 10 == 0:
            sync_available_images(doc_dir)

        try:
            if os.path.exists(target_file):
                current_mtime = os.path.getmtime(target_file)
                if current_mtime > last_mtime:
                    # Debounce to ensure file write completes
                    time.sleep(0.08)
                    if os.path.getsize(target_file) > 0:
                        last_mtime = os.path.getmtime(target_file)
                        with subscribers_lock:
                            for q in list(subscribers):
                                q.put("reload")
        except Exception:
            pass


def main():
    global pdf_file_path, doc_filename

    parser = argparse.ArgumentParser(description="Typst Live Preview Server over Network")
    parser.add_argument("input", nargs="?", default="main.typ", help="Path to input typst file (default: main.typ)")
    parser.add_argument("--port", "-p", type=int, default=8000, help="HTTP server port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--no-watch", action="store_true", help="Do not spawn 'typst watch' subprocess (only serve and monitor existing PDF)")
    args = parser.parse_args()

    input_file = os.path.abspath(args.input)
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    doc_dir = os.path.dirname(input_file)
    doc_filename = os.path.basename(input_file)
    pdf_filename = os.path.splitext(doc_filename)[0] + ".pdf"
    pdf_file_path = os.path.join(doc_dir, pdf_filename)

    # Pre-sync existing images into available_images.json before starting compilation
    sync_available_images(doc_dir)

    # Start typst watch subprocess unless disabled
    typst_proc = None
    if not args.no_watch:
        cmd = ["typst", "watch", input_file, pdf_file_path]
        print(f"-> Starting: {' '.join(cmd)}")
        typst_proc = subprocess.Popen(cmd)

    # Start file monitor thread
    monitor_thread = threading.Thread(target=file_monitor, args=(pdf_file_path, doc_dir), daemon=True)
    monitor_thread.start()

    # Network info
    ips = get_local_ips()
    print("\n" + "="*60)
    print("  Typst Live Preview Server Running")
    print("="*60)
    print(f"  Source file : {input_file}")
    print(f"  Target PDF  : {pdf_file_path}")
    print("\n  Open this in the browser on your other device:")
    
    # Highlight ethernet / 169.254 IP first if found
    for iface, ip in ips:
        star = " -> " if "169.254" in ip else "    "
        print(f"{star}http://{ip}:{args.port}  ({iface})")
    
    print("\n  Press Ctrl+C to stop.")
    print("="*60 + "\n")

    server = http.server.ThreadingHTTPServer((args.host, args.port), PDFRequestHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
        if typst_proc:
            typst_proc.terminate()
            typst_proc.wait()


if __name__ == "__main__":
    main()
