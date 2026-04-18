"""
Graph UI Server — Neural Second Brain Phase 4
=============================================
Lightweight local server for interactive graph visualization.

Run:
    python graph_ui_server.py
Open:
    http://127.0.0.1:8010
"""

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from graph_store import get_graph_snapshot, get_graph_version

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"
PORT = 8010
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, "graph_ui")


class GraphUIRequestHandler(BaseHTTPRequestHandler):
    server_version = "NSBGraphUI/1.0"

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, file_path: str, content_type: str = "text/plain; charset=utf-8"):
        if not os.path.exists(file_path):
            self._send_json({"error": "not found"}, status=404)
            return
        with open(file_path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        logger.info("GraphUI %s - %s", self.address_string(), fmt % args)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/graph":
            snapshot = get_graph_snapshot(node_limit=2500, edge_limit=5000)
            node_count = len(snapshot.get("nodes", []))
            edge_count = len(snapshot.get("edges", []))
            self._send_json(
                {
                    "ok": True,
                    "node_count": node_count,
                    "edge_count": edge_count,
                    "graph": snapshot,
                }
            )
            return

        if path == "/api/health":
            self._send_json({"ok": True, "service": "graph-ui", "port": PORT})
            return

        if path == "/api/version":
            self._send_json({"ok": True, "version": get_graph_version()})
            return

        if path in ("/", "/index.html"):
            self._send_file(os.path.join(UI_DIR, "index.html"), "text/html; charset=utf-8")
            return

        if path == "/app.js":
            self._send_file(os.path.join(UI_DIR, "app.js"), "application/javascript; charset=utf-8")
            return

        if path == "/styles.css":
            self._send_file(os.path.join(UI_DIR, "styles.css"), "text/css; charset=utf-8")
            return

        self._send_json({"error": "route not found", "path": path}, status=404)


def run_server(host: str = HOST, port: int = PORT):
    os.makedirs(UI_DIR, exist_ok=True)
    server = ThreadingHTTPServer((host, port), GraphUIRequestHandler)
    print(f"Graph UI server running at http://{host}:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("Graph UI server stopped")


if __name__ == "__main__":
    run_server()
