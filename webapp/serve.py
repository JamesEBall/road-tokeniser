"""Tiny static server with proper HTTP Range support, for testing PMTiles locally.

Python's built-in `http.server` returns 200 OK with the full body even when the
client sends a Range header — PMTiles needs 206 Partial Content. This script
adds the minimum needed for byte-range serving so MapLibre + the pmtiles plugin
can fetch the tile pyramid correctly.

Usage:
    python serve.py 8766
"""

from __future__ import annotations

import os
import re
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHandler(SimpleHTTPRequestHandler):
    # GeoJSON mime
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".geojson": "application/geo+json",
        ".pmtiles": "application/vnd.pmtiles",
    }

    def end_headers(self):
        # Always permit byte-range; helps the PMTiles client probe.
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=60")
        super().end_headers()

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        if not rng:
            return super().do_GET()

        m = RANGE_RE.match(rng)
        if not m:
            return super().do_GET()

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404, "File not found")
            return

        size = os.path.getsize(path)
        start = int(m.group(1)) if m.group(1) else 0
        end = int(m.group(2)) if m.group(2) else size - 1
        if start >= size:
            self.send_error(416, "Requested range not satisfiable")
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        end = min(end, size - 1)
        length = end - start + 1

        ctype = self.guess_type(path)
        self.send_response(206, "Partial Content")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()

        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main(port: int) -> int:
    addr = ("127.0.0.1", port)
    httpd = HTTPServer(addr, RangeHandler)
    print(f"Serving on http://{addr[0]}:{addr[1]} with HTTP Range support")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    raise SystemExit(main(port))
