import atexit
import json
import os
import queue
import socket
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _EventBus:
    def __init__(self, maxsize: int = 1000):
        self.q = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()
        self.last = []  # keep last N for quick replay
        self.max_replay = 200

    def put(self, evt: dict):
        data = json.dumps(evt, ensure_ascii=False)
        with self.lock:
            self.last.append(data)
            if len(self.last) > self.max_replay:
                self.last = self.last[-self.max_replay:]
        try:
            self.q.put_nowait(data)
        except queue.Full:
            # Drop oldest by draining one
            try:
                self.q.get_nowait()
            except Exception:
                pass
            try:
                self.q.put_nowait(data)
            except Exception:
                pass

    def stream(self):
        # generator yielding SSE frames; starts with a small replay
        with self.lock:
            replay = list(self.last[-60:])
        for d in replay:
            yield f"data: {d}\n\n"
        while True:
            try:
                d = self.q.get(timeout=1.0)
                yield f"data: {d}\n\n"
            except queue.Empty:
                # heartbeat
                yield f"data: {json.dumps({'ts': time.time(), 'type':'heartbeat'})}\n\n"


class _Handler(BaseHTTPRequestHandler):
    server_version = "HoundTelemetry/1.0"

    # Swallow common disconnect errors to avoid noisy tracebacks in the agent CLI
    def handle(self):  # noqa: N802
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            return
        except Exception:
            # Be conservative: do not crash the server thread on client hiccups
            return

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/events"):
            tok = self.server.token  # type: ignore[attr-defined]
            if not self._auth_ok(tok):
                self.send_error(HTTPStatus.UNAUTHORIZED)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                for chunk in self.server.bus.stream():  # type: ignore[attr-defined]
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except BrokenPipeError:
                return
            except Exception:
                return
            return
        elif self.path.startswith("/recent"):
            # Return last N events as JSON array (not SSE)
            tok = self.server.token  # type: ignore[attr-defined]
            if not self._auth_ok(tok):
                self.send_error(HTTPStatus.UNAUTHORIZED)
                return
            try:
                from urllib.parse import parse_qs, urlparse
                q = parse_qs(urlparse(self.path).query)
                try:
                    limit = int(q.get("limit", ["40"])[0])
                except Exception:
                    limit = 40
                with self.server.bus.lock:  # type: ignore[attr-defined]
                    items = list(self.server.bus.last[-limit:])  # type: ignore[attr-defined]
                out = []
                for s in items:
                    try:
                        out.append(json.loads(s))
                    except Exception:
                        pass
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"events": out}).encode("utf-8"))
            except Exception:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        elif self.path.startswith("/status"):
            tok = self.server.token  # type: ignore[attr-defined]
            if not self._auth_ok(tok):
                self.send_error(HTTPStatus.UNAUTHORIZED)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = {
                "project_id": self.server.project_id,  # type: ignore[attr-defined]
                "started_at": self.server.started_at,  # type: ignore[attr-defined]
                "ts": time.time(),
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):  # noqa: N802
        # Steering endpoint: POST /steer  { ... }
        if self.path.startswith("/steer"):
            tok = self.server.token  # type: ignore[attr-defined]
            if not self._auth_ok(tok):
                self.send_error(HTTPStatus.UNAUTHORIZED)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                obj = json.loads(body.decode("utf-8"))
            except Exception:
                obj = {}
            # Append to steering file in project dir
            try:
                pdir = Path(self.server.project_dir)  # type: ignore[attr-defined]
                sfile = pdir / ".hound" / "steering.jsonl"
                sfile.parent.mkdir(parents=True, exist_ok=True)
                with sfile.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": time.time(), "actor": "telemetry", **obj}) + "\n")
                self.server.bus.put({"type": "steer", "accepted": True, "payload": obj})  # type: ignore[attr-defined]
            except Exception:
                pass
            self.send_response(HTTPStatus.ACCEPTED)
            self.end_headers()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, fmt, *args):  # silence
        return

    def _auth_ok(self, tok: str) -> bool:
        # Accept token via query (?token=...) or header Authorization: Bearer <token>
        try:
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            if q.get("token", [None])[0] == tok:
                return True
        except Exception:
            pass
        if self.headers.get("Authorization", "").strip() == f"Bearer {tok}":
            return True
        return False


class TelemetryServer:
    def __init__(self, project_id: str, project_dir: Path, registry_dir: Path | None = None):
        self.project_id = project_id
        self.project_dir = Path(project_dir)
        # Respect HOUND_REGISTRY_DIR env var for consistency with the chatbot
        try:
            env_dir = os.environ.get("HOUND_REGISTRY_DIR")
        except Exception:
            env_dir = None
        default_dir = Path(os.path.expanduser("~/.local/state/hound/instances"))
        self.registry_dir = Path(registry_dir) if registry_dir else (Path(env_dir) if env_dir else default_dir)
        self.bus = _EventBus()
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.token = os.urandom(12).hex()
        self.started_at = time.time()
        self.registry_file: Path | None = None
        self.session_id: str | None = None

    def start(self):
        # Bind to localhost only
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()
        sock.close()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        # Attach attributes for handler access
        self.httpd.bus = self.bus  # type: ignore[attr-defined]
        self.httpd.project_id = self.project_id  # type: ignore[attr-defined]
        self.httpd.project_dir = str(self.project_dir)  # type: ignore[attr-defined]
        self.httpd.token = self.token  # type: ignore[attr-defined]
        self.httpd.started_at = self.started_at  # type: ignore[attr-defined]
        self.httpd.session_id = None  # type: ignore[attr-defined]

        self.thread = threading.Thread(target=self.httpd.serve_forever, name="hound-telemetry", daemon=True)
        self.thread.start()
        self._write_registry(port)
        atexit.register(self.stop)

    def _write_registry(self, port: int):
        try:
            self.registry_dir.mkdir(parents=True, exist_ok=True)
            pid = os.getpid()
            f = self.registry_dir / f"pid-{pid}.json"
            payload = {
                "pid": pid,
                "project_id": self.project_id,
                "started_at": self.started_at,
                "session_id": self.session_id,
                "telemetry": {
                    "sse_url": f"http://127.0.0.1:{port}/events",
                    "control_url": f"http://127.0.0.1:{port}",
                    "token": self.token,
                },
            }
            f.write_text(json.dumps(payload, indent=2))
            self.registry_file = f
        except Exception:
            self.registry_file = None

    def stop(self):
        # Remove registry entry and stop httpd
        try:
            if self.registry_file and self.registry_file.exists():
                self.registry_file.unlink()
        except Exception:
            pass
        if self.httpd:
            try:
                self.httpd.shutdown()
            except Exception:
                pass
            self.httpd.server_close()
            self.httpd = None

    def publish(self, evt: dict):
        try:
            evt = {"ts": time.time(), "project_id": self.project_id, **evt}
            self.bus.put(evt)
        except Exception:
            pass

    def set_session(self, session_id: str):
        self.session_id = session_id
        if self.httpd:
            self.httpd.session_id = session_id  # type: ignore[attr-defined]
