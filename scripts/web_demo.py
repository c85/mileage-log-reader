"""Serve the local, mobile-friendly Form ML-7 demonstration interface."""

import argparse
import base64
import binascii
import json
import re
import socket
import sys
import threading
import uuid
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlreader import CONFIG_DIR, OUTPUT_DIR  # noqa: E402
from mlreader.pipeline import read_form  # noqa: E402
from mlreader.review import append_correction_audit, apply_correction  # noqa: E402
from mlreader.validation import validate_document  # noqa: E402


WEB_DIR = Path(__file__).resolve().parent.parent / "web"
INDEX_FILE = WEB_DIR / "index.html"
WEB_OUTPUT_DIR = OUTPUT_DIR / "web_demo"
MAX_REQUEST_BYTES = 22 * 1024 * 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SESSIONS = {}
SESSIONS_LOCK = threading.RLock()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _public_document(document):
    """Return display evidence without local filesystem paths or full logits."""
    visible = json.loads(json.dumps(document))
    visible.pop("model_path", None)
    visible.pop("evidence", None)
    for field in (
        list(visible.get("header", {}).values())
        + [field for row in visible.get("rows", []) for field in row.get("fields", {}).values()]
        + list(visible.get("footer", {}).values())
    ):
        for character in field.get("characters", []):
            character.pop("class_probabilities", None)
    return visible


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "MileageLogDemo/1.0"

    def log_message(self, format_string, *args):
        if urlsplit(self.path).path in {
            "/favicon.ico",
            "/apple-touch-icon.png",
            "/apple-touch-icon-precomposed.png",
        }:
            return
        print("%s - %s" % (self.address_string(), format_string % args))

    def _send_bytes(self, status, content_type, body, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Invalid Content-Length.")
        if length <= 0:
            raise ValueError("The request body is empty.")
        if length > MAX_REQUEST_BYTES:
            raise OverflowError("Upload is too large. Choose an image under 15 MB.")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("The request must contain valid JSON.")

    def _same_origin(self):
        origin = self.headers.get("Origin")
        host = self.headers.get("Host")
        return not origin or not host or urlsplit(origin).netloc.lower() == host.lower()

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path in {
            "/favicon.ico",
            "/apple-touch-icon.png",
            "/apple-touch-icon-precomposed.png",
        }:
            self.send_response(204)
            self.send_header("Cache-Control", "public, max-age=86400")
            self.end_headers()
            return
        if parsed.path in {"/", "/index.html"}:
            try:
                body = INDEX_FILE.read_bytes()
            except OSError:
                self._send_json(500, {"error": "The demo page could not be loaded."})
                return
            self._send_bytes(
                200,
                "text/html; charset=utf-8",
                body,
                {"Content-Security-Policy": "default-src 'self' data: blob:; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'"},
            )
            return
        if parsed.path == "/api/cell":
            self._serve_cell(parse_qs(parsed.query))
            return
        self._send_json(404, {"error": "Not found."})

    def _serve_cell(self, query):
        run_id = (query.get("run_id") or [""])[0]
        field_name = (query.get("field") or [""])[0]
        raw_position = (query.get("position") or [""])[0]
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            self._send_json(404, {"error": "Evidence not found."})
            return
        try:
            position = int(raw_position)
        except ValueError:
            self._send_json(404, {"error": "Evidence not found."})
            return
        with SESSIONS_LOCK:
            session = SESSIONS.get(run_id)
        if session is None:
            self._send_json(404, {"error": "This demo run is no longer available."})
            return
        key = f"{field_name}.{position}"
        evidence_path = session["document"].get("evidence", {}).get(key)
        if not evidence_path:
            self._send_json(404, {"error": "No crop is available for this cell."})
            return
        run_dir = Path(session["run_dir"]).resolve()
        crop_path = Path(evidence_path).resolve()
        if not crop_path.is_relative_to(run_dir) or not crop_path.is_file():
            self._send_json(404, {"error": "Evidence not found."})
            return
        self._send_bytes(200, "image/png", crop_path.read_bytes())

    def do_POST(self):
        if not self._same_origin():
            self._send_json(403, {"error": "Cross-origin requests are not accepted."})
            return
        parsed = urlsplit(self.path)
        try:
            if parsed.path == "/api/read":
                self._handle_read()
            elif parsed.path == "/api/correct":
                self._handle_correction()
            else:
                self._send_json(404, {"error": "Not found."})
        except OverflowError as error:
            self._send_json(413, {"error": str(error)})
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:
            print(f"Demo request failed: {type(error).__name__}: {error}")
            self._send_json(500, {"error": "The reader could not process this request. Check the server terminal for details."})

    def _handle_read(self):
        payload = self._read_json()
        data_url = payload.get("data_url", "")
        match = re.fullmatch(r"data:(image/(?:jpeg|png));base64,([A-Za-z0-9+/=]+)", data_url)
        if match is None:
            raise ValueError("Choose a JPEG or PNG image.")
        mime_type, encoded = match.groups()
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("The selected image could not be decoded.")
        if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            raise OverflowError("Choose an image under 15 MB.")
        decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("The selected file is not a supported JPEG or PNG image.")
        height, width = decoded.shape[:2]
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError("This image is unusually large. Choose a smaller photo.")

        suffix = ".jpg" if mime_type == "image/jpeg" else ".png"
        filename = str(payload.get("filename", "mileage-log" + suffix))
        display_name = filename.replace("\\", "/").split("/")[-1][:120] or "mileage-log" + suffix
        run_id = uuid.uuid4().hex
        run_dir = WEB_OUTPUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        image_path = run_dir / ("input" + suffix)
        image_path.write_bytes(image_bytes)
        references = _read_json(CONFIG_DIR / "reference_data.json")
        policy = _read_json(CONFIG_DIR / "reader_policy.json")
        today_value = payload.get("as_of")
        as_of = date.fromisoformat(today_value) if today_value else None
        document = read_form(
            image_path,
            references=references,
            policy=policy,
            evidence_dir=run_dir / "crops",
            today=as_of,
        )
        with SESSIONS_LOCK:
            SESSIONS[run_id] = {
                "document": document,
                "references": references,
                "policy": policy,
                "today": as_of,
                "run_dir": run_dir,
                "display_name": display_name,
            }
        self._send_json(
            200,
            {
                "run_id": run_id,
                "filename": display_name,
                "audit_file": f"outputs/web_demo/{run_id}/correction_audit.jsonl",
                "document": _public_document(document),
            },
        )

    def _handle_correction(self):
        payload = self._read_json()
        run_id = payload.get("run_id", "")
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("The demo run has expired. Analyze the image again.")
        field_path = str(payload.get("field_path", ""))
        corrected_value = str(payload.get("value", ""))
        reason = str(payload.get("reason", ""))
        if len(corrected_value) > 12:
            raise ValueError("The corrected value is too long for a form field.")
        if len(reason) > 500:
            raise ValueError("Keep the correction reason under 500 characters.")
        with SESSIONS_LOCK:
            session = SESSIONS.get(run_id)
            if session is None:
                raise ValueError("The demo run has expired. Analyze the image again.")
            document = session["document"]
            if not document.get("registration", {}).get("ok"):
                raise ValueError("Corrected fields are available after a page is registered.")
            correction = apply_correction(document, field_path, corrected_value, reason)
            validate_document(
                document,
                session["references"],
                session["policy"],
                today=session["today"],
            )
            audit = append_correction_audit(
                Path(session["run_dir"]) / "correction_audit.jsonl",
                correction,
                document,
                field_path,
            )
            document["status"] = document["validation"]["outcome"]
        self._send_json(
            200,
            {
                "run_id": run_id,
                "correction": correction,
                "audit_outcome": audit["review_outcome"],
                "audit_file": f"outputs/web_demo/{run_id}/correction_audit.jsonl",
                "document": _public_document(document),
            },
        )


def _lan_addresses():
    try:
        addresses = {
            result[4][0]
            for result in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
            if not result[4][0].startswith("127.")
        }
    except OSError:
        return []
    return sorted(addresses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (use 0.0.0.0 for phone access on the same trusted Wi-Fi).")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not INDEX_FILE.is_file():
        parser.error(f"Demo page is missing: {INDEX_FILE}")
    try:
        server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    except OSError as error:
        parser.error(f"Could not start the demo server: {error}")
    print("Form ML-7 browser demo is ready.")
    if args.host in {"0.0.0.0", "::"}:
        addresses = _lan_addresses()
        if addresses:
            for address in addresses:
                print(f"Open on this computer or a phone on the same Wi-Fi: http://{address}:{args.port}")
        else:
            print(f"Open http://<this-computer's-Wi-Fi-IP>:{args.port} from a phone on the same Wi-Fi.")
        print("Use synthetic or made-up forms only. Stop the server with Ctrl+C when the demo ends.")
    else:
        print(f"Open http://127.0.0.1:{args.port} in this computer's browser.")
        print("For phone-camera access on the same trusted Wi-Fi, restart with --host 0.0.0.0.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down the browser demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
