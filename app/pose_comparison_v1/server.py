"""Yerel pose karşılaştırma arayüzü. Mevcut ortamla çalışır; paket kurmaz.

Çalıştırma: .venv_mediapipe/bin/python -B app/pose_comparison_v1/server.py
Adres: http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import mimetypes
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

APP = Path(__file__).resolve().parent
ROOT = APP.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pose_video_comparison_v1 as engine

DATA = ROOT / "outputs/library_comparison_app"
EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts"}
TOKEN = secrets.token_urlsafe(24)
LOCK = threading.RLock()
VIDEOS, MEDIA, JOBS, RESULTS = {}, {}, [], OrderedDict()
CANCEL = threading.Event()
WAKE = threading.Event()
SHUTDOWN = threading.Event()


def cache_result(result_id, result):
    """Keep only the two most recently viewed results in server memory."""
    with LOCK:
        RESULTS.pop(result_id, None)
        RESULTS[result_id] = result
        while len(RESULTS) > 2:
            RESULTS.pop(next(iter(RESULTS)))


def protocol_key(job):
    return (job["frame_step"], job.get("sample_limit"), tuple(sorted(job["libraries"])))


def model_signature(result):
    provenance = result.get("provenance", {})
    fingerprints = {path: entry.get("sha256") for path, entry in provenance.get("files", {}).items()}
    return hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()[:8] if fingerprints else None


def active_duplicate(video_id, key, exclude_id=None):
    return any(j["id"] != exclude_id and j["video_id"] == video_id
               and j["status"] in ("queued", "running") and protocol_key(j) == key for j in JOBS)


def batch_summary():
    """Latest attempt per video/protocol; sum counts rather than mean percentages.

    Different sampling protocols and library sets remain separate. Historical
    image pilots are never included. A latest failed attempt stays visible.
    """
    with LOCK:
        jobs = copy.deepcopy(JOBS)
    latest = {}
    for job in jobs:
        if job["status"] in ("queued", "running"):
            continue
        if (not job.get("summary") or "model_signature" not in job) and job.get("result_path") and Path(job["result_path"]).is_file():
            result = json.loads(Path(job["result_path"]).read_text())
            job["summary"] = result.get("summary", [])
            job["library_elapsed"] = {lib: row.get("elapsed_sec") for lib, row in result.get("library_status", {}).items()}
            job["model_signature"] = model_signature(result)
        latest[((protocol_key(job), job.get("model_signature")), job["video_id"])] = job
    grouped = {}
    for ((protocol, signature), _), job in latest.items():
        lookup = {row["library"]: row for row in job.get("summary", [])}
        step, limit, libraries = protocol
        abbreviations = dict(mediapipe="MP", openpose="OP", mmpose="MM", deepercut="DC")
        label = f"{step} kare · " + (f"ilk {limit} örnek" if limit else "tüm video")
        label += " · " + "/".join(abbreviations[lib] for lib in engine.LIBRARIES if lib in libraries)
        label += " · " + (signature[:4] if signature else "eski kayıt")
        for library in engine.LIBRARIES:
            if library not in libraries:
                continue
            key = (protocol, signature, library)
            if key not in grouped:
                grouped[key] = dict(library=library, protocol_label=label, frame_step=step,
                    sample_limit=limit, model_signature=signature, videos=0, sampled_frames=0, processed_frames=0,
                    native_available_total=0, native_expected_total=0,
                    common_available_total=0, common_expected_total=0,
                    common_in_image_total=0, out_of_image_total=0, low_score_total=0,
                    error_frames=0, cancelled_frames=0, no_pose_frames=0,
                    failed_videos=0, partial_videos=0, limited_videos=0,
                    elapsed_seconds=0.0, timing_available=True)
            target = grouped[key]
            target["videos"] += 1
            target["failed_videos"] += job["status"] in ("error", "cancelled", "interrupted")
            target["partial_videos"] += job["status"] == "partial"
            target["limited_videos"] += limit is not None
            row = lookup.get(library, {})
            for field in ("sampled_frames", "processed_frames", "native_available_total", "native_expected_total",
                          "common_available_total", "common_expected_total", "error_frames", "cancelled_frames", "no_pose_frames"):
                target[field] += row.get(field, 0) or 0
            # Older results contain point lists but not these new totals; mark
            # unavailable diagnostics instead of inventing zero observations.
            for dest, source in (("common_in_image_total", "common_in_image_total"),
                                 ("out_of_image_total", "native_out_of_image_total"),
                                 ("low_score_total", "low_score_total")):
                value = row.get(source)
                if value is None or target[dest] is None:
                    target[dest] = None
                else:
                    target[dest] += value
            elapsed = job.get("library_elapsed", {}).get(library)
            if elapsed is not None:
                target["elapsed_seconds"] += elapsed
            else:
                target["timing_available"] = False
    rows = list(grouped.values())
    for row in rows:
        for scope in ("native", "common"):
            denominator = row[scope+"_expected_total"]
            row[scope+"_coverage_pct"] = 100*row[scope+"_available_total"]/denominator if denominator else None
        denominator = row["common_expected_total"]
        row["common_in_image_pct"] = (100*row["common_in_image_total"]/denominator
            if denominator and row["common_in_image_total"] is not None else None)
        if not row.pop("timing_available"):
            row["elapsed_seconds"] = None
    downloads = {}
    if rows:
        payload = io.StringIO(newline="")
        writer = csv.DictWriter(payload, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
        content = payload.getvalue().encode("utf-8-sig")
        folder = DATA / "exports"; folder.mkdir(parents=True, exist_ok=True)
        path = folder / ("toplu_ozet_" + hashlib.sha256(content).hexdigest()[:12] + ".csv")
        with LOCK:
            if not path.exists():
                path.write_bytes(content)
        downloads["summary_csv"] = media_url(path)
    return dict(rows=rows, downloads=downloads,
        note="Her video, aynı örnekleme ayarı ve aynı model/kod kimliği için son deneme alınır; farklı ayarlar ayrı tutulur. "
             "Oranlar toplam nokta sayılarından hesaplanır. Hatalı kareler bulunma oranının paydasına girmez; "
             "işlenen/seçilen kare ve hata sayıları ayrıca gösterilir. Bu ölçümler anatomik doğruluk değildir.")


def media_url(path):
    path = Path(path).resolve()
    key = hashlib.sha256(str(path).encode()).hexdigest()[:24]
    with LOCK:
        MEDIA[key] = path
    return "/media/" + key


def scan(folder):
    if not str(folder).strip():
        raise ValueError("Video klasörünün yolunu yazın.")
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError("Geçerli bir video klasörü seçin.")
    found, errors = [], []
    for path in sorted(folder.iterdir(), key=lambda p: p.name.casefold()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            video = engine.inspect_video(path)
            video["thumbnail_url"] = "/thumbnail/" + video["video_id"]
            with LOCK:
                VIDEOS[video["video_id"]] = video
            found.append(video)
        except (ValueError, OSError) as exc:
            errors.append({"name": path.name, "error": str(exc)})
    return {"folder": str(folder), "videos": found, "errors": errors}


def capabilities():
    result = {}
    for library in engine.LIBRARIES:
        try:
            engine.existing.model_files([library])
            if library == "deepercut":
                docker = shutil.which("docker")
                if not docker:
                    raise ValueError("Docker bulunamadı")
                p = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"],
                                   capture_output=True, timeout=5)
                if p.returncode:
                    raise ValueError("Docker kapalı")
                p = subprocess.run([docker, "image", "inspect", "pose-tf14:latest"],
                                   capture_output=True, timeout=5)
                if p.returncode:
                    raise ValueError("Yerel Docker imajı bulunamadı")
            result[library] = {"ready": True, "message": "Dosyalar hazır"}
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            text = str(exc)
            result[library] = {"ready": False, "message": text if len(text) < 100 else "Gerekli dosya bulunamadı"}
    return result


def public_result(result, result_id, historical=False):
    result = copy.deepcopy(result)
    result["result_id"] = result_id
    result["historical"] = historical
    for sample in result["samples"]:
        sample["image_url"] = media_url(sample["image_path"])
    # Older runs saved a common-12 video. A derived native preview preserves
    # those records and only replaces the user-facing download after its source
    # result fingerprint is checked. No estimator is rerun here.
    if not historical:
        derived = DATA / "native_views" / result_id
        proof = derived / "source.json"
        if proof.is_file():
            source = Path(result.get("files", {}).get("result_json", ""))
            info = json.loads(proof.read_text())
            native_video = derived / "comparison_native.mp4"
            if (source.is_file() and native_video.is_file()
                    and info.get("source_sha256") == engine.existing.digest(source)
                    and info.get("video_sha256") == engine.existing.digest(native_video)):
                result["files"]["comparison_video"] = str(native_video)
                result["comparison_video_points"] = "Native model keypoints; derived from the preserved raw results."
    result["downloads"] = {key: media_url(path) for key, path in result.get("files", {}).items()
                            if path and Path(path).is_file()}
    return result


def pilot():
    base = ROOT / "outputs/image_feature_datasets/final_pilot_v2_release"
    manifest = json.loads((base / "manifest.json").read_text())
    native = {lib: {r["image_id"]: r for r in json.loads(
        (base / "native" / lib / "native.json").read_text())["samples"]} for lib in engine.LIBRARIES}
    samples = []
    for index, source in enumerate(manifest["samples"]):
        models = {lib: engine._model_result(lib, native[lib][source["image_id"]],
                    width=source["width"], height=source["height"]) for lib in engine.LIBRARIES}
        samples.append({"image_id": source["image_id"], "frame_index": None,
                        "timestamp_sec": None, "sample_index": index,
                        "image_path": str(base / "images" / source["staged_filename"]), "models": models})
    result = {"status": "completed", "video": {"name": "Önceki 10 resimlik çalışma",
              "width": manifest["samples"][0]["width"], "height": manifest["samples"][0]["height"],
              "fps": None, "frame_count": None, "duration": None},
              "frame_step": None, "libraries": list(engine.LIBRARIES), "samples": samples,
              "summary": engine._summaries(samples, engine.LIBRARIES), "files": {},
              "source": str(base), "visual_review": "NOT_REVIEWED"}
    return public_result(result, "pilot", historical=True)


def save_jobs():
    DATA.mkdir(parents=True, exist_ok=True)
    with LOCK:
        temporary = DATA / "jobs.next.json"
        temporary.write_text(json.dumps(JOBS, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(DATA / "jobs.json")


def load_jobs():
    path = DATA / "jobs.json"
    if path.is_file():
        for job in json.loads(path.read_text()):
            if job["status"] in ("queued", "running"):
                job.update(status="interrupted", error="Uygulama kapatıldığı için işlem yarıda kaldı.")
            JOBS.append(job)
            if job["video_id"] not in VIDEOS:
                try:
                    video = engine.inspect_video(job["video_path"])
                    if video["video_id"] == job["video_id"]:
                        video["thumbnail_url"] = "/thumbnail/" + video["video_id"]
                        VIDEOS[video["video_id"]] = video
                except (ValueError, OSError):
                    pass  # Keep historical jobs even when the source was moved.


def queue_worker():
    while not SHUTDOWN.is_set():
        WAKE.wait()
        if SHUTDOWN.is_set():
            break
        with LOCK:
            job = next((job for job in JOBS if job["status"] == "queued"), None)
            if job is None:
                WAKE.clear()
                continue
            job["status"] = "running"
            CANCEL.clear()
        save_jobs()
        def update(event):
            with LOCK:
                job["progress"] = event
        try:
            output = DATA / "runs" / job["id"]
            result = engine.analyze_video(job["video_path"], output, frame_step=job["frame_step"],
                libraries=job["libraries"], sample_limit=job["sample_limit"],
                on_progress=update, cancel_event=CANCEL,
                resume=bool(job.get("resume_requested")) and output.exists())
            with LOCK:
                cache_result(job["id"], result)
                normalized = {"COMPLETE": "completed", "PARTIAL": "partial", "ERROR": "error", "CANCELLED": "cancelled"}.get(result["status"], "error")
                warnings = [str(v) for v in (result.get("render_error"), result.get("sampling_warning")) if v]
                library_errors = [str(row["error"]) for row in result.get("library_status", {}).values() if row.get("error")]
                job.update(status=normalized, result_path=result["files"]["result_json"],
                           summary=result["summary"],
                           model_signature=model_signature(result),
                           library_elapsed={lib: row.get("elapsed_sec") for lib, row in result.get("library_status", {}).items()},
                           error=" | ".join(warnings + library_errors) or None,
                           warnings=warnings, resume_requested=False,
                           finished_at=datetime.now().isoformat(timespec="seconds"))
        except Exception as exc:
            with LOCK:
                job.update(status="cancelled" if CANCEL.is_set() else "error", error=str(exc))
        save_jobs()


class Handler(BaseHTTPRequestHandler):
    server_version = "PoseAnalysis/1"

    def log_message(self, fmt, *args):
        if not args or str(args[0]).split(" ")[0] != "GET":
            super().log_message(fmt, *args)

    def _headers(self, status=200, content_type="application/json; charset=utf-8", length=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.end_headers()

    def _json(self, value, status=200):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self._headers(status, length=len(payload))
        self.wfile.write(payload)

    def _trusted_host(self):
        host = self.headers.get("Host", "")
        return host in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def do_GET(self):
        if not self._trusted_host():
            return self._json({"error": "Geçersiz yerel adres."}, 403)
        route = urlparse(self.path).path
        try:
            if route in ("/", "/index.html", "/styles.css", "/app.js"):
                name = "index.html" if route == "/" else route[1:]
                return self._file(APP / name)
            if route == "/favicon.ico":
                return self._headers(204, length=0)
            if route == "/api/bootstrap":
                with LOCK:
                    videos = list(VIDEOS.values())
                    jobs = copy.deepcopy(JOBS)
                return self._json({"token": TOKEN, "videos": videos, "jobs": jobs,
                    "default_folder": str(ROOT / "videos/input"), "capabilities": capabilities(),
                    "schemes": {lib: {"expected": engine.NATIVE_EXPECTED[lib],
                        "edges": engine.EDGES[lib], "common_ids": list(engine.MAPS[lib].values()),
                        "common_map": engine.MAPS[lib],
                        "low_score_level": engine.LOW_SCORE_LEVELS.get(lib)} for lib in engine.LIBRARIES}})
            if route == "/api/jobs":
                with LOCK:
                    return self._json(copy.deepcopy(JOBS))
            if route == "/api/batch-summary":
                return self._json(batch_summary())
            if route == "/api/capabilities":
                return self._json({"capabilities": capabilities()})
            if route.startswith("/api/results/"):
                result_id = route.rsplit("/", 1)[-1]
                if result_id == "pilot":
                    return self._json(pilot())
                with LOCK:
                    job = next((j for j in JOBS if j["id"] == result_id), None)
                if job is None or not job.get("result_path"):
                    return self._json({"error": "Sonuç henüz hazır değil."}, 404)
                with LOCK:
                    if result_id not in RESULTS:
                        cache_result(result_id, json.loads(Path(job["result_path"]).read_text()))
                    result = RESULTS[result_id]
                return self._json(public_result(result, result_id))
            if route.startswith("/media/"):
                with LOCK:
                    path = MEDIA.get(route.rsplit("/", 1)[-1])
                if path is None:
                    return self._json({"error": "Dosya bulunamadı."}, 404)
                return self._file(path)
            if route.startswith("/thumbnail/"):
                with LOCK:
                    video = VIDEOS.get(route.rsplit("/", 1)[-1])
                if video is None:
                    return self._json({"error": "Video bulunamadı."}, 404)
                import cv2
                capture = cv2.VideoCapture(video["path"])
                try:
                    okay, image = capture.read()
                finally:
                    capture.release()
                if not okay:
                    raise ValueError("Önizleme karesi okunamadı.")
                image = cv2.resize(image, (max(1, int(image.shape[1] * 160 / image.shape[0])), 160))
                okay, data = cv2.imencode(".jpg", image)
                if not okay:
                    raise ValueError("Önizleme kodlanamadı.")
                self._headers(content_type="image/jpeg", length=len(data))
                return self.wfile.write(data.tobytes())
            return self._json({"error": "Sayfa bulunamadı."}, 404)
        except (ValueError, OSError, KeyError) as exc:
            return self._json({"error": str(exc)}, 400)

    def _file(self, path):
        if not path.is_file():
            return self._json({"error": "Dosya bulunamadı."}, 404)
        size = path.stat().st_size
        first, last = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            import re
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
            if not match:
                return self._json({"error": "Geçersiz dosya aralığı."}, 416)
            first = int(match.group(1)); last = min(last, int(match.group(2))) if match.group(2) else last
            if first > last or first >= size:
                return self._json({"error": "Dosya aralığı dışında."}, 416)
        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(last - first + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if range_header:
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        if path.suffix.lower() in (".csv", ".mp4", ".json") and "/media/" in self.path:
            self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(path.name))
        if path.suffix.lower() == ".html":
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            with path.open("rb") as handle:
                handle.seek(first)
                remaining = last - first + 1
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        origin = self.headers.get("Origin")
        expected_origins = (f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}")
        if not self._trusted_host() or (origin and origin not in expected_origins) or self.headers.get("X-Pose-Token") != TOKEN:
            return self._json({"error": "İstek bu yerel uygulamadan gelmeli."}, 403)
        route = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if route == "/api/upload":
                if not 0 < length <= 4 * 1024 ** 3:
                    raise ValueError("Dosya boş veya 4 GB sınırından büyük.")
                name = Path(unquote(self.headers.get("X-File-Name", "video.mp4"))).name
                if Path(name).suffix.lower() not in EXTENSIONS:
                    raise ValueError("Desteklenmeyen video uzantısı.")
                folder = DATA / "uploads" / uuid.uuid4().hex[:12]
                folder.mkdir(parents=True)
                path = folder / name
                remaining = length
                with path.open("xb") as sink:
                    while remaining:
                        chunk = self.rfile.read(min(1024 ** 2, remaining))
                        if not chunk:
                            raise ValueError("Dosya aktarımı yarıda kaldı.")
                        sink.write(chunk); remaining -= len(chunk)
                return self._json(scan(folder))
            if not 0 <= length <= 65536:
                raise ValueError("İstek boyutu geçersiz.")
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("JSON isteği bir nesne olmalı.")
            if route == "/api/scan":
                return self._json(scan(body.get("folder", "")))
            if route == "/api/queue":
                ids = body.get("video_ids", [])
                libraries = body.get("libraries", [])
                step, limit = body.get("frame_step", 5), body.get("sample_limit", 0)
                if (not isinstance(ids, list) or not ids or len(ids) > 200 or
                    any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids))):
                    raise ValueError("En az bir video seçin; aynı video bir istekte tekrarlanamaz.")
                if (not isinstance(libraries, list) or not libraries or
                    any(i not in engine.LIBRARIES for i in libraries) or len(libraries) != len(set(libraries))):
                    raise ValueError("En az bir geçerli kütüphane seçin.")
                if type(step) is not int or not 1 <= step <= 10000 or type(limit) is not int or not 0 <= limit <= 100000:
                    raise ValueError("Kare aralığı ve örnek sınırı geçersiz.")
                with LOCK:
                    if any(i not in VIDEOS for i in ids):
                        raise ValueError("Seçilen video listede bulunamadı.")
                    key = (step, limit or None, tuple(sorted(libraries)))
                    if any(active_duplicate(i, key) for i in ids):
                        raise ValueError("Aynı video ve ayarlar zaten kuyrukta veya işleniyor.")
                    for video_id in ids:
                        video = VIDEOS[video_id]
                        JOBS.append({"id": datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8],
                            "video_id": video_id, "video_path": video["path"], "name": video["name"],
                            "libraries": libraries, "frame_step": step, "sample_limit": limit or None,
                            "status": "queued", "progress": {}, "created_at": datetime.now().isoformat(timespec="seconds")})
                save_jobs(); WAKE.set()
                return self._json({"jobs": copy.deepcopy(JOBS)})
            if route == "/api/resume":
                job_id = body.get("job_id")
                if not isinstance(job_id, str):
                    raise ValueError("Devam edilecek işlemi seçin.")
                with LOCK:
                    job = next((j for j in JOBS if j["id"] == job_id), None)
                    if job is None or job["status"] not in ("interrupted", "cancelled", "error", "partial"):
                        raise ValueError("Yalnız yarıda kalan veya hatalı bir işlem devam ettirilebilir.")
                    if active_duplicate(job["video_id"], protocol_key(job), exclude_id=job_id):
                        raise ValueError("Aynı video ve ayarlar zaten işleniyor.")
                    video = engine.inspect_video(job["video_path"])
                    if video["video_id"] != job["video_id"]:
                        raise ValueError("Kaynak video değişmiş. Devam etmek yerine yeni analiz başlatın.")
                    video["thumbnail_url"] = "/thumbnail/" + video["video_id"]
                    VIDEOS[video["video_id"]] = video
                    job.update(status="queued", progress={}, resume_requested=True,
                               error=None, resumed_at=datetime.now().isoformat(timespec="seconds"))
                save_jobs(); WAKE.set()
                return self._json({"jobs": copy.deepcopy(JOBS),
                    "message": "İşlem sıraya alındı. Tamamlanan kütüphaneler korunur; yarım kütüphane yeniden çalışır."})
            if route == "/api/cancel":
                with LOCK:
                    for job in JOBS:
                        if job["status"] == "queued":
                            job["status"] = "cancelled"
                        elif job["status"] == "running":
                            CANCEL.set()
                save_jobs()
                return self._json({"message": "Durdurma isteği alındı."})
            return self._json({"error": "İşlem bulunamadı."}, 404)
        except (ValueError, TypeError, OSError, KeyError) as exc:
            return self._json({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    input_folder = ROOT / "videos/input"
    if input_folder.is_dir():
        scan(input_folder)
    load_jobs()
    worker = threading.Thread(target=queue_worker, daemon=True)
    worker.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Poz Analizi: http://127.0.0.1:{args.port}", flush=True)
    print("Mevcut modeller; yalnız bu bilgisayarda. Durdurmak için Ctrl+C.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        CANCEL.set()
        SHUTDOWN.set()
        WAKE.set()
        server.server_close()
        worker.join(timeout=20)


if __name__ == "__main__":
    main()
