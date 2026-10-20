"""Compare existing pose libraries on identical sampled video frames.

Only detection, native keypoint counts and visualization are performed. This
module does not calculate posture features, train models or correct coordinates.
Run with the project's existing MediaPipe Python environment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

import build_pose_image_dataset_v2 as existing
from pose_image_dataset_features_v2 import MAPS

LIBRARIES = existing.LIBRARIES
NATIVE_EXPECTED = dict(mediapipe=33, openpose=25, mmpose=17, deepercut=17)
LOW_SCORE_LEVELS = dict(mediapipe=.5, openpose=.3, mmpose=.3)
PROCESSED_STATUSES = {"POSE_RETURNED", "NO_POSE"}
MP_EDGES = ((0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(27,31),
    (24,26),(26,28),(28,30),(30,32),(28,32))
OP_EDGES = ((1,8),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(8,9),(9,10),(10,11),
    (8,12),(12,13),(13,14),(1,0),(0,15),(15,17),(0,16),(16,18),
    (14,19),(19,20),(14,21),(11,22),(22,23),(11,24),(2,17),(5,18))
COCO_EDGES = ((0,1),(0,2),(1,3),(2,4),(1,2),(3,5),(4,6),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16))
EDGES = dict(mediapipe=MP_EDGES, openpose=OP_EDGES, mmpose=COCO_EDGES, deepercut=COCO_EDGES)
COMMON_EDGES = ((11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),
    (23,24),(23,25),(25,27),(24,26),(26,28))


class AnalysisCancelled(RuntimeError):
    pass


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _cancelled(event):
    return event is not None and event.is_set()


def _notify(callback, stage, library=None, completed=0, total=0, message=""):
    if callback is not None:
        callback(dict(stage=stage, library=library, completed=completed, total=total, message=message))


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("Geçersiz JSON sayısı: " + value)))


def _unused_path(path):
    """Select a new revision without replacing a previous artifact."""
    path = Path(path)
    if not path.exists():
        return path
    revision = 2
    while True:
        candidate = path.with_name(f"{path.stem}_r{revision:04d}{path.suffix}")
        if not candidate.exists():
            return candidate
        revision += 1


def _checkpoint(output, state):
    """Immutable snapshots: a crash cannot damage the last complete checkpoint."""
    directory = output / "checkpoints"
    directory.mkdir(exist_ok=True)
    sequence = max((int(p.stem) for p in directory.glob("*.json") if p.stem.isdigit()), default=0)+1
    target = directory / f"{sequence:08d}.json"
    temporary = directory / ("."+uuid.uuid4().hex+".tmp")
    _json(temporary, state)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(target)
    return str(target)


def _latest_checkpoint(output):
    paths = sorted((output/"checkpoints").glob("[0-9]*.json"))
    if not paths:
        raise ValueError("Devam etmek için geçerli checkpoint yok; yeni çıktı klasörü seçin.")
    state = _read_json(paths[-1])
    if state.get("checkpoint_version") != 1:
        raise ValueError("Checkpoint sürümü desteklenmiyor.")
    return state


def _collect_provenance(libraries):
    """Hash existing weights/configs/scripts once; no downloads or new models."""
    files, by_library = {}, {}
    for library in libraries:
        try:
            paths = existing.model_files([library]) + [Path(__file__)]
            by_library[library] = {"files": [str(path.resolve()) for path in paths]}
            for path in paths:
                path = path.resolve()
                if str(path) not in files:
                    stat = path.stat()
                    files[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                        sha256=existing.digest(path))
        except (OSError, ValueError) as exc:
            by_library[library] = {"error": type(exc).__name__+": "+str(exc)}
    return dict(files=files, libraries=by_library)


def _verify_file(path, digest, description):
    path = Path(path)
    if not path.is_file() or existing.digest(path) != digest:
        raise ValueError(description+" kayıp veya değiştirilmiş; checkpoint kullanılmadı: "+str(path))
    return path


def inspect_video(path):
    """Read metadata and decode one frame without changing the source video."""
    import cv2
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Video bulunamadı: " + str(path))
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Video açılamadı: " + str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        okay, frame = capture.read()
        if not okay or frame is None:
            raise ValueError("Videonun ilk karesi çözülemedi.")
        if not _finite(fps) or fps <= 0 or not _finite(count) or count < 1:
            raise ValueError("Geçerli FPS/kare sayısı alınamadı; zamanlar tahmin edilmedi.")
        height, width = frame.shape[:2]
    finally:
        capture.release()
    stat = path.stat()
    identity = f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}".encode("utf-8")
    return dict(video_id=hashlib.sha256(identity).hexdigest()[:16], name=path.name,
        path=str(path), fps=fps, frame_count=int(count), duration=int(count)/fps,
        width=int(width), height=int(height), file_size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _stage_video(video, output, frame_step, sample_limit, callback, cancel_event, prior=None):
    import cv2
    images = output / "images"
    images.mkdir(exist_ok=prior is not None)
    prior_rows = {row["frame_index"]:row for row in (prior or {}).get("samples", [])}
    expected = math.ceil(video["frame_count"] / frame_step)
    if sample_limit is not None:
        expected = min(expected, sample_limit)
    capture = cv2.VideoCapture(video["path"])
    rows, decoded = [], 0
    limited = False
    try:
        if not capture.isOpened():
            raise ValueError("Video kareleri için tekrar açılamadı.")
        while not _cancelled(cancel_event):
            okay, frame = capture.read()
            if not okay:
                break
            frame_index = decoded
            decoded += 1
            if frame_index % frame_step:
                continue
            image_id = f"image_{len(rows)+1:06d}"
            target = images / (image_id + ".png")
            if target.exists():
                # An interrupted sampler may have saved a PNG before its snapshot.
                import numpy as np
                saved = cv2.imread(str(target))
                if saved is None or not np.array_equal(saved, frame):
                    raise ValueError("Önceki örnek PNG kaynak kareyle uyuşmuyor: "+str(target))
                if frame_index in prior_rows:
                    _verify_file(target, prior_rows[frame_index]["staged_sha256"], "PNG")
            elif not cv2.imwrite(str(target), frame):
                raise RuntimeError("PNG kare kaydedilemedi: " + str(target))
            height, width = frame.shape[:2]
            rows.append(dict(image_id=image_id, frame_index=frame_index,
                timestamp_sec=frame_index/video["fps"], image_path=str(target),
                staged_filename=target.name, relative_path=target.name,
                source_path=video["path"], width=width, height=height,
                input_status="READY", error=None, staged_sha256=existing.digest(target)))
            _notify(callback, "sampling", completed=len(rows), total=expected,
                message=f"Kare {frame_index}: aynı PNG bütün kütüphanelere verilecek.")
            if sample_limit is not None and len(rows) >= sample_limit:
                limited = frame_index + frame_step < video["frame_count"]
                break
    finally:
        capture.release()
    warning = (f"Decoded {decoded} frames; metadata reports {video['frame_count']}."
        if not limited and not _cancelled(cancel_event) and sample_limit is None
        and decoded < video["frame_count"] else None)
    manifest = dict(samples=rows, image_count=len(rows), video=video,
        frame_step=frame_step, decoded_frames=decoded, sample_limit=sample_limit,
        limited=limited, sampling_warning=warning, cancelled=_cancelled(cancel_event), labels_used=False,
        manual_points=False, image_decode="Sequential OpenCV decoding, lossless PNG, no resize/crop.",
        timestamp_definition="zero-based frame_index / reported FPS; fixed-frame-rate assumption")
    manifest_path = _unused_path(output / "manifest.json")
    manifest["manifest_file"] = str(manifest_path)
    _json(manifest_path, manifest)
    if not rows and not _cancelled(cancel_event):
        raise ValueError("Örneklenecek video karesi çözülemedi.")
    return manifest


def _command_for(library, output, manifest, env):
    # Reuse the existing checked model paths, worker and OpenPose parser.
    existing.model_files([library])
    args = ["--library", library, "--root", str(existing.ROOT), "--run", str(output),
            "--output", str(output / "native" / library / "native.json")]
    if library == "openpose":
        bootstrap = ("import json,os; from pathlib import Path; "
            "from build_pose_image_dataset_v2 import openpose; "
            "p=Path(os.environ['POSE_COMPARISON_RUN']); "
            "openpose(p,json.loads((p/'manifest.json').read_text()),os.environ.copy())")
        env["POSE_COMPARISON_RUN"] = str(output)
        return [str(existing.MP_PYTHON), "-B", "-c", bootstrap], None
    if library != "deepercut":
        interpreter = existing.MP_PYTHON if library == "mediapipe" else existing.MM_PYTHON
        return [str(interpreter), "-B", str(existing.WORKER), *args], None
    docker = shutil.which("docker")
    if not docker:
        raise FileNotFoundError("Mevcut Docker CLI bulunamadı; paket kurulmadı.")
    # --pull never below prevents an implicit network download.
    try:
        subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8",errors="replace")[-800:]
        raise RuntimeError("Docker servisine erişilemiyor; DeeperCut çalıştırılmadı. "+detail) from exc
    try:
        subprocess.run([docker, "image", "inspect", "pose-tf14:latest"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Yerel pose-tf14:latest Docker imajı bulunamadı; indirme yapılmadı.") from exc
    container_name = "pose-video-" + uuid.uuid4().hex[:16]
    command = [docker, "run", "--rm", "--name", container_name, "--pull", "never",
        "--platform", "linux/amd64", "--network", "none", "--read-only", "--tmpfs", "/tmp",
        "--cpus", "4", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONIOENCODING=utf-8",
        "-e", "MPLCONFIGDIR=/tmp/mpl", "-e", "TF_CUDNN_USE_AUTOTUNE=0", "-e", "OMP_NUM_THREADS=4",
        "--mount", "type=bind,src="+str(existing.ROOT)+",dst=/project,readonly",
        "--mount", "type=bind,src="+str(output)+",dst=/run,readonly",
        "--mount", "type=bind,src="+str(output/"native/deepercut")+",dst=/result",
        "-w", "/project/external/pose-tensorflow", "pose-tf14:latest", "python3", "-B",
        "/project/scripts/pose_image_dataset_worker_v2.py", "--library", "deepercut",
        "--root", "/project", "--run", "/run", "--output", "/result/native.json"]
    return command, (docker, container_name)


def _stop_process(process, container):
    if container:
        try:
            subprocess.run([container[0], "rm", "-f", container[1]],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)


def _run_library(library, output, manifest, env, callback, cancel_event):
    if _cancelled(cancel_event):
        raise AnalysisCancelled("İşlem iptal edildi.")
    command, container = _command_for(library, output, manifest, env)
    _json(output / "native" / library / "command.json", dict(command=command))
    log = output / "logs" / (library + "_worker.log")
    total = len(manifest["samples"])
    process = None
    with log.open("x", encoding="utf-8") as sink:
        process = subprocess.Popen(command, cwd=existing.ROOT, env=env,
            stdout=sink, stderr=subprocess.STDOUT, start_new_session=(os.name == "posix"))
        try:
            with log.open(encoding="utf-8", errors="replace") as reader:
                completed, last_report = 0, 0.
                while process.poll() is None:
                    if _cancelled(cancel_event):
                        raise AnalysisCancelled("İşlem iptal edildi.")
                    for line in reader.readlines():
                        if re.match(r"^image_\d+: ", line):
                            completed += 1
                    now = time.monotonic()
                    if now - last_report >= 1:
                        _notify(callback, "inference", library, min(completed, total), total,
                            f"{library}: hazır model çalışıyor.")
                        last_report = now
                    time.sleep(.1)
            if process.returncode:
                tail = log.read_text(encoding="utf-8", errors="replace")[-1600:]
                raise RuntimeError(f"{library} süreç çıkış kodu {process.returncode}. {tail}")
        except BaseException:
            _stop_process(process, container)
            raise
    native_path = output / "native" / library / "native.json"
    with native_path.open(encoding="utf-8") as handle:
        native = json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("Geçersiz JSON sayısı: " + value)))
    _notify(callback, "inference", library, total, total, f"{library}: süreç tamamlandı.")
    return native


def count_native_points(library, record, width=None, height=None):
    """Count unique, schema-valid finite native points; do not filter by score.

    score=0.2 is still an available coordinate when the worker says available.
    DeeperCut's assigned unary score is preserved and has no [0,1] cutoff.
    """
    expected = NATIVE_EXPECTED[library]
    raw_points = record.get("points", [])
    if not isinstance(raw_points, list):
        raise ValueError("Native points bir liste olmalı.")
    valid, seen, invalid_ids, duplicates = {}, set(), [], []
    for point in raw_points:
        if not isinstance(point, dict):
            raise ValueError("Native nokta kaydı nesne olmalı.")
        native_id = point.get("native_id")
        if not isinstance(native_id, int) or isinstance(native_id, bool) or not 0 <= native_id < expected:
            invalid_ids.append(native_id)
            continue
        if native_id in seen:
            duplicates.append(native_id)
        seen.add(native_id)
        if (point.get("coordinate_available") is True and _finite(point.get("x_px"))
                and _finite(point.get("y_px"))):
            valid.setdefault(native_id, point)
    common = set(MAPS[library].values())
    level = LOW_SCORE_LEVELS.get(library)
    low = [j for j, point in valid.items()
           if level is not None and _finite(point.get("score")) and point["score"] < level]
    no_score = [j for j, point in valid.items() if not _finite(point.get("score"))]
    dimensions_known = _finite(width) and _finite(height) and width > 0 and height > 0
    outside = [j for j, point in valid.items() if dimensions_known
               and not (0 <= point["x_px"] < width and 0 <= point["y_px"] < height)]
    inside = set(valid)-set(outside) if dimensions_known else None
    return dict(native_available=len(valid), native_expected=expected,
        common_available=len(common & set(valid)), common_expected=len(common),
        native_in_image=len(inside) if inside is not None else None,
        common_in_image=len(common & inside) if inside is not None else None,
        native_out_of_image=len(outside) if dimensions_known else None,
        common_out_of_image=len(common & set(outside)) if dimensions_known else None,
        low_score_count=len(low) if level is not None else None,
        low_score_native_ids=low, score_unavailable_count=len(no_score),
        out_of_image_native_ids=outside, invalid_native_ids=invalid_ids,
        duplicate_native_ids=sorted(set(duplicates)))


def _model_result(library, raw=None, error=None, status=None, width=None, height=None):
    raw = raw or dict(points=[], candidate_count=0)
    counts = count_native_points(library, raw, width, height)
    native_status = raw.get("detection_status")
    if status is None:
        if native_status in ("INFERENCE_ERROR", "INPUT_ERROR", "ERROR") or raw.get("error"):
            status = "ERROR"
            error = raw.get("error") or str(native_status)
        elif native_status not in PROCESSED_STATUSES:
            status, error = "ERROR", "Beklenmeyen native detection_status: " + str(native_status)
        else:
            status = "POSE_RETURNED" if counts["native_available"] else "NO_POSE"
    # A failed process must not contribute usable coordinates to coverage.
    if status not in PROCESSED_STATUSES:
        counts.update(native_available=0, common_available=0, native_in_image=None,
            common_in_image=None, native_out_of_image=None, common_out_of_image=None,
            low_score_count=None, score_unavailable_count=0,
            low_score_native_ids=[], out_of_image_native_ids=[])
    candidates = raw.get("candidate_count", 0)
    if not isinstance(candidates, int) or isinstance(candidates, bool) or candidates < 0:
        raise ValueError("candidate_count negatif olmayan bir tamsayı olmalı.")
    review = ("MULTIPLE_BODY_CLUSTERS" if library == "deepercut" else "MULTIPLE_CANDIDATES") if candidates > 1 else "NOT_REVIEWED"
    return dict(**counts, detection_status=status, native_detection_status=native_status,
        candidate_count=candidates, selected_index=raw.get("selected_index"),
        candidate_kind="body_cluster" if library == "deepercut" else "person_candidate",
        visual_review=review,
        points=raw.get("points", []), error=error)


def _summaries(samples, libraries, library_status=None):
    summaries = []
    for library in libraries:
        models = [sample["models"][library] for sample in samples]
        processed = [model for model in models if model["detection_status"] in PROCESSED_STATUSES]
        native = sum(model["native_available"] for model in processed)
        common = sum(model["common_available"] for model in processed)
        native_expected = len(processed)*NATIVE_EXPECTED[library]
        common_expected = len(processed)*len(MAPS[library])
        dimensions_known = bool(processed) and all(model.get("native_in_image") is not None for model in processed)
        in_native = sum(model["native_in_image"] for model in processed) if dimensions_known else None
        in_common = sum(model["common_in_image"] for model in processed) if dimensions_known else None
        summaries.append(dict(library=library, native_expected=NATIVE_EXPECTED[library], common_expected=len(MAPS[library]),
            sampled_frames=len(models), processed_frames=len(processed), native_available_total=native,
            native_expected_total=native_expected, native_coverage_pct=100*native/native_expected if native_expected else None,
            common_available_total=common, common_expected_total=common_expected,
            common_coverage_pct=100*common/common_expected if common_expected else None,
            native_in_image_total=in_native, common_in_image_total=in_common,
            native_out_of_image_total=native-in_native if in_native is not None else None,
            common_out_of_image_total=common-in_common if in_common is not None else None,
            native_in_image_coverage_pct=100*in_native/native_expected if native_expected and in_native is not None else None,
            common_in_image_coverage_pct=100*in_common/common_expected if common_expected and in_common is not None else None,
            low_score_total=sum(model["low_score_count"] or 0 for model in processed) if processed and library in LOW_SCORE_LEVELS else None,
            score_unavailable_total=sum(model["score_unavailable_count"] for model in processed),
            elapsed_sec=(library_status or {}).get(library, {}).get("elapsed_sec"),
            no_pose_frames=sum(model["detection_status"] == "NO_POSE" for model in models),
            error_frames=sum(model["detection_status"] == "ERROR" for model in models),
            cancelled_frames=sum(model["detection_status"] == "CANCELLED" for model in models),
            multiple_candidate_frames=sum(model["candidate_count"] > 1 for model in models)))
    return summaries


def _write_csv(path, columns, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _export_counts(output, result):
    columns = ["image_id", "frame_index", "timestamp_sec", "library", "detection_status",
        "native_available", "native_expected", "common_available", "common_expected",
        "native_in_image", "native_out_of_image", "common_in_image", "common_out_of_image",
        "low_score_count", "score_unavailable_count",
        "candidate_count", "candidate_kind", "visual_review", "error"]
    def rows():
        for sample in result["samples"]:
            for library, model in sample["models"].items():
                yield dict(image_id=sample["image_id"], frame_index=sample["frame_index"],
                    timestamp_sec=sample["timestamp_sec"], library=library,
                    **{key:model.get(key) for key in columns[4:]})
    def raw_rows():
        for sample in result["samples"]:
            for library, model in sample["models"].items():
                for point in model["points"]:
                    yield dict(**point, image_id=sample["image_id"], frame_index=sample["frame_index"],
                        timestamp_sec=sample["timestamp_sec"], library=library)
    paths = dict(frame_counts_csv=_unused_path(output/"frame_counts.csv"),
        summary_csv=_unused_path(output/"summary.csv"), native_keypoints_csv=_unused_path(output/"native_keypoints.csv"))
    _write_csv(paths["frame_counts_csv"], columns, rows())
    _write_csv(paths["summary_csv"], list(result["summary"][0]) if result["summary"] else ["library"], result["summary"])
    raw_columns = ["image_id", "frame_index", "timestamp_sec", "library", "native_id", "x_px", "y_px",
        "coordinate_available", "score", "score_type", "presence", "assigned_detections", "unary_score_sum"]
    _write_csv(paths["native_keypoints_csv"], raw_columns, raw_rows())
    return {key:str(path) for key,path in paths.items()}


def _render_comparison(result, output, callback, cancel_event, point_mode="native"):
    import cv2
    import numpy as np
    if point_mode not in ("native", "common"):
        raise ValueError("Nokta görünümü native veya common olmalı.")
    stem = "comparison_native" if point_mode == "native" else "comparison_common12"
    path = _unused_path(output / (stem+".mp4"))
    raw_path = _unused_path(output / (stem+"_opencv.mp4"))
    width, height = 640, 560
    fps = result["video"]["fps"] / result["frame_step"]
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2*width, 2*height))
    if not writer.isOpened():
        raise RuntimeError("MP4 video yazıcısı açılamadı.")
    written = 0
    try:
        for sample in result["samples"]:
            if _cancelled(cancel_event):
                raise AnalysisCancelled("Video oluşturma iptal edildi.")
            frame = cv2.imread(sample["image_path"])
            if frame is None:
                raise ValueError("Örnek PNG tekrar okunamadı: " + sample["image_path"])
            panels = []
            for library in LIBRARIES:
                panel = np.full((height, width, 3), 22, dtype=np.uint8)
                model = sample["models"].get(library)
                canvas = frame.copy()
                if model and model["detection_status"] in PROCESSED_STATUSES:
                    valid = {p["native_id"]:p for p in model["points"] if isinstance(p.get("native_id"),int)
                        and not isinstance(p["native_id"],bool) and 0 <= p["native_id"] < NATIVE_EXPECTED[library]
                        and p.get("coordinate_available") is True and _finite(p.get("x_px")) and _finite(p.get("y_px"))
                        and 0 <= p["x_px"] < frame.shape[1] and 0 <= p["y_px"] < frame.shape[0]}
                    def pixel(j):
                        return round(valid[j]["x_px"]), round(valid[j]["y_px"])
                    if point_mode == "common":
                        common = set(MAPS[library].values())
                        valid = {j:p for j,p in valid.items() if j in common}
                        edges = [(MAPS[library][a], MAPS[library][b]) for a,b in COMMON_EDGES]
                    else:
                        edges = EDGES[library]
                    for a, b in edges:
                        if a in valid and b in valid:
                            cv2.line(canvas, pixel(a), pixel(b), (40,210,40), 2, cv2.LINE_AA)
                    for j in valid:
                        color = (0,170,255) if j in model["low_score_native_ids"] else (255,200,0)
                        cv2.circle(canvas, pixel(j), 4, color, -1, cv2.LINE_AA)
                scale = min(width/frame.shape[1], (height-110)/frame.shape[0])
                resized = cv2.resize(canvas, (max(1,round(frame.shape[1]*scale)), max(1,round(frame.shape[0]*scale))))
                x, y = (width-resized.shape[1])//2, 110+(height-110-resized.shape[0])//2
                panel[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
                lines = [library.upper()+f"   t={sample['timestamp_sec']:.3f}s   frame={sample['frame_index']}"]
                if model and model["detection_status"] not in PROCESSED_STATUSES:
                    lines += [model["detection_status"]+" - measurements unavailable",
                        "See result.json / library log for reason."]
                elif model:
                    drawing = "native points" if point_mode == "native" else "common 12"
                    lines += [f"DRAW: {drawing} | Native {model['native_available']}/{model['native_expected']} | Common {model['common_available']}/{model['common_expected']}",
                        ("REVIEW: multiple body clusters" if library == "deepercut" else "REVIEW: multiple person candidates")
                            if model["candidate_count"]>1 else "Visual accuracy: NOT REVIEWED"]
                    lines += [f"Outside image: {model['native_out_of_image']} | Low score: "+
                        ("N/A" if library == "deepercut" else str(model['low_score_count']))+" (flags only)"]
                else:
                    lines += ["NOT SELECTED"]
                for i, line in enumerate(lines):
                    cv2.putText(panel, line, (9,23+24*i), cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (100,170,255) if "ERROR" in line else (240,240,240), 1, cv2.LINE_AA)
                panels.append(panel)
            writer.write(np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))
            written += 1
            _notify(callback, "rendering", completed=written, total=len(result["samples"]), message="Dört panelli video oluşturuluyor.")
    finally:
        writer.release()
    if written != len(result["samples"]) or not written:
        raise RuntimeError("Karşılaştırma videosu tamamlanmadı.")
    # OpenCV's MPEG-4 Part 2 is not reliably supported by HTML video players.
    # The already installed FFmpeg converts only this new output to H.264.
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("Tarayıcı için H.264 dönüşümü yapılamadı: mevcut FFmpeg bulunamadı. comparison_opencv.mp4 korundu.")
    command = [ffmpeg,"-hide_banner","-loglevel","warning","-n","-i",str(raw_path),
        "-an","-c:v","libx264","-preset","fast","-crf","23","-pix_fmt","yuv420p",
        "-threads","4","-movflags","+faststart",str(path)]
    _notify(callback,"encoding",completed=0,total=1,message="Tarayıcı için H.264 video hazırlanıyor.")
    encoder_log = _unused_path(output/"logs/video_encoder.log")
    with encoder_log.open("x",encoding="utf-8") as log:
        process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=(os.name=="posix"))
        try:
            while process.poll() is None:
                if _cancelled(cancel_event):
                    raise AnalysisCancelled("Video kodlama iptal edildi.")
                time.sleep(.1)
            if process.returncode:
                raise RuntimeError("H.264 kodlama başarısız; logs/video_encoder.log dosyasına bakın.")
        except BaseException:
            _stop_process(process,None)
            raise
    _notify(callback,"encoding",completed=1,total=1,message="H.264 video hazır.")
    check = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        while True:
            okay, frame = check.read()
            if not okay:
                break
            if frame.shape[:2] != (2*height, 2*width):
                raise RuntimeError("Çıktı video boyutu uyuşmuyor.")
            decoded += 1
        actual_fps = check.get(cv2.CAP_PROP_FPS)
    finally:
        check.release()
    if decoded != written or not math.isclose(actual_fps, fps, rel_tol=.001, abs_tol=.001):
        raise RuntimeError("Çıktı video kare sayısı/FPS kontrolü başarısız.")
    # Only this run's intermediate is removed after the final MP4 is verified.
    raw_path.unlink()
    return str(path)


def _validated_native(native, manifest, library):
    if not isinstance(native, dict) or not isinstance(native.get("samples"), list):
        raise ValueError("Native sonuç nesne ve samples listesi içermeli.")
    if native.get("library") != library:
        raise ValueError("Native sonuç başka kütüphaneye ait.")
    records = native["samples"]
    if any(not isinstance(row, dict) for row in records):
        raise ValueError("Native kare kaydı nesne olmalı.")
    lookup = {record.get("image_id"):record for record in records}
    expected = {sample["image_id"] for sample in manifest["samples"]}
    if len(records) != len(lookup) or set(lookup) != expected:
        raise ValueError("Native sonuçlarda eksik/fazla/tekrarlanan kare kimliği.")
    for row in manifest["samples"]:
        model = _model_result(library, lookup[row["image_id"]], width=row["width"], height=row["height"])
        if model["invalid_native_ids"] or model["duplicate_native_ids"]:
            raise ValueError("Native sonuçta geçersiz/tekrarlanan nokta kimliği.")
    return lookup


def _attempt_directory(output, library, manifest, resumed):
    """Workers write to new directories on retry; old raw results remain intact."""
    if not resumed:
        target = output
        (target/"native"/library).mkdir(exist_ok=False)
        return target
    attempts = output/"attempts"
    attempts.mkdir(exist_ok=True)
    number = 1
    while (attempts/f"{library}_{number:04d}").exists():
        number += 1
    target = attempts/f"{library}_{number:04d}"
    (target/"images").mkdir(parents=True)
    (target/"native"/library).mkdir(parents=True)
    (target/"logs").mkdir()
    # Hard links avoid duplicate image bytes and remain accessible in Docker's
    # read-only /run mount (symlinks pointing outside that mount would not).
    for row in manifest["samples"]:
        os.link(row["image_path"], target/"images"/row["staged_filename"])
    _json(target/"manifest.json", manifest)
    return target


def analyze_video(video_path, output_dir, frame_step=5, libraries=None,
                  on_progress=None, cancel_event=None, sample_limit=None, resume=False):
    """Sample 0,N,2N... and compare native/common point availability.

    ERROR/CANCELLED frames are excluded from coverage denominators. NO_POSE
    frames are successfully processed zero detections and remain denominators.
    Coverage measures returned coordinates, not anatomical correctness.
    """
    if isinstance(frame_step, bool) or not isinstance(frame_step, int) or frame_step < 1:
        raise ValueError("frame_step pozitif tamsayı olmalı.")
    if sample_limit is not None and (isinstance(sample_limit,bool) or not isinstance(sample_limit,int) or sample_limit<1):
        raise ValueError("sample_limit pozitif tamsayı olmalı.")
    if not isinstance(resume, bool):
        raise ValueError("resume bool olmalı.")
    selected = list(dict.fromkeys(LIBRARIES if libraries is None else libraries))
    if not selected or any(library not in LIBRARIES for library in selected):
        raise ValueError("Geçerli kütüphaneler: " + ", ".join(LIBRARIES))
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and not resume:
        raise FileExistsError("Yeni bir çıktı klasörü seçin: " + str(output))
    if resume and not output.is_dir():
        raise ValueError("Devam edilecek çıktı klasörü bulunamadı.")
    started = time.monotonic()
    video = inspect_video(video_path)
    request = dict(video_id=video["video_id"], video_path=video["path"],
        file_size=video["file_size"], mtime_ns=video["mtime_ns"], frame_step=frame_step,
        libraries=selected, sample_limit=sample_limit)
    if resume:
        checkpoint = _latest_checkpoint(output)
        if checkpoint.get("request") != request:
            raise ValueError("Kaynak video kimliği veya analiz ayarları checkpoint ile uyuşmuyor.")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/"logs").mkdir()
        (output/"native").mkdir()
        checkpoint = dict(checkpoint_version=1, request=request, manifest=None,
            libraries={}, provenance=None, state="CREATED")
        _checkpoint(output, checkpoint)
    _notify(on_progress,"provenance",message="Mevcut model ve ayar dosyalarının kimliği denetleniyor.")
    provenance = _collect_provenance(selected)
    old_provenance = checkpoint.get("provenance")
    if resume and old_provenance:
        completed = [lib for lib,entry in checkpoint["libraries"].items() if entry.get("status")=="COMPLETE"]
        for library in completed:
            before = old_provenance["libraries"].get(library,{})
            after = provenance["libraries"].get(library,{})
            if before != after or any(old_provenance["files"].get(path,{}).get("sha256") !=
                    provenance["files"].get(path,{}).get("sha256") for path in before.get("files", [])):
                raise ValueError("Tamamlanmış kütüphanenin model/kod dosyaları değişmiş: "+library)
    checkpoint["provenance"] = provenance
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="4",
        MPLCONFIGDIR=str(output/"runtime_cache/mpl"), XDG_CACHE_HOME=str(output/"runtime_cache"),
        PYTHONPATH=str(existing.ROOT/"scripts")+os.pathsep+env.get("PYTHONPATH", ""))
    planned_samples = math.ceil(video["frame_count"]/frame_step)
    if sample_limit is not None:
        planned_samples = min(planned_samples,sample_limit)
    sampling_start = time.monotonic()
    prior = None
    if checkpoint.get("manifest"):
        entry = checkpoint["manifest"]
        path = _verify_file(entry["path"], entry["sha256"], "Örnek manifesti")
        prior = _read_json(path)
        for row in prior["samples"]:
            _verify_file(row["image_path"], row["staged_sha256"], "Örnek PNG")
    if prior and not prior.get("cancelled"):
        manifest = prior
        _notify(on_progress,"sampling",completed=len(manifest["samples"]),total=len(manifest["samples"]),
            message="Doğrulanmış PNG örnekleri checkpoint'ten kullanılıyor.")
    else:
        _notify(on_progress, "sampling", total=planned_samples, message="Video kareleri okunuyor.")
        manifest = _stage_video(video, output, frame_step, sample_limit, on_progress, cancel_event,
            prior=(prior or {}) if resume else None)
    checkpoint["manifest"] = dict(path=manifest["manifest_file"],sha256=existing.digest(Path(manifest["manifest_file"])))
    checkpoint["state"] = "SAMPLED"
    _checkpoint(output,checkpoint)
    sampling_sec = time.monotonic()-sampling_start
    # Verify completed artifacts before changing their checkpoint entries. A
    # corrupted result is never silently accepted or replaced by a retry.
    if resume:
        for library,entry in checkpoint["libraries"].items():
            if entry.get("status")=="COMPLETE":
                native_path = _verify_file(entry["native_json"],entry["native_sha256"],library+" native JSON")
                lookup = _validated_native(_read_json(native_path),manifest,library)
                if any(_model_result(library,record)["detection_status"] not in PROCESSED_STATUSES for record in lookup.values()):
                    raise ValueError("Tamamlanmış checkpoint hata içeren kare taşıyor: "+library)
    samples = [dict(image_id=row["image_id"], frame_index=row["frame_index"], timestamp_sec=row["timestamp_sec"],
        image_path=row["image_path"], models={}) for row in manifest["samples"]]
    result = dict(status="RUNNING", video=video, frame_step=frame_step, libraries=selected,
        library_status={}, samples=samples, summary=[], files={},
        sample_limit=sample_limit, limited=manifest["limited"], sampling_warning=manifest["sampling_warning"],
        interpretation="Point coverage is not anatomical accuracy or movement recognition.",
        availability_definition="Unique schema-valid finite x/y coordinates flagged available by the native worker; includes low scores and out-of-image coordinates.",
        in_image_definition="0 <= x < image width and 0 <= y < image height; geometric bounds are not anatomical correctness.",
        coverage_denominator="processed_frames * expected_points; includes NO_POSE, excludes ERROR/CANCELLED",
        score_policy=dict(low_score_levels=LOW_SCORE_LEVELS, score_filtering=False,
            deepercut="Native assigned unary scores preserved, no [0,1] assumption or low-score threshold."),
        training=False, manual_coordinate_corrections=False, feature_extraction=False,
        temporal_tracking=False, running_mode="Independent still-image inference on identical sampled PNGs",
        resumed=resume, resume_granularity="Completed library outputs; unfinished library inference restarts from its first sampled image.",
        provenance=provenance, timings=dict(sampling_sec=sampling_sec),
        timing_definition="Wall seconds, sequential CPU/container runs including initialization and output I/O; reused library elapsed is its original attempt, not current-run compute time. Not a controlled speed benchmark.",
        comparison_video_points="Native model keypoints, including the face/foot/hand landmarks present in each configured schema; common 12 totals separately reported.",
        model_selection=dict(mediapipe="Existing worker: full model, IMAGE, maximum one returned pose; not proof of only one person in scene.",
            openpose="Existing BODY25 parser: largest mean common-joint confidence.",
            mmpose="Existing RTMDet person candidates above 0.3; highest detector score; RTMPose-M COCO17.",
            deepercut="Existing multicut body clusters: most supported common joints; native tie-breaks. Clusters are not person counts."),
        timestamp_definition=manifest["timestamp_definition"],
        comparison_video_timing="FPS=source FPS/frame_step; last sampled frame holds for one sample interval; H.264/yuv420p; no audio.")
    inference_start = time.monotonic()
    for library in selected:
        error, status, lookup = None, "COMPLETE", {}
        library_start, reused, attempt = time.monotonic(), False, output
        old = checkpoint["libraries"].get(library, {})
        try:
            if _cancelled(cancel_event):
                raise AnalysisCancelled("İşlem iptal edildi.")
            if resume and old.get("status") == "COMPLETE":
                native_path = _verify_file(old["native_json"],old["native_sha256"],library+" native JSON")
                lookup = _validated_native(_read_json(native_path),manifest,library)
                attempt, reused = Path(old["attempt_directory"]), True
                _notify(on_progress,"inference",library,len(samples),len(samples),library+": doğrulanmış tamamlanmış sonuç yeniden kullanıldı.")
            else:
                attempt = _attempt_directory(output,library,manifest,resume)
                checkpoint["libraries"][library] = dict(status="RUNNING",attempt_directory=str(attempt))
                checkpoint["state"] = "INFERENCE"
                _checkpoint(output,checkpoint)
                _notify(on_progress, "inference", library, total=len(samples), message=library+" hazırlanıyor.")
                native = _run_library(library, attempt, manifest, env.copy(), on_progress, cancel_event)
                lookup = _validated_native(native,manifest,library)
        except AnalysisCancelled as exc:
            status, error, lookup = "CANCELLED", str(exc), {}
        except Exception as exc:
            status, error, lookup = "ERROR", type(exc).__name__+": "+str(exc), {}
        for sample, row in zip(samples, manifest["samples"]):
            try:
                model = _model_result(library, lookup.get(sample["image_id"]), error=error,
                    status=status if status in ("ERROR","CANCELLED") else None,
                    width=row["width"], height=row["height"])
            except Exception as exc:
                model = _model_result(library, error=type(exc).__name__+": "+str(exc), status="ERROR")
            sample["models"][library] = model
        per_frame_errors = sum(sample["models"][library]["detection_status"]=="ERROR" for sample in samples)
        if status == "COMPLETE" and per_frame_errors:
            status = "ERROR" if per_frame_errors == len(samples) else "PARTIAL"
            error = f"{per_frame_errors} karede inference/çıktı hatası; kare kayıtlarına bakın."
        native_path = attempt/"native"/library/"native.json"
        worker_log = attempt/"logs"/(library+"_worker.log")
        elapsed = old.get("elapsed_sec",0.) if reused else time.monotonic()-library_start
        entry = dict(status=status, error=error, elapsed_sec=elapsed, reused=reused,
            attempt_directory=str(attempt), native_json=str(native_path) if native_path.is_file() else None,
            log=str(worker_log), native_sha256=existing.digest(native_path) if native_path.is_file() else None)
        result["library_status"][library] = entry
        # A cancelled resume must not discard a previously completed checkpoint.
        if not (status=="CANCELLED" and old.get("status")=="COMPLETE"):
            checkpoint["libraries"][library] = entry
        _checkpoint(output,checkpoint)
        if error and not worker_log.exists():
            with worker_log.open("x",encoding="utf-8") as handle:
                handle.write(error+"\n")
    result["timings"]["inference_sec"] = time.monotonic()-inference_start
    result["summary"] = _summaries(samples, selected, result["library_status"])
    result["status"] = ("CANCELLED" if _cancelled(cancel_event) else
        "ERROR" if all(row["status"]=="ERROR" for row in result["library_status"].values()) else
        "PARTIAL" if any(row["status"]!="COMPLETE" for row in result["library_status"].values()) else "COMPLETE")
    if result["status"] == "COMPLETE" and result["sampling_warning"]:
        result["status"] = "PARTIAL"
    result["files"] = dict(**_export_counts(output, result),
        comparison_video=None, result_json=str(_unused_path(output/"result.json")))
    render_start = time.monotonic()
    if samples and not _cancelled(cancel_event):
        try:
            result["files"]["comparison_video"] = _render_comparison(result, output, on_progress, cancel_event)
        except AnalysisCancelled as exc:
            result["status"], result["render_error"] = "CANCELLED", str(exc)
        except Exception as exc:
            result["render_error"] = type(exc).__name__+": "+str(exc)
            if result["status"] == "COMPLETE":
                result["status"] = "PARTIAL"
    result["timings"].update(render_sec=time.monotonic()-render_start, total_wall_sec=time.monotonic()-started)
    _json(result["files"]["result_json"], result)
    checkpoint.update(state=result["status"], latest_result=result["files"]["result_json"])
    _checkpoint(output,checkpoint)
    _notify(on_progress, "complete", completed=len(samples), total=len(samples), message="Sonuç: "+result["status"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--sample-limit", type=int)
    parser.add_argument("--libraries", nargs="+", choices=LIBRARIES)
    parser.add_argument("--resume",action="store_true",help="Aynı çıktının doğrulanmış tamamlanmış kütüphanelerinden devam et.")
    args = parser.parse_args()
    result = analyze_video(args.video, args.output, args.frame_step, args.libraries,
        on_progress=lambda event: print(json.dumps(event,ensure_ascii=False),flush=True), sample_limit=args.sample_limit,resume=args.resume)
    print(json.dumps(dict(status=result["status"],files=result["files"],summary=result["summary"]),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
