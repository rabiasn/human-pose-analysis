"""Behavioral tests; fake inference is explicit, no model downloads/training."""
import csv
import hashlib
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import pose_video_comparison_v1 as comparison


def point(native_id, x=15., y=20., score=.9, available=True):
    return dict(native_id=native_id, x_px=x, y_px=y, score=score,
        score_type="TEST_ONLY", coordinate_available=available)


def fake_inference(library, output, manifest, env, callback, cancel_event):
    """Deterministic synthetic records, never described as model performance."""
    records = []
    for row in manifest["samples"]:
        records.append(dict(image_id=row["image_id"], detection_status="POSE_RETURNED",
            candidate_count=1, selected_index=0,
            points=[point(j, x=15+j%5*8, y=20+j//5*7)
                    for j in range(comparison.NATIVE_EXPECTED[library])], error=None))
    payload = dict(library=library,samples=records, test_only=True)
    comparison._json(output/"native"/library/"native.json", payload)
    return payload


class VideoComparisonTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np
        self.temp = tempfile.TemporaryDirectory()
        self.provenance_patch = patch.object(comparison,"_collect_provenance",return_value=dict(
            files={},libraries={library:{"files":[]} for library in comparison.LIBRARIES}))
        self.provenance_patch.start()
        self.root = Path(self.temp.name)
        self.source = self.root/"iki saniye.avi"
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*"MJPG"), 10., (96,128))
        if not writer.isOpened():
            self.fail("Synthetic fixture encoder unavailable.")
        for index in range(20):
            frame = np.full((128,96,3),index*10,dtype=np.uint8)
            cv2.putText(frame,str(index),(10,60),0,.6,(255,40,40),1)
            writer.write(frame)
        writer.release()

    def tearDown(self):
        self.provenance_patch.stop()
        self.temp.cleanup()

    def test_inspect_sample_indices_shared_pixels_and_video_timing(self):
        import cv2
        info = comparison.inspect_video(self.source)
        self.assertEqual((info["frame_count"],info["width"],info["height"]),(20,96,128))
        self.assertAlmostEqual(info["duration"],2.)
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        paths_by_library = {}
        events = []
        def infer(library, output, manifest, env, callback, event):
            paths_by_library[library] = [(row["image_path"],hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest())
                for row in manifest["samples"]]
            return fake_inference(library,output,manifest,env,callback,event)
        with patch.object(comparison,"_run_library",side_effect=infer), patch.object(comparison.existing,"export") as export:
            result = comparison.analyze_video(self.source,self.root/"out",5,on_progress=events.append)
            export.assert_not_called()
        self.assertEqual(result["status"],"COMPLETE")
        self.assertEqual([row["frame_index"] for row in result["samples"]],[0,5,10,15])
        self.assertEqual([row["timestamp_sec"] for row in result["samples"]],[0.,.5,1.,1.5])
        self.assertTrue(all(paths == paths_by_library["mediapipe"] for paths in paths_by_library.values()))
        self.assertEqual(before,hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertTrue(all(row["common_coverage_pct"]==100 for row in result["summary"]))
        self.assertTrue(all(row["native_expected_total"]==4*row["native_expected"] for row in result["summary"]))
        self.assertTrue(all(Path(path).is_absolute() for path in result["files"].values() if path))
        self.assertTrue(all(set(("stage","library","completed","total","message")) <= set(event) for event in events))
        capture=cv2.VideoCapture(result["files"]["comparison_video"])
        try:
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT),4)
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS),2.)
        finally:
            capture.release()
        saved = json.loads(Path(result["files"]["result_json"]).read_text())
        self.assertEqual(saved,result)
        with Path(result["files"]["frame_counts_csv"]).open(newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))),16)

    def test_native_unique_finite_availability_and_scores_are_not_filtered(self):
        raw = dict(points=[point(11,score=.1),point(11),point(12,x=0.,y=0.,score=.49),
            point(13,available=False),point(14,x=float("nan")),point(15,y=None),
            point(99),point("16"),point(True),point(0,x=-100,score=.5)])
        counts = comparison.count_native_points("mediapipe",raw,96,128)
        self.assertEqual(counts["native_available"],3)
        self.assertEqual(counts["common_available"],2)
        self.assertEqual(counts["low_score_count"],2)
        self.assertEqual(counts["duplicate_native_ids"],[11])
        self.assertEqual(counts["out_of_image_native_ids"],[0])
        self.assertEqual((counts["native_in_image"],counts["native_out_of_image"]),(2,1))
        self.assertEqual((counts["common_in_image"],counts["common_out_of_image"]),(2,0))
        self.assertEqual(len(counts["invalid_native_ids"]),3)
        self.assertEqual(raw["points"][0]["score"],.1)

    def test_openpose_mmpose_cutoff_flags_and_deepercut_native_score(self):
        for library in ("openpose","mmpose"):
            raw=dict(points=[point(5,score=.299),point(6,score=.3),point(7,score=1.7)])
            counted=comparison.count_native_points(library,raw)
            self.assertEqual(counted["native_available"],3)
            self.assertEqual(counted["low_score_count"],1)
        raw=dict(points=[point(5,score=4.2),point(6,score=-2.3)],candidate_count=3,detection_status="POSE_RETURNED")
        counted=comparison._model_result("deepercut",raw)
        self.assertEqual(counted["native_available"],2)
        self.assertIsNone(counted["low_score_count"])
        self.assertEqual(counted["points"][0]["score"],4.2)
        self.assertEqual(counted["points"][1]["score"],-2.3)
        self.assertEqual(counted["candidate_kind"],"body_cluster")
        self.assertEqual(counted["visual_review"],"MULTIPLE_BODY_CLUSTERS")

    def test_error_is_not_no_pose_or_zero_percent_success(self):
        def infer(library,output,manifest,env,callback,event):
            if library == "deepercut":
                raise FileNotFoundError("Docker daemon unavailable (synthetic test)")
            payload=fake_inference(library,output,manifest,env,callback,event)
            payload["samples"][0].update(points=[],detection_status="NO_POSE",candidate_count=0)
            payload["samples"][1].update(points=[],detection_status="INFERENCE_ERROR",error="synthetic failure",candidate_count=0)
            return payload
        with patch.object(comparison,"_run_library",side_effect=infer), patch.object(comparison,"_render_comparison",return_value=None):
            result=comparison.analyze_video(self.source,self.root/"out",5,libraries=["mediapipe","deepercut"])
        self.assertEqual(result["status"],"PARTIAL")
        mp,dc=result["summary"]
        self.assertEqual((mp["processed_frames"],mp["error_frames"],mp["no_pose_frames"]),(3,1,1))
        self.assertEqual(mp["native_expected_total"],3*33)
        self.assertAlmostEqual(mp["native_coverage_pct"],200/3)
        self.assertEqual((dc["processed_frames"],dc["error_frames"],dc["no_pose_frames"]),(0,4,0))
        self.assertIsNone(dc["native_coverage_pct"])
        self.assertIsNone(dc["common_coverage_pct"])
        self.assertIn("Docker",result["library_status"]["deepercut"]["error"])

    def test_pilot_limit_and_existing_folder_are_explicit(self):
        with patch.object(comparison,"_run_library",side_effect=fake_inference), patch.object(comparison,"_render_comparison",return_value=None):
            result=comparison.analyze_video(self.source,self.root/"out",5,libraries=["mediapipe"],sample_limit=2)
        self.assertEqual(len(result["samples"]),2)
        self.assertTrue(result["limited"])
        with self.assertRaises(FileExistsError):
            comparison.analyze_video(self.source,self.root/"out")
        for step in (0,-1,True,1.5):
            with self.assertRaises(ValueError):
                comparison.analyze_video(self.source,self.root/"other",frame_step=step)

    def test_cancel_during_sampling_never_runs_models(self):
        event=threading.Event()
        def progress(update):
            if update["stage"]=="sampling" and update["completed"]==2:
                event.set()
        with patch.object(comparison,"_run_library") as runner:
            result=comparison.analyze_video(self.source,self.root/"out",5,on_progress=progress,cancel_event=event)
        runner.assert_not_called()
        self.assertEqual(result["status"],"CANCELLED")
        self.assertEqual(len(result["samples"]),2)
        self.assertTrue(all(row["cancelled_frames"]==2 and row["error_frames"]==0 for row in result["summary"]))
        self.assertIsNone(result["files"]["comparison_video"])

    def test_missing_worker_record_is_error_not_no_pose(self):
        def infer(library,output,manifest,env,callback,event):
            payload=fake_inference(library,output,manifest,env,callback,event)
            payload["samples"].pop()
            return payload
        with patch.object(comparison,"_run_library",side_effect=infer), patch.object(comparison,"_render_comparison",return_value=None):
            result=comparison.analyze_video(self.source,self.root/"out",5,libraries=["mediapipe"])
        self.assertEqual(result["status"],"ERROR")
        self.assertEqual(result["summary"][0]["error_frames"],4)
        self.assertEqual(result["summary"][0]["no_pose_frames"],0)
        self.assertIsNone(result["summary"][0]["native_coverage_pct"])

    def test_in_image_metrics_do_not_claim_coordinate_accuracy(self):
        model = comparison._model_result("mediapipe",dict(detection_status="POSE_RETURNED",candidate_count=1,
            points=[point(11,x=-1,score=.1),point(12,x=96),point(23,x=30,score=.2),point(0,x=20)]),width=96,height=128)
        row = comparison._summaries([dict(models={"mediapipe":model})],["mediapipe"])[0]
        self.assertEqual((row["native_available_total"],row["native_in_image_total"],row["native_out_of_image_total"]),(4,2,2))
        self.assertEqual((row["common_available_total"],row["common_in_image_total"],row["common_out_of_image_total"]),(3,1,2))
        self.assertEqual(row["low_score_total"],2)
        self.assertEqual(model["visual_review"],"NOT_REVIEWED")

    def test_native_video_draws_all_schemas_and_common_view_is_explicit(self):
        import cv2
        output=self.root/'native_video_test'
        with patch.object(comparison,'_run_library',side_effect=fake_inference),patch.object(comparison,'_render_comparison',return_value=None):
            result=comparison.analyze_video(self.source,output,5,sample_limit=1)
        circle=cv2.circle
        with patch.object(cv2,'circle',wraps=circle) as drawn:
            native_path=comparison._render_comparison(result,output,None,None)
            self.assertEqual(drawn.call_count,33+25+17+17)
        with patch.object(cv2,'circle',wraps=circle) as drawn:
            common_path=comparison._render_comparison(result,output,None,None,point_mode='common')
            self.assertEqual(drawn.call_count,12*4)
        self.assertIn('comparison_native',native_path)
        self.assertIn('comparison_common12',common_path)

    def test_resume_reuses_completed_library_and_preserves_old_artifacts(self):
        event = threading.Event()
        def infer(library,*args):
            value = fake_inference(library,*args)
            if library=="mediapipe":
                event.set()
            return value
        output = self.root/"out"
        with patch.object(comparison,"_run_library",side_effect=infer), patch.object(comparison,"_render_comparison",return_value=None):
            first = comparison.analyze_video(self.source,output,5,libraries=["mediapipe","mmpose"],cancel_event=event)
        self.assertEqual(first["status"],"CANCELLED")
        protected = {path:hashlib.sha256(path.read_bytes()).hexdigest() for path in output.rglob("*") if path.is_file()}
        event.clear()
        with patch.object(comparison,"_run_library",side_effect=fake_inference) as runner, patch.object(comparison,"_render_comparison",return_value=None):
            resumed = comparison.analyze_video(self.source,output,5,libraries=["mediapipe","mmpose"],resume=True)
        self.assertEqual([call.args[0] for call in runner.call_args_list],["mmpose"])
        self.assertTrue(resumed["library_status"]["mediapipe"]["reused"])
        self.assertEqual(resumed["status"],"COMPLETE")
        self.assertIn("result_r0002.json",resumed["files"]["result_json"])
        self.assertIn("attempts/mmpose_0001",resumed["library_status"]["mmpose"]["native_json"])
        for path,digest in protected.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest,str(path))

    def test_resume_rejects_changed_settings_source_and_corrupted_native(self):
        output = self.root/"out"
        with patch.object(comparison,"_run_library",side_effect=fake_inference),patch.object(comparison,"_render_comparison",return_value=None):
            result = comparison.analyze_video(self.source,output,5,libraries=["mediapipe"])
        for kwargs in ({"frame_step":10},{"sample_limit":2},{"libraries":["mmpose"]}):
            with self.assertRaisesRegex(ValueError,"uyuşmuyor"):
                comparison.analyze_video(self.source,output,resume=True,
                    **(dict(frame_step=5,libraries=["mediapipe"]) | kwargs))
        import os
        stat = self.source.stat()
        os.utime(self.source,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000000))
        with self.assertRaisesRegex(ValueError,"uyuşmuyor"):
            comparison.analyze_video(self.source,output,5,libraries=["mediapipe"],resume=True)
        os.utime(self.source,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        native = Path(result["library_status"]["mediapipe"]["native_json"])
        native.write_text('{"samples":[]}')
        with self.assertRaisesRegex(ValueError,"değiştirilmiş"):
            comparison.analyze_video(self.source,output,5,libraries=["mediapipe"],resume=True)

    def test_cancelled_sampling_resumes_without_overwriting_pngs(self):
        event = threading.Event()
        def progress(update):
            if update["stage"]=="sampling" and update["completed"]==2:
                event.set()
        output = self.root/"out"
        with patch.object(comparison,"_run_library") as runner:
            first = comparison.analyze_video(self.source,output,5,libraries=["mediapipe"],on_progress=progress,cancel_event=event)
        runner.assert_not_called()
        png = output/"images/image_000001.png"
        before = png.stat().st_mtime_ns
        with patch.object(comparison,"_run_library",side_effect=fake_inference),patch.object(comparison,"_render_comparison",return_value=None):
            resumed = comparison.analyze_video(self.source,output,5,libraries=["mediapipe"],resume=True)
        self.assertEqual(len(first["samples"]),2)
        self.assertEqual(len(resumed["samples"]),4)
        self.assertEqual(png.stat().st_mtime_ns,before)
        self.assertTrue((output/"manifest_r0002.json").is_file())

    def test_renderer_failure_keeps_numerical_results_and_is_partial(self):
        with patch.object(comparison,"_run_library",side_effect=fake_inference),patch.object(comparison,"_render_comparison",side_effect=RuntimeError("TEST encoder unavailable")):
            result = comparison.analyze_video(self.source,self.root/"out",5,libraries=["mediapipe"])
        self.assertEqual(result["status"],"PARTIAL")
        self.assertIn("encoder",result["render_error"])
        self.assertTrue(Path(result["files"]["frame_counts_csv"]).is_file())
        self.assertEqual(result["library_status"]["mediapipe"]["status"],"COMPLETE")
        self.assertGreaterEqual(result["timings"]["total_wall_sec"],0)

    def test_wrong_library_payload_is_error_never_relabelled(self):
        def infer(library,*args):
            payload = fake_inference(library,*args)
            payload["library"] = "mmpose"
            return payload
        with patch.object(comparison,"_run_library",side_effect=infer),patch.object(comparison,"_render_comparison",return_value=None):
            result = comparison.analyze_video(self.source,self.root/"out",5,libraries=["mediapipe"])
        self.assertEqual(result["status"],"ERROR")
        self.assertIn("başka kütüphaneye",result["library_status"]["mediapipe"]["error"])
        self.assertIsNone(result["summary"][0]["native_coverage_pct"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
