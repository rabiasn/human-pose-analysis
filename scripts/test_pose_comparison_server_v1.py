"""Yerel HTTP arayüzü için erişim ve veri eşleme kontrolleri; model çalıştırmaz."""
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pose_workbench_server', ROOT/'app/pose_comparison_v1/server.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class LocalServerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cv2
        import numpy as np
        cls.temp = tempfile.TemporaryDirectory(prefix='pose_http_test_')
        cls.addClassCleanup(cls.temp.cleanup)
        app.DATA = Path(cls.temp.name)
        input_folder = app.DATA/'input'
        input_folder.mkdir()
        writer = cv2.VideoWriter(str(input_folder/'test.avi'),
            cv2.VideoWriter_fourcc(*'MJPG'), 10., (96, 128))
        if not writer.isOpened():
            raise RuntimeError('Synthetic test video encoder unavailable.')
        try:
            for index in range(5):
                writer.write(np.full((128, 96, 3), index*40, dtype=np.uint8))
        finally:
            writer.release()
        app.VIDEOS.clear()
        app.scan(input_folder)
        cls.http = ThreadingHTTPServer(('127.0.0.1', 0), app.Handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.http.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown();cls.http.server_close();cls.thread.join();cls.temp.cleanup()

    def request(self, path, data=None, headers=None):
        h = dict(headers or {})
        payload = None
        if data is not None:
            payload = json.dumps(data).encode();h.setdefault('Content-Type','application/json')
        try:
            response = urlopen(Request(self.base+path, data=payload, headers=h), timeout=10)
        except HTTPError as exc:
            response=exc
        with response:
            raw=response.read()
            return response.status, raw, response.headers

    def setUp(self):
        app.JOBS.clear(); app.RESULTS.clear(); app.CANCEL.clear(); app.WAKE.clear(); app.SHUTDOWN.clear()

    def test_pilot_preserves_native_schemas_and_historical_source(self):
        status,raw,_=self.request('/api/results/pilot')
        self.assertEqual(status,200)
        result=json.loads(raw)
        self.assertTrue(result['historical']);self.assertEqual(len(result['samples']),10)
        expected={'mediapipe':(330,330,120),'openpose':(231,250,119),'mmpose':(170,170,120),'deepercut':(105,170,75)}
        for row in result['summary']:
            self.assertEqual((row['native_available_total'],row['native_expected_total'],row['common_available_total']),expected[row['library']])
        self.assertTrue(all(s['timestamp_sec'] is None for s in result['samples']))

    def test_mutations_require_token_and_same_origin(self):
        self.assertEqual(self.request('/api/queue',{})[0],403)
        self.assertEqual(self.request('/api/queue',{}, {'X-Pose-Token':app.TOKEN,'Origin':'https://unrelated.example'})[0],403)

    def test_host_and_path_allowlist(self):
        self.assertEqual(self.request('/api/jobs',headers={'Host':'unrelated.example'})[0],403)
        self.assertEqual(self.request('/../README.md')[0],404)
        self.assertEqual(self.request('/media/../../README.md')[0],404)

    def test_media_is_registered_and_supports_ranges(self):
        path=app.DATA/'test.bin';path.write_bytes(b'0123456789')
        url=app.media_url(path)
        status,body,headers=self.request(url,headers={'Range':'bytes=2-5'})
        self.assertEqual((status,body),(206,b'2345'))
        self.assertEqual(headers['Content-Range'],'bytes 2-5/10')
        self.assertEqual(self.request(url,headers={'Range':'bytes=20-'})[0],416)

    def test_invalid_queue_rejected_without_running_models(self):
        headers={'X-Pose-Token':app.TOKEN}
        for invalid_body in ([], 'text', False):
            self.assertEqual(self.request('/api/queue',invalid_body,headers)[0],400)
        video_id=next(iter(app.VIDEOS))
        for invalid in [{'frame_step':0},{'libraries':[]},{'video_ids':['unknown']},{'sample_limit':-1}]:
            body={'video_ids':[video_id],'libraries':['mediapipe'],'frame_step':5,'sample_limit':3,**invalid}
            self.assertEqual(self.request('/api/queue',body,headers)[0],400)

    def test_queue_keeps_requested_frame_step_and_scope(self):
        video_id=next(iter(app.VIDEOS))
        status,raw,_=self.request('/api/queue',{'video_ids':[video_id],'libraries':['mediapipe'],
            'frame_step':20,'sample_limit':2},{'X-Pose-Token':app.TOKEN})
        self.assertEqual(status,200)
        job=json.loads(raw)['jobs'][-1]
        self.assertEqual((job['frame_step'],job['sample_limit'],job['libraries'],job['status']),
                         (20,2,['mediapipe'],'queued'))
        self.assertTrue((app.DATA/'jobs.json').is_file())

    def test_duplicate_active_queue_is_rejected_and_original_kept(self):
        video_id=next(iter(app.VIDEOS)); headers={'X-Pose-Token':app.TOKEN}
        payload={'video_ids':[video_id],'libraries':['mediapipe'],'frame_step':5,'sample_limit':0}
        self.assertEqual(self.request('/api/queue',payload,headers)[0],200)
        self.assertEqual(self.request('/api/queue',payload,headers)[0],400)
        self.assertEqual(len(app.JOBS),1)

    def test_resume_keeps_identity_and_rejects_changed_video(self):
        video_id=next(iter(app.VIDEOS)); video=app.VIDEOS[video_id]
        app.JOBS.append(dict(id='retry',video_id=video_id,video_path=video['path'],name=video['name'],
            frame_step=5,sample_limit=None,libraries=['mediapipe'],status='interrupted'))
        headers={'X-Pose-Token':app.TOKEN}
        status,raw,_=self.request('/api/resume',{'job_id':'retry'},headers)
        self.assertEqual(status,200)
        self.assertTrue(json.loads(raw)['jobs'][0]['resume_requested'])
        self.assertEqual(app.JOBS[0]['id'],'retry')
        app.JOBS[0]['status']='interrupted'
        with patch.object(app.engine,'inspect_video',return_value={**video,'video_id':'changed'}):
            self.assertEqual(self.request('/api/resume',{'job_id':'retry'},headers)[0],400)
        self.assertEqual(app.JOBS[0]['status'],'interrupted')

    def test_batch_summary_deduplicates_protocols_and_uses_weighted_counts(self):
        def job(key,vid,n,den,step=5,limit=None,status='completed'):
            return dict(id=key,video_id=vid,libraries=['mediapipe'],frame_step=step,sample_limit=limit,
                status=status,summary=[dict(library='mediapipe',sampled_frames=den//12,processed_frames=den//12,
                common_available_total=n,common_expected_total=den,native_available_total=n,native_expected_total=den)])
        app.JOBS.extend([job('old','a',0,12),job('new','a',12,12),job('other','b',12,36),
                         job('different','a',1,12,step=10),job('pilot','a',1,12,limit=1),
                         job('failure','c',0,0,status='error')])
        status,raw,_=self.request('/api/batch-summary');self.assertEqual(status,200)
        result=json.loads(raw);self.assertEqual(len(result['rows']),3)
        row=next(r for r in result['rows'] if r['frame_step']==5 and r['sample_limit'] is None)
        self.assertEqual((row['videos'],row['common_available_total'],row['common_expected_total']),(3,24,48))
        self.assertEqual(row['common_coverage_pct'],50)  # not mean(100,33.3)
        self.assertEqual(row['failed_videos'],1)
        self.assertIsNone(row['out_of_image_total'])  # older diagnostics are not fake zeros
        self.assertEqual(self.request(result['downloads']['summary_csv'])[0],200)

    def test_result_cache_is_bounded(self):
        for i in range(4):app.cache_result(str(i),{'samples':[]})
        self.assertEqual(list(app.RESULTS),['2','3'])

    def test_native_preview_requires_matching_source_and_video_hashes(self):
        source=app.DATA/'original_result.json'
        result=dict(samples=[],files={'result_json':str(source),'comparison_video':None})
        source.write_text(json.dumps(result)); before=source.read_bytes()
        derived=app.DATA/'native_views'/'native_test';derived.mkdir(parents=True,exist_ok=True)
        video=derived/'comparison_native.mp4';video.write_bytes(b'test-video-only')
        proof=dict(source_sha256=app.engine.existing.digest(source),video_sha256=app.engine.existing.digest(video))
        (derived/'source.json').write_text(json.dumps(proof))
        public=app.public_result(result,'native_test')
        self.assertEqual(public['files']['comparison_video'],str(video))
        self.assertIn('comparison_video',public['downloads'])
        self.assertEqual(source.read_bytes(),before)
        video.write_bytes(b'changed')
        self.assertIsNone(app.public_result(result,'native_test')['files']['comparison_video'])
        self.assertIsNone(result['files']['comparison_video'])

    def test_batch_keeps_different_model_identities_separate(self):
        base=dict(video_id='same',libraries=['mediapipe'],frame_step=5,sample_limit=None,
            status='completed',summary=[dict(library='mediapipe',sampled_frames=1,processed_frames=1,
            common_available_total=12,common_expected_total=12)])
        app.JOBS.extend([{**base,'id':'one','model_signature':'aaaa'},{**base,'id':'two','model_signature':'bbbb'}])
        result=app.batch_summary()
        self.assertEqual(len(result['rows']),2)
        self.assertEqual({r['model_signature'] for r in result['rows']},{'aaaa','bbbb'})

    def test_worker_resumes_existing_output_and_uses_revision_result_path(self):
        video_id=next(iter(app.VIDEOS));video=app.VIDEOS[video_id]
        output=app.DATA/'runs'/'retry_worker';output.mkdir(parents=True,exist_ok=True)
        app.JOBS.append(dict(id='retry_worker',video_id=video_id,video_path=video['path'],name=video['name'],
            frame_step=5,sample_limit=2,libraries=['mediapipe'],status='queued',resume_requested=True))
        latest=str(output/'result_r0002.json')
        def analyze(*args,**kwargs):
            self.assertTrue(kwargs['resume']); app.SHUTDOWN.set()
            return dict(status='COMPLETE',files={'result_json':latest},summary=[],library_status={})
        app.WAKE.set()
        try:
            with patch.object(app.engine,'analyze_video',side_effect=analyze):app.queue_worker()
            self.assertEqual((app.JOBS[0]['status'],app.JOBS[0]['result_path']),('completed',latest))
            self.assertFalse(app.JOBS[0]['resume_requested'])
        finally:app.SHUTDOWN.clear();app.WAKE.clear()


if __name__=='__main__':
    unittest.main(verbosity=2)
