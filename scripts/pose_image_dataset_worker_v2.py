"""Inference worker for the folder dataset. Compatible with Python 3.5 Docker.

Consumes lossless, independently sized staged PNGs. Never trains, edits points,
uses labels, substitutes coordinates or mixes DeeperCut clusters.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def save(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def empty(row):
    return dict(image_id=row['image_id'], detection_status=row['input_status'],
                candidate_count=0, selected_index=None, points=[], error=row.get('error'))


def point(j, x, y, score, score_type, available=True, **extra):
    import math
    x, y = float(x), float(y)
    return dict(native_id=j, x_px=x if math.isfinite(x) else None,
        y_px=y if math.isfinite(y) else None,
        score=float(score) if score is not None and math.isfinite(float(score)) else None,
        score_type=score_type, coordinate_available=bool(available and math.isfinite(x) and math.isfinite(y)), **extra)


def process_rows(rows, run, infer):
    records = []
    for row in rows:
        if row['input_status'] != 'READY':
            records.append(empty(row)); continue
        try:
            result = infer(run / 'images' / row['staged_filename'], row)
            result.update(image_id=row['image_id'], error=None)
            result['detection_status'] = 'POSE_RETURNED' if result['points'] else 'NO_POSE'
        except Exception as exc:
            # A failed image remains a row; runtime/model initialization failures
            # are outside this handler and fail the library process explicitly.
            result = empty(row)
            result.update(detection_status='INFERENCE_ERROR', error=type(exc).__name__+': '+str(exc))
        records.append(result)
        print('{}: {}'.format(row['image_id'], result['detection_status']), flush=True)
    return records


def mediapipe(root, run, rows):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(
        model_asset_path=str(root / 'models/mediapipe/pose_landmarker_full.task')),
        running_mode=vision.RunningMode.IMAGE, num_poses=1)
    with vision.PoseLandmarker.create_from_options(options) as detector:
        def infer(path, row):
            image = cv2.imread(str(path))
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            points = [point(j,p.x*row['width'],p.y*row['height'],p.visibility,
                'landmark_visibility', presence=float(p.presence))
                for j,p in enumerate(result.pose_landmarks[0])] if result.pose_landmarks else []
            return dict(points=points, candidate_count=len(result.pose_landmarks),
                        selected_index=0 if points else None)
        return process_rows(rows, run, infer)


def mmpose(root, run, rows):
    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    import cv2
    import numpy as np
    import torch
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_topdown, init_model
    from mmpose.utils import adapt_mmdet_pipeline
    torch.set_num_threads(4)
    detector = init_detector(str(root / 'external/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'),
        str(root / 'models/mmpose/rtmdet_m_person.pth'), device='cpu')
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose = init_model(str(root / 'external/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py'),
        str(root / 'models/mmpose/rtmpose-m_coco_256x192.pth'), device='cpu')
    def infer(path, row):
        image = cv2.imread(str(path))
        pred = inference_detector(detector, image).pred_instances.cpu().numpy()
        candidates = np.flatnonzero((pred.labels == 0) & (pred.scores > .3))
        selected = int(candidates[np.argmax(pred.scores[candidates])]) if len(candidates) else None
        points = []
        boxes = [dict(index=int(i), xyxy=[float(v) for v in pred.bboxes[i]],
                      score=float(pred.scores[i])) for i in candidates]
        if selected is not None:
            # Preserve the previous experiment's float64 box representation;
            # changing float32/float64 here can shift affine sampling slightly.
            box = np.asarray([float(v) for v in pred.bboxes[selected]])
            result = inference_topdown(pose, image, box[None,:])
            if result:
                native = result[0].pred_instances.cpu().numpy()
                xy, scores = native.keypoints[0], native.keypoint_scores[0]
                assert xy.shape == (17,2) and scores.shape == (17,)
                points = [point(j,xy[j,0],xy[j,1],scores[j],'RTMPose_keypoint_score') for j in range(17)]
        return dict(points=points, candidate_count=len(candidates), selected_index=selected,
                    candidate_boxes=boxes)
    return process_rows(rows, run, infer)


def deepercut(root, run, rows, output):
    import numpy as np
    import tensorflow as tf
    import matplotlib
    matplotlib.use('Agg')
    from scipy.misc import imread
    repo = root / 'external/pose-tensorflow'
    os.chdir(str(repo)); sys.path.insert(0,str(repo))
    from util.config import load_config
    from dataset.factory import create as create_dataset
    from dataset.pose_dataset import data_to_input
    from nnet.net_factory import pose_net
    from nnet import predict
    from multiperson.detections import extract_detections
    from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
    cfg = load_config(str(repo / 'demo/pose_cfg_multi.yaml'))
    dataset = create_dataset(cfg)
    spatial = SpatialModel(cfg); spatial.load()
    inputs = tf.placeholder(tf.float32,shape=[cfg.batch_size,None,None,3])
    outputs = pose_net(cfg).test(inputs)
    restorer = tf.train.Saver()
    session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4,inter_op_parallelism_threads=2))
    session.run(tf.global_variables_initializer()); session.run(tf.local_variables_initializer())
    restorer.restore(session,cfg.init_weights)
    def infer(path, row):
        image = imread(str(path),mode='RGB')
        result = session.run(outputs,feed_dict={inputs:data_to_input(image)})
        scmap,locref,pairwise = predict.extract_cnn_output(result,cfg,dataset.pairwise_stats)
        labels,pos,unary,_,_ = eval_graph(spatial,extract_detections(cfg,scmap,locref,pairwise))
        xy = get_person_conf_multicut(spatial,labels,unary,pos)
        counts = np.zeros((len(xy),17),dtype=np.int64)
        sums = np.zeros((len(xy),17),dtype=np.float64)
        for idx,(joint,person) in enumerate(labels):
            counts[int(person),int(joint)] += 1
            sums[int(person),int(joint)] += float(unary[idx,0])
        candidates = [i for i in range(len(xy)) if np.any(sums[i,5:17]>0)]
        selected = max(candidates,key=lambda i:(int(np.count_nonzero(sums[i,5:17]>0)),
            float(np.sum(sums[i,5:17])),-i)) if candidates else None
        people = []
        for person in range(len(xy)):
            points = [point(j,xy[person,j,0],xy[person,j,1],
                sums[person,j]/counts[person,j] if counts[person,j] else None,
                'mean_assigned_unary_score', available=bool(sums[person,j]>0),
                assigned_detections=int(counts[person,j]), unary_score_sum=float(sums[person,j])) for j in range(17)]
            people.append(dict(cluster_id=person,points=points))
        np.savez_compressed(str(output.parent / (row['image_id']+'_native.npz')),
            person_xy=xy,labels=labels,positions=pos,unary_scores=unary,
            assigned_counts=counts,unary_sums=sums)
        return dict(points=people[selected]['points'] if selected is not None else [],
            candidate_count=len(candidates), total_multicut_clusters=len(xy),
            selected_index=selected,people=people)
    try:
        return process_rows(rows,run,infer)
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',choices=('mediapipe','mmpose','deepercut'),required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(str(args.output))
    rows = json.loads((args.run/'manifest.json').read_text(encoding='utf-8'))['samples']
    if not any(r['input_status']=='READY' for r in rows):
        results = [empty(r) for r in rows]
    elif args.library == 'deepercut':
        results = deepercut(args.root,args.run,rows,args.output)
    else:
        results = globals()[args.library](args.root,args.run,rows)
    save(args.output,dict(library=args.library,coordinate_unit='decoded input-image pixels',
                         z_used=False,manual_coordinate_corrections=False,training=False,samples=results))


if __name__ == '__main__':
    main()
