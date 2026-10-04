"""Same images, same anatomical points, same x/y, angle and ratio definitions.

prepare: map native points and create image comparison panels.
build: export three feature families for each library after visual review.
Existing MediaPipe pilot and raw detector outputs are never overwritten.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from extract_posture_image_features_v1 import (
    ROOT, JOINTS, CORE, ANGLE_SPECS, RATIO_SPECS, extract_features,
    schema, digest, save_json, finite,
)
from export_posture_image_csv_v1 import encoded, verify_roundtrip

BASE = ROOT / 'outputs/posture_still_images/pilot_v1'
RUN = BASE / 'library_comparison_v1'
STANDARD = RUN / 'standardized_v1'
DATA = RUN / 'datasets_v1'
LIBRARIES = ('MediaPipe', 'OpenPose', 'MMPose')
KEY = {'MediaPipe': 'mediapipe', 'OpenPose': 'openpose', 'MMPose': 'mmpose'}
MAPS = {
    'MediaPipe': {j: j for j in JOINTS},
    'OpenPose': {11: 5, 12: 2, 13: 6, 14: 3, 15: 7, 16: 4,
                 23: 12, 24: 9, 25: 13, 26: 10, 27: 14, 28: 11},
    'MMPose': {11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10,
               23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16},
}
SCORE_LEVELS = {'MediaPipe': .5, 'OpenPose': .3, 'MMPose': .3}
SCORE_NAMES = {'MediaPipe': 'landmark_visibility', 'OpenPose': 'BODY_25_confidence', 'MMPose': 'RTMPose_keypoint_score'}
SHORT = {11: 'LS', 12: 'RS', 13: 'LE', 14: 'RE', 15: 'LW', 16: 'RW',
         23: 'LH', 24: 'RH', 25: 'LK', 26: 'RK', 27: 'LA', 28: 'RA'}
EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


def canonical_point(library, joint, native, width, height):
    """Keep native coordinates/scores; OpenPose score-zero points are absent."""
    point = native.get(MAPS[library][joint])
    score = point.get('score') if point else None
    valid = bool(point and finite(point.get('x_px')) and finite(point.get('y_px')))
    if library == 'OpenPose' and (not finite(score) or score <= 0):
        valid = False
    return dict(canonical_id=joint, joint_name=JOINTS[joint], native_id=MAPS[library][joint],
                x_px=point.get('x_px') if point else None, y_px=point.get('y_px') if point else None,
                x=point['x_px'] / width if valid else None,
                y=point['y_px'] / height if valid else None,
                score=score, score_type=SCORE_NAMES[library], coordinate_available=valid)


def prepare():
    import cv2
    import numpy as np

    if STANDARD.exists():
        raise FileExistsError(STANDARD)
    manifest = json.loads((BASE / 'manifest.json').read_text())
    mp = {r['image_id']: r for r in json.loads((BASE / 'mediapipe_image_landmarks.json').read_text())['samples']}
    mm_file = RUN / 'mmpose/native_keypoints.json'
    mm = {r['image_id']: r for r in json.loads(mm_file.read_text())['samples']}
    raw_paths = [BASE / 'manifest.json', BASE / 'mediapipe_image_landmarks.json', mm_file,
                 RUN / 'openpose/run_protocol.json', Path(__file__),
                 ROOT / 'scripts/extract_posture_image_features_v1.py']
    records = []
    for row in manifest['samples']:
        image_id = row['image_id']
        w, h = manifest['width'], manifest['height']
        image_path = BASE / 'images' / row['filename']
        assert digest(image_path) == row['image_sha256']
        raw_paths.append(image_path)
        op_path = RUN / 'openpose/raw_json' / (Path(row['filename']).stem + '_keypoints.json')
        op = json.loads(op_path.read_text())
        raw_paths.append(op_path)
        people = op['people']
        # Fixed automatic rule; no anatomical edits or label-dependent selection.
        def mean_body_score(person):
            values = person['pose_keypoints_2d']
            assert len(values) == 75
            return sum(values[3*j + 2] for j in MAPS['OpenPose'].values()) / 12
        op_idx = max(range(len(people)), key=lambda i: mean_body_score(people[i])) if people else None
        op_points = ({j: dict(x_px=float(v[3*j]), y_px=float(v[3*j+1]), score=float(v[3*j+2]))
                      for j in range(25) for v in [people[op_idx]['pose_keypoints_2d']]} if op_idx is not None else {})
        mp_points = {p['landmark_id']: dict(x_px=p['x']*w, y_px=p['y']*h, score=p['visibility']) for p in mp[image_id]['points']}
        mm_points = {p['native_id']: p for p in mm[image_id]['keypoints']}
        for library, native, detected, count, selected in (
            ('MediaPipe', mp_points, mp[image_id]['detected'], int(mp[image_id]['detected']), 0 if mp_points else None),
            ('OpenPose', op_points, bool(op_points), len(people), op_idx),
            ('MMPose', mm_points, mm[image_id]['detected'], mm[image_id]['candidate_people'], mm[image_id]['selected_index'])):
            points = [canonical_point(library, j, native, w, h) for j in JOINTS]
            records.append(dict(image_id=image_id, filename=row['filename'], library=library,
                                width=w, height=h, detected=detected, people=count,
                                selected_native_person=selected, points=points))
    STANDARD.mkdir(parents=True, exist_ok=False)
    save_json(STANDARD / 'common_keypoints.json', dict(
        canonical_ids='MediaPipe body IDs used only as a common anatomical index; native_id retained separately',
        anatomical_left_right=True, joint_maps=MAPS, score_names=SCORE_NAMES,
        score_review_levels=SCORE_LEVELS,
        score_policy='Levels mark measurements for review only; model scores are not calibrated across libraries. RTMPose scores above 1 remain valid native scores, never clamped.',
        openpose_missing_policy='Native confidence <= 0 means missing point: retain native zero tuple but normalized coordinates unavailable',
        person_selection='OpenPose: largest mean confidence over 12 common joints. MMPose: RTMDet highest-score person. MediaPipe: configured num_poses=1. Subject identity must be visually checked.',
        source_sha256={str(p): digest(p) for p in raw_paths}, samples=records))
    lookup = {(r['image_id'], r['library']): r for r in records}
    def card(im, title, subtitle):
        im = cv2.resize(im, (320, round(im.shape[0] * 320 / im.shape[1])))
        im = cv2.copyMakeBorder(im, 64, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(im, title, (6, 22), 0, .48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(im, subtitle, (6, 46), 0, .37, (230, 230, 230), 1, cv2.LINE_AA)
        return im
    for source in manifest['samples']:
        image = cv2.imread(str(BASE / 'images' / source['filename']))
        panels = [card(image.copy(), f"{source['image_id']} | {source['timestamp_sec']:.3f}s", 'REFERENCE: ' + source['reference_posture'])]
        for library in LIBRARIES:
            row = lookup[source['image_id'], library]
            im = image.copy()
            points = {p['canonical_id']: p for p in row['points'] if p['coordinate_available']}
            def pixel(j):
                return round(points[j]['x_px']), round(points[j]['y_px'])
            for a, b in EDGES:
                if a in points and b in points:
                    cv2.line(im, pixel(a), pixel(b), (30, 210, 40), 2, cv2.LINE_AA)
            for j, p in points.items():
                color = (0, 170, 255) if not finite(p['score']) or p['score'] < SCORE_LEVELS[library] else (255, 200, 0)
                cv2.circle(im, pixel(j), 5, color, -1, cv2.LINE_AA)
                xy = pixel(j)
                for thickness, ink in ((3, (0, 0, 0)), (1, (255, 255, 255))):
                    cv2.putText(im, SHORT[j], (xy[0]+5, xy[1]-4), 0, .40, ink, thickness, cv2.LINE_AA)
            panels.append(card(im, library + ' | still image', f'Orange: score < {SCORE_LEVELS[library]}; labels: anatomical L/R'))
        path = STANDARD / (source['image_id'] + '_comparison.jpg')
        assert not path.exists() and cv2.imwrite(str(path), np.hstack(panels))
    print(json.dumps(dict(images=len(manifest['samples']), libraries=3, records=len(records), output=str(STANDARD))), flush=True)


def automatic_flags(dependencies, points, library):
    flags = []
    for j in sorted(set(dependencies)):
        p = points[j]
        if not p['coordinate_available']:
            flags.append(f'MISSING_XY:{j}')
            continue
        if not finite(p['score']) or p['score'] < 0:
            flags.append(f'SCORE_UNAVAILABLE:{j}')
        elif p['score'] < SCORE_LEVELS[library]:
            flags.append(f'LOW_SCORE:{j}')
        if not (0 <= p['x'] <= 1 and 0 <= p['y'] <= 1):
            flags.append(f'OUT_OF_IMAGE:{j}')
    return flags


def build():
    if DATA.exists():
        raise FileExistsError(DATA)
    raw_path = STANDARD / 'common_keypoints.json'
    review_path = RUN / 'visual_review.json'
    raw = json.loads(raw_path.read_text())
    reviews = json.loads(review_path.read_text())['samples']
    review_map = {(r['image_id'], r['library']): r for r in reviews}
    manifest = json.loads((BASE / 'manifest.json').read_text())
    sources = {r['image_id']: r for r in manifest['samples']}
    assert set(review_map) == {(r['image_id'], r['library']) for r in raw['samples']}
    xy_specs = [(JOINTS[j] + '_' + axis + '_px', (j,)) for j in JOINTS for axis in ('x', 'y')]
    families = dict(xy=xy_specs, angles=[(n, ids) for n, ids, _ in ANGLE_SPECS],
                    ratios=[(n, ids) for n, ids, _ in RATIO_SPECS])
    meta = ['image_id', 'library', 'source_frame', 'timestamp_sec', 'image_width_px', 'image_height_px',
            'reference_posture', 'pose_detected', 'people_detected', 'torso_length_px']
    tables = {(lib, family): [] for lib in LIBRARIES for family in families}
    summary = {lib: dict(images=0, detected=0, available_common_points=0, numeric_xy=0, numeric_angles=0,
                        numeric_ratios=0, automatic_flagged_features=0, observer_flagged_features=0) for lib in LIBRARIES}
    for record in raw['samples']:
        lib, image_id = record['library'], record['image_id']
        source, review = sources[image_id], review_map[image_id, lib]
        points = {p['canonical_id']: p for p in record['points']}
        # Geometry consumes original coordinates. Confidence checks stay native
        # and separate; visibility=1 here only bypasses the MP-specific QC helper.
        geometry_points = {j: dict(x=p['x'], y=p['y'], visibility=1.0)
                           for j, p in points.items() if p['coordinate_available']}
        geometry = extract_features(geometry_points, record['width'], record['height'])
        values = geometry['values'].copy()
        for j in JOINTS:
            for axis in ('x', 'y'):
                values[JOINTS[j] + '_' + axis + '_px'] = points[j][axis + '_px'] if points[j]['coordinate_available'] else None
        prefix = [image_id, lib, source['frame'], source['timestamp_sec'], record['width'], record['height'],
                  source['reference_posture'], record['detected'], record['people'], geometry['torso_length_px']]
        stats = summary[lib]
        stats['images'] += 1
        stats['detected'] += int(record['detected'])
        stats['available_common_points'] += sum(p['coordinate_available'] for p in points.values())
        for family, specs in families.items():
            auto, observer = [], []
            for name, ids in specs:
                deps = set(ids) | set(CORE) if family == 'ratios' else set(ids)
                flags = automatic_flags(deps, points, lib)
                if family != 'xy':
                    flags += [f for f in geometry['automatic'][name].split(';')
                              if f in ('DEGENERATE_ANGLE', 'DEGENERATE_TORSO', 'NONFINITE_RESULT')]
                if record['people'] > 1:
                    flags.append('MULTIPLE_PEOPLE')
                auto.append(';'.join(flags) if flags else 'NO_FLAG')
                affected = sorted(deps & set(review['review_joint_ids']))
                observer.append(';'.join('REVIEW:' + str(j) for j in affected) if affected else 'NOT_FLAGGED')
            names = [n for n, _ in specs]
            row = prefix + [values[n] for n in names] + auto + observer
            tables[lib, family].append(row)
            stats['numeric_' + family] += sum(values[n] is not None for n in names)
            stats['automatic_flagged_features'] += sum(v != 'NO_FLAG' for v in auto)
            stats['observer_flagged_features'] += sum(v.startswith('REVIEW:') for v in observer)
    DATA.mkdir(parents=True, exist_ok=False)
    checks = []
    for (lib, family), rows in tables.items():
        names = [n for n, _ in families[family]]
        columns = meta + names + ['auto_qc__' + n for n in names] + ['observer_qc__' + n for n in names]
        path = DATA / f'{KEY[lib]}_{family}.csv'
        with path.open('x', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(columns)
            writer.writerows([[encoded(v) for v in r] for r in rows])
        checks.append(dict(file=path.name, **verify_roundtrip(path, columns, rows)))
    with (DATA / 'common_keypoints_xy.csv').open('x', encoding='utf-8', newline='') as f:
        columns = ['image_id', 'library', 'joint_name', 'canonical_id', 'native_id',
                   'x_px', 'y_px', 'x_normalized', 'y_normalized', 'coordinate_available', 'native_score', 'score_type']
        writer = csv.writer(f, lineterminator='\n'); writer.writerow(columns)
        for r in raw['samples']:
            for p in r['points']:
                writer.writerow([encoded(v) for v in [r['image_id'], r['library'], p['joint_name'], p['canonical_id'],
                                 p['native_id'], p['x_px'], p['y_px'], p['x'], p['y'], p['coordinate_available'], p['score'], p['score_type']]])
    info = schema()
    info.update(xy_features='24 original-image pixel coordinates, 12 common joints. Normalized coordinates and native scores also retained in common_keypoints_xy.csv.',
                shared_anatomical_joints=JOINTS, native_maps=MAPS,
                quality_policy='Native score levels MP .5 / OP .3 / MM .3 only mark review; scores not comparable probabilities. No coordinate correction or confidence-based replacement.',
                visual_review='Independent per-library review; MP review flags never copied to another library.',
                family_feature_counts=dict(xy=24, angles=12, ratios=21))
    save_json(DATA / 'feature_schema.json', info)
    save_json(DATA / 'verification.json', dict(
        by_library=summary, csv_roundtrip=checks,
        source_sha256={str(p): digest(p) for p in (raw_path, review_path, BASE / 'manifest.json', Path(__file__), ROOT / 'scripts/extract_posture_image_features_v1.py')},
        classifier_run=False, manually_corrected_coordinates=False,
        interpretation='Detection and numerically available measurements, not keypoint localization accuracy or posture recognition success. Ten selected images from one source video.',
    ))
    assert all(digest(Path(p)) == h for p, h in raw['source_sha256'].items())
    print(json.dumps(dict(output=str(DATA), by_library=summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'build'))
    args = parser.parse_args()
    prepare() if args.phase == 'prepare' else build()
