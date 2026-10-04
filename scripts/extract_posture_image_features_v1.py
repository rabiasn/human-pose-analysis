"""Extract projected 2D angles and length ratios from the still-image pilot.

One image becomes one row. Numerical measurements use only original x/y points
and image dimensions. Labels and review notes are metadata, not calculation inputs.
The JSON table is the lossless intermediate for the delivered CSV export.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'outputs/posture_still_images/pilot_v1'
CORE = (11, 12, 23, 24)
VISIBILITY_REVIEW_LEVEL = 0.5  # Review policy only, not a posture decision.
EPSILON_PX = 1e-9  # Numerical zero guard, not a movement threshold.

JOINTS = {
    11: 'left_shoulder', 12: 'right_shoulder',
    13: 'left_elbow', 14: 'right_elbow',
    15: 'left_wrist', 16: 'right_wrist',
    23: 'left_hip', 24: 'right_hip',
    25: 'left_knee', 26: 'right_knee',
    27: 'left_ankle', 28: 'right_ankle',
}

# A-B-C: the angle is at B. Left/right refer to the person's anatomy.
ANGLE_SPECS = [
    ('elbow_left_deg', (11, 13, 15), 'Sol dirsek'),
    ('elbow_right_deg', (12, 14, 16), 'Sağ dirsek'),
    ('shoulder_left_deg', (13, 11, 23), 'Sol omuz: üst kol ile gövde yanı'),
    ('shoulder_right_deg', (14, 12, 24), 'Sağ omuz: üst kol ile gövde yanı'),
    ('hip_left_deg', (11, 23, 25), 'Sol kalça: gövde yanı ile uyluk'),
    ('hip_right_deg', (12, 24, 26), 'Sağ kalça: gövde yanı ile uyluk'),
    ('knee_left_deg', (23, 25, 27), 'Sol diz'),
    ('knee_right_deg', (24, 26, 28), 'Sağ diz'),
    ('trunk_shoulder_left_deg', (12, 11, 23), 'Gövde dörtgeninin sol üst köşesi'),
    ('trunk_shoulder_right_deg', (11, 12, 24), 'Gövde dörtgeninin sağ üst köşesi'),
    ('trunk_hip_left_deg', (11, 23, 24), 'Gövde dörtgeninin sol alt köşesi'),
    ('trunk_hip_right_deg', (12, 24, 23), 'Gövde dörtgeninin sağ alt köşesi'),
]

RATIO_SPECS = [
    ('shoulder_width_ratio', (11, 12), 'Omuzlar arası mesafe'),
    ('hip_width_ratio', (23, 24), 'Kalçalar arası mesafe'),
    ('trunk_side_left_ratio', (11, 23), 'Sol omuz–kalça mesafesi'),
    ('trunk_side_right_ratio', (12, 24), 'Sağ omuz–kalça mesafesi'),
    ('trunk_diagonal_left_right_ratio', (11, 24), 'Sol omuz–sağ kalça köşegeni'),
    ('trunk_diagonal_right_left_ratio', (12, 23), 'Sağ omuz–sol kalça köşegeni'),
    ('upper_arm_left_ratio', (11, 13), 'Sol üst kol'),
    ('upper_arm_right_ratio', (12, 14), 'Sağ üst kol'),
    ('forearm_left_ratio', (13, 15), 'Sol ön kol'),
    ('forearm_right_ratio', (14, 16), 'Sağ ön kol'),
    ('thigh_left_ratio', (23, 25), 'Sol uyluk'),
    ('thigh_right_ratio', (24, 26), 'Sağ uyluk'),
    ('lower_leg_left_ratio', (25, 27), 'Sol alt bacak'),
    ('lower_leg_right_ratio', (26, 28), 'Sağ alt bacak'),
    ('wrist_separation_ratio', (15, 16), 'El bilekleri arası mesafe'),
    ('knee_separation_ratio', (25, 26), 'Dizler arası mesafe'),
    ('ankle_separation_ratio', (27, 28), 'Ayak bilekleri arası mesafe'),
    ('shoulder_wrist_left_span_ratio', (11, 15), 'Sol omuz–el bileği düz mesafesi; kol boyunca toplam uzunluk değil'),
    ('shoulder_wrist_right_span_ratio', (12, 16), 'Sağ omuz–el bileği düz mesafesi; kol boyunca toplam uzunluk değil'),
    ('hip_ankle_left_span_ratio', (23, 27), 'Sol kalça–ayak bileği düz mesafesi; bacak boyunca toplam uzunluk değil'),
    ('hip_ankle_right_span_ratio', (24, 28), 'Sağ kalça–ayak bileği düz mesafesi; bacak boyunca toplam uzunluk değil'),
]
FEATURE_NAMES = [s[0] for s in ANGLE_SPECS + RATIO_SPECS]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def pixel_points(landmarks, width, height):
    """Restore image aspect ratio before Euclidean geometry."""
    if not finite(width) or not finite(height) or width <= 0 or height <= 0:
        raise ValueError('Positive finite image width and height required')
    result = {}
    for joint, point in landmarks.items():
        if finite(point.get('x')) and finite(point.get('y')):
            xy = (point['x'] * width, point['y'] * height)
            if all(math.isfinite(v) for v in xy):
                result[joint] = xy
    return result


def angle_degrees(a, b, c):
    """Cosine law: angle ABC in degrees; None if an arm has zero length."""
    if not all(math.isfinite(v) for point in (a, b, c) for v in point):
        return None
    ab, bc, ac = math.dist(a, b), math.dist(b, c), math.dist(a, c)
    if min(ab, bc) <= EPSILON_PX:
        return None
    cosine = (ab * ab + bc * bc - ac * ac) / (2 * ab * bc)
    if not math.isfinite(cosine):
        return None
    # Round-off can put an otherwise valid cosine just outside [-1, 1].
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def torso_length(points):
    """Shoulder midpoint to hip midpoint, in image pixels."""
    if not all(j in points for j in CORE):
        return None
    shoulder = tuple((points[11][k] + points[12][k]) / 2 for k in (0, 1))
    hip = tuple((points[23][k] + points[24][k]) / 2 for k in (0, 1))
    length = math.dist(shoulder, hip)
    return length if math.isfinite(length) else None


def quality_flags(dependencies, landmarks, pixels):
    flags = []
    for joint in sorted(set(dependencies)):
        if joint not in pixels:
            flags.append(f'MISSING_XY:{joint}')
            continue
        point = landmarks[joint]
        visibility = point.get('visibility')
        if not finite(visibility) or not 0 <= visibility <= 1:
            flags.append(f'VISIBILITY_UNAVAILABLE:{joint}')
        elif visibility < VISIBILITY_REVIEW_LEVEL:
            flags.append(f'LOW_VISIBILITY:{joint}')
        if not (0 <= point['x'] <= 1 and 0 <= point['y'] <= 1):
            flags.append(f'OUT_OF_IMAGE:{joint}')
    return flags


def extract_features(landmarks, width, height, observer_joints=None):
    """Return raw numbers, automatic flags, and separate observer-review flags."""
    pixels = pixel_points(landmarks, width, height)
    length = torso_length(pixels)
    values, automatic, observer = {}, {}, {}
    for name, ids, _ in ANGLE_SPECS + RATIO_SPECS:
        is_angle = name.endswith('_deg')
        dependencies = set(ids) if is_angle else set(ids) | set(CORE)
        flags = quality_flags(dependencies, landmarks, pixels)
        value = None
        if all(j in pixels for j in dependencies):
            if is_angle:
                value = angle_degrees(*(pixels[j] for j in ids))
                if value is None:
                    flags.append('DEGENERATE_ANGLE')
            elif length is not None and length > EPSILON_PX:
                value = math.dist(pixels[ids[0]], pixels[ids[1]]) / length
            else:
                flags.append('DEGENERATE_TORSO')
        if value is not None and not math.isfinite(value):
            value = None
            flags.append('NONFINITE_RESULT')
        values[name] = value
        automatic[name] = ';'.join(flags) if flags else 'NO_FLAG'
        if observer_joints is None:
            observer[name] = 'NOT_REVIEWED'
        else:
            affected = sorted(dependencies & set(observer_joints))
            observer[name] = ';'.join(f'REVIEW:{j}' for j in affected) if affected else 'NOT_FLAGGED'
    return dict(values=values, automatic=automatic, observer=observer, torso_length_px=length)


def schema():
    features = []
    for name, ids, description in ANGLE_SPECS + RATIO_SPECS:
        is_angle = name.endswith('_deg')
        features.append(dict(name=name, description_tr=description,
                             kind='angle' if is_angle else 'distance_ratio',
                             unit='degree' if is_angle else 'dimensionless',
                             point_ids=ids, vertex_id=ids[1] if is_angle else None,
                             quality_dependencies=sorted(set(ids) if is_angle else set(ids) | set(CORE))))
    return dict(
        feature_count=len(features), angles=len(ANGLE_SPECS), ratios=len(RATIO_SPECS),
        features=features, joint_names=JOINTS,
        coordinates='x_px = x * image_width_px; y_px = y * image_height_px; z unused',
        angle_formula='acos((AB^2 + BC^2 - AC^2) / (2*AB*BC)) * 180/pi; vertex B',
        ratio_formula='Euclidean distance of the numerator pair / torso_length_px',
        denominator='distance((left_shoulder + right_shoulder)/2, (left_hip + right_hip)/2)',
        numerator_distances='Straight-line 2D endpoint distances, not 3D bone lengths or curved paths',
        missing_policy='Undefined measurements are null in JSON and empty cells in CSV; zero is never imputed. Finite suspect measurements remain with separate flags.',
        automatic_quality='Checks missing/nonfinite x,y, visibility < 0.5, unavailable visibility, out-of-image points and numerical degeneracy. NO_FLAG is not proof of correct landmarks.',
        observer_quality='Visual review flags from the existing review file, separately propagated through feature dependencies. They never change coordinates or values. NOT_FLAGGED is not proof of correctness.',
        visibility_review_level=VISIBILITY_REVIEW_LEVEL, numerical_zero_px=EPSILON_PX,
        invariance='Translation, uniform image scaling and image-plane rotation; not arbitrary camera viewpoint',
        selection_rationale='An initial interpretable inventory: 8 limb angles, 4 trunk corners, 21 segment/width/diagonal/endpoint ratios. No claim of optimal posture discrimination.',
        reference_posture='Existing manual selection label, not a classifier output; exclude metadata and quality fields from any future numeric feature matrix',
        quality_column_prefixes=dict(automatic='auto_qc__', observer='observer_qc__'),
    )


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def build(input_dir, output_dir):
    manifest_path = input_dir / 'manifest.json'
    landmark_path = input_dir / 'mediapipe_image_landmarks.json'
    review_path = input_dir / 'visual_review.json'
    sources = [manifest_path, landmark_path, review_path, Path(__file__)]
    manifest = json.loads(manifest_path.read_text())
    landmark_file = json.loads(landmark_path.read_text())
    review_file = json.loads(review_path.read_text())
    def indexed(rows):
        result = {r['image_id']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate image ID')
        return result
    detections = indexed(landmark_file['samples'])
    reviews = indexed(review_file['samples'])
    selections = indexed(manifest['samples'])
    if set(detections) != set(selections) or set(reviews) != set(selections):
        raise ValueError('Selection, detections and review image IDs differ')
    for row in selections.values():
        path = input_dir / 'images' / row['filename']
        if digest(path) != row['image_sha256']:
            raise ValueError(f'Source image changed: {path}')
        sources.append(path)
    hashes = {str(p.resolve()): digest(p) for p in sources}
    metadata = ['image_id', 'library', 'running_mode', 'source_video', 'source_frame',
                'timestamp_sec', 'image_width_px', 'image_height_px', 'reference_posture',
                'reference_posture_source', 'pose_detected', 'torso_length_px']
    columns = metadata + FEATURE_NAMES + ['auto_qc__' + n for n in FEATURE_NAMES] + ['observer_qc__' + n for n in FEATURE_NAMES]
    rows, details = [], []
    for image_id, source in selections.items():
        detection, review = detections[image_id], reviews[image_id]
        raw = {p['landmark_id']: p for p in detection['points']}
        if len(raw) != len(detection['points']):
            raise ValueError(f'Duplicate landmark ID in {image_id}')
        if bool(raw) != detection['detected']:
            raise ValueError(f'Inconsistent detection flag in {image_id}')
        result = extract_features(raw, manifest['width'], manifest['height'], review['review_joint_ids'])
        row = [image_id, 'MediaPipe', landmark_file['mode'], Path(manifest['source']).name,
               source['frame'], source['timestamp_sec'], manifest['width'], manifest['height'],
               source['reference_posture'], source['original_reference_record']['posture_source'],
               detection['detected'], result['torso_length_px']]
        row += [result['values'][n] for n in FEATURE_NAMES]
        row += [result['automatic'][n] for n in FEATURE_NAMES]
        row += [result['observer'][n] for n in FEATURE_NAMES]
        assert len(row) == len(columns)
        rows.append(row)
        details.append(dict(image_id=image_id,
                            numeric_features=sum(v is not None for v in result['values'].values()),
                            automatic_flagged_features=[n for n in FEATURE_NAMES if result['automatic'][n] != 'NO_FLAG'],
                            observer_flagged_features=[n for n in FEATURE_NAMES if result['observer'][n].startswith('REVIEW:')]))
    output_dir.mkdir(parents=True, exist_ok=False)
    numeric = ['source_frame', 'timestamp_sec', 'image_width_px', 'image_height_px', 'torso_length_px'] + FEATURE_NAMES
    save_json(output_dir / 'feature_table.json', dict(columns=columns, rows=rows, numeric_columns=numeric, feature_columns=FEATURE_NAMES))
    save_json(output_dir / 'feature_schema.json', schema())
    save_json(output_dir / 'extraction_summary.json', dict(images=len(rows), features_per_image=len(FEATURE_NAMES),
              total_numeric_features=sum(r['numeric_features'] for r in details),
              classifier_run=False, coordinates_modified=False, quality_flags_modify_values=False,
              source_sha256=hashes, samples=details))
    assert all(digest(Path(p)) == h for p, h in hashes.items())
    print(json.dumps(dict(images=len(rows), angles=12, ratios=21, numeric_features=sum(r['numeric_features'] for r in details),
                          columns=len(columns), output=str(output_dir)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_INPUT / 'features_v1')
    args = parser.parse_args()
    build(args.input_dir.resolve(), args.output_dir.resolve())
