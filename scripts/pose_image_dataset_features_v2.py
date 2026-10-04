"""Shared anatomical mapping and identical 2D geometry for four libraries."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

from extract_posture_image_features_v1 import (
    JOINTS, CORE, ANGLE_SPECS, RATIO_SPECS, EPSILON_PX,
    angle_degrees, torso_length, finite, schema, save_json,
)
from export_posture_image_csv_v1 import encoded, verify_roundtrip

COCO_MAP = {11:5,12:6,13:7,14:8,15:9,16:10,23:11,24:12,25:13,26:14,27:15,28:16}
MAPS = dict(mediapipe={j:j for j in JOINTS},
    openpose={11:5,12:2,13:6,14:3,15:7,16:4,23:12,24:9,25:13,26:10,27:14,28:11},
    mmpose=COCO_MAP.copy(), deepercut=COCO_MAP.copy())
LEVELS = dict(mediapipe=.5, openpose=.3, mmpose=.3)
FAMILIES = dict(xy=[(JOINTS[j]+'_'+axis+'_px',(j,)) for j in JOINTS for axis in ('x','y')],
    angles=[(n,ids) for n,ids,_ in ANGLE_SPECS],ratios=[(n,ids) for n,ids,_ in RATIO_SPECS])
META = ['image_id','source_file','library','image_width_px','image_height_px',
        'detection_status','native_candidate_count','selected_native_index','torso_length_px','visual_review']


def canonical(library, row, result):
    native = {p['native_id']:p for p in result['points']}
    points = []
    for joint,name in JOINTS.items():
        p = native.get(MAPS[library][joint],{})
        available = bool(p.get('coordinate_available') and finite(p.get('x_px')) and finite(p.get('y_px')))
        if library == 'openpose':
            available = available and finite(p.get('score')) and p['score']>0
        if library == 'deepercut':
            available = available and p.get('assigned_detections',0)>0 and p.get('unary_score_sum',0)>0
        points.append(dict(canonical_id=joint,joint_name=name,native_id=MAPS[library][joint],
            coordinate_available=available,x_px=p.get('x_px'),y_px=p.get('y_px'),
            x=p['x_px']/row['width'] if available else None,
            y=p['y_px']/row['height'] if available else None,
            score=p.get('score'),score_type=p.get('score_type'),
            assigned_detections=p.get('assigned_detections'),unary_score_sum=p.get('unary_score_sum')))
    return points


def measurements(points):
    xy = {p['canonical_id']:(p['x_px'],p['y_px']) for p in points if p['coordinate_available']}
    length = torso_length(xy)
    values, numerical_flags = {}, {}
    for joint,name in JOINTS.items():
        for axis,index in (('x',0),('y',1)):
            values[name+'_'+axis+'_px'] = xy[joint][index] if joint in xy else None
    for name,ids,_ in ANGLE_SPECS+RATIO_SPECS:
        deps = set(ids) if name.endswith('_deg') else set(ids)|set(CORE)
        value,flags = None,[]
        if deps<=set(xy):
            if name.endswith('_deg'):
                value = angle_degrees(*(xy[j] for j in ids))
                if value is None: flags.append('DEGENERATE_ANGLE')
            elif length is not None and length>EPSILON_PX:
                value = math.dist(xy[ids[0]],xy[ids[1]])/length
            else: flags.append('DEGENERATE_TORSO')
        if value is not None and not finite(value):
            value = None; flags.append('NONFINITE_RESULT')
        values[name] = value
        numerical_flags[name] = flags
    return values,length,numerical_flags


def flags(library, deps, points):
    result = []
    by_id = {p['canonical_id']:p for p in points}
    for j in sorted(set(deps)):
        p = by_id[j]
        if not p['coordinate_available']:
            result.append('MISSING_XY:'+str(j)); continue
        if not finite(p['score']): result.append('SCORE_UNAVAILABLE:'+str(j))
        elif library in LEVELS and p['score']<LEVELS[library]: result.append('LOW_SCORE:'+str(j))
        if library=='deepercut' and p['assigned_detections']>1:
            result.append('MULTIPLE_ASSIGNED_DETECTIONS:'+str(j))
        if not (0<=p['x']<=1 and 0<=p['y']<=1): result.append('OUT_OF_IMAGE:'+str(j))
    return result


def write_csv(path, columns, rows):
    with path.open('x',encoding='utf-8',newline='') as handle:
        writer = csv.writer(handle,lineterminator='\n'); writer.writerow(columns)
        writer.writerows([[encoded(v) for v in row] for row in rows])
    return dict(file=str(path),**verify_roundtrip(path,columns,rows))


def export(run, libraries):
    manifest = json.loads((run/'manifest.json').read_text())
    all_common,checks,summary = [],[],{}
    data = run/'csv'; data.mkdir(exist_ok=False)
    combined = run/'combined'; combined.mkdir(exist_ok=False)
    for library in libraries:
        raw = json.loads((run/'native'/library/'native.json').read_text())
        lookup = {r['image_id']:r for r in raw['samples']}
        expected = {r['image_id'] for r in manifest['samples']}
        if set(lookup)!=expected or len(raw['samples'])!=len(expected):
            raise ValueError('Missing/duplicate native image records for '+library)
        tables = {family:[] for family in FAMILIES}
        combined_rows = []
        stats = dict(images=len(expected),detection_status=dict(Counter(r['detection_status'] for r in raw['samples'])),
            available_common_points=0,numeric_xy=0,numeric_angles=0,numeric_ratios=0)
        for row in manifest['samples']:
            result = lookup[row['image_id']]
            points = canonical(library,row,result)
            values,length,numeric_flags = measurements(points)
            prefix = [row['image_id'],row['relative_path'],library,row['width'],row['height'],
                result['detection_status'],result['candidate_count'],result['selected_index'],length,'NOT_REVIEWED']
            common = dict(image_id=row['image_id'],library=library,width=row['width'],height=row['height'],
                detection_status=result['detection_status'],candidate_count=result['candidate_count'],
                selected_index=result['selected_index'],points=points)
            all_common.append(common)
            stats['available_common_points'] += sum(p['coordinate_available'] for p in points)
            combined_values,combined_qc = [],[]
            for family,specs in FAMILIES.items():
                qc = []
                for name,ids in specs:
                    deps = set(ids)|set(CORE) if family=='ratios' else set(ids)
                    reasons = flags(library,deps,points)+numeric_flags.get(name,[])
                    if result['candidate_count']>1:
                        reasons.append('MULTIPLE_BODY_CLUSTERS' if library=='deepercut' else 'MULTIPLE_PEOPLE')
                    if result['detection_status']!='POSE_RETURNED': reasons.append(result['detection_status'])
                    qc.append(';'.join(reasons) if reasons else 'NO_FLAG')
                names = [n for n,_ in specs]
                tables[family].append(prefix+[values[n] for n in names]+qc)
                combined_values += [values[n] for n in names]
                combined_qc += qc
                stats['numeric_'+family] += sum(values[n] is not None for n in names)
            combined_rows.append(prefix+combined_values+combined_qc)
        for family,rows in tables.items():
            names = [n for n,_ in FAMILIES[family]]
            checks.append(write_csv(data/(library+'_'+family+'.csv'),META+names+['qc__'+n for n in names],rows))
        all_names = [n for specs in FAMILIES.values() for n,_ in specs]
        checks.append(write_csv(combined/(library+'_features.csv'),META+all_names+['qc__'+n for n in all_names],combined_rows))
        summary[library] = stats
    save_json(run/'common_keypoints.json',dict(joint_maps=MAPS,samples=all_common))
    columns = ['image_id','library','joint_name','canonical_id','native_id','x_px','y_px',
               'x_normalized','y_normalized','coordinate_available','score','score_type',
               'assigned_detections','unary_score_sum']
    rows = [[r['image_id'],r['library'],p['joint_name'],p['canonical_id'],p['native_id'],
        p['x_px'],p['y_px'],p['x'],p['y'],p['coordinate_available'],p['score'],p['score_type'],
        p['assigned_detections'],p['unary_score_sum']] for r in all_common for p in r['points']]
    checks.append(write_csv(run/'common_keypoints_xy.csv',columns,rows))
    info = schema()
    info.update(native_maps=MAPS,feature_families={family:[n for n,_ in specs] for family,specs in FAMILIES.items()},
        combined_feature_columns=[n for specs in FAMILIES.values() for n,_ in specs],
        metadata_columns=META,xy_coordinate_unit='Decoded input image pixels. No resizing before measurements.',
        quality_policy='QC columns never change finite measurements. Blank = missing/nonfinite/undefined, never filled with zero. NO_FLAG is not visual verification; visual_review stays NOT_REVIEWED.',
        score_policy='MP visibility, BODY25 confidence, RTMPose scores, and mean assigned DeeperCut unary scores are different quantities. Native RTMPose values above 1 are retained. DeeperCut has no added score cutoff.',
        label_policy='No posture/head labels read, inferred or predicted. No label enters features. No manual coordinate corrections.',
        library_score_review_levels=LEVELS)
    save_json(run/'feature_schema.json',info)
    return all_common,summary,checks
