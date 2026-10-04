"""Independently verify a saved folder run against native points and geometry."""
import argparse
import csv
import json
import math
from pathlib import Path

from extract_posture_image_features_v1 import digest
from pose_image_dataset_features_v2 import JOINTS,CORE,FAMILIES,MAPS


def verify(run):
    import cv2
    import numpy as np
    manifest=json.loads((run/'manifest.json').read_text())
    summary=json.loads((run/'run_summary.json').read_text())
    protocol=json.loads((run/'run_protocol.json').read_text())
    sources={r['image_id']:r for r in manifest['samples']}
    common=json.loads((run/'common_keypoints.json').read_text())['samples']
    libraries=protocol['libraries']
    assert len(common)==len(sources)*len(libraries)
    assert summary['csv_count']==4*len(libraries)
    for path,sha in protocol['source_sha256'].items(): assert digest(Path(path))==sha,path
    for row in sources.values():
        source=Path(row['source_path']); assert digest(source)==row['source_sha256']
        if row['input_status']=='READY':
            staged=run/'images'/row['staged_filename']
            assert digest(staged)==row['staged_sha256']
            a=cv2.imdecode(np.frombuffer(source.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            b=cv2.imread(str(staged)); np.testing.assert_array_equal(a,b)
            assert b.shape[:2]==(row['height'],row['width'])
    tables={}
    native={lib:{r['image_id']:r for r in json.loads((run/'native'/lib/'native.json').read_text())['samples']} for lib in libraries}
    for lib in libraries:
        for family in FAMILIES:
            with (run/'csv'/(lib+'_'+family+'.csv')).open(newline='',encoding='utf-8') as f:
                rows=list(csv.DictReader(f))
            assert len(rows)==len(sources) and {r['image_id'] for r in rows}==set(sources)
            assert all(r['visual_review']=='NOT_REVIEWED' and r['library']==lib for r in rows)
            tables[lib,family]={r['image_id']:r for r in rows}
        with (run/'combined'/(lib+'_features.csv')).open(newline='',encoding='utf-8') as f:
            joined=list(csv.DictReader(f))
        assert len(joined)==len(sources)
        for row in joined:
            for family,specs in FAMILIES.items():
                separate=tables[lib,family][row['image_id']]
                for name,_ in specs:
                    assert row[name]==separate[name]
                    assert row['qc__'+name]==separate['qc__'+name]
    maximum_angle_error=0.
    for record in common:
        lib,identifier=record['library'],record['image_id']
        src=sources[identifier]
        raw=native[lib][identifier]
        by_id={p['native_id']:p for p in raw['points']}
        for p in record['points']:
            assert p['native_id']==MAPS[lib][p['canonical_id']]
            if p['coordinate_available']:
                original=by_id[p['native_id']]
                assert p['x_px']==original['x_px'] and p['y_px']==original['y_px']
                assert p['score']==original['score']
                assert math.isclose(p['x']*src['width'],p['x_px'],abs_tol=1e-10)
                assert math.isclose(p['y']*src['height'],p['y_px'],abs_tol=1e-10)
        xy={p['canonical_id']:(p['x_px'],p['y_px']) for p in record['points'] if p['coordinate_available']}
        length=math.hypot((xy[11][0]+xy[12][0]-xy[23][0]-xy[24][0])/2,
            (xy[11][1]+xy[12][1]-xy[23][1]-xy[24][1])/2) if set(CORE)<=set(xy) else None
        for j,name in JOINTS.items():
            for axis,k in (('x',0),('y',1)):
                cell=tables[lib,'xy'][identifier][name+'_'+axis+'_px']
                assert (cell=='')==(j not in xy)
                if j in xy: assert float(cell)==xy[j][k]
        for name,(a,b,c) in FAMILIES['angles']:
            cell=tables[lib,'angles'][identifier][name]
            if not {a,b,c}<=set(xy): assert cell==''; continue
            u=(xy[a][0]-xy[b][0],xy[a][1]-xy[b][1]); v=(xy[c][0]-xy[b][0],xy[c][1]-xy[b][1])
            if min(math.hypot(*u),math.hypot(*v))<=1e-9: assert cell==''; continue
            expected=math.degrees(math.atan2(abs(u[0]*v[1]-u[1]*v[0]),u[0]*v[0]+u[1]*v[1]))
            error=abs(float(cell)-expected); maximum_angle_error=max(maximum_angle_error,error)
            assert error<1e-6,(identifier,lib,name,error)
        for name,(a,b) in FAMILIES['ratios']:
            cell=tables[lib,'ratios'][identifier][name]
            if not {a,b,*CORE}<=set(xy) or length is None or length<=1e-9:
                assert cell==''; continue
            expected=math.hypot(xy[a][0]-xy[b][0],xy[a][1]-xy[b][1])/length
            assert math.isclose(float(cell),expected,rel_tol=1e-12,abs_tol=1e-12),(identifier,lib,name)
        if lib=='deepercut' and raw['detection_status']=='POSE_RETURNED':
            pack=np.load(run/'native/deepercut'/(identifier+'_native.npz'))
            positions=pack['positions']; unary=pack['unary_scores']; labels=pack['labels']
            counts=np.zeros_like(pack['assigned_counts']); sums=np.zeros_like(pack['unary_sums'])
            weighted=np.zeros_like(pack['person_xy'])
            for i,(joint,person) in enumerate(labels):
                counts[person,joint]+=1; sums[person,joint]+=unary[i,0]
                weighted[person,joint]+=positions[i]*unary[i,0]
            np.testing.assert_array_equal(counts,pack['assigned_counts'])
            np.testing.assert_allclose(sums,pack['unary_sums'],rtol=0,atol=1e-12)
            candidates=[i for i in range(len(sums)) if np.any(sums[i,5:17]>0)]
            selected=max(candidates,key=lambda i:(np.count_nonzero(sums[i,5:17]>0),np.sum(sums[i,5:17]),-i))
            assert selected==raw['selected_index']
            for p in raw['points']:
                j=p['native_id']; assert p['coordinate_available']==bool(sums[selected,j]>0)
                if p['coordinate_available']:
                    np.testing.assert_allclose([p['x_px'],p['y_px']],weighted[selected,j]/sums[selected,j],rtol=0,atol=1e-12)
                    assert math.isclose(p['score'],sums[selected,j]/counts[selected,j],abs_tol=1e-12)
    return dict(images=len(sources),libraries=len(libraries),method_csvs=3*len(libraries),combined_csvs=len(libraries),
                maximum_independent_angle_error_degrees=maximum_angle_error,
                staged_pixels_and_inputs_unchanged=True,native_coordinates_and_missing_values_preserved=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    print(json.dumps(verify(parser.parse_args().run),indent=2))
