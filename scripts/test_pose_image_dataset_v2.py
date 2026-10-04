"""Behavioral checks for folder input, native mappings and feature CSVs."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from build_pose_image_dataset_v2 import find_images,validate_output,stage
from pose_image_dataset_features_v2 import (
    MAPS,JOINTS,CORE,FAMILIES,canonical,measurements,flags,export,
)
from extract_posture_image_features_v1 import save_json


def native_point(i,x=20.,y=30.,score=.9,available=True,**extra):
    return dict(native_id=i,x_px=x,y_px=y,score=score,score_type='test',coordinate_available=available,**extra)


class FolderDatasetTests(unittest.TestCase):
    def test_recursive_input_unicode_duplicate_names_and_extensions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'yan yatış').mkdir()
            for name in ('aynı.PNG','yan yatış/aynı.PNG','test.jpg','skip.csv'):
                (root/name).write_bytes(b'test')
            self.assertEqual(len(find_images(root)),2)
            self.assertEqual(len(find_images(root,True)),3)
            self.assertEqual(len({p.relative_to(root).as_posix() for p in find_images(root,True)}),3)

    def test_output_refuses_overwrite_and_input_subfolder(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(FileExistsError): validate_output(root,root)
            with self.assertRaises(ValueError): validate_output(root,root/'new')
            validate_output(root,root.parent/(root.name+'_new'))

    def test_staging_keeps_different_dimensions_and_bad_image_record(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); source=root/'src'; source.mkdir(); output=root/'new'
            for name,shape in (('a.png',(40,80)),('b.png',(90,30))):
                self.assertTrue(cv2.imwrite(str(source/name),np.zeros((*shape,3),dtype=np.uint8)))
            (source/'bozuk.png').write_bytes(b'not an image')
            manifest=stage(source,output,find_images(source))
            self.assertEqual([(r['height'],r['width']) for r in manifest['samples'][:2]],[(40,80),(90,30)])
            self.assertEqual(manifest['samples'][2]['input_status'],'INPUT_ERROR')
            self.assertFalse(manifest['labels_used'])

    def test_missing_points_origin_and_native_score_above_one(self):
        row=dict(width=100,height=200)
        native=dict(points=[native_point(MAPS['openpose'][11],0.,0.,0.)])
        self.assertFalse(canonical('openpose',row,native)[0]['coordinate_available'])
        native=dict(points=[native_point(MAPS['mmpose'][11],0.,0.,1.2)])
        p=canonical('mmpose',row,native)[0]
        self.assertTrue(p['coordinate_available']); self.assertEqual(p['score'],1.2)
        self.assertEqual(flags('mmpose',[11],[p]),[])
        native=dict(points=[native_point(MAPS['deepercut'][11],0.,0.,.5,assigned_detections=0,unary_score_sum=0.)])
        self.assertFalse(canonical('deepercut',row,native)[0]['coordinate_available'])

    def test_deepercut_origin_valid_when_supported(self):
        p=canonical('deepercut',dict(width=100,height=200),dict(points=[native_point(5,0.,0.,.5,
            assigned_detections=1,unary_score_sum=.5)]))[0]
        self.assertTrue(p['coordinate_available']); self.assertEqual(p['x'],0.)

    def test_ratio_quality_includes_denominator_dependencies(self):
        native=dict(points=[native_point(MAPS['mmpose'][j],score=.1 if j==11 else .9) for j in JOINTS])
        points=canonical('mmpose',dict(width=100,height=200),native)
        for _,ids in FAMILIES['ratios']:
            self.assertIn('LOW_SCORE:11',flags('mmpose',set(ids)|set(CORE),points))
        self.assertNotIn('LOW_SCORE:11',flags('mmpose',[24,26,28],points))

    def test_geometry_scale_invariance_and_independent_angles(self):
        xy={11:(20.,20.),12:(80.,20.),13:(10.,40.),14:(90.,50.),15:(30.,70.),16:(80.,90.),
            23:(30.,110.),24:(70.,110.),25:(25.,150.),26:(75.,155.),27:(30.,190.),28:(80.,200.)}
        def points(scale):
            return [dict(canonical_id=j,x_px=x*scale,y_px=y*scale,coordinate_available=True) for j,(x,y) in xy.items()]
        values,length,_=measurements(points(1)); larger,length2,_=measurements(points(3))
        self.assertAlmostEqual(length2,length*3)
        for family in ('angles','ratios'):
            for name,_ in FAMILIES[family]: self.assertAlmostEqual(values[name],larger[name],places=9)
        for name,(a,b,c) in FAMILIES['angles']:
            u=(xy[a][0]-xy[b][0],xy[a][1]-xy[b][1]); v=(xy[c][0]-xy[b][0],xy[c][1]-xy[b][1])
            expected=math.degrees(math.atan2(abs(u[0]*v[1]-u[1]*v[0]),u[0]*v[0]+u[1]*v[1]))
            self.assertAlmostEqual(values[name],expected,places=9)

    def test_degenerate_torso_is_blank_and_zero_distance_remains_zero(self):
        points=[dict(canonical_id=j,x_px=10.,y_px=20.,coordinate_available=True) for j in JOINTS]
        values,length,reason=measurements(points)
        self.assertEqual(length,0.)
        self.assertIsNone(values['shoulder_width_ratio']); self.assertIn('DEGENERATE_TORSO',reason['shoulder_width_ratio'])
        for p in points:
            if p['canonical_id'] in (23,24): p['y_px']=100.
        values,_,_=measurements(points)
        self.assertEqual(values['shoulder_width_ratio'],0.)

    def test_four_libraries_three_families_arbitrary_row_count_no_manual_file(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp); (run/'native').mkdir()
            rows=[dict(image_id='image_'+str(i),relative_path='x'+str(i)+'.png',width=100,height=200,
                input_status='READY') for i in range(3)]
            save_json(run/'manifest.json',dict(samples=rows))
            for library in MAPS:
                (run/'native'/library).mkdir()
                save_json(run/'native'/library/'native.json',dict(samples=[dict(image_id=r['image_id'],
                    points=[],detection_status='NO_POSE',candidate_count=0,selected_index=None) for r in rows]))
            _,summary,checks=export(run,list(MAPS))
            self.assertEqual(len(list((run/'csv').glob('*.csv'))),12)
            self.assertEqual(len(list((run/'combined').glob('*.csv'))),4)
            for library in MAPS:
                self.assertEqual(summary[library]['images'],3)
                for family,specs in FAMILIES.items():
                    with (run/'csv'/(library+'_'+family+'.csv')).open(newline='') as f:
                        table=list(csv.DictReader(f))
                    self.assertEqual(len(table),3)
                    self.assertTrue(all(r['visual_review']=='NOT_REVIEWED' for r in table))
                    self.assertTrue(all(r[name]=='' for r in table for name,_ in specs))
            self.assertEqual(len(checks),17)


if __name__=='__main__': unittest.main(verbosity=2)
