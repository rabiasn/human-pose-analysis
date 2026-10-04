"""One folder -> four ready pose models -> 12 method CSVs and 4 combined CSVs.

Run with the existing .venv_mediapipe/bin/python. No installation/training.
Output must be new and outside the input folder. Old project files are preserved.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from extract_posture_image_features_v1 import ROOT, digest, save_json
from pose_image_dataset_features_v2 import MAPS, JOINTS, flags, export

LIBRARIES = ('mediapipe','openpose','mmpose','deepercut')
EXTENSIONS = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}
WORKER = ROOT/'scripts/pose_image_dataset_worker_v2.py'
MP_PYTHON = ROOT/'.venv_mediapipe/bin/python'
MM_PYTHON = ROOT/'.venv_mmpose/bin/python'
OP = ROOT/'external/openpose/build/examples/openpose/openpose.bin'
OP_MODELS = ROOT/'external/openpose/models'


def find_images(folder, recursive=False):
    if not folder.is_dir(): raise ValueError('Input is not a directory: '+str(folder))
    paths = folder.rglob('*') if recursive else folder.iterdir()
    return sorted((p for p in paths if p.is_file() and p.suffix.lower() in EXTENSIONS),
                  key=lambda p:p.relative_to(folder).as_posix())


def validate_output(input_dir, output):
    if output.exists(): raise FileExistsError('Choose a NEW output directory: '+str(output))
    if output == input_dir or input_dir in output.parents:
        raise ValueError('Output must be outside the input directory.')


def model_files(libraries):
    result = [Path(__file__),WORKER,ROOT/'scripts/pose_image_dataset_features_v2.py',
        ROOT/'scripts/extract_posture_image_features_v1.py',ROOT/'scripts/export_posture_image_csv_v1.py']
    if 'mediapipe' in libraries:
        result += [MP_PYTHON,ROOT/'models/mediapipe/pose_landmarker_full.task']
    if 'openpose' in libraries:
        result += [OP,OP_MODELS/'pose/body_25/pose_iter_584000.caffemodel',OP_MODELS/'pose/body_25/pose_deploy.prototxt']
    if 'mmpose' in libraries:
        result += [MM_PYTHON,ROOT/'models/mmpose/rtmdet_m_person.pth',ROOT/'models/mmpose/rtmpose-m_coco_256x192.pth',
            ROOT/'external/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py',
            ROOT/'external/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py']
    if 'deepercut' in libraries:
        repo = ROOT/'external/pose-tensorflow'
        result += [repo/'demo/pose_cfg_multi.yaml',repo/'models/coco/coco-resnet-101.data-00000-of-00001',
            repo/'models/coco/coco-resnet-101.index',repo/'models/coco/coco-resnet-101.meta',
            repo/'multiperson/predict.py',repo/'multiperson/detections.py',repo/'dataset/mscoco.py',
            repo/'nnet/predict.py',repo/'nnet/pose_net.py']
        result += sorted(p for p in (repo/'models/coco/pairwise').glob('*') if p.is_file())
    for path in result:
        if not path.is_file(): raise FileNotFoundError('Missing existing dependency: '+str(path))
    return result


def stage(input_dir, output, paths):
    import cv2
    import numpy as np
    (output/'images').mkdir(parents=True,exist_ok=False)
    records = []
    for index,path in enumerate(paths,1):
        image_id = 'image_{:06d}'.format(index)
        payload = path.read_bytes()
        import hashlib
        row = dict(image_id=image_id,relative_path=path.relative_to(input_dir).as_posix(),
            source_path=str(path),source_sha256=hashlib.sha256(payload).hexdigest(),
            staged_filename=image_id+'.png',width=None,height=None,input_status='INPUT_ERROR',error=None)
        try:
            image = cv2.imdecode(np.frombuffer(payload,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None: raise ValueError('Image could not be decoded')
            height,width = image.shape[:2]
            if min(width,height)<=0: raise ValueError('Image has no pixels')
            target = output/'images'/row['staged_filename']
            if not cv2.imwrite(str(target),image): raise RuntimeError('Lossless PNG staging failed')
            row.update(width=width,height=height,input_status='READY',staged_sha256=digest(target))
        except Exception as exc:
            row['error'] = type(exc).__name__+': '+str(exc)
        records.append(row)
    manifest = dict(input_directory=str(input_dir),samples=records,
        image_decode='OpenCV IMREAD_COLOR; JPEG EXIF orientation applied by decoder. No crop/rescale. Every model sees the same lossless PNG pixels. Geometry uses that decoded image width/height.',
        image_count=len(records),labels_used=False,manual_points=False)
    save_json(output/'manifest.json',manifest)
    return manifest


def execute(command, log, env):
    with log.open('x',encoding='utf-8') as handle:
        process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        try:
            for line in process.stdout:
                handle.write(line); handle.flush()
                if 'image_' in line or 'Traceback' in line or 'Error:' in line:
                    print(line.rstrip(),flush=True)
            status = process.wait()
        except BaseException:
            process.terminate(); process.wait(); raise
    if status:
        raise RuntimeError('Library command exited {}. See {}'.format(status,log))


def openpose(output, manifest, env):
    from pose_image_dataset_worker_v2 import empty, point
    native = output/'native/openpose'; raw = native/'raw_json'; raw.mkdir()
    command = [str(OP),'--image_dir',str(output/'images'),'--model_folder',str(OP_MODELS),
        '--model_pose','BODY_25','--net_resolution','-1x256','--keypoint_scale','0',
        '--display','0','--render_pose','0','--write_json',str(raw)]
    if any(r['input_status']=='READY' for r in manifest['samples']):
        execute(command,output/'logs/openpose.log',env)
    records = []
    for row in manifest['samples']:
        if row['input_status']!='READY':
            records.append(empty(row)); continue
        source = raw/(row['image_id']+'_keypoints.json')
        people = json.loads(source.read_text())['people']
        for person in people:
            if len(person['pose_keypoints_2d'])!=75: raise ValueError('Expected BODY25 native points')
        def score(person):
            return sum(person['pose_keypoints_2d'][3*j+2] for j in MAPS['openpose'].values())/12
        selected = max(range(len(people)),key=lambda i:score(people[i])) if people else None
        values = people[selected]['pose_keypoints_2d'] if selected is not None else []
        points = [point(j,values[3*j],values[3*j+1],values[3*j+2],'BODY_25_confidence',
            available=values[3*j+2]>0) for j in range(25)] if values else []
        records.append(dict(image_id=row['image_id'],detection_status='POSE_RETURNED' if points else 'NO_POSE',
            candidate_count=len(people),selected_index=selected,points=points,error=None))
    save_json(native/'native.json',dict(library='openpose',samples=records,z_used=False,
        coordinate_unit='decoded input-image pixels',command=command,training=False,manual_coordinate_corrections=False))


def previews(output, manifest, common, libraries):
    import cv2
    import numpy as np
    from compare_posture_image_libraries_v1 import EDGES,SHORT
    target = output/'previews'; target.mkdir()
    lookup = {(r['image_id'],r['library']):r for r in common}
    def card(image,title,subtitle):
        scale = min(320/image.shape[1],480/image.shape[0])
        resized = cv2.resize(image,(max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))))
        canvas = np.full((544,320,3),25,dtype=np.uint8)
        x,y = (320-resized.shape[1])//2,64+(480-resized.shape[0])//2
        canvas[y:y+resized.shape[0],x:x+resized.shape[1]] = resized
        cv2.putText(canvas,title,(6,22),0,.47,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(canvas,subtitle,(6,46),0,.34,(230,230,230),1,cv2.LINE_AA)
        return canvas
    for row in manifest['samples']:
        image = cv2.imread(str(output/'images'/row['staged_filename'])) if row['input_status']=='READY' else None
        if image is None: image = np.full((480,320,3),25,dtype=np.uint8)
        panels = [card(image.copy(),row['image_id']+' | input',row['input_status'])]
        for library in libraries:
            record = lookup[row['image_id'],library]
            im = image.copy()
            points = {p['canonical_id']:p for p in record['points'] if p['coordinate_available'] and
                      0<=p['x_px']<image.shape[1] and 0<=p['y_px']<image.shape[0]}
            def pixel(j): return round(points[j]['x_px']),round(points[j]['y_px'])
            for a,b in EDGES:
                if a in points and b in points:
                    cv2.line(im,pixel(a),pixel(b),(30,210,40),2,cv2.LINE_AA)
            for joint in points:
                color = (0,170,255) if flags(library,[joint],record['points']) else (255,200,0)
                cv2.circle(im,pixel(joint),5,color,-1,cv2.LINE_AA)
                x,y = pixel(joint)
                for thickness,ink in ((3,(0,0,0)),(1,(255,255,255))):
                    cv2.putText(im,SHORT[joint],(x+5,y-4),0,.40,ink,thickness,cv2.LINE_AA)
            panels.append(card(im,library,record['detection_status']+' | NOT REVIEWED'))
        if not cv2.imwrite(str(target/(row['image_id']+'.jpg')),np.hstack(panels)):
            raise RuntimeError('Comparison preview write failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='PNG/JPG/JPEG/BMP/TIFF/WEBP folder')
    parser.add_argument('--output',type=Path,help='New output folder (default: timestamped project outputs)')
    parser.add_argument('--recursive',action='store_true',help='Include subfolders; preserve relative source paths')
    parser.add_argument('--libraries',nargs='+',choices=LIBRARIES,default=list(LIBRARIES))
    parser.add_argument('--no-previews',action='store_true',help='Skip skeleton comparison JPGs')
    args = parser.parse_args()
    libraries = list(dict.fromkeys(args.libraries))
    input_dir = args.input.expanduser().resolve()
    output = (args.output.expanduser().resolve() if args.output else ROOT/'outputs/image_feature_datasets'/
        datetime.now(ZoneInfo('Europe/Istanbul')).strftime('run_%Y%m%d_%H%M%S'))
    validate_output(input_dir,output)
    paths = find_images(input_dir,args.recursive)
    if not paths: raise ValueError('No supported images in input folder.')
    sources = model_files(libraries)
    docker = shutil.which('docker')
    image_info = None
    if 'deepercut' in libraries:
        if not docker: raise FileNotFoundError('Existing Docker CLI not found')
        subprocess.run([docker,'info','--format','{{.ServerVersion}}'],check=True,stdout=subprocess.DEVNULL,timeout=30)
        image_info = json.loads(subprocess.check_output([docker,'image','inspect','pose-tf14:latest'],text=True))[0]
    hashes = {str(p):digest(p) for p in sources}
    output.mkdir(parents=True,exist_ok=False)
    (output/'logs').mkdir(); (output/'native').mkdir()
    manifest = stage(input_dir,output,paths)
    commands = {}
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='4',
        MPLCONFIGDIR=str(output/'runtime_cache/mpl'),XDG_CACHE_HOME=str(output/'runtime_cache'))
    save_json(output/'run_protocol.json',dict(libraries=libraries,source_sha256=hashes,
        docker_image_id=image_info['Id'] if image_info else None,docker_image='pose-tf14:latest' if image_info else None,
        methods=['original x,y','cosine-law 2D angles','distances / torso midpoint length'],
        model_selection=dict(mediapipe='PoseLandmarker full, IMAGE, max 1 pose',openpose='BODY25 CPU, -1x256, highest mean score over common joints',
            mmpose='RTMDet-M person score > 0.3, highest person score; RTMPose-M COCO17',
            deepercut='Existing COCO ResNet101 and native multicut; most supported common joints, tie largest unary sum then lowest cluster ID'),
        quality_levels_are_not_posture_thresholds=True,model_training=False,manual_coordinate_corrections=False,
        scope='Folder of still images; no posture/head/movement classifier, no temporal video feature packing.'))
    try:
        for library in libraries:
            (output/'native'/library).mkdir()
            print('Running {} on {} images...'.format(library,len(paths)),flush=True)
            if library=='openpose':
                openpose(output,manifest,env); continue
            common_args = ['--library',library,'--root',str(ROOT),'--run',str(output),
                           '--output',str(output/'native'/library/'native.json')]
            if library=='deepercut':
                command = [docker,'run','--rm','--pull','never','--platform','linux/amd64','--network','none',
                    '--read-only','--tmpfs','/tmp','--cpus','4','-e','PYTHONDONTWRITEBYTECODE=1',
                    '-e','PYTHONIOENCODING=utf-8','-e','MPLCONFIGDIR=/tmp/mpl','-e','TF_CUDNN_USE_AUTOTUNE=0','-e','OMP_NUM_THREADS=4',
                    '--mount','type=bind,src='+str(ROOT)+',dst=/project,readonly',
                    '--mount','type=bind,src='+str(output)+',dst=/run,readonly',
                    '--mount','type=bind,src='+str(output/'native/deepercut')+',dst=/result',
                    '-w','/project/external/pose-tensorflow','pose-tf14:latest','python3','-B',
                    '/project/scripts/pose_image_dataset_worker_v2.py','--library','deepercut',
                    '--root','/project','--run','/run','--output','/result/native.json']
            else:
                command = [str(MP_PYTHON if library=='mediapipe' else MM_PYTHON),'-B',str(WORKER),*common_args]
            commands[library] = command
            execute(command,output/'logs'/(library+'.log'),env)
        common,summary,checks = export(output,libraries)
        if not args.no_previews: previews(output,manifest,common,libraries)
        for p,h in hashes.items():
            if digest(Path(p))!=h: raise RuntimeError('Dependency changed during run: '+p)
        for row in manifest['samples']:
            if digest(Path(row['source_path']))!=row['source_sha256']:
                raise RuntimeError('Input changed during run: '+row['source_path'])
            if row['input_status']=='READY' and digest(output/'images'/row['staged_filename'])!=row['staged_sha256']:
                raise RuntimeError('Staged image changed during inference: '+row['image_id'])
        errors = sum(r['detection_status'] in ('INPUT_ERROR','INFERENCE_ERROR') for library in libraries
            for r in json.loads((output/'native'/library/'native.json').read_text())['samples'])
        save_json(output/'run_summary.json',dict(status='COMPLETED_WITH_IMAGE_ERRORS' if errors else 'COMPLETED',
            image_count=len(paths),csv_count=4*len(libraries),method_csv_count=3*len(libraries),
            combined_csv_count=len(libraries),by_library=summary,csv_roundtrip=checks,
            commands=commands,inputs_and_model_files_unchanged=True,
            interpretation='Counts are availability, not pose accuracy/posture recognition success. All fresh records are NOT_REVIEWED.',
            output_directory=str(output)))
        print(json.dumps(dict(status='COMPLETED_WITH_IMAGE_ERRORS' if errors else 'COMPLETED',images=len(paths),
            method_csvs=3*len(libraries),combined_csvs=len(libraries),output=str(output)),ensure_ascii=False),flush=True)
    except Exception as exc:
        save_json(output/'run_failure.json',dict(status='FAILED',error=type(exc).__name__+': '+str(exc),
            commands=commands,note='Partial new files retained; no success claimed. Use a NEW output directory on retry.'))
        raise


if __name__ == '__main__':
    main()
