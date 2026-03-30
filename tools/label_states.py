from __future__ import annotations
"""
A minimal labeling helper for vehicle states on mini-val videos.
- Reads per-frame detections dumped by demo (--save + --dump-states later) or runs detector live.
- Allows keyboard labeling for active tracks: [1]=going_straight, [2]=lane_change, [3]=turning, [4]=parking.
- Saves labels to data/nuscenes/annotations/states_user_mini_val.json

This is intentionally simple to bootstrap annotations. For real projects, consider CVAT/Label-Studio integration.
"""
import os
import json
import time
import argparse
import cv2
import numpy as np
from pathlib import Path

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import _init_paths  # adds src and src/lib into PYTHONPATH

# Ensure singletons across mixed import paths (lib.* vs bare names)
import lib.config as _lib_config
import lib.dataset as _lib_dataset
import lib.detector as _lib_detector
sys.modules.setdefault('config', _lib_config)
sys.modules.setdefault('dataset', _lib_dataset)
sys.modules.setdefault('detector', _lib_detector)

from lib.utils.simple_tracker import SimpleTracker
from lib.utils.state_classifier import RuleBasedStateClassifier
from lib.detector import Detector
from lib.dataset import getDataset
from lib.config import config, updateConfig, updateDatasetAndModelConfig
from lib.utils.utils import createFolder
from lib.utils.image import getAffineTransform, affineTransform
from lib.dataset.datasets.nuscenes import nuScenes as NuscDS
from lib.dataset.datasets.nuscenes import nuScenes as NuscClass
from lib.utils.pointcloud import RadarPointCloudWithVelocity as RadarPointCloud

STATE_KEYS = {
    ord('1'): 'going_straight',
    ord('2'): 'going_left',
    ord('3'): 'going_right',
    ord('4'): 'parking',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--split', type=str, default='mini_val')
    parser.add_argument('--save', action='store_true', help='save video frames with overlays')
    parser.add_argument('--min', type=int, default=0, help='ignore number of scenes from front')
    parser.add_argument('--max', type=int, default=999999, help='max number of scenes to iterate')
    parser.add_argument('--sample', type=str, default=None, help='specific sample token/id')
    parser.add_argument('--auto-play', action='store_true', help='auto play without pausing (default off for labeling)')
    # 保留与主 demo 参数接口一致的占位（无需实际使用）
    parser.add_argument('--single', action='store_true', help='(unused placeholder)')
    parser.add_argument('--not-show', dest='not_show', action='store_true', help='(unused placeholder)')
    parser.add_argument('--show-attention', dest='show_attention', action='store_true', help='(unused placeholder)')
    # Keep parity with main.py/updateConfig expecting args.opts
    parser.add_argument(
        'opts',
        nargs=argparse.REMAINDER,
        default=None,
        help='Modify config options from command line'
    )
    args = parser.parse_args()

    updateConfig(config, args)
    dataset = getDataset(config.DATASET.DATASET)
    updateDatasetAndModelConfig(config, dataset, output_dir='output')

    # Build detector (show=False so we start from raw image without all boxes)
    det = Detector(config, show=False)

    # Init tracker & rule-based classifier (as warm start helper for annotator)
    tracker = SimpleTracker()
    rb = RuleBasedStateClassifier()

    # Lazy import to avoid static analysis issues if nuscenes-devkit not resolved by linter
    from nuscenes.nuscenes import NuScenes
    # Load nuScenes SDK directly to avoid constructing Demo (which builds another Detector)
    nusc = NuScenes(
        version=NuscDS.SPLITS[args.split],
        dataroot=os.path.join(config.DATASET.ROOT, 'nuscenes'),
        verbose=True,
    )

    labels = { 'tracks': [] }
    track_label_map = {}  # tid -> state
    paused = not args.auto_play
    selected_tid = None
    max_show = 20
    only_vehicle = True  # force only vehicles
    show_help = True

    # mouse selection
    click_state = {'x': -1, 'y': -1, 'clicked': False}
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            click_state['x'] = x; click_state['y'] = y; click_state['clicked'] = True
    cv2.namedWindow('label_states', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('label_states', on_mouse)

    # Iterate scenes with frame navigation & caching
    total_scenes = len(nusc.scene)
    start_i = max(0, int(args.min))
    end_i = min(total_scenes, int(args.max)) if args.max is not None else total_scenes
    for si in range(start_i, end_i):
        scene = nusc.scene[si]
        # Build ordered list of sample tokens for this scene
        frame_tokens = []
        stoken = scene['first_sample_token']
        while stoken:
            smp = nusc.get('sample', stoken)
            frame_tokens.append(smp['token'])
            stoken = smp['next'] if smp['next'] != '' else None

        # Caches per frame
        ret_images = [None] * len(frame_tokens)
        dets_list = [None] * len(frame_tokens)
        mapping_list = [None] * len(frame_tokens)
        time_list = [None] * len(frame_tokens)
        suggests_list = [None] * len(frame_tokens)

        frame_idx = 0
        dirty_display = True  # force first render

        while 0 <= frame_idx < len(frame_tokens):
            # Build tracker up to current frame (recompute if moved backwards or uncached)
            tracker = SimpleTracker()
            suggestions = {}
            for fi in range(0, frame_idx + 1):
                if dets_list[fi] is None:
                    # load sample/frame data & run detector
                    smp = nusc.get('sample', frame_tokens[fi])
                    cam = 'CAM_FRONT'
                    sd_token = smp['data'][cam]
                    sd = nusc.get('sample_data', sd_token)
                    frame = cv2.imread(os.path.join(nusc.dataroot, sd['filename']))
                    _, _, camera_intrinsic = nusc.get_sample_data(sd_token)
                    calib = np.eye(4, dtype=np.float32); calib[:3,:3] = camera_intrinsic; calib = calib[:3]
                    img_info = {
                        'width': sd['width'], 'height': sd['height'], 'camera_intrinsic': camera_intrinsic,
                        'calib': calib.tolist(), 'target': None,
                    }
                    ret, _infer = det.run(frame, [img_info], radar_pc=None)
                    predicts = ret['predictBoxes'][0]
                    # Prepare transform
                    input_h, input_w = config.MODEL.INPUT_SIZE
                    center = np.array([sd['width'] / 2.0, sd['height'] / 2.0], dtype=np.float32)
                    scale = max(sd['height'], sd['width']) * 1.0
                    trans_out = getAffineTransform(center, scale, 0, (input_w, input_h))
                    dets = []
                    # Collect radar points for this sample (multi-sweep, multi-radar associated with CAM_FRONT)
                    radar_channels = NuscClass.RADARS_FOR_CAMERA.get(cam, [])
                    all_radar_points = np.zeros((18,0))
                    try:
                        for rc in radar_channels:
                            radar_pc, _ = RadarPointCloud.from_file_multisweep(nusc, smp, rc, cam, nsweeps=4)
                            all_radar_points = np.hstack((all_radar_points, radar_pc.points))
                    except Exception:
                        pass
                    for p in predicts:
                        loc_np = p['location'].detach().cpu().numpy() if hasattr(p['location'], 'detach') else np.array(p['location'])
                        radar_feat = None
                        if all_radar_points.shape[1] > 0:
                            pts = all_radar_points
                            # distance in ground plane x,z
                            dists = np.linalg.norm(np.stack([pts[0]-loc_np[0], pts[2]-loc_np[2]], axis=1), axis=1)
                            mask = dists < 2.0
                            sel = pts[:, mask]
                            if sel.shape[1] > 0:
                                vx = sel[8]; vz = sel[9]
                                vr = np.sqrt(vx**2 + vz**2)
                                rcs = sel[5]
                                # 质量过滤：is_quality_valid (idx 10)
                                valid_mask = (sel[10] == 1)
                                valid_sel = sel[:, valid_mask] if valid_mask.sum() > 0 else sel
                                n = valid_sel.shape[1] if valid_sel.shape[1] > 0 else 1
                                # dyn_prop (idx 3): 0=moving, 1=stationary, 2=oncoming, 5=crossing, 7=stopped
                                dyn = valid_sel[3] if valid_sel.shape[1] > 0 else np.array([], dtype=np.int32)
                                dyn_moving = float(np.sum(dyn == 0)) / n
                                dyn_stationary = float(np.sum(dyn == 1)) / n
                                dyn_oncoming = float(np.sum(dyn == 2)) / n
                                dyn_cross = float(np.sum((dyn == 5) | (dyn == 6))) / n
                                dyn_stopped = float(np.sum(dyn == 7)) / n
                                radar_feat = {
                                    'count': int(sel.shape[1]),
                                    'vx_mean': float(np.mean(vx)),
                                    'vz_mean': float(np.mean(vz)),
                                    'vr_mean': float(np.mean(vr)),
                                    'vr_std': float(np.std(vr)),
                                    'rcs_mean': float(np.mean(rcs)),
                                    'rcs_std': float(np.std(rcs)),
                                    'dyn_moving': dyn_moving,
                                    'dyn_stationary': dyn_stationary,
                                    'dyn_oncoming': dyn_oncoming,
                                    'dyn_cross': dyn_cross,
                                    'dyn_stopped': dyn_stopped,
                                    'vx_rms_mean': float(np.mean(valid_sel[16])) if valid_sel.shape[1] > 0 else 0.0,
                                    'vy_rms_mean': float(np.mean(valid_sel[17])) if valid_sel.shape[1] > 0 else 0.0,
                                    'valid_ratio': float(valid_mask.sum()) / max(sel.shape[1], 1),
                                }
                        dets.append({
                            'class': int(p['class']),
                            'score': float(p['score']),
                            'location': loc_np,
                            'yaw': float(p['yaw'] if not hasattr(p['yaw'], 'detach') else p['yaw'].detach().cpu().numpy()),
                            'velocity': p.get('velocity', np.zeros(3)),
                            'bbox_pix': (
                                affineTransform(
                                    (p['bboxes'].view(-1,2).detach().cpu().numpy() if hasattr(p['bboxes'], 'detach') else np.array(p['bboxes']).reshape(-1,2)),
                                    trans_out
                                ).astype(np.int32).tolist()
                                if 'bboxes' in p else None
                            ),
                            'radar_feat': radar_feat,
                        })
                    dets_list[fi] = dets
                    ret_images[fi] = ret['images'][0].copy()
                    time_list[fi] = time.time()
                # advance tracker
                mapping = tracker.update(dets_list[fi], time_list[fi])
                mapping_list[fi] = mapping
                # update suggestions for active tracks at this frame
                suggestions = {tid: rb.classify_one(tr.history, radar_stats=tr.radar_stats) for tid, tr in tracker.get_active_tracks().items()}
                suggests_list[fi] = suggestions

            # Display current frame if needed (dirty_display set when labels changed or frame changed)
            if dirty_display:
                vis = ret_images[frame_idx].copy()
                mapping = mapping_list[frame_idx]
                suggestions = suggests_list[frame_idx]
                active = list(tracker.get_active_tracks().items())
                if only_vehicle:
                    active = [(tid, tr) for tid, tr in active if 1 <= tr.cls <= 5]
                def last_score(tr):
                    return float(tr.history[-1][4]) if tr.history else 0.0
                active.sort(key=lambda it: last_score(it[1]), reverse=True)
                active = active[:max_show]
                tid_to_bbox = {}
                for tid, tr in active:
                    det_rec = mapping.get(tid)
                    bb = None
                    if det_rec is not None and det_rec.get('bbox_pix'):
                        b = det_rec['bbox_pix']
                        try:
                            x1, y1 = b[0]; x2, y2 = b[1]
                            bb = (int(x1), int(y1), int(x2), int(y2))
                        except Exception:
                            bb = None
                    if bb is not None:
                        tid_to_bbox[tid] = bb
                        color = (0,255,255) if tid == selected_tid else (0,128,255)
                        cv2.rectangle(vis, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                        label_txt = f"id={tid} {suggestions.get(tid,'?')} / {track_label_map.get(tid,'unset')}"
                        cv2.putText(vis, label_txt, (bb[0], max(0, bb[1]-6)), 0, 0.5, (255,255,0), 1)
                y0 = 18
                if show_help:
                    cv2.putText(vis, 'Space=Pause/Play  n=Next  b=Prev  Click=Select  1/2/3/4=Label  s=Save  v=ToggleTopK  h=Help  q=Quit', (10, y0), 0, 0.5, (0,255,255), 1)
                    y0 += 18
                cv2.putText(vis, f'Scene {si+1}/{end_i}  Frame {frame_idx+1}/{len(frame_tokens)}  Paused={paused}  Selected={selected_tid if selected_tid else "None"}', (10, y0), 0, 0.5, (0,255,0), 1)
                cv2.imshow('label_states', vis)
                dirty_display = False

            # Save snapshot helper (current track states)
            def save_snapshot():
                labels['tracks'] = []
                # use latest tracker (rebuilt up to frame_idx)
                for tid, tr in tracker.get_active_tracks().items():
                    labels['tracks'].append({
                        'track_id': tid,
                        'state': track_label_map.get(tid, suggests_list[frame_idx].get(tid, 'going_straight')),
                        'history': [
                            {
                                't': float(t),
                                'location': loc.tolist(),
                                'yaw': float(yaw),
                                'velocity': vel.tolist(),
                                'score': float(score),
                                'radar': tr.radar_stats[i] if i < len(tr.radar_stats) else None,
                            } for (i, (t, loc, yaw, vel, score)) in enumerate(tr.history)
                        ]
                    })
                out = Path('data/nuscenes/annotations/states_user_mini_val.json')
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(labels, indent=2))
                print(f'Saved labels to {out}')

            # Handle interaction loop
            k = cv2.waitKey(0 if paused else 1) & 0xFF
            if click_state['clicked']:
                cx, cy = click_state['x'], click_state['y']
                click_state['clicked'] = False
                # rebuild tid_to_bbox for selection (if not dirty we can recompute quickly)
                mapping = mapping_list[frame_idx]
                active = list(tracker.get_active_tracks().items())
                if only_vehicle:
                    active = [(tid, tr) for tid, tr in active if 1 <= tr.cls <= 5]
                tid_to_bbox = {}
                for tid, tr in active:
                    det_rec = mapping.get(tid)
                    if det_rec is None or not det_rec.get('bbox_pix'): continue
                    b = det_rec['bbox_pix']
                    try:
                        x1,y1 = b[0]; x2,y2 = b[1]
                        if x1 <= cx <= x2 and y1 <= cy <= y2:
                            selected_tid = tid
                            dirty_display = True
                            break
                    except Exception:
                        pass
            if k == ord(' '):
                paused = not paused
                dirty_display = True
            elif k == ord('n'):
                frame_idx += 1
                paused = True
                dirty_display = True
            elif k == ord('b'):
                frame_idx = max(0, frame_idx - 1)
                paused = True
                dirty_display = True
            elif k in STATE_KEYS:
                if selected_tid is not None:
                    track_label_map[selected_tid] = STATE_KEYS[k]
                    dirty_display = True
                else:
                    print('请先用鼠标点击目标框以选中一个 track 再标注。')
            elif k == ord('s'):
                save_snapshot()
            elif k == ord('v'):
                max_show = 999 if max_show < 100 else 20
                dirty_display = True
            elif k == ord('h'):
                show_help = not show_help
                dirty_display = True
            elif k == ord('q'):
                cv2.destroyAllWindows(); raise SystemExit
            # Auto-advance when not paused
            if not paused and k == 255:  # no key pressed in non-blocking mode
                frame_idx += 1
                dirty_display = True
        # next scene loop continues


if __name__ == '__main__':
    main()
