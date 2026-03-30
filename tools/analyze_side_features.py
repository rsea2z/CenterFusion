from __future__ import annotations
"""Analyze side state feature distributions: training vs inference.
Usage:
  python tools/analyze_side_features.py --labels data/nuscenes/annotations/states_user_mini_val.json --inference-csv output/side_feats/run_enhanced.csv [--use-radar]
Outputs per-feature mean/std and z-score shift, plus simple suggestions.
"""
import argparse, json, csv, statistics, math
from pathlib import Path
import numpy as np

STATE_TO_ID = {
    'going_straight': 0,
    'going_left': 1,
    'going_right': 2,
    'parking': 3,
}

# Keep a mirror of current training feature extractor (must stay in sync)
def extract_features(track: dict, use_radar: bool=False):
    hist = track['history'][-10:]
    xs, zs, ts, yaws = [], [], [], []
    for h in hist:
        loc = h['location']
        xs.append(float(loc[0])); zs.append(float(loc[2])); ts.append(float(h.get('t',0.0))); yaws.append(float(h['yaw']))
    vfs, vls, speeds = [], [], []
    for i in range(1, len(xs)):
        dz = zs[i]-zs[i-1]; dx = xs[i]-xs[i-1]
        vf = dz/0.5; vl = dx/0.5
        vfs.append(vf); vls.append(vl); speeds.append((vf*vf+vl*vl)**0.5)
    if not vfs:
        vfs=[0.0]; vls=[0.0]; speeds=[0.0]
    def agg(x):
        a=np.array(x); return [float(a.mean()), float(a.std()), float(a.max()), float(a.min())]
    yaw_rate=0.0
    if len(yaws)>=2:
        dy=np.diff(np.unwrap(np.array(yaws)))
        yaw_rate=float(dy.mean()/0.5)
    feat=[]; feat+=agg(vfs); feat+=agg(vls); feat+=agg(speeds); feat+=[yaw_rate]
    acc_f=[]; acc_l=[]
    for i in range(1,len(vfs)):
        acc_f.append( (vfs[i]-vfs[i-1])/0.5 )
        acc_l.append( (vls[i]-vls[i-1])/0.5 )
    def agg2(x):
        if not x: return [0,0,0,0]
        a=np.array(x); return [float(a.mean()), float(a.std()), float(a.max()), float(a.min())]
    feat+=agg2(acc_f); feat+=agg2(acc_l)
    dir_changes=0
    for i in range(1,len(vls)):
        if vls[i-1]*vls[i]<0 and abs(vls[i]-vls[i-1])>0.1:
            dir_changes+=1
    mean_abs_vf=float(np.mean(np.abs(vfs)))
    mean_abs_vl=float(np.mean(np.abs(vls)))
    ratio_lat_forward=float(mean_abs_vl/(mean_abs_vf+1e-6))
    max_vl=float(max(np.abs(vls))) if vls else 0.0
    max_yaw_rate=abs(yaw_rate)
    feat += [float(dir_changes), mean_abs_vf, mean_abs_vl, ratio_lat_forward, max_vl, max_yaw_rate]
    if use_radar:
        # Aggregate same radar stats keys if present
        radar_keys=['count','vx_mean','vz_mean','vr_mean','vr_std','rcs_mean','rcs_std']
        def aggR(vals):
            if not vals: return [0,0,0,0]
            a=np.array(vals); return [float(a.mean()), float(a.std()), float(a.max()), float(a.min())]
        # gather per history entry
        for key in radar_keys:
            series=[]
            for h in hist:
                r=h.get('radar')
                if r and r.get('count',0)>0 and key in r:
                    series.append(r[key])
            feat.extend(aggR(series))
    return np.array(feat, dtype=np.float32)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--labels', required=True)
    ap.add_argument('--inference-csv', required=True)
    ap.add_argument('--use-radar', action='store_true')
    args=ap.parse_args()

    # Training features
    data=json.loads(Path(args.labels).read_text())
    train_feats=[]
    for tr in data.get('tracks', []):
        state=tr.get('state','going_straight')
        if state not in STATE_TO_ID: continue
        f=extract_features(tr, use_radar=args.use_radar)
        train_feats.append(f)
    if not train_feats:
        print('No training features extracted.')
        return
    train_mat=np.stack(train_feats,0)
    tr_mean=train_mat.mean(0); tr_std=train_mat.std(0)+1e-6

    # Inference CSV (raw features columns start at f0)
    with open(args.inference_csv,'r') as f:
        reader=csv.reader(f)
        rows=list(reader)
    if not rows:
        print('Empty inference csv'); return
    header=rows[0]
    feat_idx=[i for i,c in enumerate(header) if c.startswith('f')]
    inf_feats=[]
    for r in rows[1:]:
        if len(r) <= max(feat_idx): continue
        vals=[float(r[i]) for i in feat_idx]
        inf_feats.append(vals)
    if not inf_feats:
        print('No inference feature rows parsed'); return
    inf_mat=np.array(inf_feats, dtype=np.float32)
    if inf_mat.shape[1] != train_mat.shape[1]:
        print(f'Feature dim mismatch train {train_mat.shape[1]} vs inference {inf_mat.shape[1]}')
    n=min(train_mat.shape[1], inf_mat.shape[1])
    inf_mean=inf_mat[:,:n].mean(0); inf_std=inf_mat[:,:n].std(0)+1e-6
    z_shift=(inf_mean - tr_mean[:n]) / tr_std[:n]

    print('Feature Shift Summary (first 30 dims or less):')
    for i in range(min(30,n)):
        print(f'f{i:02d}: train_mean={tr_mean[i]:.3f} inf_mean={inf_mean[i]:.3f} z_shift={z_shift[i]:+.2f} train_std={tr_std[i]:.3f} inf_std={inf_std[i]:.3f}')

    large = [ (i, z_shift[i]) for i in range(n) if abs(z_shift[i])>2.5 ]
    if large:
        print('\nLarge mean shifts (|z|>2.5):', large[:20])
    else:
        print('\nNo large mean shift (>2.5 std).')

    near_zero_std = [ (i, tr_std[i], inf_std[i]) for i in range(n) if tr_std[i]<1e-3 or inf_std[i]<1e-3 ]
    if near_zero_std:
        print('\nZero-variance dimensions:', near_zero_std[:20])

if __name__=='__main__':
    main()
