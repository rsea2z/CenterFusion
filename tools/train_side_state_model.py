from __future__ import annotations
"""
Two-stage vehicle state classifier:
- Stage 1 (Rule): dyn_stationary > threshold → parking
- Stage 2 (ML): RF on 3-class moving problem (straight/left/right)

Usage:
  python tools/train_side_state_model.py --labels data/nuscenes/annotations/states_user_mini_val.json \
    --out models/side_state_model.npz --use-radar --two-stage --dyn-threshold 0.4 --kfold 5 --model rf
"""
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

STATE_TO_ID = {
    'going_straight': 0,
    'going_left': 1,
    'going_right': 2,
    'parking': 3,
}
ID_TO_STATE = {v: k for k, v in STATE_TO_ID.items()}


def extract_features(track: dict, use_radar: bool = False):
    hist = track['history']
    hist2 = hist[-20:] if len(hist) >= 2 else hist

    xs, zs, ts, yaws = [], [], [], []
    for h in hist2:
        loc = h['location']
        xs.append(float(loc[0])); zs.append(float(loc[2]))
        ts.append(float(h.get('t', 0.0))); yaws.append(float(h['yaw']))

    vfs, vls, speeds = [], [], []
    for i in range(1, len(xs)):
        dt = 0.5
        dz = zs[i] - zs[i-1]; dx = xs[i] - xs[i-1]
        vf = dz / dt; vl = dx / dt
        vfs.append(vf); vls.append(vl); speeds.append((vf*vf+vl*vl)**0.5)
    if not vfs:
        vfs=[0.0]; vls=[0.0]; speeds=[0.0]

    def agg(x):
        if len(x) == 0: return [0,0,0,0]
        arr = np.array(x)
        return [float(np.mean(arr)), float(np.std(arr)), float(np.max(arr)), float(np.min(arr))]

    yaw_rate = 0.0
    if len(yaws) >= 2:
        dy = np.diff(np.unwrap(np.array(yaws)))
        yaw_rate = float(np.mean(dy) / 0.5)

    feat = []
    feat += agg(vfs); feat += agg(vls); feat += agg(speeds); feat += [yaw_rate]

    # Acceleration
    acc_f, acc_l = [], []
    for i in range(1, len(vfs)):
        acc_f.append((vfs[i]-vfs[i-1])/0.5)
        acc_l.append((vls[i]-vls[i-1])/0.5)
    feat += agg(acc_f); feat += agg(acc_l)

    dir_changes = sum(1 for i in range(1,len(vls)) if vls[i-1]*vls[i]<0 and abs(vls[i]-vls[i-1])>0.1)
    mean_abs_vf = float(np.mean(np.abs(vfs)))
    mean_abs_vl = float(np.mean(np.abs(vls)))
    ratio_lat = float(mean_abs_vl / (mean_abs_vf + 1e-6))
    feat.extend([float(dir_changes), mean_abs_vf, mean_abs_vl, ratio_lat])
    feat.extend([float(max(np.abs(vls)) if vls else 0), abs(yaw_rate)])

    if use_radar:
        radar_counts, vx_means, vz_means, vr_means = [], [], [], []
        vr_stds, rcs_means, rcs_stds = [], [], []
        dyn_movings, dyn_stationary, dyn_oncoming, dyn_cross, dyn_stopped = [], [], [], [], []
        valid_ratios, vx_rms_means, vy_rms_means = [], [], []
        for hfull in hist2:
            radar = hfull.get('radar')
            if radar and radar.get('count', 0) > 0:
                radar_counts.append(radar.get('count', 0))
                vx_means.append(radar.get('vx_mean', 0.0))
                vz_means.append(radar.get('vz_mean', 0.0))
                vr_means.append(radar.get('vr_mean', 0.0))
                vr_stds.append(radar.get('vr_std', 0.0))
                rcs_means.append(radar.get('rcs_mean', 0.0))
                rcs_stds.append(radar.get('rcs_std', 0.0))
                dyn_movings.append(radar.get('dyn_moving', 0.0))
                dyn_stationary.append(radar.get('dyn_stationary', 0.0))
                dyn_oncoming.append(radar.get('dyn_oncoming', 0.0))
                dyn_cross.append(radar.get('dyn_cross', 0.0))
                dyn_stopped.append(radar.get('dyn_stopped', 0.0))
                valid_ratios.append(radar.get('valid_ratio', 1.0))
                vx_rms_means.append(radar.get('vx_rms_mean', 0.0))
                vy_rms_means.append(radar.get('vy_rms_mean', 0.0))

        def agg1(x):
            if len(x)==0: return [0.0,0.0,0.0,0.0]
            arr = np.array(x, dtype=np.float32)
            return [float(np.mean(arr)),float(np.std(arr)),float(np.max(arr)),float(np.min(arr))]

        for lst in [radar_counts, vx_means, vz_means, vr_means, vr_stds, rcs_means, rcs_stds,
                    dyn_movings, dyn_stationary, dyn_oncoming, dyn_cross, dyn_stopped,
                    valid_ratios, vx_rms_means, vy_rms_means]:
            feat += agg1(lst)

    return np.array(feat, dtype=np.float32)


def compute_dyn_stationary(track: dict) -> float:
    """Average dyn_stationary from last 5 radar observations."""
    hist = track['history'][-5:]
    vals = []
    for h in hist:
        radar = h.get('radar')
        if radar and radar.get('count', 0) > 0:
            vals.append(radar.get('dyn_stationary', 0.0))
    return float(np.mean(vals)) if vals else 0.0


def two_stage_predict(clf, feat: np.ndarray, dyn_stationary: float,
                      threshold: float, feat_mean: np.ndarray,
                      feat_std: np.ndarray, standardized: bool) -> int:
    """Predict using two-stage logic: rule for parking, ML for others."""
    if dyn_stationary >= threshold:
        return STATE_TO_ID['parking']
    # Standardize and predict moving class
    x = feat.reshape(1, -1)
    if standardized:
        x = (x - feat_mean) / (feat_std + 1e-6)
    return int(clf.predict(x)[0])


def build_clf(model_type, class_weight='balanced'):
    if model_type == 'gbdt':
        return GradientBoostingClassifier()
    if model_type == 'rf':
        return RandomForestClassifier(n_estimators=400, max_depth=8,
                                      class_weight=class_weight,
                                      random_state=42, n_jobs=-1)
    if model_type == 'logreg':
        return LogisticRegression(max_iter=500, class_weight=class_weight, multi_class='auto')
    raise ValueError('Unknown model')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', type=str, default='data/nuscenes/annotations/states_user_mini_val.json')
    ap.add_argument('--out', type=str, default='models/side_state_model.npz')
    ap.add_argument('--use-radar', action='store_true', help='Include radar aggregated stats')
    ap.add_argument('--no-standardize', action='store_true', help='Disable z-score normalization')
    ap.add_argument('--minimal', action='store_true', help='Use minimal 10-feature set')
    ap.add_argument('--model', type=str, default='rf', choices=['gbdt','rf','logreg'],
                    help='Classifier type')
    ap.add_argument('--kfold', type=int, default=5, help='StratifiedKFold folds (0=train only)')
    ap.add_argument('--two-stage', action='store_true',
                    help='Use dyn_stationary rule for parking + ML for moving (recommended)')
    ap.add_argument('--dyn-threshold', type=float, default=0.4,
                    help='dyn_stationary threshold for two-stage parking detection')
    ap.add_argument('--smote', action='store_true',
                    help='Apply SMOTE oversampling (for non-two-stage only)')
    args = ap.parse_args()

    data = json.loads(Path(args.labels).read_text())
    X_all, y_all, dyn_all = [], [], []
    for tr in data.get('tracks', []):
        state = tr.get('state', 'going_straight')
        if state not in STATE_TO_ID:
            continue
        if args.minimal:
            h = tr['history'][-10:]
            xs, zs, yaws = [], [], []
            for hrec in h:
                loc = hrec['location']
                xs.append(float(loc[0])); zs.append(float(loc[2])); yaws.append(float(hrec['yaw']))
            vfs, vls, speeds = [], [], []
            for i in range(1, len(xs)):
                dz=zs[i]-zs[i-1]; dx=xs[i]-xs[i-1]
                vf=dz/0.5; vl=dx/0.5; vfs.append(vf); vls.append(vl); speeds.append((vf*vf+vl*vl)**0.5)
            if not vfs: vfs=[0.0]; vls=[0.0]; speeds=[0.0]
            arr=np.array
            def mean_std(a):
                a=np.array(a); return [float(a.mean()),float(a.std())]
            vf_m,vf_s=mean_std(vfs); vl_m,vl_s=mean_std(vls); sp_m,sp_s=mean_std(speeds)
            yaw_rate=0.0
            if len(yaws)>=2:
                dy=np.diff(np.unwrap(np.array(yaws))); yaw_rate=float(dy.mean()/0.5)
            mean_abs_vl=float(np.mean(np.abs(vls))); mean_abs_vf=float(np.mean(np.abs(vfs)))
            ratio_lat=float(mean_abs_vl/(mean_abs_vf+1e-6))
            dir_changes=sum(1 for i in range(1,len(vls)) if vls[i-1]*vls[i]<0 and abs(vls[i]-vls[i-1])>0.1)
            feats=np.array([vf_m,vf_s,vl_m,vl_s,sp_m,sp_s,yaw_rate,ratio_lat,float(dir_changes),mean_abs_vl],dtype=np.float32)
        else:
            feats = extract_features(tr, use_radar=(args.use_radar and not args.minimal))
        dyn_all.append(compute_dyn_stationary(tr))
        X_all.append(feats)
        y_all.append(STATE_TO_ID[state])

    X_all = np.stack(X_all, axis=0)
    y_all = np.array(y_all, dtype=np.int64)
    dyn_all = np.array(dyn_all, dtype=np.float32)
    uniq, cnt = np.unique(y_all, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
    print(f'[INFO] Total={len(X_all)}  dist={dist}')
    print(f'[INFO] dyn_stationary: parking={dyn_all[y_all==3].mean():.3f}  moving={dyn_all[y_all!=3].mean():.3f}')

    # ── Two-stage evaluation ──────────────────────────────────────────────────
    if args.two_stage:
        print(f'[Two-Stage] dyn_thresh={args.dyn_threshold}  model={args.model}')
        kfolds = args.kfold if args.kfold > 1 else 1
        skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)
        all_yte, all_ypred = [], []
        fold_metrics = []
        cm_sum = None

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all), 1):
            Xtr, Xte = X_all[tr_idx], X_all[te_idx]
            ytr, yte = y_all[tr_idx], y_all[te_idx]
            dyn_tr, dyn_te = dyn_all[tr_idx], dyn_all[te_idx]

            # Filter to moving classes only for stage-2 training
            moving_mask_tr = ytr != STATE_TO_ID['parking']
            Xtr_mv = Xtr[moving_mask_tr]
            ytr_mv = ytr[moving_mask_tr]

            # Fit stage-2 classifier on moving classes only
            clf = build_clf(args.model, class_weight='balanced')

            std = not args.no_standardize
            if std:
                f_mean = Xtr.mean(axis=0); f_std = Xtr.std(axis=0) + 1e-6
                Xtr_s = (Xtr_mv - f_mean) / f_std
            else:
                f_mean = np.zeros_like(Xtr[0]); f_std = np.ones_like(Xtr[0])
                Xtr_s = Xtr_mv

            clf.fit(Xtr_s, ytr_mv)

            # Predict test set
            preds = []
            for i in range(len(te_idx)):
                p = two_stage_predict(clf, Xte[i], dyn_te[i],
                                       args.dyn_threshold, f_mean, f_std, std)
                preds.append(p)
            preds = np.array(preds)

            all_yte.extend(yte.tolist()); all_ypred.extend(preds.tolist())
            cm = confusion_matrix(yte, preds, labels=list(range(4)))
            if cm_sum is None: cm_sum = cm
            else: cm_sum += cm

            p_m, r_m, f_m, _ = precision_recall_fscore_support(yte, preds, average='macro', zero_division=0)
            acc = accuracy_score(yte, preds)
            fold_metrics.append((p_m, r_m, f_m, acc))
            if kfolds > 1:
                print(f'--- Fold {fold} ---')
            print(f'  Acc={acc:.3f}  MacroF1={f_m:.3f}')
            print(classification_report(yte, preds, target_names=list(STATE_TO_ID.keys()), zero_division=0))
            print('CM:\n', cm)

        print('[Two-Stage] Combined:')
        print(f'  Macro P/R/F1/Acc: {np.mean(fold_metrics,0)}  std: {np.std(fold_metrics,0)}')
        print('Cumulative CM:\n', cm_sum)
        print(classification_report(np.array(all_yte), np.array(all_ypred),
                                   target_names=list(STATE_TO_ID.keys()), zero_division=0))

        # Train final model on all data
        moving_mask = y_all != STATE_TO_ID['parking']
        clf_final = build_clf(args.model, class_weight='balanced')
        if std:
            feat_mean = X_all.mean(axis=0); feat_std = X_all.std(axis=0) + 1e-6
            X_all_s = (X_all[moving_mask] - feat_mean) / feat_std
        else:
            feat_mean = np.zeros_like(X_all[0]); feat_std = np.ones_like(X_all[0])
            X_all_s = X_all[moving_mask]
        clf_final.fit(X_all_s, y_all[moving_mask])

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out,
                 model=clf_final,
                 STATE_TO_ID=STATE_TO_ID,
                 feature_dim=X_all.shape[1],
                 feature_mean=feat_mean,
                 feature_std=feat_std,
                 standardized=std,
                 minimal=args.minimal,
                 two_stage=True,
                 dyn_threshold=args.dyn_threshold)
        print(f'Saved two-stage model to {args.out}')
        return

    # ── Standard (single-stage) evaluation ────────────────────────────────────
    X, y = X_all, y_all
    feat_mean = X.mean(axis=0); feat_std = X.std(axis=0) + 1e-6
    if args.no_standardize:
        feat_mean = np.zeros_like(feat_mean); feat_std = np.ones_like(feat_std)
    X_s = (X - feat_mean) / feat_std if not args.no_standardize else X

    try:
        from imblearn.over_sampling import SMOTE
        HAS_SMOTE = True
    except ImportError:
        HAS_SMOTE = False

    if args.kfold > 1:
        print(f'[KFold] {args.kfold}-fold  model={args.model}')
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
        all_yte, all_ypred = [], []; fold_metrics = []; cm_sum = None
        for fold,(tr_idx,te_idx) in enumerate(skf.split(X_s,y),1):
            Xtr_s,Xte_s = X_s[tr_idx], X_s[te_idx]; ytr,yte = y[tr_idx],y[te_idx]
            clf = build_clf(args.model); clf.fit(Xtr_s, ytr)
            ypred = clf.predict(Xte_s)
            all_yte.extend(yte.tolist()); all_ypred.extend(ypred.tolist())
            cm = confusion_matrix(yte,ypred,labels=list(range(4)))
            if cm_sum is None: cm_sum = cm
            else: cm_sum += cm
            p_m,r_m,f_m,_ = precision_recall_fscore_support(yte,ypred,average='macro',zero_division=0)
            acc = accuracy_score(yte,ypred); fold_metrics.append((p_m,r_m,f_m,acc))
            print(f'--- Fold {fold} ---')
            print(classification_report(yte,ypred,target_names=list(STATE_TO_ID.keys()),zero_division=0))
            print('CM:\n',cm)
        print('[KFold] Cumulative:'); print('CM:\n',cm_sum)
        print(f'Macro avg: {np.mean(fold_metrics,0)}  std: {np.std(fold_metrics,0)}')
        print(classification_report(np.array(all_yte),np.array(all_ypred),
                                   target_names=list(STATE_TO_ID.keys()),zero_division=0))
        clf = build_clf(args.model); clf.fit(X_s, y)
    else:
        clf = build_clf(args.model); clf.fit(X_s, y)
        ypred = clf.predict(X_s)
        print(classification_report(y,ypred,target_names=list(STATE_TO_ID.keys()),zero_division=0))
        print('CM:\n', confusion_matrix(y,ypred,labels=list(range(4))))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             model=clf, STATE_TO_ID=STATE_TO_ID,
             feature_dim=X.shape[1],
             feature_mean=feat_mean, feature_std=feat_std,
             standardized=not args.no_standardize,
             minimal=args.minimal,
             two_stage=False, dyn_threshold=0.0)
    print(f'Saved model to {args.out}')


if __name__ == '__main__':
    main()
