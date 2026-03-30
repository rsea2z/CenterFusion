from __future__ import annotations
"""
Train a simple side model (XGBoost-like via sklearn GradientBoosting) for vehicle state classification
using labels produced by tools/label_states.py. This avoids touching the main detector.
python tools/train_side_state_model.py --labels data/nuscenes/annotations/states_user_mini_val.json --out models/side_state_model.npz
"""
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
try:
    # imbalanced-learn for SMOTE
    from imblearn.over_sampling import SMOTE
except ImportError:  # graceful if dependency missing
    SMOTE = None

STATE_TO_ID = {
    'going_straight': 0,
    'going_left': 1,
    'going_right': 2,
    'parking': 3,
}


def extract_features(track: dict, use_radar: bool = False):
    hist = track['history']
    if len(hist) < 2:
        hist2 = hist
    else:
        hist2 = hist[-20:]  # last N

    # 位置差分推导速度（更鲁棒，避免 detector velocity 头为零）
    xs, zs, ts, yaws = [], [], [], []
    for h in hist2:
        loc = h['location']
        xs.append(float(loc[0])); zs.append(float(loc[2]))
        ts.append(float(h.get('t', 0.0)))
        yaws.append(float(h['yaw']))
    vfs, vls, speeds = [], [], []
    # 固定 dt=0.5s（nuScenes 2Hz），避免真实执行时间抖动导致速度接近 0
    for i in range(1, len(xs)):
        dt = 0.5
        dz = zs[i] - zs[i-1]
        dx = xs[i] - xs[i-1]
        vf = dz / dt
        vl = dx / dt
        vfs.append(vf); vls.append(vl); speeds.append((vf**2 + vl**2)**0.5)
    # 若只有一个点，填充 0
    if len(vfs) == 0:
        vfs = [0.0]; vls = [0.0]; speeds = [0.0]

    def agg(x):
        if len(x) == 0:
            return [0,0,0,0]
        arr = np.array(x)
        return [float(np.mean(arr)), float(np.std(arr)), float(np.max(arr)), float(np.min(arr))]

    # yaw rate (平均角速度)
    yaw_rate = 0.0
    if len(yaws) >= 2:
        y_arr = np.unwrap(np.array(yaws))
        dy = np.diff(y_arr)
        if len(ts) == len(yaws) and len(ts) >= 2 and (ts[-1] - ts[0]) > 0:
            dt_arr = []
            for i in range(1, len(ts)):
                dt_arr.append(max(ts[i] - ts[i-1], 1e-3))
            dt_arr = np.array(dt_arr)
            yaw_rate = float(np.mean(dy / dt_arr))
        else:
            yaw_rate = float(np.mean(dy) / 0.5)  # fallback 0.5s

    feat = []
    feat += agg(vfs)            # 4
    feat += agg(vls)            # 8
    feat += agg(speeds)         # 12
    feat += [yaw_rate]          # 13
    # 额外运动特征
    # 加速度（基于速度序列差分）
    acc_f, acc_l = [], []
    for i in range(1, len(vfs)):
        dt = 0.5
        acc_f.append( (vfs[i]-vfs[i-1]) / dt )
        acc_l.append( (vls[i]-vls[i-1]) / dt )
    def agg_or0(x):
        if len(x)==0:
            return [0,0,0,0]
        a=np.array(x)
        return [float(np.mean(a)), float(np.std(a)), float(np.max(a)), float(np.min(a))]
    feat += agg_or0(acc_f)      # 17
    feat += agg_or0(acc_l)      # 21
    # 方向变化次数（横向速度符号变化）
    dir_changes = 0
    for i in range(1, len(vls)):
        if vls[i-1]*vls[i] < 0 and abs(vls[i]-vls[i-1]) > 0.1:
            dir_changes += 1
    feat += [float(dir_changes)]  # 22
    # 平均绝对横/前速度与比例
    if len(vfs)==0: vfs_tmp=[0.0]; vls_tmp=[0.0]
    else: vfs_tmp=vfs; vls_tmp=vls
    mean_abs_vf = float(np.mean(np.abs(vfs_tmp)))
    mean_abs_vl = float(np.mean(np.abs(vls_tmp)))
    ratio_lat_forward = float(mean_abs_vl / (mean_abs_vf + 1e-6))
    feat += [mean_abs_vf, mean_abs_vl, ratio_lat_forward]  # 25
    # 最大横向速度 / 最大角速度
    max_vl = float(max(np.abs(vls_tmp))) if vls_tmp else 0.0
    max_yaw_rate = abs(yaw_rate)
    feat += [max_vl, max_yaw_rate]  # 27
    if use_radar:
        # collect per-frame radar stats if present
        radar_counts = []
        vx_means = []
        vz_means = []
        vr_means = []
        vr_stds = []
        rcs_means = []
        rcs_stds = []
        # dyn_prop 特征
        dyn_movings, dyn_stationary, dyn_oncoming, dyn_cross, dyn_stopped = [], [], [], [], []
        # 质量特征
        valid_ratios, vx_rms_means, vy_rms_means = [], [], []
        for hfull in hist2:
            # original saved format: history entries have keys including 'radar'
            radar = hfull.get('radar')
            if radar and radar.get('count', 0) > 0:
                radar_counts.append(radar.get('count', 0))
                vx_means.append(radar.get('vx_mean', 0.0))
                vz_means.append(radar.get('vz_mean', 0.0))
                vr_means.append(radar.get('vr_mean', 0.0))
                vr_stds.append(radar.get('vr_std', 0.0))
                rcs_means.append(radar.get('rcs_mean', 0.0))
                rcs_stds.append(radar.get('rcs_std', 0.0))
                # dyn_prop 比例
                dyn_movings.append(radar.get('dyn_moving', 0.0))
                dyn_stationary.append(radar.get('dyn_stationary', 0.0))
                dyn_oncoming.append(radar.get('dyn_oncoming', 0.0))
                dyn_cross.append(radar.get('dyn_cross', 0.0))
                dyn_stopped.append(radar.get('dyn_stopped', 0.0))
                # 质量特征
                valid_ratios.append(radar.get('valid_ratio', 1.0))
                vx_rms_means.append(radar.get('vx_rms_mean', 0.0))
                vy_rms_means.append(radar.get('vy_rms_mean', 0.0))
        def agg1(x):
            if len(x) == 0:
                return [0.0, 0.0, 0.0, 0.0]
            arr = np.array(x, dtype=np.float32)
            return [float(np.mean(arr)), float(np.std(arr)), float(np.max(arr)), float(np.min(arr))]
        feat += agg1(radar_counts)  # +4
        feat += agg1(vx_means)      # +4
        feat += agg1(vz_means)      # +4
        feat += agg1(vr_means)      # +4
        feat += agg1(vr_stds)       # +4
        feat += agg1(rcs_means)     # +4
        feat += agg1(rcs_stds)      # +4
        # dyn_prop 特征聚合
        feat += agg1(dyn_movings)      # +4
        feat += agg1(dyn_stationary)  # +4
        feat += agg1(dyn_oncoming)     # +4
        feat += agg1(dyn_cross)        # +4
        feat += agg1(dyn_stopped)     # +4
        # 质量特征聚合
        feat += agg1(valid_ratios)   # +4
        feat += agg1(vx_rms_means)  # +4
        feat += agg1(vy_rms_means)   # +4
    return np.array(feat, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', type=str, default='data/nuscenes/annotations/states_user_mini_val.json')
    ap.add_argument('--test-size', type=float, default=0.1)
    ap.add_argument('--out', type=str, default='models/side_state_model.npz')
    ap.add_argument('--smote', action='store_true', help='Apply SMOTE oversampling on training split only')
    ap.add_argument('--smote-k', type=int, default=5, help='k_neighbors for SMOTE (will auto-reduce if class counts too small)')
    ap.add_argument('--use-radar', action='store_true', help='Use radar aggregated stats (if present in history entries)')
    # 旧模型若使用 detector velocity 头，可关闭差分；默认开启差分以获得非零速度
    ap.add_argument('--no-derive-vel', action='store_true', help='(保留参数占位, 当前实现始终使用差分推导速度)')
    ap.add_argument('--no-standardize', action='store_true', help='Disable feature standardization (mean/std).')
    ap.add_argument('--minimal', action='store_true', help='Use minimal feature set (ignore radar & acceleration; only core motion stats).')
    ap.add_argument('--model', type=str, default='gbdt', choices=['gbdt','rf','logreg'], help='Classifier type: gbdt (GradientBoosting), rf (RandomForest), logreg (LogisticRegression)')
    ap.add_argument('--kfold', type=int, default=0, help='>1 时启用 StratifiedKFold 交叉验证 (对整套数据评估，不拆测试集)')
    args = ap.parse_args()

    data = json.loads(Path(args.labels).read_text())
    X, y = [], []
    for tr in data.get('tracks', []):
        state = tr.get('state', 'going_straight')
        if state not in STATE_TO_ID:
            continue
        if args.minimal:
            # 精简特征：vf_mean, vf_std, vl_mean, vl_std, speed_mean, speed_std, yaw_rate, ratio_lat_forward, dir_changes, mean_abs_vl
            h = tr['history'][-10:]
            xs, zs, yaws = [], [], []
            for hrec in h:
                loc = hrec['location']; xs.append(float(loc[0])); zs.append(float(loc[2])); yaws.append(float(hrec['yaw']))
            vfs, vls, speeds = [], [], []
            for i in range(1,len(xs)):
                dz = zs[i]-zs[i-1]; dx = xs[i]-xs[i-1]
                vf = dz/0.5; vl = dx/0.5; vfs.append(vf); vls.append(vl); speeds.append((vf*vf+vl*vl)**0.5)
            if not vfs:
                vfs=[0.0]; vls=[0.0]; speeds=[0.0]
            def mean_std(a):
                arr=np.array(a); return [float(arr.mean()), float(arr.std())]
            vf_m, vf_s = mean_std(vfs)
            vl_m, vl_s = mean_std(vls)
            sp_m, sp_s = mean_std(speeds)
            yaw_rate=0.0
            if len(yaws)>=2:
                dy=np.diff(np.unwrap(np.array(yaws))); yaw_rate=float(dy.mean()/0.5)
            # 横向/前向比例
            mean_abs_vl = float(np.mean(np.abs(vls)))
            mean_abs_vf = float(np.mean(np.abs(vfs)))
            ratio_lat = float(mean_abs_vl / (mean_abs_vf + 1e-6))
            dir_changes=0
            for i in range(1,len(vls)):
                if vls[i-1]*vls[i] < 0 and abs(vls[i]-vls[i-1])>0.1:
                    dir_changes+=1
            feats = np.array([vf_m, vf_s, vl_m, vl_s, sp_m, sp_s, yaw_rate, ratio_lat, float(dir_changes), mean_abs_vl], dtype=np.float32)
        else:
            feats = extract_features(tr, use_radar=(args.use_radar and (not args.minimal)))
        X.append(feats)
        y.append(STATE_TO_ID[state])

    if not X:
        print('[ERROR] 未解析到任何样本，请检查 labels JSON 文件。')
        return
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    uniq, cnt = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
    print(f'[INFO] 样本总数={len(X)} 类别分布(id->count)={dist}')

    # 自适应划分逻辑
    can_stratify = (len(uniq) > 1) and (cnt.min() >= 2)
    do_split = True
    if len(X) < 5 or len(uniq) == 1:
        print('[WARN] 样本过少或只有一个类别，跳过测试集划分，全部用于训练。')
        do_split = False
    if do_split and args.kfold <= 1:
        test_size = args.test_size
        if cnt.min() < 3:
            test_size = min(test_size, 0.2)
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y,
                test_size=test_size,
                random_state=42,
                stratify=y if can_stratify else None
            )
        except ValueError as e:
            print(f'[WARN] train_test_split 失败({e})，取消划分。')
            do_split = False
    if (not do_split) or args.kfold > 1:
        X_tr, y_tr = X, y
        X_te, y_te = None, None

    # Optional SMOTE only on training set to avoid leakage
    if args.smote:
        if SMOTE is None:
            print('[SMOTE] imbalanced-learn 未安装，跳过 SMOTE。请在 requirements.txt 中添加 imbalanced-learn 并安装。')
        else:
            # Determine minimal class count
            unique, counts = np.unique(y_tr, return_counts=True)
            class_count_dict = {int(k): int(v) for k, v in zip(unique, counts)}
            print(f'[SMOTE] 原始训练集类别分布: {class_count_dict}')
            min_count = counts.min()
            if min_count < 2:
                print('[SMOTE] 存在样本数 <2 的类别，无法进行 SMOTE，跳过。')
            else:
                k_neighbors = min(args.smote_k, min_count - 1)
                if k_neighbors < 1:
                    print('[SMOTE] 计算后 k_neighbors<1，跳过 SMOTE。')
                else:
                    try:
                        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
                        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
                        unique2, counts2 = np.unique(y_tr, return_counts=True)
                        class_count_dict2 = {int(k): int(v) for k, v in zip(unique2, counts2)}
                        print(f'[SMOTE] 过采样后训练集类别分布: {class_count_dict2} (k_neighbors={k_neighbors})')
                    except Exception as e:
                        print(f'[SMOTE] 失败，跳过。原因: {e}')

    feat_mean = X_tr.mean(axis=0)
    feat_std = X_tr.std(axis=0) + 1e-6
    if not args.no_standardize:
        X_tr = (X_tr - feat_mean) / feat_std
        if X_te is not None:
            X_te = (X_te - feat_mean) / feat_std
    else:
        feat_mean = np.zeros_like(feat_mean)
        feat_std = np.ones_like(feat_std)

    def build_clf():
        if args.model == 'gbdt':
            return GradientBoostingClassifier()
        if args.model == 'rf':
            return RandomForestClassifier(n_estimators=400, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
        if args.model == 'logreg':
            return LogisticRegression(max_iter=500, class_weight='balanced', multi_class='auto')
        raise ValueError('Unknown model')

    if args.kfold > 1 and can_stratify:
        print(f'[KFold] 使用 {args.kfold} 折交叉验证 (整集标准化统计基于训练折) model={args.model}')
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
        reports = []
        cms = []
        fold_metrics = []
        for fold,(tr_idx, te_idx) in enumerate(skf.split(X_tr, y_tr), 1):
            Xtr, Xte = X_tr[tr_idx], X_tr[te_idx]
            ytr, yte = y_tr[tr_idx], y_tr[te_idx]
            f_mean = Xtr.mean(axis=0); f_std = Xtr.std(axis=0)+1e-6
            if not args.no_standardize:
                Xtr_s = (Xtr - f_mean)/f_std
                Xte_s = (Xte - f_mean)/f_std
            else:
                Xtr_s, Xte_s = Xtr, Xte
            clf_f = build_clf(); clf_f.fit(Xtr_s, ytr)
            ypred = clf_f.predict(Xte_s)
            print(f'--- Fold {fold} ---')
            print(classification_report(yte, ypred, target_names=list(STATE_TO_ID.keys())))
            cm = confusion_matrix(yte, ypred, labels=list(range(len(STATE_TO_ID))))
            print('Confusion matrix (rows=true, cols=pred):\n', cm)
            reports.append((yte, ypred)); cms.append(cm)
            from sklearn.metrics import precision_recall_fscore_support, accuracy_score
            p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(yte, ypred, average='macro', zero_division=0)
            acc = accuracy_score(yte, ypred); fold_metrics.append((p_macro, r_macro, f_macro, acc))
        if cms:
            cm_sum = sum(cms); print('[KFold] 累计混淆矩阵:\n', cm_sum)
        if fold_metrics:
            import numpy as _np
            fm = _np.array(fold_metrics); avg = fm.mean(0); std = fm.std(0)
            print(f"[KFold] Macro Precision/Recall/F1/Acc 平均: {avg}  std: {std}")
        clf = build_clf(); clf.fit(X_tr, y_tr)
    else:
        clf = build_clf()
        clf.fit(X_tr, y_tr)
        if X_te is not None:
            y_pred = clf.predict(X_te)
            print(classification_report(y_te, y_pred, target_names=list(STATE_TO_ID.keys())))
            cm = confusion_matrix(y_te, y_pred, labels=list(range(len(STATE_TO_ID))))
            print('Confusion matrix (rows=true, cols=pred):\n', cm)

    # save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, 
             model=clf,
             STATE_TO_ID=STATE_TO_ID,
             feature_dim=X.shape[1],
             feature_mean=feat_mean,
             feature_std=feat_std,
             standardized= (not args.no_standardize),
             minimal=args.minimal)
    print(f'Saved side model to {args.out}')


if __name__ == '__main__':
    main()
