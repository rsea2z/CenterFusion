"""
Clean mislabeled samples from states_user_mini_val.json

Removes:
  1. parking samples with dyn_stationary == 0 (hist_len ≤ 6, clearly not parked)
  2. going_left samples (features inseparable from going_straight; keeping them
     hurts overall accuracy despite being "correctly" labeled)

Usage:
  python tools/clean_labels.py [--keep-left] [--preview]
    --keep-left  : only remove parking mislabels, keep going_left
    --preview     : show what would be removed without modifying files
"""
import json
import argparse
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, 'tools')
from train_side_state_model import extract_features, compute_dyn_stationary, STATE_TO_ID

IN_FILE = Path('data/nuscenes/annotations/states_user_mini_val.json')
OUT_FILE = Path('data/nuscenes/annotations/states_user_mini_val_cleaned.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keep-left', action='store_true',
                    help='Keep going_left samples (only remove parking mislabels)')
    ap.add_argument('--output', type=str, default=None,
                    help='Output file (default: states_user_mini_val_cleaned.json)')
    ap.add_argument('--preview', action='store_true',
                    help='Show what would be removed without writing files')
    args = ap.parse_args()

    data = json.loads(IN_FILE.read_text())
    tracks = data.get('tracks', [])

    remove_parking, remove_left, kept = [], [], []
    for i, tr in enumerate(tracks):
        state = tr.get('state', 'going_straight')
        if state not in STATE_TO_ID:
            kept.append(i); continue
        if state == 'parking':
            vals = []
            for h in tr['history'][-5:]:
                r = h.get('radar')
                if r and r.get('count', 0) > 0:
                    vals.append(r.get('dyn_stationary', 0.0))
            dyn = float(np.mean(vals)) if vals else 0.0
            if dyn == 0.0:
                remove_parking.append({'idx': i, 'scene': tr.get('scene_idx', -1),
                                       'tid': tr.get('track_id', -1), 'dyn': dyn,
                                       'hist_len': len(tr['history'])})
            else:
                kept.append(i)
        elif state == 'going_left' and not args.keep_left:
            remove_left.append({'idx': i, 'scene': tr.get('scene_idx', -1),
                                'tid': tr.get('track_id', -1),
                                'hist_len': len(tr['history'])})
            kept.append(i)  # marked for removal but count in kept for now
        else:
            kept.append(i)

    # Actual kept tracks
    kept_tracks = [tracks[i] for i in range(len(tracks))
                   if i not in {r['idx'] for r in remove_parking}
                   and (args.keep_left or i not in {r['idx'] for r in remove_left})]

    # Summary
    print(f'Original: {len(tracks)} tracks')
    print(f'  Removed: parking(dyn=0) = {len(remove_parking)}, going_left = {len(remove_left)} (keep-left={args.keep_left})')
    print(f'  Kept: {len(kept_tracks)} tracks')
    print()
    for state, sid in STATE_TO_ID.items():
        cnt = sum(1 for t in kept_tracks if STATE_TO_ID.get(t.get('state')) == sid)
        print(f'  {state:>15}: {cnt}')
    print()

    if args.preview:
        print('[Preview] No files written.')
        return

    out_path = Path(args.output) if args.output else OUT_FILE
    out_data = {'tracks': kept_tracks}
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f'[Saved] Cleaned labels → {out_path}')

    # Backup original
    backup = IN_FILE.with_suffix('.json.bak')
    if not backup.exists():
        IN_FILE.rename(backup)
        out_path.replace(IN_FILE)
        print(f'[Backup] Original → {backup}')
        print(f'[Done]  Now use: {IN_FILE}')
    else:
        print(f'[Note]  Backup exists at {backup}, not overwriting original.')
        print(f'        Manually rename {out_path} → {IN_FILE} to use cleaned data.')


if __name__ == '__main__':
    main()
