"""
channel_disjoint_analysis.py — train/dev에 등장한 채널이 test에도 등장하는지 확인하고,
channel-clean test 서브셋을 만든다. GPU 불필요 (text_meta.json/gt_all.json만 사용).

Usage:
  python data_pipeline/channel_disjoint_analysis.py
"""
import json, random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

TRAIN_RATIO, DEV_RATIO, SEED = 0.6, 0.2, 42


def load_gt(max_sec: int = 150) -> list[dict]:
    data = json.load(open(DATA_DIR / "gt_results" / "gt_all.json", encoding="utf-8"))
    out = []
    for v in data:
        v["frame_probabilities"] = [f for f in v["frame_probabilities"] if f["second"] <= max_sec]
        if v["frame_probabilities"]:
            out.append(v)
    return out


def split_data(gt_list, train_ratio, dev_ratio, seed):
    random.seed(seed)
    abnormal = [v for v in gt_list if v["label"] == "abnormal"]
    normal   = [v for v in gt_list if v["label"] == "normal"]

    def _split(lst):
        lst = lst[:]
        random.shuffle(lst)
        n_tr = int(len(lst) * train_ratio)
        n_dv = int(len(lst) * dev_ratio)
        return lst[:n_tr], lst[n_tr:n_tr + n_dv], lst[n_tr + n_dv:]

    ab_tr, ab_dv, ab_te = _split(abnormal)
    no_tr, no_dv, no_te = _split(normal)
    return ab_tr + no_tr, ab_dv + no_dv, ab_te + no_te


if __name__ == "__main__":
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    text_meta = json.load(open(DATA_DIR / "text_meta.json", encoding="utf-8"))

    def channel_of(vname):
        return (text_meta.get(vname, {}) or {}).get("channel")

    train_dev_names = {v["video_name"] for v in trainset} | {v["video_name"] for v in devset}
    test_names = {v["video_name"] for v in testset}

    # test 중 채널 정보 있는/없는 영상
    test_with_channel = {v: channel_of(v) for v in test_names if channel_of(v)}
    test_without_channel = test_names - set(test_with_channel.keys())

    # train/dev에 등장한 채널 집합
    train_dev_channels = {channel_of(v) for v in train_dev_names if channel_of(v)}

    # ── 1. 겹치는 채널/영상 표 ──
    overlap_by_channel = defaultdict(list)
    for vname, ch in test_with_channel.items():
        if ch in train_dev_channels:
            overlap_by_channel[ch].append(vname)

    print(f"\n=== 1. Train/Dev와 겹치는 채널이 있는 test 영상 ===")
    print(f"{'채널명':<20} {'겹치는 test 영상 수':<10}")
    total_overlap_videos = 0
    for ch, vids in sorted(overlap_by_channel.items(), key=lambda x: -len(x[1])):
        print(f"{ch:<20} {len(vids):<10}  {vids}")
        total_overlap_videos += len(vids)
    print(f"\n겹치는 채널 수: {len(overlap_by_channel)}")
    print(f"겹치는 test 영상 수: {total_overlap_videos}")

    # ── 2. channel-clean test 서브셋 ──
    contaminated_names = {v for vids in overlap_by_channel.values() for v in vids}
    clean_testset = [v for v in testset if v["video_name"] not in contaminated_names]
    print(f"\n=== 2. Channel-clean test 서브셋 ===")
    print(f"원래 test: {len(testset)}개 -> channel-clean: {len(clean_testset)}개 "
          f"({len(contaminated_names)}개 제외)")

    # ── 5. 채널 정보 커버리지 한계 ──
    print(f"\n=== 5. 채널 정보 커버리지 (한계) ===")
    print(f"Test set {len(test_names)}개 중 채널 정보 있음: {len(test_with_channel)}개 "
          f"({len(test_with_channel)/len(test_names)*100:.1f}%)")
    print(f"Test set {len(test_names)}개 중 채널 정보 없음: {len(test_without_channel)}개 "
          f"({len(test_without_channel)/len(test_names)*100:.1f}%) — 이 영상들은 channel-disjoint 여부 확인 불가")

    out = {
        "split": {"train": len(trainset), "dev": len(devset), "test": len(testset), "seed": SEED},
        "test_with_channel_info": len(test_with_channel),
        "test_without_channel_info": len(test_without_channel),
        "overlap_channels": {ch: vids for ch, vids in overlap_by_channel.items()},
        "n_overlap_channels": len(overlap_by_channel),
        "n_contaminated_test_videos": len(contaminated_names),
        "contaminated_video_names": sorted(contaminated_names),
        "clean_test_video_names": sorted(v["video_name"] for v in clean_testset),
        "clean_test_size": len(clean_testset),
    }
    out_path = DATA_DIR / "channel_disjoint_analysis.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")
