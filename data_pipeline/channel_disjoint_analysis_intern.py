"""
channel_disjoint_analysis_intern.py — channel_disjoint_analysis.py의 InternVL 전용 버전.

utils_intern.py의 실제 split_data/load_gt는 Qwen(utils.py)과 다른 알고리즘이다:
  - load_gt: max_sec 필터링 없음 (원본 그대로)
  - split_data: label별 stratify 없이 통째로 섞음, random.Random(seed) 로컬 인스턴스 사용
같은 SEED=42/TRAIN_RATIO/DEV_RATIO를 써도 Qwen과 InternVL의 실제 test 영상 목록은 다르다
(개수는 같아도 구성이 다름). InternVL 체크포인트 기반 재채점(eval_channel_disjoint.py 등)에는
반드시 이 파일(channel_disjoint_analysis_intern.json)을 써야 한다 — Qwen용
channel_disjoint_analysis.json을 잘못 섞어 쓰면 교집합이 왜곡된다(실제로 91개여야 할 게
17개로 나온 사고가 있었음).

Usage:
  python data_pipeline/channel_disjoint_analysis_intern.py
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

TRAIN_RATIO, DEV_RATIO, SEED = 0.6, 0.2, 42


def load_gt() -> list[dict]:
    with open(DATA_DIR / "gt_results" / "gt_all.json", encoding="utf-8") as f:
        return json.load(f)


def split_data(gt_list, train_ratio, dev_ratio, seed):
    import random
    rng = random.Random(seed)
    d = gt_list[:]
    rng.shuffle(d)
    n_train = int(len(d) * train_ratio)
    n_dev   = int(len(d) * dev_ratio)
    return d[:n_train], d[n_train:n_train + n_dev], d[n_train + n_dev:]


if __name__ == "__main__":
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Train={len(trainset)}  Dev={len(devset)}  Test={len(testset)}")

    text_meta = json.load(open(DATA_DIR / "text_meta.json", encoding="utf-8"))

    def channel_of(vname):
        return (text_meta.get(vname, {}) or {}).get("channel")

    train_dev_names = {v["video_name"] for v in trainset} | {v["video_name"] for v in devset}
    test_names = {v["video_name"] for v in testset}

    test_with_channel = {v: channel_of(v) for v in test_names if channel_of(v)}
    test_without_channel = test_names - set(test_with_channel.keys())

    train_dev_channels = {channel_of(v) for v in train_dev_names if channel_of(v)}

    overlap_by_channel = defaultdict(list)
    for vname, ch in test_with_channel.items():
        if ch in train_dev_channels:
            overlap_by_channel[ch].append(vname)

    print(f"\n=== 1. Train/Dev와 겹치는 채널이 있는 test 영상 (InternVL split) ===")
    total_overlap_videos = 0
    for ch, vids in sorted(overlap_by_channel.items(), key=lambda x: -len(x[1])):
        print(f"{ch:<20} {len(vids):<3}  {vids}")
        total_overlap_videos += len(vids)
    print(f"\n겹치는 채널 수: {len(overlap_by_channel)}")
    print(f"겹치는 test 영상 수: {total_overlap_videos}")

    contaminated_names = {v for vids in overlap_by_channel.values() for v in vids}
    clean_testset = [v for v in testset if v["video_name"] not in contaminated_names]
    print(f"\n=== 2. Channel-clean test 서브셋 (InternVL) ===")
    print(f"원래 test: {len(testset)}개 -> channel-clean: {len(clean_testset)}개 "
          f"({len(contaminated_names)}개 제외)")

    print(f"\n=== 5. 채널 정보 커버리지 (InternVL, 한계) ===")
    print(f"Test set {len(test_names)}개 중 채널 정보 있음: {len(test_with_channel)}개 "
          f"({len(test_with_channel)/len(test_names)*100:.1f}%)")
    print(f"Test set {len(test_names)}개 중 채널 정보 없음: {len(test_without_channel)}개 "
          f"({len(test_without_channel)/len(test_names)*100:.1f}%) — channel-disjoint 확인 불가")

    out = {
        "backbone_split": "InternVL (utils_intern.py — non-stratified, random.Random(seed))",
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
    out_path = DATA_DIR / "channel_disjoint_analysis_intern.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")
