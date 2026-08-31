"""
channel_stats.py — 채널별 영상 수 / 채널 총 개수 / 영상 길이 분포 집계.

채널-disjoint split 검토(및 리뷰 대응)용 진단 스크립트. 모델 로드 없이 순수
데이터 집계만 하므로 GPU 불필요.

Usage:
  python data_pipeline/channel_stats.py
"""
import json
from collections import Counter
from pathlib import Path

import cv2

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
GT_ALL    = DATA_DIR / "gt_results" / "gt_all.json"
TEXT_META = DATA_DIR / "text_meta.json"
VIDEO_DIR = DATA_DIR / "videos"
LABELS_CSV = DATA_DIR / "LABELS.csv"


def video_duration_sec(video_path: Path) -> float | None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0:
        return None
    return frames / fps


def main():
    gt_all    = json.load(open(GT_ALL, encoding="utf-8"))
    text_meta = json.load(open(TEXT_META, encoding="utf-8"))

    total = len(gt_all)
    label_counts = Counter(v["label"] for v in gt_all)

    # ── 채널 커버리지 ────────────────────────────────────────────────────────
    with_channel = {v["video_name"]: text_meta[v["video_name"]]["channel"]
                     for v in gt_all
                     if v["video_name"] in text_meta and text_meta[v["video_name"]].get("channel")}
    without_channel = total - len(with_channel)

    ch_counter = Counter(with_channel.values())

    print("=" * 60)
    print("  채널 통계")
    print("=" * 60)
    print(f"전체 영상: {total}개  (label 분포: {dict(label_counts)})")
    print(f"채널 정보 있음: {len(with_channel)}개 ({len(with_channel)/total:.1%})")
    print(f"채널 정보 없음: {without_channel}개 ({without_channel/total:.1%})  ← channel-disjoint split 불가 영역")
    print(f"고유 채널 수: {len(ch_counter)}개")
    print()

    print("── 채널당 영상 수 분포 ──")
    dist = Counter(ch_counter.values())  # {영상수: 채널수}
    for n_videos in sorted(dist.keys()):
        print(f"  영상 {n_videos:>3}개 보유 채널: {dist[n_videos]:>3}개")
    multi = {c: n for c, n in ch_counter.items() if n > 1}
    print(f"\n2개 이상 영상을 가진 채널: {len(multi)}개 "
          f"(총 {sum(multi.values())}개 영상, 전체의 {sum(multi.values())/total:.1%})")

    print("\n── 상위 15개 채널 ──")
    for ch, n in ch_counter.most_common(15):
        # 이 채널의 최종 라벨 분포
        vids = [vn for vn, c in with_channel.items() if c == ch]
        labels = Counter()
        gt_by_name = {v["video_name"]: v for v in gt_all}
        for vn in vids:
            labels[gt_by_name[vn]["label"]] += 1
        print(f"  {ch:<20} {n:>3}개  {dict(labels)}")

    # ── 영상 길이 분포 (실제 영상 파일 기준) ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  영상 길이 분포 (실제 파일, 초 단위)")
    print("=" * 60)
    durations = []
    missing = 0
    for v in gt_all:
        vfiles = list(VIDEO_DIR.glob(f"{v['video_name']}.*"))
        if not vfiles:
            missing += 1
            continue
        d = video_duration_sec(vfiles[0])
        if d is not None:
            durations.append(d)

    if missing:
        print(f"[경고] 영상 파일 없음: {missing}개 (길이 분포에서 제외)")

    if durations:
        import statistics as st
        durations.sort()
        n = len(durations)
        print(f"영상 수: {n}")
        print(f"평균: {st.mean(durations):.1f}s   중앙값: {st.median(durations):.1f}s   "
              f"표준편차: {st.pstdev(durations):.1f}s")
        print(f"최소: {durations[0]:.1f}s   최대: {durations[-1]:.1f}s")
        for pct in (10, 25, 50, 75, 90, 95):
            idx = min(n - 1, int(n * pct / 100))
            print(f"  p{pct}: {durations[idx]:.1f}s")

        buckets = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, float("inf"))]
        print("\n구간별 분포:")
        for lo, hi in buckets:
            cnt = sum(1 for d in durations if lo <= d < hi)
            hi_s = f"{hi:.0f}" if hi != float("inf") else "∞"
            print(f"  [{lo:>3}, {hi_s:>3})s : {cnt:>4}개  ({cnt/n:.1%})")
    else:
        print("영상 파일을 찾을 수 없어 길이 분포를 계산할 수 없습니다.")

    # ── 결과 저장 ────────────────────────────────────────────────────────────
    out = {
        "total_videos":        total,
        "label_counts":        dict(label_counts),
        "videos_with_channel": len(with_channel),
        "videos_without_channel": without_channel,
        "unique_channels":     len(ch_counter),
        "channel_video_counts": dict(ch_counter.most_common()),
        "duration_stats": {
            "n": len(durations),
            "mean": sum(durations) / len(durations) if durations else None,
            "min": durations[0] if durations else None,
            "max": durations[-1] if durations else None,
        },
    }
    out_path = DATA_DIR / "channel_stats.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
