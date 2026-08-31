"""
eval_tool_usage.py — 완료된 InternVL binary CYREN 체크포인트(results_intern/
tempo_cov_intern_evolve_checkpoint.json)에 남아있는 학습된 fb_mem으로 test set(121개)의
실제 tool 라우팅 결과를 계산한다.

VLM 추론 전혀 안 함 — retrieve_tools()는 CPU 임베더(bge-m3)와 저장된 fb_mem만 있으면
되므로, 재학습·GPU 없이 그대로 재현 가능. em_core_tool2.py의 FeedbackMemory.retrieve_tools
로직을 그대로 복제.

Usage:
  python experiment_intern/eval_tool_usage.py
"""
import csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
CKPT_PATH = ROOT / "results_intern" / "tempo_cov_intern_evolve_checkpoint.json"

TRAIN_RATIO, DEV_RATIO, SEED = 0.6, 0.2, 42
TOP_K, SIM_THRESHOLD = 3, 0.5
CLASSES = ["보이스피싱", "투자사기", "로맨스스캠", "부업사기", "정상"]


def load_gt() -> list[dict]:
    """utils_intern.py의 실제 load_gt()와 동일 — max_sec 필터링 없음(원본 그대로 반환).
    Qwen(utils.py)의 load_gt()는 필터링을 하지만 InternVL 쪽은 안 함 — 여기서도 그대로 맞춤."""
    with open(DATA_DIR / "gt_results" / "gt_all.json", encoding="utf-8") as f:
        return json.load(f)


def split_data(gt_list, train_ratio, dev_ratio, seed):
    """utils_intern.py의 실제 split_data와 동일 로직으로 복제 — Qwen(utils.py)의
    stratified 방식과 다름: 여기는 label 구분 없이 통째로 섞고, random.Random(seed)
    로컬 인스턴스를 씀(global random.seed()가 아님). 이 함수는 GPU/vLLM 로드 없이
    가벼운 tool-routing 분석만 하려고 만든 standalone 복제본이라, 진짜 utils_intern.py를
    import하지 않고 로직만 그대로 베낀 것."""
    import random
    rng = random.Random(seed)
    d = gt_list[:]
    rng.shuffle(d)
    n_train = int(len(d) * train_ratio)
    n_dev   = int(len(d) * dev_ratio)
    return d[:n_train], d[n_train:n_train + n_dev], d[n_train + n_dev:]


def _extract_vid_id(link: str):
    m = re.search(r'(?:v=|/shorts/|youtu\.be/)([a-zA-Z0-9_-]{11})', link)
    if m:
        return m.group(1)
    m = re.search(r'tiktok\.com/.*/video/(\d+)', link)
    if m:
        return m.group(1)
    return None


def load_type_map() -> dict:
    type_map = {}
    with open(DATA_DIR / "LABELS.csv", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row[0] == "라벨":
                continue
            vid_id = _extract_vid_id(row[2] if len(row) > 2 else "")
            if vid_id:
                type_map[vid_id] = row[1]
    return type_map


def scam_type_of(vname, label, type_map):
    if label == "normal":
        return "정상"
    yt_id = vname.split("_", 1)[1] if "_" in vname else vname
    t = type_map.get(yt_id)
    return t if t in CLASSES else (t or "미상")


def retrieve_tools(embedder, memory: list[dict], query: str) -> list[str]:
    """em_core_tool2.FeedbackMemory.retrieve_tools와 동일 로직 (top_k=3, threshold=0.5)."""
    if not memory:
        return []
    patterns = [m.get("pattern") or m["feedback"] for m in memory]
    q_emb  = embedder.encode([query], normalize_embeddings=True)[0]
    p_embs = embedder.encode(patterns, normalize_embeddings=True)
    sims   = p_embs @ q_emb
    top    = np.argsort(sims)[-TOP_K:][::-1]
    tools: set[str] = set()
    for i in top:
        if sims[i] >= SIM_THRESHOLD:
            tools.update(m for m in memory[i].get("tools", []))
    return sorted(tools)


if __name__ == "__main__":
    print("=== 체크포인트 로드 ===")
    ckpt = json.load(open(CKPT_PATH, encoding="utf-8"))
    early_fb_mem = ckpt["early_fb_mem"]
    print(f"early_fb_mem entries: {len(early_fb_mem)}")
    print(f"(late_fb_mem은 프로덕션에서 _use_l_tools=False라 애초에 안 쓰임: {len(ckpt['late_fb_mem'])}개 저장은 돼있음)")

    print("\n=== test set 로드 (SEED=42, 동일 split) ===")
    gt_list = load_gt()
    trainset, devset, testset = split_data(gt_list, TRAIN_RATIO, DEV_RATIO, SEED)
    print(f"Test={len(testset)}")

    text_meta = json.load(open(DATA_DIR / "text_meta.json", encoding="utf-8"))
    type_map  = load_type_map()

    def get_video_query(vname):
        title = (text_meta.get(vname, {}).get("title") or "").strip()
        return title if title else vname

    print("\n=== 임베더 로드 (CPU, bge-m3) ===")
    embedder = SentenceTransformer("BAAI/bge-m3", device="cpu")

    print("\n=== 영상별 tool 라우팅 계산 ===")
    per_video = []
    tool_counter = Counter()
    by_type_tool = defaultdict(Counter)
    n_activated = 0

    for v in testset:
        vname, label = v["video_name"], v["label"]
        query = f"early {get_video_query(vname)}"
        tools = retrieve_tools(embedder, early_fb_mem, query)
        stype = scam_type_of(vname, label, type_map)

        if tools:
            n_activated += 1
        for t in tools:
            tool_counter[t] += 1
            by_type_tool[stype][t] += 1

        per_video.append({
            "video_name": vname, "label": label, "scam_type": stype,
            "tools": tools, "n_tools": len(tools),
        })

    total = len(testset)
    activation_rate = n_activated / total

    print(f"\n=== 결과 ===")
    print(f"전체 test 영상: {total}")
    print(f"도구 1개 이상 활성화: {n_activated}개 ({activation_rate*100:.1f}%)")
    print(f"도구 없음(시각 정보만): {total - n_activated}개 ({(1-activation_rate)*100:.1f}%)")
    print(f"\n도구별 활성화 빈도 (활성화된 영상 {n_activated}개 기준 분모):")
    for tool, cnt in tool_counter.most_common():
        pct = cnt / n_activated * 100 if n_activated else 0
        print(f"  {tool:12s}: {cnt}회  ({pct:.1f}%)")

    print(f"\n사기 유형별 도구 사용 분포:")
    for stype in CLASSES:
        c = by_type_tool.get(stype)
        if not c:
            print(f"  {stype}: 활성화된 도구 없음")
            continue
        total_type = sum(1 for r in per_video if r["scam_type"] == stype)
        detail = ", ".join(f"{t}={n}" for t, n in c.most_common())
        print(f"  {stype} (n={total_type}): {detail}")

    out = {
        "source_checkpoint": str(CKPT_PATH.relative_to(ROOT)),
        "backbone": "InternVL2.5-78B-AWQ",
        "note": "VLM 재추론 없이, 완료된 학습의 fb_mem으로 test set tool 라우팅만 재계산. "
                "Late 세그먼트는 프로덕션에서 tool 미사용(_use_l_tools=False)이라 제외.",
        "total_test_videos": total,
        "n_tool_activated": n_activated,
        "activation_rate": activation_rate,
        "tool_frequency": dict(tool_counter),
        "by_scam_type": {k: dict(v) for k, v in by_type_tool.items()},
        "per_video": per_video,
    }
    out_path = ROOT / "results_intern" / "tool_usage_result.json"
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n저장: {out_path}")
