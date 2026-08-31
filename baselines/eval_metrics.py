"""공유 메트릭 유틸리티 — metrics_core.py(single source of truth)의 재수출.

예전엔 이 파일에 compute_metrics/find_best_threshold가 직접 정의돼 있었고, utils.py 쪽과
달리 미정의(단일 클래스) 구간에서 None 대신 0.0을 반환하는 동작 차이가 있었다 (리뷰에서 발견).
metrics_core.py로 통합하면서 그 차이도 없앴다 — 이제 미정의면 None을 반환하므로,
print_metrics도 None을 안전하게 포맷하도록 같이 고쳤다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics_core import (
    compute_metrics_from_preds as compute_metrics,
    find_best_threshold_from_preds as find_best_threshold,
)


def _fmt(v):
    return f"{v:.4f}" if v is not None else "None"


def print_metrics(metrics: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}AUC : {_fmt(metrics['auc'])}")
    print(f"  {prefix}MAP : {_fmt(metrics['map'])}")
    print(f"  {prefix}F1  : {_fmt(metrics['f1'])}")
    print(f"  {prefix}ACC : {_fmt(metrics['acc'])}")
    if "f1_opt" in metrics:
        print(f"  {prefix}F1_opt  : {_fmt(metrics['f1_opt'])}  (threshold={metrics.get('best_threshold')})")
    if "acc_opt" in metrics:
        print(f"  {prefix}ACC_opt : {_fmt(metrics['acc_opt'])}")
