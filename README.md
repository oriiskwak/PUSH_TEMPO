# TEMPO: Tool-Enhanced Multi-Pass Optimization for Scam Video Detection

온라인 동영상 플랫폼의 사이버 사기 영상(로맨스스캠, 투자사기, 보이스피싱, 부업사기)을 탐지하는 프레임워크.
**Qwen2.5-VL-72B-Instruct-AWQ** + **EM 기반 프롬프트 최적화**를 사용합니다.

---

## 전체 파이프라인

```
1. 데이터 수집 및 GT 생성     (data_pipeline/)
2. 텍스트 메타데이터 추출      (data_pipeline/preprocess_text.py)
3. RAG 패턴 DB 생성          (data_pipeline/build_scam_patterns.py)
4. 실험 실행                 (experiment/)
5. 베이스라인 비교            (baselines/)
```

---

## Step 1. 데이터 수집 및 GT 생성 (`data_pipeline/`)

```bash
# data_pipeline/data/labels.csv 에 label,link 형식으로 URL 추가 후 실행
python data_pipeline/prepare_data.py
```

| 단계 | 내용 |
|------|------|
| Step 0 | `labels.csv` 링크 → yt-dlp로 `data/videos/` 저장 |
| Step 1 | 2초 간격 프레임을 GPT로 채점 → `data/gt_results/` 저장 |
| Step 2 | 사기 프레임(prob≥0.5) 비율 기준으로 label 재분류 |
| Step 3 | early/late 구간 분할 → `data/segmentation.json` 저장 |

`labels.csv` 형식:
```
label,link
abnormal,https://www.youtube.com/shorts/xxx
normal,https://www.tiktok.com/@user/video/yyy
```

---

## Step 2. 텍스트 메타데이터 추출

```bash
python data_pipeline/preprocess_text.py
# → data/text_meta.json 생성 (title, asr, ocr, channel, subscriber_count)
```

---

## Step 3. RAG 패턴 DB 생성

```bash
python data_pipeline/build_scam_patterns.py
# → data/scam_patterns.json 생성
```

---

## Step 4. 실험 실행 (`experiment/`)

### 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `VIDEO_DIR` | `/path/to/videos` | `abnormal_*.mp4`, `normal_*.mp4` 가 있는 폴더 |
| `TP_SIZE` | `4` | Tensor parallel size (GPU 수) |
| `MM_LIMIT` | `1` | 프롬프트당 최대 이미지 수 |

### 실험 스크립트

| 스크립트 | 방법 | 툴 | 시간 분할 | Coverage Diversity |
|----------|------|----|-----------|--------------------|
| `uniapo_total.py` | Baseline EM | ✗ | ✗ | ✗ |
| `uniapo_total_tool.py` | Tool EM | ✅ | ✗ | ✗ |
| `uniapo_total_tool2.py` | Tool EM v2 (안정 라우팅) | ✅ MIN_FB=3 | ✗ | ✗ |
| `uniapo_total_tool3.py` | Tool EM v3 (channel ctx, n_cands=5) | ✅ MIN_FB=2 | ✗ | ✗ |
| `uniapo_total_temporal.py` | Temporal EM | ✗ | ✅ Early/Late | ✗ |
| `tempo_total.py` | TEMPO (Temporal + Tool) | ✅ | ✅ Early/Late | ✗ |
| `tempo_cov_total.py` | TEMPO-COV | ✅ | ✅ Early/Late | ✅ |
| `tempo_cov_total2.py` | TEMPO-COV v2 (안정 라우팅) | ✅ MIN_FB=2 | ✅ Early/Late | ✅ |

### 실행 예시

```bash
# Baseline EM
CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 python experiment/uniapo_total.py

# Tool EM v3 (권장)
CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 python experiment/uniapo_total_tool3.py

# TEMPO-COV (제안 방법)
CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 python experiment/tempo_cov_total.py
```

체크포인트는 `results/`에 자동 저장되며, 중단 후 재실행 시 이어서 시작합니다.

---

## Step 5. 베이스라인 비교 (`baselines/`)

| 스크립트 | 방법 |
|----------|------|
| `apo_video.py` | APO |
| `evoprompt_video.py` | EvoPrompt |
| `erm_video.py` | ERM |
| `eval_metrics.py` | 결과 메트릭 계산 |

---

## 결과 저장

```
results/
├── baseline_em_result.json
├── tool_em_result.json
├── tool_em2_result.json
├── tool_em3_result.json
├── temporal_em_result.json
├── tempo_cov_total_result.json
└── em_logs/          ← 이터레이션별 로그 (.jsonl)
```

각 결과 파일 구조:
- `best_prompt` — 최종 최적화된 프롬프트
- `baseline_test` / `best_test` — baseline vs 최적화 후 테스트 메트릭 (MAP, AUC, F1, ACC)
- `log` — 라운드별 train_map, dev_map, dev_auc, 피드백

---

## 주요 모듈

| 모듈 | 역할 |
|------|------|
| `em_core_tool.py` | EM 코어: FeedbackMemory, PromptMemory, 프레임 중복 제거, 클러스터링, 피드백/후보 생성 |
| `utils.py` / `utils2.py` | vLLM 초기화, 데이터 로딩, 배치 추론, 메트릭 계산 |
| `tools.py` | title / ASR / channel / web_search / RAG 주입 |

### EM 루프 구조

```
E-step: 틀린 프레임 수집 → 시각적 중복 제거 → 클러스터링 → 피드백 생성 → FeedbackMemory 저장
M-step: 피드백 기반 후보 N개 생성 → mini-eval → best 선택 → dev 평가 → PromptMemory 저장
```

### 핵심 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `PATIENCE` | 3 | 조기 종료 patience |
| `N_ROUNDS` | 10 | 최대 라운드 수 |
| `DEV_SUBSAMPLE` | 50 | dev 평가 샘플 수 |
| `MINI_EVAL` | 15 | 후보 비교용 mini-eval 샘플 수 |
| `MIN_FB_ENTRIES` | 2 (v3) | 툴 라우팅 활성화 최소 피드백 수 |
| `n_candidates` | 5 (v3) | M-step 프롬프트 후보 수 |
| `DIVERSITY_ALPHA` | 0.05 | COV 다양성 가중치 |
