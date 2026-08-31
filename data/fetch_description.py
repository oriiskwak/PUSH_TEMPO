"""
fetch_description.py — 600개 YouTube 영상의 설명(description)·조회수·좋아요를
yt-dlp로 수집해 text_meta.json에 추가.

서버에서 한 번만 실행:
  python data/fetch_description.py

- 영상 재다운로드 없음 (--no-download)
- 10개마다 중간 저장 (중단 후 재실행 시 이어서 진행)
- 삭제·비공개 영상은 description="" 로 마킹해 재시도 방지
"""
import json, time
from pathlib import Path

META_PATH = Path(__file__).parent / "text_meta.json"
SLEEP_SEC = 0.8    # 요청 간 딜레이 (rate limit 방지)
MAX_DESC  = 500    # description 저장 최대 글자 수


def _yt_id(vname: str) -> str:
    """abnormal_0JUr7e9RGeA → 0JUr7e9RGeA"""
    return vname.split("_", 1)[1]


def _fetch(yt_id: str) -> dict:
    import yt_dlp
    url = f"https://www.youtube.com/watch?v={yt_id}"
    opts = {
        "quiet":            True,
        "skip_download":    True,
        "no_warnings":      True,
        "extract_flat":     False,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return {
            "description": (info.get("description") or "")[:MAX_DESC].strip(),
            "view_count":  info.get("view_count"),
            "like_count":  info.get("like_count"),
        }
    except Exception as e:
        print(f"ERR {e}")
        return {}


def main():
    meta    = json.load(open(META_PATH, encoding="utf-8"))
    total   = len(meta)
    updated = 0

    for i, (vname, info) in enumerate(meta.items()):
        if "description" in info:   # 이미 처리됨 → 스킵
            continue

        yt_id = _yt_id(vname)
        print(f"[{i+1}/{total}] {vname}", end=" ... ", flush=True)

        fields = _fetch(yt_id)
        if fields:
            info.update(fields)
            updated += 1
            print(f"OK  desc={len(fields.get('description',''))}chars  "
                  f"views={fields.get('view_count')}  likes={fields.get('like_count')}")
        else:
            info["description"] = ""   # 실패 마킹
            print("FAIL (private/deleted)")

        if (i + 1) % 10 == 0:
            json.dump(meta, open(META_PATH, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            print(f"  → 중간 저장 ({updated}개 업데이트)")

        time.sleep(SLEEP_SEC)

    json.dump(meta, open(META_PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n완료: {updated}/{total}개 업데이트")


if __name__ == "__main__":
    main()
