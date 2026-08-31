"""
demo_api_call.py — OpenAI API 실제 호출 데모용 스크립트.
계정/사용 화면 캡처 요청에 대응: 요청을 보내고 응답을 받는 과정을 터미널에 그대로 출력한다.

Usage:
  OPENAI_API_KEY=... python gpt_experiment/demo_api_call.py
"""

import os
from openai import OpenAI

MODEL = os.environ.get("GPT_MODEL", "gpt-4o-mini")
client = OpenAI()

print(f"[1] 사용 모델: {MODEL}")
print(f"[2] API 키 적용 여부: {'설정됨' if os.environ.get('OPENAI_API_KEY') else '없음 (환경변수 확인 필요)'}")

prompt = "Predict the probability (0.0-1.0) that this sentence is about a cyber scam: 'Click this link to claim your prize now!'"
print(f"\n[3] 요청 전송 중...\n    prompt: {prompt}")

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=20,
    temperature=0.0,
)

print(f"\n[4] 응답 수신 완료")
print(f"    model: {response.model}")
print(f"    content: {response.choices[0].message.content.strip()}")
print(f"    usage: prompt_tokens={response.usage.prompt_tokens}, "
      f"completion_tokens={response.usage.completion_tokens}, "
      f"total_tokens={response.usage.total_tokens}")
