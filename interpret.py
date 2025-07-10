import os
import re
import json
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from openai import AsyncOpenAI
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

app = FastAPI()

# 디바이스 설정 (GPU 또는 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 감정 분석 파이프라인
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=0 if device.type == "cuda" else -1
)

# 번역기 파이프라인 설정
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="kor_Hang",
    tgt_lang="eng_Latn",
    device=0 if device.type == "cuda" else -1
)

# OpenAI 비동기 클라이언트
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 요청 모델
class DreamRequest(BaseModel):
    content: str

# 응답 모델
class DreamResponse(BaseModel):
    original_text: str
    overview: str
    theme_analysis: str
    detail_analysis: str
    real_life_connection: str
    comforting_advice: str

@app.post("/interpret", response_model=DreamResponse)
async def interpret_dream(request: DreamRequest):
    try:
        # 1. 원문
        dream_text = request.content

        # 2. 번역
        translation_result = translator(dream_text)
        if not translation_result or "translation_text" not in translation_result[0]:
            raise HTTPException(status_code=500, detail="번역 결과가 올바르지 않습니다.")
        translated_text = translation_result[0]["translation_text"]

        # 3. 감정 분석
        emotion_result = classifier(translated_text)
        if not emotion_result:
            raise HTTPException(status_code=500, detail="감정 분석 결과가 없습니다.")
        if len(emotion_result) == 1 and isinstance(emotion_result[0], list):
            emotion_result = emotion_result[0]

        top_emotion = max(emotion_result, key=lambda x: x["score"])
        if top_emotion["label"] == "neutral":
            filtered = [e for e in emotion_result if e["label"] != "neutral"]
            if filtered:
                top_emotion = max(filtered, key=lambda x: x["score"])

        # 4. GPT 프롬프트
        prompt = f"""
다음은 악몽에 대한 심리학적 해석 요청입니다. 아래 감정 및 꿈 내용을 기반으로, 요청한 5가지 항목을 각각 **한국어로** 해석해 주세요.

- 꿈 내용: "{translated_text}"
- 감정: "{top_emotion['label']}" (score: {top_emotion['score']:.2f})

🎯 아래 형식을 **정확히 그대로** 사용하여 JSON 응답을 만들어 주세요. 다른 텍스트나 설명은 절대 포함하지 마세요.

{{
  "overview": "...",
  "theme_analysis": "...",
  "detail_analysis": "...",
  "real_life_connection": "...",
  "comforting_advice": "..."
}}

❗ 반드시 위 JSON 구조만 포함하며, 각 값은 간결하고 희망적인 문장으로 작성해 주세요.
❗ 절대 마크다운 코드블럭(```json 등)을 사용하지 마세요.
❗ 다른 설명, 해석, 텍스트는 절대 포함하지 마세요.
"""

        # 5. GPT 호출
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a dream interpretation expert who always replies in JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        # 6. GPT 응답 파싱
        gpt_raw_response = completion.choices[0].message.content.strip()
        clean_response = re.sub(r"^```json|```$", "", gpt_raw_response, flags=re.MULTILINE).strip()

        try:
            gpt_data = json.loads(clean_response)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="GPT 응답이 JSON 형식이 아닙니다:\n" + gpt_raw_response)

        # 7. 키 유효성 검사
        required_keys = [
            "overview",
            "theme_analysis",
            "detail_analysis",
            "real_life_connection",
            "comforting_advice"
        ]
        missing_keys = [k for k in required_keys if k not in gpt_data]
        if missing_keys:
            raise HTTPException(
                status_code=500,
                detail=f"GPT 응답에서 누락된 키: {', '.join(missing_keys)}"
            )

        # 8. 응답 반환
        return DreamResponse(
            original_text=dream_text,
            overview=gpt_data["overview"],
            theme_analysis=gpt_data["theme_analysis"],
            detail_analysis=gpt_data["detail_analysis"],
            real_life_connection=gpt_data["real_life_connection"],
            comforting_advice=gpt_data["comforting_advice"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
