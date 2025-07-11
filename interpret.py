import os
import re
import json
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from openai import AsyncOpenAI
from dotenv import load_dotenv


# 환경변수 로드
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 또는 ["http://localhost:3000"] 등
    allow_credentials=True,
    allow_methods=["*"],  # "POST", "GET", "OPTIONS" 등
    allow_headers=["*"],
)

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
        
- 꿈 내용: "{translated_text}"
- 감정: "{top_emotion['label']}" (score: {top_emotion['score']:.2f})
다음은 악몽에 대한 심리학적 해석 요청입니다. 아래 감정 및 꿈 내용을 기반으로, 요청한 5가지 항목을 각각 **한국어로** 해석해 주세요.
1. **개요 (Overview)**:
악몽의 주요 내용과 지배적인 감정을 3줄 이상 4줄 이내로 요약해주세요.

2. **꿈의 주제 기반 분석 (Dream Theme Analysis)**:
핵심 주제를 바탕으로 긍정적인 메시지나 희망적인 해석을 5줄이상 6줄 이내로설명해주세요.theme_analysis에 이 해석을 대입해주세요.

3. **세부요소 분석 (Detailed Element Analysis)**:
꿈의 구체적인 상황과 감정({top_emotion['label']})을 분석하고, 핵심단어 2~3개를 뽑아서 각 핵심단어별로 2줄 이상 3줄 이내로설명해주세요.
다음의 예시의 형식만 참고로 하고 내용은 절대 따라하지 말고 반드시 새로 생성해야한다. **한 문장씩 줄바꿈(\n)으로 구분된 하나의 문자열** 형태로 작성해 주세요:

당신이 추출한 핵심단어명을 여기에 작성 : 분석한 내용을 이곳에 작성 이 뒤에 줄바꿈(\n)을 포함해주세요. 
이형식을 절대 벗어나거나 다른 텍스트나 설명은 절대 포함하지 마세요.
무조건 한국어로 작성하세요.

그리고 이를 detail_analysis 변수에 문자열(String) 형태로 넣어주세요.

4. **생활 속 연관성 유추 (Real-life Connection Inference)**:
이 꿈이 일상, 고민, 무의식과 어떻게 연결되는지 4줄이상 6줄 이내로 설명해주세요. 이를 real_life_connection 변수에 넣어주세요.

5. **위로 조언 및 결론 (Comforting Advice and Conclusion)**:
감정({top_emotion['label']})에 공감하며 따뜻한 위로와 조언을 포함하며 위의 모든 해석 내용을 포괄하는 결론을 작성해주세요.
4줄이상 6줄 이내로 작성해주세요.
이를 comforting_advice 에 넣어주세요.

 아래 형식을 **정확히 그대로** 사용하여 JSON 응답을 만들어 주세요. 다른 텍스트나 설명은 절대 포함하지 마세요.

{{
  "overview": "...",
  "theme_analysis": "...",
  "detail_analysis": "...",
  "real_life_connection": "...",
  "comforting_advice": "..."
}}
분석 내용은 반드시 다음 5가지 키를 가진 단일 JSON 객체 형식이어야 합니다: "overview", "theme_analysis", "detail_analysis", "real_life_connection", "comforting_advice".
모든 값은 한국어로 작성해주세요.
앞뒤에 어떠한 설명, 해석, 언급도 붙이지 마세요.
JSON 안의 값 외에 아무것도 작성하지 마세요.

 반드시 위 JSON 구조만 포함하며, 각 값은 희망적인 문장으로 작성해 주세요.
절대 마크다운 코드블럭(```json 등)을 사용하지 마세요.
 다른 설명, 해석, 텍스트는 절대 포함하지 마세요.
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

        # ⬇️ 여기에 추가
        print("🔥 GPT 원본 응답:")
        print(gpt_raw_response)

        clean_response = re.sub(r"^```json|```$", "", gpt_raw_response, flags=re.MULTILINE).strip()

        # ⬇️ 여기에도 추가 가능
        print("✅ 정제된 응답:")
        print(clean_response)

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
