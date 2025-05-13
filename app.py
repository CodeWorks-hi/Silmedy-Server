# 표준 라이브러리 
import os
import re
import time
import uuid
import json
import logging
import random
import string
from datetime import datetime, timedelta
from io import BytesIO
from urllib.parse import unquote
from typing import Any, Dict, List, Optional

#  서드파티 라이브러리 
import numpy as np
import requests
import boto3
import yaml
from PIL import Image
from flask import Flask, Blueprint, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    create_refresh_token,
    get_jwt_identity,
    jwt_required,
)
from flasgger import Swagger
from dotenv import load_dotenv
from requests.exceptions import ReadTimeout, RequestException
from boto3.dynamodb.conditions import Attr
from boto3.dynamodb.types import TypeDeserializer
from openai import OpenAI
from openai.types.chat import ChatCompletion
from transformers import AutoModelForCausalLM, AutoTokenizer

# Firebase 관련 
import firebase_admin
from firebase_admin import credentials, firestore, db

#  TensorFlow Lite (푸쉬 할때 바꿔서) 
from tflite_runtime.interpreter import Interpreter
#from tensorflow.lite.python.interpreter import Interpreter

# TensorFlow Lite 인터프리터
interpreter = Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# ---- 기본 세팅 ----
app = Flask(__name__, static_url_path='/static')
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
app.secret_key = os.getenv("JWT_SECRET_KEY")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=30)
jwt = JWTManager(app)



# ---- JWT 토큰 갱신 ----
@app.route('/token/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh_token():
    identity = get_jwt_identity()
    access_token = create_access_token(identity=identity)
    return jsonify(access_token=access_token), 200

with open('api-doc.yaml', 'r', encoding='utf-8-sig') as f:
    swagger_template = yaml.safe_load(f)

swagger = Swagger(app, template=swagger_template)

CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

interpreter = Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "ap-northeast-2")

KAKAO_API_KEY = os.getenv("KAKAO_REST_API_KEY")


# Hugging Face Inference API 설정
load_dotenv()
HF_API_KEY  = os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL    = "mistralai/Mistral-Small-24B-Instruct-2501"
HF_API_URL  = "https://router.huggingface.co/together/v1/chat/completions"
HEADERS     = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type":  "application/json"
}

# HF API 호출 공용 함수
def query(payload: dict) -> dict:
    """HF Inference API 에 안전하게 POST 한 뒤 JSON 리턴 (3회 재시도)"""
    for attempt in range(1, 4):
        try:
            resp = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=(5, 60))
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"[HF_API] attempt {attempt} failed: {e}")
            time.sleep(attempt)
    raise RuntimeError("HF API 호출 3회 모두 실패")


dynamodb = boto3.resource(
    'dynamodb',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
cred = credentials.Certificate('silmedy-23a1b-firebase-adminsdk-fbsvc-1e8c6b596b.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://silmedy-23a1b-default-rtdb.firebaseio.com'
})
fs_db = firestore.client()




# ---- 테이블 목록 ----
# Realtime
calls_ref = db.reference('calls')

# Firestore
collection_patients = fs_db.collection('patients')
collection_doctors = fs_db.collection('doctors')
collection_admins = fs_db.collection('admins')
collection_counters = fs_db.collection('fs_counters')
collcetion_call_text = fs_db.collection('call_text')
collection_consult_text = fs_db.collection('consult_text')
collection_diagnosis_text = fs_db.collection('diagnosis_text')

# DynamoDB
table_ai_consults = dynamodb.Table('ai_consults')
table_care_requests = dynamodb.Table('care_requests')
table_counters = dynamodb.Table('counters')
table_diagnosis_records = dynamodb.Table('diagnosis_records')
table_diseases_teachable = dynamodb.Table('diseases_teachable')
table_diseases = dynamodb.Table('diseases')
table_diseases_similar = dynamodb.Table('diseases_similar')
table_drug_deliveries = dynamodb.Table('drug_deliveries')
table_drugs = dynamodb.Table('drugs')
table_hospitals = dynamodb.Table('hospitals')
table_pharmacies = dynamodb.Table('pharmacies')
table_prescription_records = dynamodb.Table('prescription_records')

# 전처리 함수들
def normalize(text: str) -> str:
    """
    한글과 숫자만 남기고:
    - 모든 문자를 소문자화
    - 공백 제거
    """
    # 한글(jamo 제외)과 숫자만 남김
    cleaned = re.sub(r"[^가-힣0-9\s]", "", text)
    # 소문자화 후 공백 제거
    return cleaned.lower().replace(" ", "")

def clean_symptom(text: str) -> str:
    """
    사용자 증상 입력에서:
      1) 불필요한 표현 제거
      2) 한국어 조사 제거
      3) 연속된 공백 제거
    """
    unnecessary = [
        "설명해 줘", "말해 줘", "알려주세요", "알려 줘", "가요",
        "알겠습니다", "네", "아니요", "혹시", "요","심해",
        "너무","많이","아주","조금"
    ]
    # 긴 표현 우선 제거
    unnecessary.sort(key=len, reverse=True)
    for token in unnecessary:
        text = re.sub(token, "", text, flags=re.IGNORECASE)
    # 한국어 조사 제거
    text = re.sub(r"(을|를|이|가|은|는)\b", "", text)
    # 공백 정리
    return re.sub(r"\s+", "", text).strip()

# DynamoDB AttributeValue -> Python 타입 변환
deserializer = TypeDeserializer()

def deserialize_item(av_map: Dict[str, Any]) -> Dict[str, Any]:
    """
    DynamoDB AttributeValue 맵을 순수 Python 타입으로 변환
    """
    return {k: deserializer.deserialize(v) for k, v in av_map.items()}

def to_list(val: Any) -> List[Any]:
    """
    입력값이 리스트가 아니면 리스트로 감싸서 반환
    """
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    return [val]





# 외과 긴급 키워드 목록
class HybridLlamaService:
    # 외과 긴급 키워드 목록
    SURGICAL_KEYWORDS = ["골절", "뼈 부러짐", "상처", "출혈", "멍"]
        # 이비인후과/안과(기타) 키워드 목록
    ENT_OPHTH_KEYWORDS = [
        # 안과 키워드
        "눈", "시야", "충혈", "안구", "시력", "눈물", "눈부심", "안통", "눈통증",
        # 이비인후과 키워드
        "귀", "이명", "청력", "코막힘", "콧물", "재채기", "목", "목통증", "인후통"
    ]

    def classify_text(self, text: str) -> str:
        lower = text.lower()
        # 1) 외과 응급 키워드
        if any(kw in lower for kw in self.SURGICAL_KEYWORDS):
            return "외과"
        # 2) 이비인후과/안과 키워드 -> 기타
        if any(kw in lower for kw in self.ENT_OPHTH_KEYWORDS):
            return "기타"
        # 3) 그 외는 내과
        return "내과"

    def load_rules(self) -> List[Dict[str, Any]]:
        try:
            items = table_diseases.scan().get("Items", [])
            if not items:
                return []
            first = next(iter(items[0].values()))
            if isinstance(first, dict) and ("S" in first or "L" in first):
                return [deserialize_item(i) for i in items]
            return items
        except Exception as e:
            logger.warning(f"룰 로드 실패: {e}")
            return []


    def call_llm_for_symptom(self, patient_id: str, messages: List[str]) -> str:
        """
        patient_id: 환자 식별자 (로그 등에 활용)
        messages:   환자 발화 문자열의 리스트
        """
        system_msg = "너는 친절한 내과 상담 AI야. 한국어로 환자가 불안해지지 않도록 부드럽게 답변해줘."
        # 메시지 리스트를 하나의 대화(dialog)로 합칩니다.
        dialog = "\n".join(f"환자: {m}" for m in messages)

        prompt = (
            f"{dialog}\n\n"
            "아래 양식을 **엄격히** 준수해 **200자 이내**로 답변해주세요.\n"
            "※ 환자가 말한 부위를 **명사 형태로 한 단어**로 다시 제시해주세요. 예: “속쓰림이 너무 심해요” → “속쓰림”\n"
            "※ 의료 맥락에 맞지 않는 단어(예: 굴욕, 굉장 등) 사용 금지\n"
            "※ 마지막 두 문구를 반드시 포함하세요:\n"
            "  - 정확한 진단은 전문가 상담을 통해 진행하세요.\n"
            "  - 비대면 진료가 필요하면 '예'라고 답해주세요.\n\n"
            "## 출력 양식\n"
            "- patient_symptom   :  (1가지)  # 환자가 말한 부위를 명사로\n\n"
            "- disease_symptoms  : (1~2가지)\n"
            "- main_symptoms     : (1~2가지)\n"
            "- home_actions      : (1~2가지)\n"
            "- guideline         : (1~2가지)\n"
            "- emergency_advice  : (1~2가지)\n\n"
            "## 예시\n"
            "patient_symptoms  : 속쓰림"
            "disease_symptoms  : 만성 위염, 위염 "
            "main_symptoms     : 속쓰림, 구역 "
            "home_actions      : 식사량 조절, 충분한 휴식  "
            "guideline         : 제산제 복용 권장, 스트레스 관리 필요 "
            "emergency_advice  : 흑색변·혈변 시 병원 방문 "
            "※ 정확한 진단은 전문가 상담을 통해 진행하세요.  "
            "※ 비대면 진료가 필요하면 '예'라고 답해주세요."
        )
        payload = {
            "model": HF_MODEL,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 400
        }
        result = query(payload)
        return result["choices"][0]["message"]["content"].strip()


    def generate_llama_response(self, patient_id: str, chat_history: List[Any]) -> Dict[str, Any]:
        """
        1) 외과 키워드 → 즉시 외과 안내  
        2) 분류 → 내과/외과 결정  
        3) 내과 → LLM 호출 후 룰 매칭  
        4) 기타 과목 → 선택 유도  
        5) 외과 fallback  
        """
        raw      = chat_history[-1]
        last_msg = (raw.get("patient_text") if isinstance(raw, dict) else str(raw)).strip()
        symptom  = clean_symptom(last_msg)
        norm     = normalize(symptom)
        logger.info(f"[generate] patient_id={patient_id}, symptom={symptom}")
        

        # 1) 외과 긴급 키워드
        if any(kw in norm for kw in self.SURGICAL_KEYWORDS):
            return {"category":"외과","text":(
                "외과 진료가 필요해 보여요.\n"
                "편하실 때 촬영을 통해 증상을 확인해 보실 수 있습니다.\n"
                "지금 터치로 증상 확인 페이지로 이동해 보시겠어요? (예/아니오)"
            )}

        # 2) 분류
        category = self.classify_text(last_msg)


        # 3) 내과 처리
        if category == "내과":
            # 3-1) LLM 호출
            ai_text = self.call_llm_for_symptom("\n".join([m if isinstance(m,str) else m.get("patient_text","") 
                                                         for m in chat_history]),
                                               last_msg).strip()

            # 3-2) DB 룰 매칭 (후보 키워드)
            if not ai_text:   
                for rule in self.load_rules():
                    kws = (
                        to_list(rule.get("sub_category")) +
                        to_list(rule.get("main_symptoms")) +
                        to_list(rule.get("symptom_synonyms")) +
                        ([rule.get("name_ko")] if rule.get("name_ko") else [])
                    )
                    kws_norm = [normalize(k) for k in kws if isinstance(k, str)]
                    if any(k in norm for k in kws_norm):
                        name     = to_list(rule.get("name_ko")) or ["정보 없음"]
                        home_act = to_list(rule.get("home_actions")) or ["정보 없음"]
                        emerg    = to_list(rule.get("emergency_advice")) or ["정보 없음"]
                        db_text = (
                            f"disease_symptoms: {', '.join(name)}\n"
                            f"home_actions: {', '.join(home_act)}\n"
                            f"emergency_advice: {', '.join(emerg)}\n"
                            "비대면 진료가 필요하면 '예'라고 답해주세요.\n"
                            "(※ 정확한 진단은 전문가 상담을 통해 진행하세요.)"
                        )
                    return {"category":"내과", "text": db_text}
                    
            return {"category":"내과","text":ai_text}
        
        # 4) 기타 과목
        if category not in ("내과","외과"):
            return {"category":"기타","text":(
                "전문의 상담이 필요해 보이는 증상입니다.\n"
                "내과 또는 외과 중 추가로 원하시는 상담이 있나요? (내과/외과)"
            )}

        # 5) 외과 fallback
        return {"category":"외과","text":(
            "외과 진료가 필요해 보여요.\n"
            "편하실 때 촬영을 통해 증상을 확인해 보실 수 있습니다.\n"
            "지금 터치로 증상 확인 페이지로 이동해 보시겠어요? (예/아니오)"
        )}
    

# ---------채팅 요약 ---------
def summarize_dialog(dialog: str) -> str:
    """대화를 100자 내외로 요약"""
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role":"system","content":"당신은 의학전문 요약가입니다. 핵심만 간결하게 요약하세요."},
            {"role":"user","content":f"아래 환자↔AI 대화를 100자 내외로 한국어로 요약해주세요.\n\n{dialog}"}
        ],
        "temperature":0.1,
        "max_tokens":120
    }
    return query(payload)["choices"][0]["message"]["content"].strip()


# ── 부위별 증상 매핑
PART_SYMPTOMS = {
    "귀":       ["귀 통증", "청력 저하", "이명", "귀가 먹먹함", "귀가 아픔"],
    "코":       ["코막힘", "콧물", "코답답", "콧물분비", "재채기", "물콧물"],
    "유방":     ["유방 종괴", "피부 함몰", "유방 통증", "유두 분비물"],
    "잇몸":     ["잇몸 출혈", "발적", "잇몸 부종", "치은염"],
    "가슴":     ["호흡곤란", "가슴 통증", "심계항진", "흉통", "숨가쁨", "숨답답"],
    "전신":     ["부종", "식은땀", "수면 곤란", "피로감", "무증상", "근육통", "전신 통증", "피로", "발열", "권태감"],
    "근육":     ["근육통", "근육 경련", "근육 약화", "근육 경직", "근육 뻐근"],
    "관절":     ["관절강직", "관절 통증", "관절 부종", "관절염", "관절 붓기"],
    "비뇨기":   ["화장실자주감", "배뇨 곤란", "야뇨", "긴박뇨", "단백뇨", "배뇨 통증"],
    "입":       ["목마름", "구강 건조증", "입마름", "구취", "입안 통증"],
    "허리":     ["요통", "허리 통증", "허리 아픔"],
    "뇌":       ["기억력 저하", "인지기능 장애", "우울감", "무기력", "흥미상실", "편측 마비", "언어장애"],
    "머리":     ["두통", "머리아픔", "편두통", "현기증", "어지러움"],
    "위":       ["구역", "속쓰림", "신트림", "상복부 통증", "소화 불량"],
    "배":       ["설사", "복통"],
    "하복부":   ["하복부 통증", "경련"],
    "눈":       ["충혈", "눈곱", "시야 흐림", "눈부심", "눈물", "눈통증", "시력 저하"],
    "목":       ["인후통", "목 통증", "목이 아픔", "목소리 변화"],
    "피부":     ["발진", "가려움증", "피부 발적", "피부 건조증", "피부 통증"],
    "폐":       ["기침", "잦은기침", "천명음"],
    "치아":     ["치아통증", "치통", "과민반응"]
}



def extract_structured_info_manual(ai_text: str) -> dict:
    """
    ai_text에서  disease_symptoms, main_symptoms, home_actions,
    guideline, emergency_advice를 파싱하고,
    main_symptoms를 기준으로 symptom_part까지 매핑해 반환합니다.
    """
    patterns = {

        "disease_symptoms":  r"-\s*disease_symptoms\s*[:：]\s*([^\n]+)",
        "main_symptoms":     r"-\s*main_symptoms\s*[:：]\s*([^\n]+)",
        "home_actions":      r"-\s*home_actions\s*[:：]\s*([^\n]+)",
        "guideline":         r"-\s*guideline\s*[:：]\s*([^\n]+)",
        "emergency_advice":  r"-\s*emergency_advice\s*[:：]\s*([^\n]+)",
    }

    parsed = {k: [] for k in patterns}
    for key, pat in patterns.items():
        m = re.search(pat, ai_text, re.IGNORECASE)
        if m:
            parsed[key] = [tok.strip() for tok in m.group(1).split(",") if tok.strip()]

    # symptom_part 매핑
    symptom_part = []
    for ms in parsed["main_symptoms"]:
        for part, syms in PART_SYMPTOMS.items():
            if normalize(ms) in {normalize(s) for s in syms}:
                symptom_part.append(part)
                break
    symptom_part = list(dict.fromkeys(symptom_part))

    return {

        "disease_symptoms":  parsed["disease_symptoms"],
        "main_symptoms":     parsed["main_symptoms"],
        "home_actions":      parsed["home_actions"],
        "guideline":         parsed["guideline"],         
        "emergency_advice":  parsed["emergency_advice"],
        "symptom_part":      symptom_part
    }   

# ---------채팅 요약 ---------
def summarize_dialog(dialog: str) -> str:
    """대화를 100자 내외로 요약"""
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role":"system","content":"당신은 의학전문 요약가입니다. 핵심만 간결하게 요약하세요."},
            {"role":"user","content":f"아래 환자↔AI 대화를 80자 내외로 한국어로 요약해주세요.\n\n{dialog}"}
        ],
        "temperature":0.1,
        "max_tokens":120
    }
    return query(payload)["choices"][0]["message"]["content"].strip()


# ── 부위별 증상 매핑
PART_SYMPTOMS = {
    "귀":       ["귀 통증", "청력 저하", "이명", "귀가 먹먹함", "귀가 아픔"],
    "코":       ["코막힘", "콧물", "코답답", "콧물분비", "재채기", "물콧물"],
    "유방":     ["유방 종괴", "피부 함몰", "유방 통증", "유두 분비물"],
    "잇몸":     ["잇몸 출혈", "발적", "잇몸 부종", "치은염"],
    "가슴":     ["호흡곤란", "가슴 통증", "심계항진", "흉통", "숨가쁨", "숨답답"],
    "전신":     ["부종", "식은땀", "수면 곤란", "피로감", "무증상", "근육통", "전신 통증", "피로", "발열", "권태감"],
    "근육":     ["근육통", "근육 경련", "근육 약화", "근육 경직", "근육 뻐근"],
    "관절":     ["관절강직", "관절 통증", "관절 부종", "관절염", "관절 붓기"],
    "비뇨기":   ["화장실자주감", "배뇨 곤란", "야뇨", "긴박뇨", "단백뇨", "배뇨 통증"],
    "입":       ["목마름", "구강 건조증", "입마름", "구취", "입안 통증"],
    "허리":     ["요통", "허리 통증", "허리 아픔"],
    "뇌":       ["기억력 저하", "인지기능 장애", "우울감", "무기력", "흥미상실", "편측 마비", "언어장애"],
    "머리":     ["두통", "머리아픔", "편두통", "현기증", "어지러움"],
    "위":       ["구역", "속쓰림", "신트림", "상복부 통증", "소화 불량"],
    "배":       ["설사", "복통"],
    "하복부":   ["하복부 통증", "경련"],
    "눈":       ["충혈", "눈곱", "시야 흐림", "눈부심", "눈물", "눈통증", "시력 저하"],
    "목":       ["인후통", "목 통증", "목이 아픔", "목소리 변화"],
    "피부":     ["발진", "가려움증", "피부 발적", "피부 건조증", "피부 통증"],
    "폐":       ["기침", "잦은기침", "천명음"],
    "치아":     ["치아통증", "치통", "과민반응"]
}



def extract_structured_info_manual(ai_text: str) -> dict:
    """
    ai_text에서 patient_symptom,disease_symptoms, main_symptoms, home_actions,
    guideline, emergency_advice를 한 번에 파싱하고,
    main_symptoms를 기준으로 symptom_part까지 매핑해 반환합니다.
    """
    # 1) 파싱할 섹션의 패턴을 한 곳에 정의 (더 유연하게, 대시(-) 없이도, 한글 콜론(：) 허용)
    patterns = {
        "patient_symptom" : r"\bpatient_symptoms\s*[:：]\s*([^-\n]+)",
        "disease_symptoms": r"\bdisease_symptoms\s*[:：]\s*([^-\n]+)",
        "main_symptoms"   : r"\bmain_symptoms\s*[:：]\s*([^-\n]+)",
        "home_actions"    : r"\bhome_actions\s*[:：]\s*([^-\n]+)",
        "guideline"        : r"\bguideline\s*[:：]\s*([^-\n]+)",
        "emergency_advice": r"\bemergency_advice\s*[:：]\s*([^-\n]+)",
    }

    # 2) 각 항목을 정규식으로 한 번만 순회하며 파싱
    parsed = {k: [] for k in patterns}
    for key, pat in patterns.items():
        m = re.search(pat, ai_text, re.IGNORECASE)
        if m:
            parsed[key] = [tok.strip() for tok in m.group(1).split(",") if tok.strip()]


    # 3) PART_SYMPTOMS 매핑을 이용해 symptom_part 생성
    symptom_part = []
    for ms in parsed["main_symptoms"]:
        for part, syms in PART_SYMPTOMS.items():
            # 필요시 normalize/clean_symptom 적용
            if ms in syms:
                symptom_part.append(part)
                break
    symptom_part = list(dict.fromkeys(symptom_part))  # 중복 제거

    # 4) 최종 반환 형식 맞춰 리턴
    return {
        "patient_symptom":  parsed["patient_symptom"],
        "disease_symptoms": parsed["disease_symptoms"],
        "main_symptoms":    parsed["main_symptoms"],
        "home_actions":     parsed["home_actions"],
        "guideline":         parsed["guideline"],
        "emergency_advice": parsed["emergency_advice"],
        "symptom_part":     symptom_part
    }




# ---- 환자 회원가입 ----
@app.route('/patient/signup', methods=['POST'])
def patient_signup():
    logger.info(f"REQUEST: {request.json}")
    try:
        body = request.get_json()
        email = body['email']

        # 이메일 중복 확인 (반드시 doc id 생성 전에)
        existing_user = collection_patients.where("email", "==", email).limit(1).stream()
        if next(existing_user, None):
            return jsonify({'error': '이미 사용 중인 이메일입니다.'}), 409

        # Counter 가져오기 (이메일 중복 확인 이후에 진행)
        counter_doc = collection_counters.document('patients').get()
        if counter_doc.exists:
            current_id = counter_doc.to_dict().get('current_id', 0)
        else:
            current_id = 0

        new_doc_id = current_id + 1

        item = {
            'email': email,
            'password': body['password'],
            'name': body['name'],
            'contact': body['contact'],
            'postal_code': body['postal_code'],
            'address': body['address'],
            'address_detail': body['address_detail'],
            'birth_date': body['birth_date'],
            'created_at': datetime.utcnow().isoformat(),
            'sign_language_needed': body.get('sign_language_needed', False),
            'is_default_address': body.get('is_default_address', False)
        }

        # 새로운 환자 등록
        collection_patients.document(str(new_doc_id)).set(item)
        logger.info(f"Inserted patient: {item}")

        # 카운터 업데이트 (등록 완료 후)
        collection_counters.document('patients').update({'current_id': new_doc_id})

        return jsonify({'message': '환자 등록 성공'}), 200

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500


# ---- 환자 로그인 ----
@app.route('/patient/login', methods=['POST'])
def patient_login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400

        user_query = collection_patients.where("email", "==", email).limit(1).stream()
        user_doc = next(user_query, None)

        if user_doc and user_doc.exists:
            item = user_doc.to_dict()
            if item.get('password') == password:
                patient_id = user_doc.id  # ✔️ 실제 로그인한 환자의 ID

                access_token = create_access_token(
                    identity=patient_id,
                    additional_claims={"name": item.get('name', '')}
                )
                refresh_token = create_refresh_token(identity=patient_id)

                return jsonify({
                    'message': 'Login successful',
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'username': item.get('name', ''),
                    'fcm_token': item.get('fcm_token', '')
                }), 200
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
        else:
            return jsonify({'error': 'Invalid credentials'}), 401

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ---- FCM 토큰 등록 ----
@app.route('/patient/fcm-token', methods=['POST'])
@jwt_required()
def register_fcm_token():
    try:
        patient_id = get_jwt_identity()  # JWT의 sub (환자 ID 또는 이메일)

        data = request.get_json()
        fcm_token = data.get('fcm_token')
        if not fcm_token:
            return jsonify({'error': 'FCM 토큰이 필요합니다.'}), 400

        # Firestore에서 이메일 또는 ID 기반 문서 찾기
        user_query = collection_patients.document(patient_id).get()
        if not user_query.exists:
            return jsonify({'error': '사용자를 찾을 수 없습니다.'}), 404

        user_query.reference.update({'fcm_token': fcm_token})
        return jsonify({'message': 'FCM 토큰이 성공적으로 저장되었습니다.'}), 200

    except Exception as e:
        logger.error(f"FCM 토큰 저장 오류: {e}")
        return jsonify({'error': '서버 오류로 저장에 실패했습니다.'}), 500


# ---- 우편번호 검색 ----
@app.route('/postal_code', methods=['GET'])
def search_postal_code():
    keyword = request.args.get('keyword')
    if not keyword:
        return jsonify({'error': 'Keyword is required'}), 400

    postcode_key =  os.getenv("POSTAL_CODE_KEY")
    try:
        encoded_keyword = requests.utils.quote(keyword, encoding='utf-8')
        apiUrl = f"https://business.juso.go.kr/addrlink/addrLinkApi.do?currentPage=1&countPerPage=100&keyword={encoded_keyword}&confmKey={postcode_key}&resultType=json"
        response = requests.get(apiUrl)
        response.raise_for_status()
        data = response.json()

        juso_list = data.get("results", {}).get("juso", [])
        result = [{"zipNo": j.get("zipNo"), "roadAddr": j.get("roadAddr")} for j in juso_list]

        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 환자 비밀번호 변경 ----
@app.route('/patient/repassword', methods=['POST'])
def patient_change_password():
    try:
        data = request.get_json()
        email = data.get('email')
        new_password = data.get('new_password')

        if not email or not new_password:
            return jsonify({'error': 'Email and new password required'}), 400

        user_query = collection_patients.where("email", "==", email).limit(1).stream()
        user_doc = next(user_query, None)

        if user_doc and user_doc.exists:
            user_doc.reference.update({'password': new_password})
            return jsonify({'message': '비밀번호 변경 완료'}), 200
        else:
            return jsonify({'error': '사용자 없음'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 로그아웃----
@app.route('/patient/logout', methods=['POST'])
@jwt_required()
def logout():
    patient_id = get_jwt_identity()
    logger.info(f"Patient {patient_id} logged out.")
    return jsonify({'message': '로그아웃 처리 완료'}), 200


# ---- 환자 마이페이지 조회 ----
@app.route('/patient/mypage', methods=['GET'])
@jwt_required()
def get_mypage():
    patient_id = get_jwt_identity()
    if not patient_id:
        return jsonify({'error': 'Patient ID required'}), 400

    doc = collection_patients.document(patient_id).get()
    if doc.exists:
        return jsonify(doc.to_dict()), 200
    else:
        return jsonify({'error': 'User not found'}), 404
    

# ---- 회원 정보 수정  ----
@app.route('/patient/update', methods=['POST'])
@jwt_required()
def update_patient_info():
    try:
        data = request.get_json()
        patient_id = get_jwt_identity()
        updates = data.get('updates')  # 수정할 필드들 (딕셔너리)

        if not updates:
            return jsonify({'error': '업데이트 항목이 필요합니다.'}), 400

        user_doc = collection_patients.document(patient_id).get()
        if user_doc.exists:
            collection_patients.document(patient_id).update(updates)
            return jsonify({'message': '회원 정보 수정 완료'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 회원 탈퇴   ----
@app.route('/patient/delete', methods=['DELETE'])
@jwt_required()
def delete_patient():
    patient_id = get_jwt_identity()
    if not patient_id:
        return jsonify({'error': 'Patient ID required'}), 400

    doc_ref = collection_patients.document(patient_id)
    doc = doc_ref.get()

    if doc.exists:
        doc_ref.delete()
        return jsonify({'message': '회원 탈퇴 완료'}), 200
    else:
        return jsonify({'error': 'User not found'}), 404
    

# 임의 인증번호 저장
verification_codes = {}

# 전화번호 유효성 검사 함수
def is_valid_phone_number(phone):
    # 010으로 시작하고 숫자만 있는 10자리 또는 11자리
    return re.fullmatch(r"^010\d{7,8}$", phone)

# ---- 인증번호 요청 ----
@app.route("/request-verification-code", methods=["POST"])
def request_verification_code():
    data = request.json
    phone_number = data.get("phone_number")

    if not phone_number:
        return jsonify({"success": False, "message": "전화번호가 필요합니다."}), 400

    # 전화번호 형식 검사
    if not is_valid_phone_number(phone_number):
        return jsonify({"success": False, "message": "올바른 전화번호 형식이 아닙니다. 예: 01012345678"}), 400

    # 인증번호 생성 및 저장
    verification_code = ''.join(random.choices(string.digits, k=6))
    verification_codes[phone_number] = verification_code

    logging.info(f"[DEV] '{phone_number}'로 인증번호 발송됨: {verification_code}")

    return jsonify({
        "message": "인증번호가 발송되었습니다.",
        "debug_code": verification_code  # 테스트용. 실제 서비스에선 제거
    }), 200


# ---- 인증번호 확인 ----
@app.route("/verify-code", methods=["POST"])
def verify_code():
    data = request.json
    phone_number = data.get("phone_number")
    code = data.get("code")

    if not phone_number or not code:
        return jsonify({"success": False, "message": "전화번호와 인증번호가 필요합니다."}), 400

    if verification_codes.get(phone_number) == code:
        return jsonify({"success": True, "message": "인증 성공"}), 200
    else:
        return jsonify({"success": False, "message": "인증 실패"}), 400
    


# --- 인증 후 이메일 확인 ---
@app.route("/verify-code-get-email", methods=["POST"])
def verify_and_get_email():
    
    phone = request.json.get("phone_number", "").strip()  # 전화번호에서 하이픈 제거
    code = request.json.get("code", "").strip()  # 인증코드

    # 인증번호 확인 (여기서는 예시로 임의의 코드 비교)
    if verification_codes.get(phone) != code:  # verification_codes는 인증번호 저장소
        return jsonify({"success": False, "message": "인증번호가 일치하지 않습니다."}), 400

    try:
        # Firestore에서 해당 전화번호(contact)를 가진 유저 찾기
        # 입력된 전화번호에서 하이픈을 추가하여 Firestore에서 조회할 전화번호와 형식 맞추기
        formatted_phone = f"{phone[:3]}-{phone[3:7]}-{phone[7:]}"  # 하이픈 추가
        users_ref = collection_patients.where("contact", "==", formatted_phone).limit(1).stream()
        user_doc = next(users_ref, None)

        if not user_doc:  # user_doc가 없으면 유저가 없다는 메시지
            return jsonify({"success": False, "message": "해당 번호로 등록된 유저가 없습니다."}), 404

        # 유저의 이메일 가져오기
        user_data = user_doc.to_dict()
        email = user_data.get("email", "이메일 없음")
        
        return jsonify({
            "success": True,
            "email": email
        }), 200

    except Exception as e:
        return jsonify({"success": False, "message": f"오류 발생: {str(e)}"}), 500
    

# ---- 인증 후 비밀번호 변경 ----
@app.route("/verify-code-check-user", methods=["POST"])
def verify_code_check_user():
    email = request.json.get("email", "").strip()
    phone = request.json.get("phone_number", "").replace("-", "").strip()
    code = request.json.get("code", "").strip()

    # 인증번호 확인
    if verification_codes.get(phone) != code:
        return jsonify({"success": False, "message": "인증번호가 일치하지 않습니다."}), 400

    # 전화번호에 하이픈 추가 (파이어스토어에 저장된 형식에 맞추기)
    formatted_phone = f"{phone[:3]}-{phone[3:7]}-{phone[7:]}"  # 예: 01012341234 → 010-1234-1234

    try:
        # Firestore에서 이메일과 전화번호 둘 다 일치하는 유저 찾기 (doc id가 아닌 필드 기반)
        user_query = (
            collection_patients
            .where("email", "==", email)
            .where("contact", "==", formatted_phone)
            .limit(1)
            .stream()
        )
        user_doc = next(user_query, None)

        if user_doc and user_doc.exists:
            return jsonify({"success": True, "message": "사용자 확인 완료"}), 200
        else:
            return jsonify({"success": False, "message": "가입된 사용자가 아닙니다."}), 404

    except Exception as e:
        return jsonify({"success": False, "message": f"오류 발생: {str(e)}"}), 500
    


# ---- 환자 이름 반환 ----
@app.route('/patient/name', methods=['GET'])
@jwt_required()
def get_patient_name():
    try:
        patient_id = get_jwt_identity()
        if not patient_id:
            return jsonify({'error': 'Unauthorized'}), 401

        doc = collection_patients.document(str(patient_id)).get()
        if not doc.exists:
            return jsonify({'error': 'User not found'}), 404

        data = doc.to_dict()
        name = data.get('name', '')

        return jsonify({'name': name}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



 # ---- 증상으로 관련 정보 조회 ----
@app.route('/info-by-symptom', methods=['POST'])
def get_disease_info_by_symptom():
    try:
        data = request.get_json()
        symptom = data.get('symptom')

        if not symptom:
            return jsonify({'error': 'symptom is required'}), 400

        response = table_diseases_teachable.scan(
            FilterExpression=Attr('name_ko').eq(symptom)
        )
        items = response.get('Items', [])
        if not items:
            return jsonify({'error': 'No matching disease found'}), 404

        item = items[0]
        department = item.get('department')
        sub_departments = item.get('sub_department', '')
        image_url = item.get('image_url')

        if isinstance(sub_departments, str):
            sub_departments = sub_departments.strip('{}').replace('"', '').split(',')

        return jsonify({
            'department': department,
            'sub_department': sub_departments[0] if sub_departments else None,
            'image_url': image_url
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    



# ─────── AI 채팅 저장 ───────

# ── 전역 서비스 인스턴스 ───
service = HybridLlamaService()

@app.route('/chat/save', methods=['POST'])
@jwt_required()
def save_chat():
    try:
        patient_id = get_jwt_identity()
        data = request.get_json()
        patient_text = data.get('patient_text', '').strip()
        if not patient_id or not patient_text:
            return jsonify({"error": "Missing required fields"}), 400

        now  = datetime.utcnow()
        ts   = now.strftime("%Y-%m-%d %H:%M:%S")
        coll = collection_consult_text.document(str(patient_id)).collection("chats")

        # 최초 구분자
        if not any(coll.stream()):
            sep = now - timedelta(milliseconds=1)
            sid = sep.strftime("%Y%m%d%H%M%S%f")
            coll.document(sid).set({
                'chat_id': sid, 'sender_id':'',
                'text':'',
                'created_at': sep.strftime("%Y-%m-%d %H:%M:%S"),
                'is_separator': True
            })

        # 1) 환자 메시지 저장
        pid = now.strftime("%Y%m%d%H%M%S%f")
        coll.document(pid).set({
            'chat_id': pid,
            'sender_id': '나',
            'text': patient_text,
            'created_at': ts,
            'is_separator': False
        })

        # 2) 분류 및 응답 생성
        resp = service.generate_llama_response(patient_id, [patient_text])
        category = resp.get('category')
        ai_text  = resp.get('text', '')

        # 3) 외과 혹은 기타일 경우 즉시 반환
        if category in ("외과", "기타"):
            return jsonify({
                "category": category,
                "message": ai_text,
                "chat_ids": [pid]
            }), 200

        # 4) 내과 응답 저장
        aid = (now + timedelta(milliseconds=1)).strftime("%Y%m%d%H%M%S%f")
        coll.document(aid).set({
            'chat_id': aid,
            'sender_id': 'AI',
            'text': ai_text,
            'created_at': ts,
            'is_separator': False
        })

        # 5) 결과 반환
        return jsonify({
            "category": category,
            "chat_ids": [pid, aid],
            "ai_text": ai_text,
            "message": "Chat saved"
        }), 200

    except Exception as e:
        logger.error(f"Error saving chat: {e}")
        return jsonify({'error': str(e)}), 500


# ---- 의사 진료 가능 시간 + 수어 필요 여부 통합 확인 ----
@app.route('/request/availability-signcheck', methods=['GET'])
@jwt_required()
def get_availability_and_signcheck():
    try:
        patient_id = get_jwt_identity()
        license_number = request.args.get('license_number')

        if not license_number:
            return jsonify({'error': 'license_number is required'}), 400

        # Fetch doctor's reservations
        doc_ref = collection_doctors.document(license_number)
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({'error': 'Doctor not found'}), 404

        doctor_id = doc.id
        today = datetime.utcnow().date()
        tomorrow = today + timedelta(days=1)

        dates = [today.isoformat(), tomorrow.isoformat()]
        reservations = []

        for date_str in dates:
            response = table_care_requests.scan()
            for item in response.get('Items', []):
                item_doctor_id = str(item.get('doctor_id', '')).strip()
                item_book_date = item.get('book_date', '')

                if item_doctor_id == str(doctor_id) and item_book_date == date_str:
                    if isinstance(item.get('symptom_part'), set):
                        item['symptom_part'] = list(item['symptom_part'])
                    if isinstance(item.get('symptom_type'), set):
                        item['symptom_type'] = list(item['symptom_type'])
                    reservations.append(item)

        # Fetch patient sign language need
        patient_doc = collection_patients.document(str(patient_id)).get()
        if not patient_doc.exists:
            return jsonify({'error': 'User not found'}), 404

        patient_data = patient_doc.to_dict()
        sign_language_needed = patient_data.get('sign_language_needed', False)

        return jsonify({
            'reservations': reservations,
            'sign_language_needed': sign_language_needed
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/request/confirmed', methods=['POST'])
@jwt_required()
def confirm_reservation():
    try:
        data = request.get_json()
        patient_id = get_jwt_identity()
        required_fields = [
            "doctor_id", "department",
            "symptom_part", "symptom_type", "book_date",
            "book_hour", "sign_language_needed"
        ]

        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required reservation information."}), 400

        # 카운터 조회 및 증가
        counter_response = table_counters.get_item(Key={"counter_name": "request_id"})
        counter = counter_response.get("Item", {})
        current_id = int(counter.get("current_id", 0)) + 1
        table_counters.put_item(Item={"counter_name": "request_id", "current_id": current_id})

        # 예약 정보 구성 (DynamoDB 저장용)
        reservation_item = {
            "request_id": current_id,
            "patient_id": patient_id,
            "doctor_id": int(data["doctor_id"]),
            "department": data["department"],
            "symptom_part": set(data["symptom_part"]),
            "symptom_type": set(data["symptom_type"]),
            "book_date": data["book_date"],
            "book_hour": data["book_hour"],
            "sign_language_needed": data["sign_language_needed"],
            "is_solved": False,
            "requested_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 실제 응답으로 사용할 JSON 직렬화 가능한 dict
        reservation_response = {
            "request_id": current_id,
            "patient_id": patient_id,
            "doctor_id": int(data["doctor_id"]),
            "department": data["department"],
            "symptom_part": list(data["symptom_part"]),
            "symptom_type": list(data["symptom_type"]),
            "book_date": data["book_date"],
            "book_hour": data["book_hour"],
            "sign_language_needed": data["sign_language_needed"],
            "is_solved": False,
            "requested_at": reservation_item["requested_at"]
        }

        table_care_requests.put_item(Item=reservation_item)

        return jsonify({
            "message": "진료 예약이 확정되었습니다.",
            "reservation": reservation_response
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
   

# ---- 처방전 삽입 정보 반환 ----
# @app.route('/prescription/data', methods=['GET'])
# @jwt_required()
# def get_prescription_url():
#     try:
#         patient_id = get_jwt_identity()
#         diagnosis_id = request.args.get('diagnosis_id')
#         if not diagnosis_id:
#             return jsonify({'error': 'diagnosis_id is required'}), 400

#         prescription_items = table_prescription_records.scan(
#             FilterExpression=Attr('diagnosis_id').eq(int(diagnosis_id))
#         ).get('Items', [])
#         if not prescription_items:
#             return jsonify({'error': 'Prescription not found for this diagnosis'}), 404
        
#         prescription_item = prescription_items[0]
#         prescription_id = prescription_item.get('prescription_id')
#         doctor_id = prescription_item.get('doctor_id')
#         medication_days = prescription_item.get('medication_days', [])
#         medication_list = prescription_item.get('medication_list', [])
#         # 안전한 타입 체크 및 반복
#         if isinstance(medication_list, list):
#             for med in medication_list:
#                 if isinstance(med, dict):
#                     disease_id = med.get('disease_id')
#                     drug_id = med.get('drug_id')
#                     print(f"Disease: {disease_id}, Drug: {drug_id}")


#         patient_items = collection_patients.document(str(patient_id)).get()
#         if not patient_items.exists:
#             return jsonify({'error': 'User not found'}), 404
        
#         patient_data = patient_items.to_dict()
#         patient_name = patient_data.get('name', '')
#         patient_identity = patient_data.get('birth_date', '').replace("-", "").strip()[2:]


#         doctor_items = collection_doctors.document(str(doctor_id)).get()
#         if not doctor_items.exists:
#             return jsonify({'error': 'Doctor not found'}), 404
        
#         doctor_data = doctor_items.to_dict()
#         doctor_name = doctor_data.get('name', '')
#         hospital_id = doctor_data.get('hospital_id', '')
        

#         hospital_items = table_hospitals.scan(
#             FilterExpression=Attr('hospital_id').eq(hospital_id)
#         ).get('Items', [])
#         if not hospital_items:
#             return jsonify({'error': 'Hospital not found'}), 404
        
#         hospital_item = hospital_items[0]
#         hospital_name = hospital_item.get('name', '')
#         hospital_contact = hospital_item.get('contact', '')


#         # Step 2: Check if delivery exists for this prescription_id
#         delivery_items = table_drug_deliveries.scan(
#             FilterExpression=Attr('prescription_id').eq(int(prescription_id))
#         ).get('Items', [])

#         is_made = bool(delivery_items)

#         return jsonify({
#             'prescription_id': prescription_id,
#             'doctor_id': doctor_id,
#             'medication_days': medication_days,
#             'medication_list': medication_list,
#             'patient_name': patient_name,
#             'patient_identity': patient_identity,
#             'doctor_name': doctor_name,
#             'hospital_name': hospital_name,
#             'hospital_contact': hospital_contact,
#             'is_made': is_made
#         }), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    


# ---- 처방전 URL 반환 ----
@app.route('/prescription/url', methods=['GET'])
@jwt_required()
def get_prescription_url():
    try:
        patient_id = get_jwt_identity()
        diagnosis_id = request.args.get('diagnosis_id')
        if not diagnosis_id:
            return jsonify({'error': 'diagnosis_id is required'}), 400

        # Step 1: Find prescription record matching diagnosis_id
        prescription_items = table_prescription_records.scan(
            FilterExpression=Attr('diagnosis_id').eq(int(diagnosis_id))
        ).get('Items', [])

        if not prescription_items:
            return jsonify({'error': 'Prescription not found for this diagnosis'}), 404

        prescription_item = prescription_items[0]
        prescription_id = prescription_item.get('prescription_id')
        prescription_url = prescription_item.get('prescription_url')

        # Step 2: Check if delivery exists for this prescription_id
        delivery_items = table_drug_deliveries.scan(
            FilterExpression=Attr('prescription_id').eq(int(prescription_id))
        ).get('Items', [])

        is_made = bool(delivery_items)

        return jsonify({
            'prescription_id': prescription_id,
            'prescription_url': prescription_url,
            'is_made': is_made
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 진료 내역 반환 ----
@app.route('/diagnosis/list', methods=['GET'])
@jwt_required()
def get_diagnosis_by_patient():
    try:
        patient_id = get_jwt_identity()

        response = table_diagnosis_records.scan()
        items = []
        for item in response.get('Items', []):
            if str(item.get('patient_id', '')).strip() == patient_id:
                diagnosis_id = item.get('diagnosis_id')
                diagnosed_at = item.get('diagnosed_at', '')
                doctor_id = item.get('doctor_id', '')
                symptoms = list(item.get('symptoms', [])) if isinstance(item.get('symptoms'), set) else item.get('symptoms', [])

                try:
                    diagnosed_date = datetime.strptime(diagnosed_at, "%Y-%m-%d %H:%M:%S").date().isoformat()
                except ValueError:
                    diagnosed_date = diagnosed_at  # fallback if format is wrong

                # doctor_id 기반 hospital_id 가져오기 및 hospital_name 조회
                doctor_doc = collection_doctors.document(str(doctor_id)).get()
                hospital_id = doctor_doc.to_dict().get('hospital_id') if doctor_doc.exists else None

                hospital_name = None
                if hospital_id:
                    hospital_doc = table_hospitals.get_item(Key={'hospital_id': int(hospital_id)})
                    if 'Item' in hospital_doc:
                        hospital_name = hospital_doc['Item'].get('name')

                result = {
                    'diagnosis_id': diagnosis_id,
                    'diagnosed_at': diagnosed_date,
                    'symptoms': symptoms,
                    'hospital_name': hospital_name
                }
                items.append(result)

        return jsonify({'records': items}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 환자 기본 주소 반환 ----
@app.route('/patient/default-address', methods=['GET'])
@jwt_required()
def get_default_address():
    try:
        patient_id = get_jwt_identity()

        doc = collection_patients.document(str(patient_id)).get()
        if not doc.exists:
            return jsonify({'error': 'User not found'}), 404

        data_doc = doc.to_dict()
        is_default = data_doc.get('is_default_address', False)

        if is_default:
            return jsonify({
                'is_default_address': True,
                'postal_code': data_doc.get('postal_code', ''),
                'address': data_doc.get('address', ''),
                'address_detail': data_doc.get('address_detail', ''),
                'name': data_doc.get('name', ''),
                'contact': data_doc.get('contact', '')
            }), 200
        else:
            return jsonify({
                'is_default_address': False,
                'postal_code': data_doc.get('postal_code', ''),
                'address': data_doc.get('address', ''),
                'address_detail': data_doc.get('address_detail', ''),
                'name': data_doc.get('name', ''),
                'contact': data_doc.get('contact', '')
                }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 배송 요청 등록 ----
@app.route('/delivery/register', methods=['POST'])
@jwt_required()
def register_delivery():
    try:
        data = request.get_json()
        required_fields = ['is_delivery', 'prescription_id']
        if not all(field in data for field in required_fields):
            return jsonify({'error': '필수 항목 누락'}), 400

        patient_id = get_jwt_identity()
        is_delivery = data['is_delivery']

        # 환자 정보 업데이트: is_default_address 반영
        if 'is_default_address' in data:
            collection_patients.document(patient_id).update({
                'is_default_address': data['is_default_address']
            })

        if 'patient_contact' in data:
            contact = data['patient_contact']
        else:
            contact = collection_patients.document(patient_id).get().to_dict().get('contact', '')

        # 배송 요청일 경우 필수 필드 확인
        if is_delivery:
            if not all(k in data for k in ['address', 'postal_code']):
                return jsonify({'error': '주소 및 우편번호는 필수입니다 (is_delivery=True)'})
        
        # delivery_id 발급
        counter_response = table_counters.get_item(Key={"counter_name": "delivery_id"})
        counter = counter_response.get("Item", {})
        current_id = int(counter.get("current_id", 0)) + 1
        table_counters.put_item(Item={"counter_name": "delivery_id", "current_id": current_id})

        # 배송 정보 구성
        if 'pharmacy_id' not in data:
            delivery = {
                "delivery_id": current_id,
                "patient_id": patient_id,
                "is_delivery": is_delivery,
                "patient_contact": contact,
                "prescription_id": data["prescription_id"],
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "is_received": False
            }
        else:
            delivery = {
                "delivery_id": current_id,
                "patient_id": patient_id,
                "is_delivery": is_delivery,
                "patient_contact": contact,
                "pharmacy_id": data["pharmacy_id"],
                "prescription_id": data["prescription_id"],
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "is_received": False
            }

        if is_delivery:
            delivery["address"] = data["address"]
            delivery["postal_code"] = data["postal_code"]
            if "delivery_request" in data:
                delivery["delivery_request"] = data["delivery_request"]

        table_drug_deliveries.put_item(Item=delivery)

        return jsonify({
            "message": "배송 요청이 등록되었습니다.",
            "delivery": delivery
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 배송 완료 처리 ----
@app.route('/delivery/complete', methods=['POST'])
@jwt_required()
def mark_delivery_as_received():
    try:
        data = request.get_json()
        delivery_id = data.get('delivery_id')
        patient_id = get_jwt_identity()

        if not delivery_id:
            return jsonify({'error': 'delivery_id is required'}), 400

        # 해당 배송 건이 존재하는지 확인
        response = table_drug_deliveries.get_item(Key={'delivery_id': int(delivery_id)})
        item = response.get('Item')

        if not item or item.get('patient_id') != patient_id:
            return jsonify({'error': '해당 배송 내역을 찾을 수 없습니다.'}), 404

        # 배송 수령 완료 처리
        table_drug_deliveries.update_item(
            Key={'delivery_id': int(delivery_id)},
            UpdateExpression='SET is_received = :val1, delivered_at = :val2',
            ExpressionAttributeValues={
                ':val1': True,
                ':val2': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            }
        )

        return jsonify({'message': '배송 완료 처리되었습니다.', 'delivery_id': delivery_id}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 전화 대기 ----
@app.route('/call/waiting-doctor', methods=['POST'])
def update_waiting_call():
    try:
        data = request.get_json()
        room_code = data.get('room_code')

        if not room_code:
            return jsonify({'error': 'room_code is required'}), 400

        # Locate the document by room_code (doc_id)
        doc_ref = calls_ref.child(room_code)
        if not doc_ref.get():
            return jsonify({'error': 'Call not found'}), 404

        # Update status to 'accepted'
        doc_ref.update({'status': 'accepted'})

        return jsonify({'message': 'Call status updated to accepted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# ---- 전화 연결 ----
@app.route('/call/accept', methods=['POST'])
def accept_call():
    try:
        data = request.get_json()
        call_id = data.get('call_id')

        if not call_id:
            return jsonify({'error': 'call_id is required'}), 400

        doc_ref = calls_ref.document(str(call_id))
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({'error': 'Call not found'}), 404

        doc_ref.update({
            'is_accepted': True,
            'started_at': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        })
        return jsonify({'message': 'Call accepted successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500




# ---- 화면에 의사 이름 및 정보 표시 (room_code 기반) ----
@app.route('/call/doctor-id', methods=['POST'])
def get_doctor_info_from_room_code():
    try:
        data = request.get_json()
        room_code = data.get('room_code')

        if not room_code:
            return jsonify({'error': 'room_code is required'}), 400

        # Find the call document by room_code
        call_doc_ref = calls_ref.child(room_code)
        call_doc = call_doc_ref.get()

        if not call_doc:
            return jsonify({'error': 'Call not found'}), 404

        doctor_id = call_doc.get('doctor_id')
        if not doctor_id:
            return jsonify({'error': 'Doctor ID not found in call'}), 404

        # Fetch doctor information from doctors collection
        doctor_doc = collection_doctors.document(str(doctor_id)).get()
        if not doctor_doc.exists:
            return jsonify({'error': 'Doctor not found'}), 404

        doctor_data = doctor_doc.to_dict()

        return jsonify({
            'doctor_id': doctor_id,
            'name': doctor_data.get('name', '')
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 환자 텍스트 저장 ----
@app.route('/call/add-patient-text', methods=['POST'])
def add_patient_text():
    try:
        data = request.get_json()
        room_code = data.get('room_code')
        patient_id = data.get('patient_id')
        patient_text = data.get('patient_text')

        if not room_code or not patient_id or not patient_text:
            return jsonify({'error': 'room_code, patient_id, and patient_text are required'}), 400

        doc_ref = calls_ref.child(room_code)
        doc = doc_ref.get()

        if not doc:
            return jsonify({'error': 'Call not found'}), 404

        # Retrieve existing patient_texts for this room
        existing_texts = doc.get('patient_texts', {})

        if not isinstance(existing_texts, dict):
            existing_texts = {}

        # Append new text to patient's list
        patient_text_list = existing_texts.get(patient_id, [])
        patient_text_list.append(patient_text.strip())
        existing_texts[patient_id] = patient_text_list

        doc_ref.update({'patient_texts': existing_texts})

        return jsonify({'message': 'Patient text added successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# ---- 의사 텍스트 화면에 표출 ----
@app.route('/call/latest-doctor-text', methods=['POST'])
def get_latest_doctor_text():
    try:
        data = request.get_json()
        room_code = data.get('room_code')

        if not room_code:
            return jsonify({'error': 'room_code is required'}), 400

        doc_ref = calls_ref.child(room_code)
        doc = doc_ref.get()

        if not doc:
            return jsonify({'error': 'Call not found'}), 404

        doctor_text_list = doc.get('doctor_text', [])
        if not isinstance(doctor_text_list, list) or not doctor_text_list:
            return jsonify({'message': 'No doctor text found'}), 404

        return jsonify({'latest_doctor_text': doctor_text_list[-1]}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 전화 종료 ----
@app.route('/call/end', methods=['POST'])
def end_call():
    try:
        data = request.get_json()
        room_code = data.get('room_code')

        if not room_code:
            return jsonify({'error': 'room_code is required'}), 400

        doc_ref = calls_ref.child(room_code)
        doc = doc_ref.get()

        if not doc:
            return jsonify({'error': 'Call not found'}), 404

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        doc_ref.update({
            'ended_at': now,
            'status': 'ended'
        })

        return jsonify({'message': 'Call ended successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    

# ----    음성-> 텍스트 변환     ───────────────────────────────


# 1) HF Whisper API 설정
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
HF_API_KEY  = os.getenv("HUGGINGFACE_API_KEY")
HEADERS     = {
    "Authorization": f"Bearer {HF_API_KEY}"

    }

# 2) 확장자 → Content-Type 매핑
CONTENT_TYPES = {
    ".flac": "audio/flac",
    ".wav":  "audio/wav",
    ".mp3":  "audio/mpeg",
    ".ogg":  "audio/ogg",
    ".m4a":  "audio/mp4",
}

def get_content_type(filename: str, default: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return CONTENT_TYPES.get(ext, default)

def query_whisper_bytes(data: bytes, content_type: str) -> dict:
    resp = requests.post(
        API_URL,
        headers={**HEADERS, "Content-Type": content_type},
        data=data,
        timeout=60
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # 상세 에러 로그 출력
        print(f"[HF ERROR] status={resp.status_code}, body={resp.text}")
        raise
    return resp.json()


# ---- 음성→텍스트 변환 (실시간, 저장 없이) ----
@app.route('/voice/add-txt', methods=['POST'])
def add_txt():
    """
    요청: multipart/form-data, 필드명 'audio'
    응답: { "text": "전사된 한글 문자열" }
    """
    if 'audio' not in request.files:
        return jsonify({"error": "audio 파일이 없습니다"}), 400

    f = request.files['audio']
    raw = f.read()
    content_type = get_content_type(f.filename, f.mimetype or "application/octet-stream")

    try:
        result = query_whisper_bytes(raw, content_type)
        text = result.get("text", "")
        return jsonify({"text": text.strip()})
    except requests.HTTPError as e:
        # e.response 가 아니라 resp.text 를 이미 로깅했으니 간단히
        return jsonify({
            "error": "Upstream API 오류",
            "status": e.response.status_code if e.response else None,
            "detail": e.response.text if e.response else str(e)
        }), 502
    except Exception as e:
        print(f"[SERVER ERROR] {e}")
        return jsonify({"error": f"서버 내부 오류: {e}"}), 500








# ── 채팅 구분선 이후 요약 & 구조화 정보 저장 ───────────────────────────────
@app.route('/chat/add-separator', methods=['POST'])
@jwt_required()
def add_chat_separator():
    try:
        patient_id = get_jwt_identity()
        if not patient_id:
            return jsonify({"error": "patient_id is required"}), 400

        # 1) Firestore에서 separator 이후 대화 읽기 (내림차순)
        coll = collection_consult_text.document(str(patient_id)).collection("chats")
        from firebase_admin import firestore as _fs
        docs = list(coll.order_by("created_at", direction=_fs.Query.DESCENDING).stream())

        # 2) 마지막 separator 시각 찾기
        last_sep = None
        for doc in docs:
            d = doc.to_dict()
            if d.get("is_separator") or d.get("is_separater"):
                last_sep = d["created_at"]
                break

        # 3) separator 이후의 환자 발화·AI 발화를 차례로 뽑아냄
        last_patient = last_ai = None
        for doc in docs:
            d = doc.to_dict()
            if last_sep and d["created_at"] <= last_sep:
                continue
            if d["sender_id"] == "나" and not last_patient:
                # patient_text 필드가 우선
                last_patient = d.get("patient_text") or d.get("text")
                last_patient = last_patient.strip()
            elif d["sender_id"] == "AI" and not last_ai:
                # ai_text 필드가 우선
                last_ai = d.get("ai_text") or d.get("text")
                last_ai = last_ai.strip()
            if last_patient and last_ai:
                break

        if not (last_patient and last_ai):
            return jsonify({"error": "대화 내역이 충분하지 않습니다."}), 400

        # 4) AI 응답에서 구조화 정보 추출
        info = extract_structured_info_manual(last_ai)
        # manual 함수에서 guideline 키로 뽑아왔는지 꼭 확인하세요
        main_symptoms     = info["main_symptoms"]
        disease_symptoms  = info["disease_symptoms"]
        home_actions      = info["home_actions"]
        guideline_actions = info["guideline"]
        emerg_actions     = info["emergency_advice"]
        symptom_part      = info["symptom_part"]

        # 5) summary 생성
        if main_symptoms:
            pati_txt = main_symptoms[0]
        else:
            # “속쓰림이 너무 심해요” → “속쓰림”
            pati_txt = re.sub(r"(이|가).*", "", last_patient)

        disease_txt = ", ".join(disease_symptoms) or "정보 없음"
        home_txt    = ", ".join(home_actions)     or "정보 없음"
        guide_txt   = " 및 ".join(guideline_actions) or "정보 없음"
        emerg_txt   = emerg_actions[0] if emerg_actions else "필요 시 병원 방문"

        summary = (
            f"환자는 {pati_txt}에 대해 불편함을 호소하여, AI는 {disease_txt}일 가능성을 제시하고, "
            f"{home_txt}를 권장했으며, {guide_txt}를 추천했습니다. "
            f"{emerg_txt}을 권유하였습니다."
        )

        # 6) consult_id 카운터 증가 & DynamoDB 저장
        ctr = table_counters.get_item(Key={"counter_name": "consult_id"})
        new_id = int(ctr.get("Item", {}).get("current_id", 0)) + 1
        table_counters.put_item(Item={"counter_name": "consult_id", "current_id": new_id})

        table_ai_consults.put_item(Item={
            "consult_id":       new_id,
            "patient_id":       int(patient_id),
            "disease_symptoms": disease_symptoms,
            "symptom_part":     symptom_part,
            "analysis":         summary
        })

        # 7) Firestore 에 separator 한 줄 추가
        sep_time = datetime.utcnow()
        sep_id   = sep_time.strftime("%Y%m%d%H%M%S%f")
        coll.document(sep_id).set({
            'chat_id':      sep_id,
            'sender_id':    '',
            'text':         '',
            'created_at':   sep_time.strftime("%Y-%m-%d %H:%M:%S"),
            'is_separator': True
        })

        # 8) 결과 반환
        return jsonify({
            "patient_id":       int(patient_id),
            "disease_symptoms": disease_symptoms,
            "symptom_part":     symptom_part,
            "analysis":         summary,
            "message":          "Chat saved"
        }), 200

    except Exception as e:
        logger.error(f"Error in add_chat_separator: {e}")
        return jsonify({"error": str(e)}), 500

# ---- 채팅 구분선 생성 및 외과 이동 ----
@app.route('/chat/move-to-body', methods=['POST'])
@jwt_required()
def move_with_separator():
    try:
        patient_id = get_jwt_identity()
        if not patient_id:
            return jsonify({"error": "patient_id is required"}), 400

        # 1) Firestore에서 separator 이후 대화 읽기 (내림차순)
        coll = collection_consult_text.document(str(patient_id)).collection("chats")

        sep = datetime.utcnow() - timedelta(milliseconds=1)
        sid = sep.strftime("%Y%m%d%H%M%S%f")
        coll.document(sid).set({
            'chat_id': sid, 'sender_id':'',
            'text':'',
            'created_at': sep.strftime("%Y-%m-%d %H:%M:%S"),
            'is_separator': True
        })

        response = {
            "answer" : "사진 기반 진단으로 이동합니다."
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in add_chat_separator: {e}")
        return jsonify({"error": str(e)}), 500


# ---- 보건소+의사 통합 검색 ----
@app.route('/health-centers-with-doctors', methods=['GET'])
def health_centers_with_doctors():
    try:
        lat_str = request.args.get('lat')
        lng_str = request.args.get('lng')
        department = request.args.get('department')
        gender = request.args.get('gender')

        if not lat_str or not lng_str or not department:
            return jsonify({"error": "Missing 'lat', 'lng' or 'department' parameter"}), 400

        try:
            lat = float(lat_str)
            lng = float(lng_str)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid 'lat' or 'lng' value"}), 400

        headers = {
            "Authorization": f"KakaoAK {KAKAO_API_KEY}"
        }
        params = {
            "query": "보건소",
            "x": str(lng),
            "y": str(lat),
            "radius": 10000,
            "sort": "distance"
        }
        kakao_url = "https://dapi.kakao.com/v2/local/search/keyword.json"

        response = requests.get(kakao_url, headers=headers, params=params)
        if response.status_code != 200:
            return jsonify({"error": "Kakao API error"}), 500

        kakao_data = response.json()
        documents = kakao_data.get("documents", [])

        clinic_list = []
        for doc in documents:
            place_name = doc.get("place_name", "")
            if place_name.endswith("보건소") and " " not in place_name:
                clinic_list.append(place_name)
            if len(clinic_list) >= 5:
                break

        if not clinic_list:
            return jsonify({"error": "No health centers found nearby"}), 404

        matched_hospitals = []
        for item in table_hospitals.scan().get('Items', []):
            if item.get('name') in clinic_list:
                matched_hospitals.append({
                    "hospital_id": item.get('hospital_id'),
                    "name": item.get('name')
                })

        doctors = []
        for doc in collection_doctors.stream():
            data = doc.to_dict()
            data["license_number"] = doc.id  # 여기서 직접 추가
            for hospital in matched_hospitals:
                if (
                    data.get("hospital_id") == hospital["hospital_id"]
                    and data.get("department") == department
                    and (not gender or gender == "전체" or data.get("gender") == gender)
                ):
                    doctors.append({
                        "hospital_id": data.get("hospital_id"),
                        "hospital_name": hospital["name"],
                        "profile_url": data.get("profile_url"),
                        "name": data.get("name"),
                        "department": data.get("department"),
                        "gender": data.get("gender"),
                        "contact": data.get("contact"),
                        "email": data.get("email"),
                        "availability": data.get("availability"),
                        "license_number": data.get("license_number")
                    })
                    break

        return jsonify(doctors), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# ---- 약국 근처 상세 정보 통합 반환 ----
@app.route('/pharmacies/nearby-info', methods=['GET'])
@jwt_required()
def search_pharmacies_with_details():
    try:
        patient_id = get_jwt_identity()
        if not patient_id:
            return jsonify({'error': 'Unauthorized'}), 401

        lat_str = request.args.get('lat')
        lng_str = request.args.get('lng')

        if not lat_str or not lng_str:
            return jsonify({"error": "Missing 'lat' or 'lng' parameter"}), 400

        try:
            lat = float(lat_str)
            lng = float(lng_str)
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid 'lat' or 'lng' value"}), 400

        logger.info(f"[pharmacies] lat: {lat}, lng: {lng}")

        headers = {
            "Authorization": f"KakaoAK {KAKAO_API_KEY}"
        }
        params = {
            "query": "약국",
            "x": str(lng),
            "y": str(lat),
            "radius": 10000,
            "sort": "distance"
        }

        kakao_url = "https://dapi.kakao.com/v2/local/search/keyword.json"
        response = requests.get(kakao_url, headers=headers, params=params)

        if response.status_code != 200:
            return jsonify({"error": "Kakao API error"}), 500

        kakao_data = response.json()
        documents = kakao_data.get("documents", [])
        logger.info(f"[pharmacies] 검색된 place_names: {[doc.get('place_name') for doc in documents]}")

        matched_pharmacies = []
        for doc in documents:
            name = doc.get("place_name", "").strip()
            phone = doc.get("phone", "").strip()

            if name.endswith("약국") and " " not in name and phone:
                scan_result = table_pharmacies.scan(
                    FilterExpression=Attr('name').eq(name) & Attr('contact').eq(phone)
                )
                items = scan_result.get('Items', [])
                if items:
                    item = items[0]
                    matched_pharmacies.append({
                        'pharmacy_id': item.get('pharmacy_id'),
                        'pharmacy_name': item.get('name'),
                        'open_hour': item.get('open_hour'),
                        'close_hour': item.get('close_hour'),
                        'address': item.get('address'),
                        'contact': item.get('contact')
                    })
                if len(matched_pharmacies) >= 10:
                    break

        return jsonify(matched_pharmacies), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)