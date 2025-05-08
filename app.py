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
import torch
import numpy as np
import requests
import boto3
import toml
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
# from tflite_runtime.interpreter import Interpreter
from tensorflow.lite.python.interpreter import Interpreter


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
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL   = "mistralai/Mistral-7B-Instruct-v0.3"
client = OpenAI(
    base_url=f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}/v1",
    api_key=HF_API_KEY,
)


HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type":  "application/json"
}


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
        "알겠습니다", "네", "아니요", "혹시", "요"
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

def _deserialize_item(av_map: Dict[str, Any]) -> Dict[str, Any]:
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
     # HF API 호출 공용 함수

def query(payload: Dict[str, Any]) -> Dict[str, Any]:
    """HF Inference API 에 안전하게 POST 한 뒤 JSON 리턴"""
    for attempt in range(1, 4):
        try:
            r = requests.post(HF_API_URL, headers=HEADERS, json=payload, timeout=(5,60))
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            logger.warning(f"[HF_API] attempt {attempt} failed: {e}")
            time.sleep(attempt)
    raise RuntimeError("HF API 호출 3회 모두 실패")

# 외과 긴급 키워드 목록
class HybridLlamaService:
    # 외과 긴급 키워드 목록
    SURGICAL_KEYWORDS = ["골절", "뼈 부러짐", "상처", "출혈", "멍"]

    def _classify_text(self, text: str) -> str:
        lower = text.lower()
        return "외과" if any(kw in lower for kw in self.SURGICAL_KEYWORDS) else "내과"

    def _load_rules(self) -> List[Dict[str, Any]]:
        try:
            items = table_diseases.scan().get("Items", [])
            if not items:
                return []
            first = next(iter(items[0].values()))
            if isinstance(first, dict) and ("S" in first or "L" in first):
                return [_deserialize_item(i) for i in items]
            return items
        except Exception as e:
            logger.warning(f"룰 로드 실패: {e}")
            return []

    def _call_llm_for_symptom(self, symptom: str) -> str:
        """
        1) OpenAI‐style HF router client 호출 시도
        2) 실패 시 HTTP POST(query) 폴백
        """
        prompt = (
            "너는 친절한 내과 상담 AI야. 예상 원인 1~2가지, 자가관리법 1~2가지, "
            "이럴 땐 병원 방문 안내를 2~3줄로 알려줘.\n\n"
            f"Symptom: {symptom}"
        )

        # 1) OpenAI‐style client 호출
        try:
            completion = client.chat.completions.create(
                model=HF_MODEL,
                messages=[
                    {"role": "system", "content": "너는 친절한 내과 상담 AI야."},
                    {"role": "user",   "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3,
                top_p=0.9,
                n=1
            )
            # 정상 응답이 있으면 바로 반환
            msg = completion.choices[0].message.content.strip()
            if msg:
                return msg
        except Exception as e:
            logger.warning(f"[HF_CLIENT] OpenAI‐style 호출 실패: {e}")

        # 2) HTTP POST 폴백
        payload = {
            "model": HF_MODEL,
            "messages": [
                {"role": "system", "content": "너는 친절한 내과 상담 AI야."},
                {"role": "user",   "content": prompt}
            ],
            "max_tokens": 150,
            "temperature": 0.3,
            "top_p": 0.9,
            "n": 1
        }
        try:
            data = query(payload)
            # chat-completion 응답 포맷 처리
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"].strip()
            # fallback: text-generation 형식
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
        except Exception as e:
            logger.error(f"[HF_API] 폴백 호출 전체 실패: {e}")

        return "죄송합니다. 현재 해당 증상에 대한 정보를 생성할 수 없습니다."

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
        


        # # 2) AI 응답 생성
        # ai_resp = service.generate_llama_response(patient_id, [patient_text])
        # ai_text = ai_resp.get("text") if isinstance(ai_resp, dict) else str(ai_resp)

        # 1) 외과 긴급 키워드
        if any(kw in norm for kw in self.SURGICAL_KEYWORDS):
            return {
                "category": "외과",
                "text": (
                    "외과 진료가 필요해 보여요.\n"
                    "편하실 때 촬영을 통해 증상을 확인해 보실 수 있습니다.\n"
                    "지금 터치로 증상 확인 페이지로 이동해 보시겠어요? (예/아니오)"
                )
            }

        # 2) 분류
        category = self._classify_text(last_msg)

        # 3) 내과 처리
        if category == "내과":
            # 1차: LLM 호출
            ai_text = self._call_llm_for_symptom(last_msg)

            # 2차: DB 룰 매칭 (sub_category, main_symptoms, symptom_synonyms, name_ko)
            # Clean and normalize user symptom
            symptom = last_msg
            norm = normalize(clean_symptom(symptom))
            for rule in self._load_rules():
                # gather potential match keywords
                main_syms = to_list(rule.get("main_symptoms"))
                syns      = to_list(rule.get("symptom_synonyms"))
                subs      = to_list(rule.get("sub_category"))
                name_ko   = [rule.get("name_ko")] if rule.get("name_ko") else []
                kws       = subs + main_syms + syns + name_ko
                # normalize keywords
                kws_norm  = [normalize(kw) for kw in kws if isinstance(kw, str)]
                # match against normalized user symptom
                if any(kw in norm for kw in kws_norm):
                    # format outputs
                    name = to_list(rule.get("name_ko")) or ["정보 없음"]
                    home_act  = to_list(rule.get("home_actions"))       or ["정보 없음"]
                    emerg     = to_list(rule.get("emergency_advice"))   or ["정보 없음"]
                    return {
                        "category": "내과",
                        "text": (
                            f"예상 질환: {', '.join(name)}\n"
                            f"자가관리: {', '.join(home_act)}\n"
                            f"이럴 땐 병원: {', '.join(emerg)}\n"
                            "비대면 진료가 필요하면 '예'라고 답해주세요.\n"
                            "(※ 정확한 진단은 전문가 상담을 통해 진행하세요.)"
                        )
                    }



        # 4) 기타 과목 → 전문의 선택 유도
        if category not in ("내과", "외과"):
            return {
                "category": "기타",
                "text": (
                    "전문의 상담이 필요해 보이는 증상입니다.\n"
                    "내과 또는 외과 중 추가로 원하시는 상담이 있나요? (내과/외과)"
                )
            }

        # 5) 외과 fallback
        return {
            "category": "외과",
            "text": (
                "외과 진료가 필요해 보여요.\n"
                "편하실 때 촬영을 통해 증상을 확인해 보실 수 있습니다.\n"
                "지금 터치로 증상 확인 페이지로 이동해 보시겠어요? (예/아니오)"
            )
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
                'is_separater': True
            })

        # 1) 환자 메시지 저장
        pid = now.strftime("%Y%m%d%H%M%S%f")
        coll.document(pid).set({
            'chat_id': pid, 
            'sender_id':'나',
            'text': patient_text, 
            'created_at': ts,
            'is_separater': False
        })

        # 2) AI 응답 생성
        ai_resp = service.generate_llama_response(patient_id, [patient_text])
        ai_text = ai_resp.get("text") if isinstance(ai_resp, dict) else str(ai_resp)

        # 외과 단답형 조기 반환
        if ai_text.strip() == "외과":
            return jsonify({"message": ai_text, "chat_ids":[pid]}), 200

        # 3) AI 메시지 저장
        aid = (now + timedelta(milliseconds=1)).strftime("%Y%m%d%H%M%S%f")
        coll.document(aid).set({
            'chat_id': aid, 
            'sender_id':'AI',
            'text': ai_text, 
            'created_at': ts,
            'is_separater': False
        })

        return jsonify({
            "message": "Chat saved",
            "chat_ids": [pid, aid],
            "ai_text": ai_text
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
                summary_text = list(item.get('summary_text', [])) if isinstance(item.get('summary_text'), set) else item.get('summary_text', [])

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
                    'summary_text': summary_text,
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


# ---- 채팅 구분선 추가 ----
@app.route('/chat/add-separator', methods=['POST'])
@jwt_required()
def add_chat_separator():
    try:
        identity = get_jwt_identity()
        patient_id = identity

        if not identity:
            return jsonify({"error": "patient_id is required"}), 400

        # Step 1: Fetch recent chats in descending order
        chat_collection = collection_consult_text.document(str(patient_id)).collection("chats")
        from firebase_admin import firestore as _firestore
        chats = chat_collection.order_by("created_at", direction=_firestore.Query.DESCENDING).stream()

        chat_list = []
        last_separator_time = None

        for chat in chats:
            chat_data = chat.to_dict()
            chat_list.append(chat_data)
            # Accept both is_separator and is_separater for compatibility
            if chat_data.get("is_separator") or chat_data.get("is_separater"):
                last_separator_time = chat_data.get("created_at")
                break

        # Step 2: Filter chats after the last separator
        texts_to_summarize = []
        if last_separator_time:
            # Since chat_list is descending, reverse for ascending
            for chat in reversed(chat_list):
                # Compare created_at as string (lexicographically works for "%Y-%m-%d %H:%M:%S")
                if chat.get("created_at") > last_separator_time:
                    patient_text = chat.get("patient_text", "")
                    ai_text = chat.get("ai_text", "")
                    if patient_text:
                        texts_to_summarize.append(patient_text.strip())
                    if ai_text:
                        texts_to_summarize.append(ai_text.strip())
        else:
            # No separator found, summarize everything
            for chat in reversed(chat_list):
                patient_text = chat.get("patient_text", "")
                ai_text = chat.get("ai_text", "")
                if patient_text:
                    texts_to_summarize.append(patient_text.strip())
                if ai_text:
                    texts_to_summarize.append(ai_text.strip())

        # Step 3: Simulate summary result (TODO: replace with real model call)
        symptom_part = ["전신"]  # 예시 출력
        symptom_type = ["두통", "구토"]  # 예시 출력

        # Step 4: Add separator chat document
        now = datetime.utcnow()
        chat_id = now.strftime("%Y%m%d%H%M%S%f")
        chat_data = {
            'chat_id': chat_id,
            'is_separater': True,
            'sender_id': '',
            'text': '',
            'created_at': now.strftime("%Y-%m-%d %H:%M:%S")
        }

        chat_collection.document(chat_id).set(chat_data)
        logger.info(f"[Firestore 구분선 저장됨] patient_id={patient_id}, chat_id={chat_id}")

        return jsonify({
            "message": "Separator added",
            "chat_id": chat_id,
            "symptom_part": symptom_part,
            "symptom_type": symptom_type
        }), 200

    except Exception as e:
        logger.error(f"Error adding separator: {e}")
        return jsonify({'error': str(e)}), 500


# ---- 보건소+의사 통합 검색 ----
@app.route('/health-centers-with-doctors', methods=['GET'])
def health_centers_with_doctors():
    try:
        lat_str = request.args.get('lat')
        lng_str = request.args.get('lng')
        department = request.args.get('department')

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
                if data.get("hospital_id") == hospital["hospital_id"] and data.get("department") == department:
                    doctors.append({
                        "hospital_id": data.get("hospital_id"),
                        "hospital_name": hospital["name"],
                        "profile_url": data.get("profile_url"),
                        "name": data.get("name"),
                        "department": data.get("department"),
                        "gender": data.get("gender"),
                        "contact": data.get("contact"),
                        "email": data.get("email"),
                        "bio": data.get("bio"),
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