from io import BytesIO
from flask import Flask, Blueprint, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, create_refresh_token
import firebase_admin
import boto3
import logging
from datetime import datetime, timedelta
from firebase_admin import credentials, firestore, db
import requests
import toml
from flask_cors import CORS
from dotenv import load_dotenv
import os
import random
import string
import re
from boto3.dynamodb.conditions import Attr
import uuid
from urllib.parse import unquote
from flasgger import Swagger
import yaml
from flask import Flask, request, jsonify
import numpy as np
from tflite_runtime.interpreter import Interpreter
from PIL import Image
from openai import OpenAI
from typing import Any, Optional
from openai.types.chat import ChatCompletion




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
table_diseases = dynamodb.Table('diseases')
table_drug_deliveries = dynamodb.Table('drug_deliveries')
table_drugs = dynamodb.Table('drugs')
table_hospitals = dynamodb.Table('hospitals')
table_pharmacies = dynamodb.Table('pharmacies')
table_prescription_records = dynamodb.Table('prescription_records')



def generate_llama_response(patient_id, chat_history):
    """
    1) 외과 긴급 키워드 감지 → '외과' 반환
    2) few-shot prompt로 LLM 분류 → '내과' or '외과'
    """
    last_msg = chat_history[-1]
    logger.info(f"[Llama 호출] patient_id={patient_id}, message=\"{last_msg}\"")

    # 1) 외과 긴급 키워드 목록 (최소화)
    SURGICAL_KEYWORDS = ["골절", "뼈 부러짐", "상처", "출혈"]
    lower = last_msg.lower()
    if any(kw in lower for kw in SURGICAL_KEYWORDS):
        logger.info("[Llama 중단] 외과 키워드 감지됨. ‘외과’ 반환.")
        return "외과"

    # 2) 환경변수에서 API 설정 읽기
    api_key = os.getenv("HUGGINGFACE_API_KEY") 
    api_url = os.getenv("HUGGINGFACE_API_URL") 

    if not api_key or not api_url:
        logger.error("Hugging Face API 키 또는 URL이 설정되지 않았습니다.")
        return "AI 응답을 가져오지 못했습니다."
    # fallback: ensure a string is always returned
    return "AI 응답을 가져오지 못했습니다."


def classify_or_prompt(self, prompt: str, sentence: str, cb: Any, category: str) -> None:
        lower = sentence.lower()

        # 3) 외과 긴급 키워드 감지
        for kw in self.SURGICAL_EMERGENCY_KEYWORDS:
            if kw in lower:
                cb.on_surgical_question(
                    "외과 진료가 필요해 보여요.\n"
                    "편하실 때 촬영을 통해 증상을 확인해 보실 수 있습니다.\n"
                    "지금 터치로 증상 확인 페이지로 이동해 보시겠어요? (예/아니오)"
                )
                return

        # 2) few-shot prompt로 LLM 분류
        try:
            system_prompt = (
                "너는 병원 접수 담당자야. 환자 문장을 보면 반드시 '내과' 또는 '외과'로 분류해.\n"
                "다음 예시를 참고해:\n"
                "예시1: \"손가락이 아파요\" → 내과\n"
                "예시2: \"팔이 부러진 것 같아요\" → 외과\n"
                "예시3: \"배가 너무 아파요\" → 내과\n"
                "예시4: \"심하게 베였어요\" → 외과\n"
                f"이제 문장: \"{sentence}\"\n"
                "출력은 한 단어(내과 또는 외과)만."
            )


    # 3) LLM 호출
            resp: ChatCompletion = self.client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sentence}
                ],
                temperature=0.0,
                max_tokens=5
            )
            cat = resp.choices[0].message.content.strip()
            if cat not in ("내과", "외과"):
                logger.warning(f"Unexpected classification '{cat}', fallback to '내과'")
                cat = "내과"
            cb.on_classification(cat, category)

        except Exception as e:
            cb.on_error(e)


def send_chat_stream(
        self,
        user_id: str,
        prev_symptom: Optional[str],
        user_message: str,
        cb: Any
    ) -> None:
        """
        AI 문진·케어 스트리밍 헬퍼
        """
        try:
            # 1) 외과 긴급 키워드 감지
            lower = user_message.lower()
            if any(kw in lower for kw in self.SURGICAL_EMERGENCY_KEYWORDS):
                cb.on_error(Exception("외과 증상 감지, 스트림 중단"))
                return

            # 2) 시스템 프롬프트 생성
            is_combined = bool(prev_symptom)
            system_prompt = self._build_system_prompt(prev_symptom or "", user_message, is_combined)

            # 3) 스트리밍 호출
            stream = self.client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                max_tokens=512,
                stream=True
            )
            # 4) 델타 단위로 콜백
            for chunk in stream:
                delta = chunk.choices[0].delta.get("content")
                if delta:
                    cb.on_chunk(delta)
            cb.on_complete()

        except Exception as e:
            cb.on_error(e)

        def _build_system_prompt(self, prev: str, msg: str, combined: bool) -> str:
            sb = [
                   "안녕하세요!🩺 저는 Slimedy AI 닥터링(Dr.Link)입니다.","궁금한 증상을 말씀해 주시면 쉽게 안내해 드릴게요.\n"]
            if is_combined:
                prompt += (
                    f"✏️ [복합 증상 분석]\n"
                    f"- 증상1: {prev_symptom}\n"
                    f"- 증상2: {user_message}\n"
                    f"- 원인 예시:\n"
                    f"  1) {prev_symptom}과 {user_message} 간 관련성 가능\n"
                    f"  2) 스트레스 또는 일시적 피로\n\n"
                )
            else:
                prompt += (
                    f"✏️ [증상 분석]\n"
                    f"- 증상: {user_message}\n"
                    f"- 원인 예시:\n"
                    f"  1) 일시적 피로\n"
                    f"  2) 환경 변화\n\n"
                )
            prompt += (
                "🏠 [집에서 할 수 있는 관리]\n"
                "  → 30분마다 미지근한 물을 조금씩 마셔보세요\n"
                "    (단, 신장질환·심부전·부종·삼킴 곤란·금식 등은 의료진과 상의)\n"
                "  → 1~2시간 편하게 쉬세요\n"
                "  → 증상과 체온을 기록해 두세요\n"
            )
            if "열" in user_message:
                prompt += "  → 38.5℃ 이상이면 해열제 복용\n"
            if "통증" in user_message:
                prompt += "  → 아픈 부위 5분간 찜질\n"
            if any(x in user_message for x in ["구토", "설사"]):
                prompt += "  → 이온음료(전해질 음료)로 수분 보충\n"
            prompt += (
                "\n⚠️ [이럴 땐 병원 방문]\n"
                "• 증상이 6시간 이상 계속됨\n"
            )
            if any(x in user_message for x in ["흉통", "호흡곤란"]):
                prompt += "• 가슴 답답/숨쉬기 힘듦 → 119 신고\n"
            if "의식저하" in user_message:
                prompt += "• 의식이 흐려짐/이상 행동\n"
            prompt += "\n\n비대면 진료가 필요하시면 '예'라고 답해주세요."
            return prompt

 

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
                # Generate new document ID based on fs_counters
                counter_doc = collection_counters.document('patients').get()
                if counter_doc.exists:
                    current_id = counter_doc.to_dict().get('current_id', 0)
                else:
                    current_id = 0
                new_doc_id = current_id + 1

                access_token = create_access_token(
                    identity=str(new_doc_id),
                    additional_claims={"name": item.get('name', '')}
                )
                refresh_token = create_refresh_token(identity=str(new_doc_id))
                return jsonify({
                    'message': 'Login successful',
                    'access_token': access_token,
                    'refresh_token': refresh_token,
                    'username': item.get('name', '')
                }), 200
            else:
                return jsonify({'error': 'Invalid credentials'}), 401
        else:
            return jsonify({'error': 'Invalid credentials'}), 401

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    

 # ---- 증상으로 관련 정보 조회 ----
@app.route('/info-by-symptom', methods=['POST'])
def get_disease_info_by_symptom():
    try:
        data = request.get_json()
        symptom = data.get('symptom')

        if not symptom:
            return jsonify({'error': 'symptom is required'}), 400

        response = table_diseases.scan(
            FilterExpression=Attr('name_ko').eq(symptom)
        )
        items = response.get('Items', [])
        if not items:
            return jsonify({'error': 'No matching disease found'}), 404

        item = items[0]
        department = item.get('department')
        sub_departments = item.get('sub_department', '')
        desc_url = item.get('desc_url')

        if isinstance(sub_departments, str):
            sub_departments = sub_departments.strip('{}').replace('"', '').split(',')

        return jsonify({
            'department': department,
            'sub_department': sub_departments[0] if sub_departments else None,
            'desc_url': desc_url
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    

@app.route('/chat/save', methods=['POST'])
@jwt_required()
def save_chat():
    try:
        identity = get_jwt_identity()
        patient_id = identity
        data = request.get_json()
        patient_text = data.get('patient_text')

        if not patient_id or not patient_text:
            return jsonify({"error": "Missing required fields"}), 400

        now = datetime.utcnow()
        created_at = now.strftime("%Y-%m-%d %H:%M:%S")
        chat_collection = collection_consult_text.document(str(patient_id)).collection("chats")

        # Add separator only if there are NO documents at all in the subcollection
        if not any(chat_collection.stream()):
            separator_time = now - timedelta(milliseconds=1)
            separator_id = separator_time.strftime("%Y%m%d%H%M%S%f")
            separator_data = {
                'chat_id': separator_id,
                'sender_id': '',
                'text': '',
                'created_at': separator_time.strftime("%Y-%m-%d %H:%M:%S"),
                'is_separater': True
            }
            chat_collection.document(separator_id).set(separator_data)

        # 1. Save patient text
        patient_chat_id = now.strftime("%Y%m%d%H%M%S%f")
        patient_chat_data = {
            'chat_id': patient_chat_id,
            'sender_id': '나',
            'text': patient_text.strip(),
            'created_at': created_at,
            'is_separater': False
        }
        chat_collection.document(patient_chat_id).set(patient_chat_data)

        # 2. Generate LLM response using helper function
        chat_history = [patient_text]  # 필요시 과거 채팅 내역도 포함 가능
        ai_response = generate_llama_response(patient_id, chat_history)

        # 외과 키워드 응답 시 저장 중단 및 조기 반환
        if ai_response.strip() == "외과":
            logger.info("[Chat 저장 중단] 외과 키워드로 인해 AI 응답 저장 생략")
            return jsonify({
                "message": "외과 관련 질문으로 판단되어 AI 응답은 생략되었습니다.",
                "chat_ids": [patient_chat_id]
            }), 200

        # 3. Save AI response
        ai_chat_id = (now + timedelta(milliseconds=1)).strftime("%Y%m%d%H%M%S%f")
        ai_chat_data = {
            'chat_id': ai_chat_id,
            'sender_id': 'AI',
            'text': ai_response.strip(),
            'created_at': created_at,
            'is_separater': False
        }
        chat_collection.document(ai_chat_id).set(ai_chat_data)

        logger.info(f"[Firestore 저장됨] patient_id={patient_id}, patient_chat_id={patient_chat_id}, ai_chat_id={ai_chat_id}")
        return jsonify({
            "message": "Chat saved",
            "chat_ids": [patient_chat_id, ai_chat_id],
            "ai_text": ai_response.strip()
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
def get_prescription_url():
    try:
        prescription_id = request.args.get('prescription_id')
        if not prescription_id:
            return jsonify({'error': 'prescription_id is required'}), 400

        response = table_prescription_records.get_item(Key={'prescription_id': int(prescription_id)})
        item = response.get('Item')

        if item and 'prescription_url' in item:
            return jsonify({'prescription_url': item['prescription_url']}), 200
        else:
            return jsonify({'error': 'Prescription not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 진료 내역 반환 ----
@app.route('/diagnosis/list', methods=['GET'])
@jwt_required()
def get_diagnosis_by_patient():
    try:
        patient_id = get_jwt_identity()
        # if not patient_id:
        #     return jsonify({'error': 'Patient ID is required'}), 400

        response = table_diagnosis_records.scan()
        items = []
        for item in response.get('Items', []):
            if str(item.get('patient_id', '')).strip() == patient_id:
                diagnosis_id = item.get('diagnosis_id')
                diagnosed_at = item.get('diagnosed_at', '')
                try:
                    diagnosed_date = datetime.strptime(diagnosed_at, "%Y-%m-%d %H:%M:%S").date().isoformat()
                except ValueError:
                    diagnosed_date = diagnosed_at  # fallback if format is wrong

                delivery_scan = table_drug_deliveries.scan(
                    FilterExpression=Attr('prescription_id').eq(diagnosis_id)
                )
                delivery_items = delivery_scan.get('Items', [])
                delivery_info = delivery_items[0] if delivery_items else {}

                result = {
                    'diagnosis_id': diagnosis_id,
                    'summary_text': list(item.get('summary_text', [])) if isinstance(item.get('summary_text'), set) else item.get('summary_text', []),
                    'diagnosed_at': diagnosed_date,
                    'is_delivery': delivery_info.get('is_delivery', False),
                    'is_received': delivery_info.get('is_received', False)
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
                'address_detail': data_doc.get('address_detail', '')
            }), 200
        else:
            return jsonify({'is_default_address': False}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 배송 요청 등록 ----
@app.route('/delivery/register', methods=['POST'])
@jwt_required()
def register_delivery():
    try:
        data = request.get_json()
        required_fields = ['is_delivery', 'patient_contact', 'pharmacy_id', 'prescription_id']
        if not all(field in data for field in required_fields):
            return jsonify({'error': '필수 항목 누락'}), 400

        patient_id = get_jwt_identity()
        is_delivery = data['is_delivery']

        # 환자 정보 업데이트: is_default_address 반영
        if 'is_default_address' in data:
            collection_patients.document(patient_id).update({
                'is_default_address': data['is_default_address']
            })

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
        delivery = {
            "delivery_id": current_id,
            "patient_id": patient_id,
            "is_delivery": is_delivery,
            "patient_contact": data["patient_contact"],
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