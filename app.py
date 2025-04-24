from flask import Flask, Blueprint, request, jsonify
import firebase_admin
import boto3
import logging
from datetime import datetime
from firebase_admin import credentials, firestore
import requests
import toml
from flask_cors import CORS
from dotenv import load_dotenv
import os
import random
import string
import re
from boto3.dynamodb.conditions import Attr

# ---- 기본 세팅 ----
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logger = logging.getLogger()
logger.setLevel(logging.INFO)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "ap-northeast-2")

dynamodb = boto3.resource(
    'dynamodb',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
cred = credentials.Certificate('silmedy-23a1b-firebase-adminsdk-fbsvc-1e8c6b596b.json')
firebase_admin.initialize_app(cred)
db = firestore.client()


# ---- 테이블 목록 ----
# Firestore
collection_patients = db.collection('patients')
collection_doctors = db.collection('doctors')
collection_admins = db.collection('admins')
collection_calls = db.collection('calls')
collection_counters = db.collection('fb_counters')

# DynamoDB
table_ai_consults = dynamodb.Table('ai_consults')
table_care_requests = dynamodb.Table('care_requests')
table_consult_text = dynamodb.Table('consult_text')
table_counters = dynamodb.Table('counters')
table_diagnosis_records = dynamodb.Table('diagnosis_records')
table_diagnosis_text = dynamodb.Table('diagnosis_text')
table_diseases = dynamodb.Table('diseases')
table_drug_deliveries = dynamodb.Table('drug_deliveries')
table_drugs = dynamodb.Table('drugs')
table_hospitals = dynamodb.Table('hospitals')
table_pharmacies = dynamodb.Table('pharmacies')
table_prescription_records = dynamodb.Table('prescription_records')


# ---- 환자 회원가입 ----
@app.route('/patient/signup', methods=['POST'])
def patient_signup():
    logger.info(f"REQUEST: {request.json}")
    try:
        body = request.get_json()

        item = {
            'email': body['email'],
            'password': body['password'],
            'name': body['name'],
            'contact': body['contact'],
            'postal_code': body['postal_code'],
            'address': body['address'],
            'address_detail': body['address_detail'],
            'created_at': datetime.utcnow().isoformat()
        }

        collection_patients.document(item['email']).set(item)
        logger.info(f"Inserted patient: {item}")

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

        doc_ref = collection_patients.document(email)
        doc = doc_ref.get()

        if doc.exists:
            item = doc.to_dict()
            if item.get('password') == password:
                return jsonify({
                    'message': 'Login successful',
                    'name': item.get('name', '')
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

        doc_ref = collection_patients.document(email)
        doc = doc_ref.get()

        if doc.exists:
            doc_ref.update({'password': new_password})
            return jsonify({'message': '비밀번호 변경 완료'}), 200
        else:
            return jsonify({'error': '사용자 없음'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 로그아웃----
@app.route('/patient/logout', methods=['POST'])
def logout():
    return jsonify({'message': '로그아웃 처리 완료'}), 200


# ---- 환자 마이페이지 조회 ----
@app.route('/patient/mypage', methods=['GET'])
def get_mypage():
    email = request.args.get('email')
    if not email:
        return jsonify({'error': 'Email required'}), 400

    doc = collection_patients.document(email).get()
    if doc.exists:
        return jsonify(doc.to_dict()), 200
    else:
        return jsonify({'error': 'User not found'}), 404
    

# ---- 회원 정보 수정  ----
@app.route('/patient/update', methods=['POST'])
def update_patient_info():
    try:
        data = request.get_json()
        email = data.get('email')
        updates = data.get('updates')  # 수정할 필드들 (딕셔너리)

        if not email or not updates:
            return jsonify({'error': 'Email and update data required'}), 400

        doc_ref = collection_patients.document(email)
        doc = doc_ref.get()

        if doc.exists:
            doc_ref.update(updates)
            return jsonify({'message': '회원 정보 수정 완료'}), 200
        else:
            return jsonify({'error': 'User not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 회원 탈퇴   ----
@app.route('/patient/delete', methods=['DELETE'])
def delete_patient():
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({'error': 'Email required'}), 400

    doc_ref = collection_patients.document(email)
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
        # Firestore에서 이메일과 전화번호 둘 다 일치하는 유저 찾기
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
    
    
@app.route('/request/disease-image', methods=['POST'])
def request_disease_image():
    try:
        data = request.get_json()
        analysis = data.get('analysis')

        if not analysis:
            return jsonify({'error': 'Analysis is required'}), 400

        response = table_diseases.scan(
            FilterExpression=Attr('name_ko').eq(analysis)
        )
        items = response.get('Items', [])
        item = items[0] if items else None

        if item and 'desc_url' in item:
            return jsonify({'desc_url': item['desc_url']}), 200
        else:
            return jsonify({'error': 'No matching disease found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# ---- 채팅 저장 ----
@app.route('/chat/save', methods=['POST'])
def save_chat():
    try:
        data = request.get_json()
        consult_id = data.get('consult_id')
        original_text = data.get('original_text')

        # 입력 유효성 검증
        if not consult_id or not isinstance(original_text, list):
            return jsonify({"message": "Invalid input"}), 400

        # DynamoDB 저장
        table_consult_text.put_item(
            Item={
                'consult_id': int(consult_id),
                'original_text': original_text
            }
        )

        logger.info(f"[저장됨] consult_id={consult_id}, text={original_text}")
        return jsonify({"message": "Chat saved", "consult_id": consult_id}), 200

    except Exception as e:
        logger.error(f"Error saving chat: {e}")
        return jsonify({'error': str(e)}), 500
    

    
# ---- 진료 예약 페이지 안내 ----
@app.route('/chat/reserve', methods=['POST'])
def go_to_reservation_page():
    return jsonify({"message": "진료 예약 화면으로 이동하세요."}), 200


# ---- 의사 목록 반환 ----
@app.route('/chat/doctors', methods=['GET'])
def get_doctor_list():
    try:
        doctors = [doc.to_dict() for doc in collection_doctors.stream()]
        return jsonify(doctors), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- 의사 진료 가능 시간 확인 ----
@app.route('/chat/availability', methods=['GET'])
def get_doctor_availability():
    return jsonify({
        "availability": {
            "start": "09:00",
            "end": "18:00"
        }
    }), 200


# ---- 수어 필요 여부 자동 확인 ----
@app.route('/chat/signcheck', methods=['GET'])
def check_sign_language_required():
    # 향후 로직으로 사용자의 프로필 기반 분석 가능
    return jsonify({"sign_language_needed": True}), 200


# ---- 진료 예약 확정 처리 ----
@app.route('/chat/confirmed', methods=['POST'])
def confirm_reservation():
    data = request.get_json()
    required_fields = ["name", "time", "doctor"]

    # 필수 필드 확인
    if not all(field in data for field in required_fields):
        return jsonify({
            "error": "Missing required reservation information."
        }), 400

    return jsonify({
        "message": "진료 예약이 확정되었습니다.",
        "reservation": data
    }), 200
   

if __name__ == '__main__':
    app.run(debug=True)