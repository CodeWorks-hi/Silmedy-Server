from flask import Flask, Blueprint, request, jsonify
import firebase_admin
import boto3
import logging
from datetime import datetime, timedelta
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
        if not consult_id or not isinstance(original_text, str):
            return jsonify({"message": "Invalid input"}), 400

        # 기존 아이템 조회
        existing_item = table_consult_text.get_item(Key={'consult_id': int(consult_id)}).get('Item')
        if existing_item:
            original_list = existing_item.get('original_text', [])
            if not isinstance(original_list, list):
                original_list = []
            original_list.append(original_text.strip())
        else:
            original_list = [original_text.strip()]

        # DynamoDB 저장
        table_consult_text.put_item(
            Item={
                'consult_id': int(consult_id),
                'original_text': original_list
            }
        )

        logger.info(f"[저장됨] consult_id={consult_id}, text={original_list}")
        return jsonify({"message": "Chat saved", "consult_id": consult_id}), 200

    except Exception as e:
        logger.error(f"Error saving chat: {e}")
        return jsonify({'error': str(e)}), 500


# ---- 의사 목록 반환 ----
@app.route('/request/doctors', methods=['GET'])
def get_doctor_list():
    try:
        hospital_id = request.args.get('hospital_id')
        if not hospital_id:
            return jsonify({'error': 'hospital_id is required'}), 400

        doctors = []
        for doc in collection_doctors.where("hospital_id", "==", int(hospital_id)).stream():
            data = doc.to_dict()
            doctors.append({
                "hospital_id": data.get("hospital_id"),
                "profile_url": data.get("profile_url"),
                "name": data.get("name"),
                "department": data.get("department"),
                "gender": data.get("gender"),
                "contact": data.get("contact"),
                "email": data.get("email"),
                "bio": data.get("bio"),
                "availability": data.get("availability"),
            })
        return jsonify(doctors), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- 의사 진료 가능 시간 확인 ----
@app.route('/request/availability', methods=['GET'])
def get_doctor_availability():
    try:
        license_number = request.args.get('license_number')
        if not license_number:
            return jsonify({'error': 'license_number is required'}), 400
        
        # 의사 문서 조회 (license_number가 문서 ID로 사용됨)
        doc_ref = collection_doctors.document(license_number)
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({'error': 'Doctor not found'}), 404

        doctor_id = doc.id

        # 오늘과 내일 날짜 구하기
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

        return jsonify({'reservations': reservations}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 수어 필요 여부 자동 확인 ----
@app.route('/chat/signcheck', methods=['GET'])
def check_sign_language_required():
    email = request.args.get('email')
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    doc = collection_patients.document(email).get()
    if not doc.exists:
        return jsonify({'error': 'User not found'}), 404

    data = doc.to_dict()
    sign_language_needed = data.get('sign_language_needed', False)
    return jsonify({'sign_language_needed': sign_language_needed}), 200


# ---- 진료 예약 확정 처리 ----
@app.route('/chat/confirmed', methods=['POST'])
def confirm_reservation():
    try:
        data = request.get_json()
        required_fields = [
            "patient_id", "doctor_id", "department",
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
            "patient_id": data["patient_id"],
            "doctor_id": data["doctor_id"],
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
            "patient_id": data["patient_id"],
            "doctor_id": data["doctor_id"],
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
def get_diagnosis_by_patient():
    try:
        from urllib.parse import unquote
        from datetime import datetime
        raw_email = request.args.get('email', '')
        email = unquote(raw_email).strip().replace("'", "")
        if not email:
            return jsonify({'error': 'Email is required'}), 400

        response = table_diagnosis_records.scan()
        items = []
        for item in response.get('Items', []):
            if str(item.get('patient_id', '')).strip() == email:
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
def get_default_address():
    try:
        data = request.get_json()
        email = data.get('email') if data else None
        if not email:
            return jsonify({'error': 'Email is required'}), 400

        doc = collection_patients.document(email).get()
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
def register_delivery():
    try:
        data = request.get_json()
        required_fields = ['patient_id', 'is_delivery', 'patient_contact', 'pharmacy_id', 'prescription_id']
        if not all(field in data for field in required_fields):
            return jsonify({'error': '필수 항목 누락'}), 400

        is_delivery = data['is_delivery']

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
            "patient_id": data["patient_id"],
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


@app.route('/delivery/complete', methods=['POST'])
def mark_delivery_as_received():
    try:
        data = request.get_json()
        delivery_id = data.get('delivery_id')
        patient_id = data.get('patient_id')

        if not delivery_id or not patient_id:
            return jsonify({'error': 'delivery_id and patient_id are required'}), 400

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


# ---- 대기중인 의사 조회 ----
@app.route('/call/waiting-doctor', methods=['POST'])
def get_waiting_doctor_for_patient():
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')

        if not patient_id:
            return jsonify({'error': 'patient_id is required'}), 400

        query = collection_calls \
            .where("patient_id", "==", patient_id) \
            .where("is_accepted", "==", False) \
            .limit(1) \
            .stream()

        call_doc = next(query, None)
        if not call_doc:
            return jsonify({'message': 'No waiting doctor found'}), 404

        doctor_id = call_doc.to_dict().get("doctor_id")
        return jsonify({'doctor_id': doctor_id}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/call/accept', methods=['POST'])
def accept_call():
    try:
        data = request.get_json()
        call_id = data.get('call_id')

        if not call_id:
            return jsonify({'error': 'call_id is required'}), 400

        doc_ref = collection_calls.document(str(call_id))
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



# ---- Call에서 doctor_id 반환 ----
@app.route('/call/doctor-id', methods=['POST'])
def get_doctor_id_from_call():
    try:
        data = request.get_json()
        call_id = data.get('call_id')

        if not call_id:
            return jsonify({'error': 'call_id is required'}), 400

        doc_ref = collection_calls.document(str(call_id))
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({'error': 'Call not found'}), 404

        doctor_id = doc.to_dict().get('doctor_id')
        return jsonify({'doctor_id': doctor_id}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# ---- Add patient_text to a call ----
@app.route('/call/add-patient-text', methods=['POST'])
def add_patient_text():
    try:
        data = request.get_json()
        call_id = data.get('call_id')
        new_text = data.get('text')

        if not call_id or not new_text:
            return jsonify({'error': 'call_id and text are required'}), 400

        doc_ref = collection_calls.document(str(call_id))
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({'error': 'Call not found'}), 404

        call_data = doc.to_dict()
        patient_text_list = call_data.get('patient_text', [])

        if not isinstance(patient_text_list, list):
            patient_text_list = []

        patient_text_list.append(new_text.strip())
        doc_ref.update({'patient_text': patient_text_list})

        return jsonify({'message': 'Text added to patient_text'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# ---- Get latest doctor_text from a call ----
@app.route('/call/latest-doctor-text', methods=['POST'])
def get_latest_doctor_text():
    try:
        data = request.get_json()
        call_id = data.get('call_id')

        if not call_id:
            return jsonify({'error': 'call_id is required'}), 400

        doc_ref = collection_calls.document(str(call_id))
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({'error': 'Call not found'}), 404

        doctor_text_list = doc.to_dict().get('doctor_text', [])
        if not isinstance(doctor_text_list, list) or not doctor_text_list:
            return jsonify({'message': 'No doctor text found'}), 404

        return jsonify({'latest_doctor_text': doctor_text_list[-1]}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- End a call ----
@app.route('/call/end', methods=['POST'])
def end_call():
    try:
        data = request.get_json()
        call_id = data.get('call_id')

        if not call_id:
            return jsonify({'error': 'call_id is required'}), 400

        doc_ref = collection_calls.document(str(call_id))
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({'error': 'Call not found'}), 404

        doc_ref.update({'ended_at': datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")})
        return jsonify({'message': 'Call ended successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)