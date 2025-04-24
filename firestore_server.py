import firebase_admin
from firebase_admin import credentials, firestore
from flask import Blueprint, Flask, request, jsonify
import logging
from datetime import datetime

# ---- 기본 세팅 ----
firestore_api_patient = Blueprint('firestore_api_patient', __name__, url_prefix='/patient')
firestore_api_doctor = Blueprint('firestore_api_doctor', __name__, url_prefix='/doctor')
firestore_api_admin = Blueprint('firestore_api_admin', __name__, url_prefix='/admin')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

cred = credentials.Certificate('silmedy-23a1b-firebase-adminsdk-fbsvc-1e8c6b596b.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

collection_patients = db.collection('patients')
collection_doctors = db.collection('doctors')
collection_admins = db.collection('admins')


# ---- 환자 회원가입 ----
@firestore_api_patient.route('/signup', methods=['POST'])
def patient_signup():
    logger.info(f"REQUEST: {request.json}")
    try:
        body = request.get_json()

        item = {
            'email': body['email'],
            'password': body['password'],
            'name': body['name'],
            'phone': body['phone'],
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
@firestore_api_patient.route('/login', methods=['POST'])
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


# ---- 의사 회원가입 ----
@firestore_api_doctor.route('/signup', methods=['POST'])
def doctor_signup():
    return jsonify({'message': 'Doctor signup placeholder'}), 200

# ---- 관리자 회원가입 ----
@firestore_api_admin.route('/signup', methods=['POST'])
def admin_signup():
    return jsonify({'message': 'Admin signup placeholder'}), 200

# Register blueprints (to be used in app.py)
__all__ = ['firestore_api_patient', 'firestore_api_doctor', 'firestore_api_admin']