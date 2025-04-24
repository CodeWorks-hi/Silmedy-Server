from flask import Flask, Blueprint, request, jsonify
import firebase_admin
import boto3
import logging
from datetime import datetime
from firebase_admin import credentials, firestore

# ---- 기본 세팅 ----
app = Flask(__name__)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2')
cred = credentials.Certificate('silmedy-23a1b-firebase-adminsdk-fbsvc-1e8c6b596b.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# ---- 테이블 목록 ----
# Firestore
collection_patients = db.collection('patients')
collection_doctors = db.collection('doctors')
collection_admins = db.collection('admins')
collection_calls = db.collection('calls')

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



if __name__ == '__main__':
    app.run(debug=True)