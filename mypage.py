from flask import Flask, Blueprint, request, jsonify
import firebase_admin
import boto3
import logging
from datetime import datetime
from firebase_admin import credentials, firestore
import requests
import toml
import os

# ---- Flask 기본 세팅 ----
app = Flask(__name__)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---- DynamoDB 초기화 ----
dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2')

# ---- Firebase 초기화 ----
FIREBASE_KEY_PATH = os.path.join(os.getcwd(), 'silmedy-23a1b-firebase-adminsdk-fbsvc-1e8c6b596b.json')

if not firebase_admin._apps:  # 중복 초기화 방지
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ---- TOML 설정 로드 ----
TOML_PATH = os.path.join(os.getcwd(), 'api_keys.toml')

try:
    config = toml.load(TOML_PATH)
except Exception as e:
    logger.error(f"⚠️ TOML 설정 파일 로드 실패: {e}")
    config = {}


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


# ---- 환자 마이페이지 ----


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