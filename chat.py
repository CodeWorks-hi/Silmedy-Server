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

# ---- 기본 세팅 ----
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logger = logging.getLogger()
logger.setLevel(logging.INFO)

load_dotenv()

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


# ---- 진료 신청----


# ---- 채팅 ----

# 채팅 저장 API
@app.route('/save', methods=['POST'])
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




