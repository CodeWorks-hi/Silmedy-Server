# dynamo_server.py
from flask import Blueprint, request, jsonify
import boto3
import logging
from datetime import datetime


# ---- 기본 세팅 ----
dynamo_api = Blueprint('dynamo_api', __name__)
dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2')

table_patients = dynamodb.Table('patients')
table_doctors = dynamodb.Table('doctors')
table_admins = dynamodb.Table('admins')

# ---- API 1 ----
# @dynamo_api.route('/login', methods=['POST'])
