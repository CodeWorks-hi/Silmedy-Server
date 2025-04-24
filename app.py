# app.py
from flask import Flask
from firestore_server import firestore_api_patient, firestore_api_doctor, firestore_api_admin
from dynamo_server import dynamo_api

app = Flask(__name__)

# 각 Blueprint 등록
app.register_blueprint(firestore_api_patient)
app.register_blueprint(firestore_api_doctor)
app.register_blueprint(firestore_api_admin)
app.register_blueprint(dynamo_api)

if __name__ == '__main__':
    app.run(debug=True)