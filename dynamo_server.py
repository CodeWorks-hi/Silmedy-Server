from flask import Flask, request, jsonify
import boto3

app = Flask(__name__)
dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-2')

table_patients = dynamodb.Table('patients')
table_doctors = dynamodb.Table('doctors')
table_admins = dynamodb.Table('admins')

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400

        response = table_patients.get_item(Key={'email': email})
        item = response.get('Item')

        if item and item.get('password') == password:
            return jsonify({
                'message': 'Login successful',
                'name': item.get('name', '')
            }), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)