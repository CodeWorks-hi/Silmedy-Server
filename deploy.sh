#!/bin/bash

cd ~/Silmedy-Server || exit 1

# 🔐 .env 및 Firebase JSON 복원
echo "📦 Restoring secrets..."
echo "${ENV_B64}" | base64 -d > .env
echo "${FIREBASE_JSON_B64}" | base64 -d > silmedy-23a1b-firebase-adminsdk-fbsvc-1e8c6b596b.json

# 1. Python3 설치 여부 확인
if ! command -v python3 &> /dev/null; then
  echo "❌ Python3 not found. Please install it."
  exit 1
fi

# 2. 패키지 설치 (시스템 Python에 설치)
echo "⬆️  Upgrading pip and installing dependencies..."
pip3 install --upgrade pip setuptools
pip3 install -r requirements.txt

# 3. 기존 Flask 프로세스 종료
echo "🛑 Killing previous app.py process..."
pkill -f "python3 app.py" || echo "No existing app.py process."

# 4. 로그 파일 이름 설정
LOG_FILE="flask_$(date +%Y%m%d_%H%M%S).log"

# 5. Flask 서버 백그라운드 실행 (nohup + setsid)
echo "🚀 Starting Flask app with nohup..."
nohup python3 app.py > "$LOG_FILE" 2>&1 &
echo "✅ Deploy complete. Logs: $LOG_FILE"