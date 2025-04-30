#!/bin/bash

cd ~/Silmedy-Server || exit 1

# 1. Python3 설치 여부 확인
if ! command -v python3 &> /dev/null; then
  echo "❌ Python3 not found. Please install it."
  exit 1
fi

# 2. 가상환경 생성 및 활성화
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

# 3. pip 업그레이드 및 패키지 설치
echo "⬆️  Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. 기존 Flask 프로세스 종료
echo "🛑 Killing previous app.py process..."
pkill -f "venv/bin/python app.py" || echo "No existing app.py process."

# 5. 로그 파일 이름 설정
LOG_FILE="flask_$(date +%Y%m%d_%H%M%S).log"

# 6. Flask 서버 백그라운드 실행
echo "🚀 Starting Flask app..."
nohup venv/bin/python app.py > "$LOG_FILE" 2>&1 &

echo "✅ Deploy complete. Logs: $LOG_FILE"