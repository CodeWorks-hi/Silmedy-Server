#!/bin/bash

cd ~/Silmedy-Server

# 의존성 설치 (시스템 전역 Python 사용)
pip install --upgrade pip
pip install -r requirements.txt

# 기존 Flask 백그라운드 프로세스 종료 (선택 사항)
pkill -f app.py

# Flask 앱 백그라운드 실행 및 로그 저장
nohup python app.py > flask.log 2>&1 &