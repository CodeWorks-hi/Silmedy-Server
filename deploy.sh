#!/bin/bash

cd ~/Silmedy-Server

# 가상환경 활성화 (필요 시)
source ~/venv/bin/activate

# 필요한 패키지 설치
pip install -r requirements.txt

# 서버 실행 (백그라운드로, 로그 저장)
nohup python app.py > flask.log 2>&1 &