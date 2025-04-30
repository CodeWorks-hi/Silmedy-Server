#!/bin/bash

cd ~/Silmedy-Server

# pip3로 설치 (시스템 전역에)
pip3 install --upgrade pip
pip3 install -r requirements.txt

# 기존 프로세스 종료
pkill -f app.py

# Flask 실행
nohup python3 app.py > flask.log 2>&1 &