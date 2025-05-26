# 🏥 Silmedy 환자용 백엔드 API

### Firebase와 AWS 기반의 환자 진료 및 처방, AI 챗봇 진단 기능을 제공하는 Flask 기반 백엔드 API

---

## 📌 프로젝트 개요

Silmedy는 환자 중심의 비대면 진료 환경을 제공하기 위한 통합 헬스케어 플랫폼입니다. 본 백엔드 API는 환자용 앱의 서버로서 사용자 인증, 진료 예약, AI 챗 기반 진단, 처방 및 약 배송 등 다양한 기능을 제공합니다.

---

## 🛠️ 주요 기술 스택

| 분야                 | 라이브러리 및 플랫폼                |
|----------------------|------------------------------------|
| 웹 프레임워크         | Flask, Flasgger                    |
| 인증 및 보안         | JWT (flask_jwt_extended)          |
| DB 및 실시간 데이터  | Firebase Firestore, Realtime DB   |
| 클라우드 데이터베이스 | AWS DynamoDB                      |
| 음성 인식 및 AI      | Hugging Face API, TFLite          |
| 주소 검색            | Kakao 주소 API                    |

---

## 🔑 주요 기능

### 👤 회원 기능
- 회원가입, 로그인, 로그아웃, 마이페이지, 비밀번호 재설정
- 전화번호 인증 및 이메일 확인
- FCM 푸시 알림 토큰 등록

### 🩺 진료 기능
- 진료 예약 생성 및 확인
- 진료 기록 조회, 처방전 PDF 조회
- 약 배송 요청 및 수령 확인

### 🤖 AI 챗봇 기반 진단
- 환자 챗 기록 저장
- 증상 분석 후 내과/외과 분류 및 구조화 응답
- 요약 및 데이터 저장 (DynamoDB + Firestore)

### 🔍 기타 기능
- 우편번호 검색 (카카오 API 연동)
- 의사/병원 검색, 약국 검색

---

## 📂 프로젝트 구조

```
📁 Silmedy-Server/
├── app.py                    # Flask 메인 앱
├── api-doc.yaml              # Swagger 기반 API 명세 (v2)
├── requirements.txt          # 종속 라이브러리 목록
├── .env                      # 환경 변수 (JWT 키, API 키 등)
├── silmedy-23a1b-*.json      # Firebase 인증키
├── model_unquant.tflite      # TFLite 모델
├── static/
│   └── js/                   # 정적 파일(js 등)
```

---

## 🌐 실행 방법

1. `.env` 파일 생성 후 환경변수 설정
2. Firebase 인증 JSON 키 파일 포함
3. 필요한 라이브러리 설치:

```bash
pip install -r requirements.txt
```

4. Flask 실행:

```bash
python app.py
```

5. Swagger 문서 확인:  
   http://43.201.73.161:5000/apidocs

---

## 🔐 환경 변수 (.env)

```
JWT_SECRET_KEY=your_jwt_secret
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
HUGGINGFACE_API_KEY=your_hf_key
POSTAL_CODE_KEY=your_kakao_key
```
