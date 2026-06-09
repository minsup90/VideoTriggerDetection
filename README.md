# Video-based Triggerless Object Detection Program

연속된 비디오 영상을 기반으로 AI 또는 Template Matching을 통해 특정 시점 Event Trigger를 발생시키는 프로그램입니다.

## 기능 요약

### 핵심 기능
1. **RTSP 비디오 스트림 수신** - CCTV 카메라에서 실시간 영상 수신
2. **Template Matching** - 다중 패턴 템플릿 매칭 지원
3. **AI Classification** - CPU 기반 AI 분류 (추후 확장용)
4. **이미지 수집 및 템플릿 등록** - GUI 기반 직관적인 템플릿 등록
5. **ROI 설정** - 마우스 드래그 앤 드롭으로 ROI 영역 설정
6. **선명도 분석** - Tenengrade 알고리즘을 사용한 가장 선명한 이미지 선택
7. **FTP 전송** - 감지된 이미지를 FTP 서버로 자동 전송
8. **로그 관리** - 날짜별 로그 파일 자동 관리
9. **시스템 트레이** - 백그라운드 실행 지원

## 설치 방법

### 필수 요구사항
- Python 3.8 이상
- Windows 10/11

### 의존성 설치
```bash
pip install -r requirements.txt
```

### 실행 방법
```bash
python gui_main.py
```

## 사용 방법

### 1. 설정
- 프로그램 실행 후 "설정" 탭에서 RTSP URL, 패턴 설정, 버퍼 설정 등을 구성합니다.
- "설정 저장" 버튼을 클릭하여 설정을 저장합니다.

### 2. 템플릿 등록
1. "이미지 수집" 버튼을 클릭하여 설정된 시간 동안 프레임을 수집합니다.
2. "템플릿 등록" 버튼을 클릭하여 수집된 이미지를 확인합니다.
3. 원하는 이미지를 클릭하여 선택합니다.
4. "ROI 설정" 버튼을 누르고 마우스로 드래그하여 검사 영역을 설정합니다.
5. "Template 설정" 버튼을 누르고 마우스로 드래그하여 템플릿 영역을 설정합니다.
6. 패턴 인덱스를 변경하여 여러 템플릿을 등록할 수 있습니다.

### 3. 검출 시작
- "START" 버튼을 클릭하여 검출을 시작합니다.
- 템플릿 매칭이 임계값을 넘으면 버퍼에 이미지가 저장됩니다.
- 버퍼가 가득 차면 가장 선명한 이미지가 저장되고 FTP로 전송됩니다.

### 4. 로그 확인
- "로그" 탭에서 프로그램 동작 로그를 확인할 수 있습니다.

## 설정 파일 (config.yaml)

```yaml
# RTSP 카메라 설정
rtsp:
  url: "rtsp://username:password@192.168.1.100:554/stream1"
  reconnect_interval: 5

# Template Matching 설정
template_matching:
  patterns:
    - index: 1
      template_path: "templates/template_1.png"
      score_threshold: 0.85
      roi:
        x: 100
        y: 100
        width: 300
        height: 300
  require_all_patterns: true

# 버퍼 설정
buffer:
  size: 10

# FTP 서버 설정
ftp:
  enabled: true
  host: "192.168.1.200"
  port: 21
  username: "ftpuser"
  password: "ftppassword"
  remote_dir: "/uploads"

# 파일 저장 설정
file_storage:
  save_dir: "saved_images"
  retention_days: 30
```

## Windows EXE 빌드

### 일반 빌드 (콘솔 없음)
```bash
python build_exe.py
```

### 디버그 빌드 (콘솔 포함)
```bash
python build_exe.py --debug
```

빌드가 완료되면 `dist` 폴더에 `VideoTriggerDetection.exe` 파일이 생성됩니다.

## 프로젝트 구조

```
video_trigger_detection/
├── config.yaml              # 설정 파일
├── requirements.txt          # Python 의존성
├── config_manager.py         # 설정 관리 모듈
├── logger.py                # 로그 관리 모듈
├── rtsp_stream.py           # RTSP 스트림 모듈
├── template_matching.py     # Template Matching 엔진
├── ftp_manager.py           # FTP 전송 모듈
├── ai_classifier.py         # AI 분류 모듈 (추후 확장용)
├── gui_main.py              # 메인 GUI
├── build_exe.py             # EXE 빌드 스크립트
├── templates/               # 템플릿 이미지 저장 폴더
├── saved_images/            # 저장된 이미지 폴더
├── logs/                    # 로그 파일 폴더
└── gathered_images/         # 수집된 이미지 폴더
```

## 기술 스택

- **GUI**: PyQt5
- **비디오 처리**: OpenCV
- **이미지 처리**: NumPy
- **설정 관리**: PyYAML
- **FTP**: ftplib
- **패키징**: PyInstaller

## 주의사항

1. RTSP URL은 카메라 설정에 맞게 수정해야 합니다.
2. FTP 서버 설정은 선택사항입니다.
3. 첫 실행 시 필요한 디렉토리가 자동 생성됩니다.
4. GPU가 없는 환경에서도 동작합니다 (CPU 기반).

## 라이선스

이 프로젝트는 내부용으로 개발되었습니다.

## 개발 정보

- 작성일: 2026-03-06
- 개발자: Roo Development Team
- 버전: 1.0.0
