# THIRD_PARTY_NOTICES

VideoTriggerDetection은 GNU General Public License v3.0(GPL-3.0)으로 배포됩니다.
프로젝트 자체의 전체 라이선스 전문은 배포물에 함께 포함되는 `LICENSE` 파일을
확인하세요.

이 문서는 `requirements.txt`에 명시된 직접 의존성에 대한 제3자 오픈소스 고지
요약입니다. 각 라이브러리의 저작권과 라이선스 조건은 해당 프로젝트/패키지의
라이선스 파일과 공식 배포 페이지를 따릅니다. 이 문서는 법률 자문이 아니며,
실제 배포 전에는 빌드에 포함된 wheel, DLL, shared library, Qt 플러그인 등의
라이선스 파일을 다시 확인해야 합니다.

## 직접 의존성

| 패키지 | 버전 | 라이선스/고지 요약 | 공식/배포 페이지 |
| --- | ---: | --- | --- |
| OpenCV Python (`opencv-python`) | 4.9.0.80 | `opencv-python` 패키징 코드는 MIT 라이선스입니다. OpenCV 본체는 Apache License 2.0이며, wheel에는 FFmpeg 등 별도 라이선스를 가진 구성요소가 포함될 수 있습니다. | <https://pypi.org/project/opencv-python/> |
| NumPy (`numpy`) | 1.26.4 | BSD 계열 라이선스로 배포됩니다. | <https://numpy.org/> |
| PyQt5 (`PyQt5`) | 5.15.10 | GPL v3 또는 Riverbank 상용 라이선스 중 하나로 사용할 수 있습니다. 이 프로젝트는 GPL-3.0으로 배포하여 GPL 버전의 PyQt5와 호환되도록 합니다. | <https://pypi.org/project/PyQt5/> |
| Qt (`PyQt5-Qt5`) | 5.15.2 | PyQt5 wheel에 포함되는 Qt 라이브러리는 Qt의 별도 라이선스 조건을 따릅니다. 배포 대상 플랫폼에 포함된 Qt 라이선스/고지 파일을 확인해야 합니다. | <https://pypi.org/project/PyQt5-Qt5/> |
| PyQt SIP (`PyQt5-sip`) | 12.13.0 | SIP/PyQt 지원 패키지입니다. 배포 시 설치된 패키지의 라이선스 파일을 확인하세요. | <https://pypi.org/project/PyQt5-sip/> |
| PyYAML (`pyyaml`) | 6.0.1 | MIT 라이선스로 배포됩니다. | <https://pypi.org/project/PyYAML/> |
| PyInstaller (`pyinstaller`) | 6.3.0 | GPL 라이선스와 PyInstaller bootloader 예외가 적용됩니다. 생성된 실행 파일 배포 시 애플리케이션 및 포함 의존성의 라이선스 조건을 준수해야 합니다. | <https://pyinstaller.org/en/stable/license.html> |
| Pillow (`Pillow`) | 10.2.0 | PIL Software License 계열의 permissive 라이선스로 배포됩니다. | <https://pypi.org/project/Pillow/> |
| pyftpdlib (`pyftpdlib`) | 1.5.9 | MIT 라이선스로 배포됩니다. | <https://pypi.org/project/pyftpdlib/> |

## 배포 시 포함해야 할 파일

PyInstaller 등으로 실행 파일을 배포할 때는 최소한 다음 파일을 배포 폴더에 함께
포함하세요.

1. `LICENSE` — VideoTriggerDetection의 GPL-3.0 전체 전문
2. `THIRD_PARTY_NOTICES.md` — 이 제3자 오픈소스 고지 문서
3. 각 wheel/패키지에 포함된 라이선스 및 고지 파일
   - 특히 PyQt5/Qt 관련 라이선스 파일
   - OpenCV wheel 및 OpenCV가 포함하는 제3자 구성요소 고지
   - PyInstaller bootloader 라이선스/예외 고지

## 추가 주의사항

- GPL-3.0 배포물은 해당 버전의 소스 코드 제공 의무를 함께 고려해야 합니다.
- `config.yaml`에 실제 RTSP 주소, FTP 계정, 비밀번호 등 운영 환경의 민감정보가
  들어간 상태로 배포하지 마세요.
- 의존성 버전을 변경하거나 새 패키지를 추가하면 이 문서도 함께 갱신하세요.
