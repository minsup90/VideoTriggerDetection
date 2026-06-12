# Third-Party Notices

This project is distributed under the GNU General Public License v3.0. This
notice summarizes the direct third-party Python dependencies listed in
`requirements.txt`. Each dependency remains governed by its own license terms.

> This file is an attribution and release checklist aid, not legal advice.
> Before distributing a binary build, verify the exact license files bundled in
> the wheels/artifacts used for that build.

## Direct dependencies

| Dependency | Version pinned in `requirements.txt` | License / notice summary | Upstream |
| --- | ---: | --- | --- |
| OpenCV Python (`opencv-python`) | 4.9.0.80 | `opencv-python` packaging scripts are MIT; OpenCV is Apache-2.0. Wheels may include additional third-party components such as FFmpeg, governed by their own notices. | <https://pypi.org/project/opencv-python/> |
| NumPy (`numpy`) | 1.26.4 | BSD license. | <https://numpy.org/> |
| PyQt5 (`PyQt5`) | 5.15.10 | GPL v3 or Riverbank commercial license. This project is released as GPL-3.0 to use the GPL version. | <https://pypi.org/project/PyQt5/> |
| Qt wheel (`PyQt5-Qt5`) | 5.15.2 | Qt libraries bundled with PyQt wheels are licensed separately by Qt; verify the wheel license files for the platform being distributed. | <https://pypi.org/project/PyQt5-Qt5/> |
| PyQt SIP support (`PyQt5-sip`) | 12.13.0 | BSD-2-Clause / SIP-related license terms; verify the installed artifact. | <https://pypi.org/project/PyQt5-sip/> |
| PyYAML (`pyyaml`) | 6.0.1 | MIT license. | <https://pypi.org/project/PyYAML/> |
| PyInstaller (`pyinstaller`) | 6.3.0 | GPL with a bootloader/bundling exception; generated application bundles may be shipped under the application license if dependency licenses are followed. | <https://pyinstaller.org/en/stable/license.html> |
| Pillow (`Pillow`) | 10.2.0 | Historical Permission Notice and Disclaimer (HPND)-style permissive license; verify the installed artifact. | <https://pypi.org/project/Pillow/> |
| pyftpdlib (`pyftpdlib`) | 1.5.9 | MIT license; verify the installed artifact. | <https://pypi.org/project/pyftpdlib/> |

## Binary distribution checklist

When distributing `dist/VideoTriggerDetection` or any other packaged binary:

1. Include this project `LICENSE` file.
2. Include this `THIRD_PARTY_NOTICES.md` file.
3. Keep the corresponding source code available under GPL-3.0.
4. Include license files/notices produced by the installed dependency wheels,
   especially PyQt5/Qt and OpenCV's third-party notices.
5. Do not publish production secrets in `config.yaml` such as RTSP credentials
   or FTP usernames/passwords.
