# 🐱 Face Landmark Sticker

얼굴 랜드마크를 탐지하고 원하는 위치에 스티커를 합성하는 Jupyter 노트북 프로젝트입니다.

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [기능](#기능)
- [설치 및 환경 설정](#설치-및-환경-설정)
- [사용 방법](#사용-방법)
- [코드 구조](#코드-구조)
- [주요 파라미터 조정](#주요-파라미터-조정)
- [문제 해결](#문제-해결)

## 프로젝트 개요

이 프로젝트는 dlib 라이브러리를 사용하여 얼굴의 68개 랜드마크를 탐지하고, OpenCV를 이용해 특정 위치(예: 코, 이마)에 스티커 이미지를 합성합니다.

### 주요 라이브러리

- **dlib**: 얼굴 탐지 및 랜드마크 예측
- **OpenCV (cv2)**: 이미지 처리 및 합성
- **matplotlib**: 이미지 시각화
- **numpy**: 배열 연산

## 기능

1. ✅ 얼굴 자동 탐지
2. ✅ 68개 랜드마크 포인트 추출
3. ✅ 랜드마크 시각화
4. ✅ 투명 배경 스티커 합성
5. ✅ 얼굴 크기에 맞춘 자동 스티커 크기 조정
6. ✅ 특정 랜드마크 위치에 스티커 배치 (코, 이마 등)

## 설치 및 환경 설정

### 1. 필수 라이브러리 설치

```bash
pip install opencv-python dlib matplotlib numpy
```

### 2. 프로젝트 구조

```
~/work/camera_sticker/
├── models/
│   └── shape_predictor_68_face_landmarks.dat
├── images/
│   ├── selfie.png        # 얼굴 이미지
│   └── cat.png           # 스티커 이미지
└── notebook.ipynb        # Jupyter 노트북 파일
```

### 3. 모델 다운로드

코드의 2번 셀을 실행하면 자동으로 dlib의 얼굴 랜드마크 모델을 다운로드합니다.

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

## 사용 방법

### Step 1: 이미지 준비

`~/work/camera_sticker/images/` 폴더에 다음 이미지를 업로드합니다:

- `selfie.png`: 얼굴이 포함된 이미지
- `cat.png`: 합성할 스티커 이미지 (투명 배경 권장)

Jupyter 인터페이스에서 파일 업로드:
1. 좌측 파일 브라우저 열기
2. `images` 폴더로 이동
3. Upload 버튼 클릭 또는 드래그&드롭

### Step 2: 코드 실행

Jupyter 노트북에서 셀을 순서대로 실행합니다:

1. **1번 셀**: 라이브러리 임포트 및 matplotlib 설정
2. **2번 셀**: 랜드마크 모델 다운로드
3. **3번 셀**: 이미지 경로 설정 및 파일 존재 확인
4. **4번 셀**: 이미지 로드
5. **5번 셀**: 얼굴 탐지 및 랜드마크 예측
6. **6번 셀**: 랜드마크 시각화
7. **7번 셀**: 스티커 합성 및 결과 저장

### Step 3: 결과 확인

- 6번 셀: 랜드마크가 노란색 점으로 표시된 이미지
- 7번 셀: 스티커가 합성된 최종 이미지
- 결과는 `result_with_cat.png`로 자동 저장됩니다

## 코드 구조

### 1️⃣ 라이브러리 임포트 및 설정

```python
%matplotlib inline
import matplotlib.pyplot as plt
import cv2
import dlib
import os
import numpy as np
```

- `%matplotlib inline`: Jupyter에서 그래프를 인라인으로 표시
- OpenCV, dlib, numpy를 사용한 이미지 처리

### 2️⃣ 모델 다운로드

```bash
!wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
!mkdir -p ~/work/camera_sticker/models
!mv shape_predictor_68_face_landmarks.dat.bz2 ~/work/camera_sticker/models/
!bzip2 -d ~/work/camera_sticker/models/shape_predictor_68_face_landmarks.dat.bz2
```

- `!` 접두사: Jupyter에서 Shell 명령어 실행
- `wget`: 파일 다운로드
- `bzip2 -d`: 압축 해제

### 3️⃣ 이미지 경로 설정

```python
home_dir = os.path.expanduser('~')
img_path = os.path.join(home_dir, 'work/camera_sticker/images/selfie.png')
sticker_path = os.path.join(home_dir, 'work/camera_sticker/images/cat.png')
```

- `os.path.expanduser('~')`: 홈 디렉토리 경로 반환
- `os.path.join()`: 운영체제에 맞는 경로 결합

### 4️⃣ 이미지 로드

```python
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
sticker_img = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
```

- OpenCV는 BGR 형식으로 이미지를 읽음
- `cv2.cvtColor()`: BGR → RGB 변환 (matplotlib 표시용)
- `cv2.IMREAD_UNCHANGED`: 알파 채널 포함하여 읽기

### 5️⃣ 얼굴 탐지 및 랜드마크 예측

```python
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(model_path)

dlib_rects = detector(img_rgb, 1)

for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)
```

- `detector()`: 이미지에서 얼굴 영역 탐지
- `landmark_predictor()`: 얼굴 영역에서 68개 랜드마크 예측
- 각 랜드마크는 (x, y) 좌표로 저장

### 6️⃣ 랜드마크 시각화

```python
for landmark in list_landmarks:
    for point in landmark:
        cv2.circle(img_show, point, 2, (0, 255, 255), -1)
```

- `cv2.circle()`: 각 랜드마크 위치에 노란색 원 그리기
- `-1`: 원을 채움 (양수면 테두리 두께)

### 7️⃣ 스티커 합성

```python
def overlay_transparent(background, overlay, x, y):
    # 알파 채널을 이용한 투명 배경 합성
    ...

# 코 위치 계산
nose_x = landmark[30][0]
nose_y = landmark[30][1]

# 스티커 크기 조정
face_width = abs(landmark[16][0] - landmark[0][0])
sticker_width = int(face_width * 0.5)

# 스티커 합성
img_result = overlay_transparent(img_result, resized_sticker, sticker_x, sticker_y)
```

- 알파 채널을 사용한 투명 배경 처리
- 얼굴 크기에 비례한 스티커 크기 자동 조정
- 특정 랜드마크 위치에 스티커 배치

## 주요 파라미터 조정

### 스티커 위치 변경

```python
# 코 위치 (랜드마크 30번)
nose_x = landmark[30][0]
nose_y = landmark[30][1]

# 이마 위치 (랜드마크 27번)
forehead_x = landmark[27][0]
forehead_y = landmark[27][1] - 100  # 위로 100픽셀 이동

# 왼쪽 눈 위 (랜드마크 19번)
left_eye_x = landmark[19][0]
left_eye_y = landmark[19][1]
```

### 스티커 크기 조정

```python
# 얼굴 너비의 50%
sticker_width = int(face_width * 0.5)

# 얼굴 너비의 30% (더 작게)
sticker_width = int(face_width * 0.3)

# 얼굴 너비의 70% (더 크게)
sticker_width = int(face_width * 0.7)
```

### 미세 위치 조정

```python
# 기본 위치에서 조정
sticker_x = nose_x - sticker_width // 2 + 10  # 오른쪽으로 10픽셀
sticker_y = nose_y - sticker_height // 2 - 20  # 위로 20픽셀
```

## 68개 랜드마크 인덱스

주요 랜드마크 번호:

- **0-16**: 턱선 (0: 왼쪽 턱, 8: 턱 중앙, 16: 오른쪽 턱)
- **17-21**: 왼쪽 눈썹
- **22-26**: 오른쪽 눈썹
- **27-35**: 코 (27: 코 위, 30: 코끝, 33: 코 아래)
- **36-41**: 왼쪽 눈
- **42-47**: 오른쪽 눈
- **48-67**: 입 (48: 왼쪽 입꼬리, 54: 오른쪽 입꼬리)

## 문제 해결

### ❌ 얼굴을 찾을 수 없습니다

**원인**: 
- 이미지가 너무 어둡거나 흐림
- 얼굴이 너무 작거나 측면을 향함
- 이미지 해상도가 너무 낮음

**해결**:
```python
# 탐지 정확도 높이기 (더 느리지만 정확)
dlib_rects = detector(img_rgb, 2)  # 1 → 2로 변경
```

### ❌ 이미지를 찾을 수 없습니다

**원인**: 파일 경로가 잘못되었거나 파일이 없음

**해결**:
```python
# 파일 존재 확인
!ls -la ~/work/camera_sticker/images/

# 절대 경로 사용
img_path = '/home/jovyan/work/camera_sticker/images/selfie.png'
```

### ❌ plt.show()에서 이미지가 안 보임

**원인**: Jupyter matplotlib 백엔드 설정 문제

**해결**:
```python
# 노트북 첫 셀에 추가
%matplotlib inline
```

### ❌ 스티커가 이상한 위치에 나타남

**원인**: 랜드마크 인덱스 또는 위치 계산 오류

**해결**:
```python
# 랜드마크 시각화로 확인 (6번 셀)
# 원하는 위치의 랜드마크 번호 확인 후 수정

# 위치 미세 조정
sticker_y = nose_y - sticker_height // 2 - 10  # y좌표 조정
```

### ❌ 투명 배경이 제대로 합성되지 않음

**원인**: 스티커 이미지에 알파 채널이 없음

**해결**:
- PNG 형식의 투명 배경 이미지 사용
- 이미지 편집 도구로 배경 제거 후 PNG로 저장

## 참고 자료

- [dlib 공식 문서](http://dlib.net/)
- [OpenCV 공식 문서](https://docs.opencv.org/)
- [dlib Face Landmark Detection](http://dlib.net/face_landmark_detection.py.html)

## 라이선스

이 프로젝트는 교육 목적으로 자유롭게 사용할 수 있습니다.

---

**만든 날짜**: 2026-01-15  
**버전**: 1.0
