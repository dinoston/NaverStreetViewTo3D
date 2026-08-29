# Naver Street View to 3D

## English

Naver Street View to 3D is a restartable reconstruction pipeline for converting street-level panoramas and perspective screenshots into aligned camera data, 3D Gaussian Splatting scenes, point clouds, and extractable meshes.

### Pipeline

1. Validate 2:1 equirectangular panoramas and input images.
2. Generate configurable perspective views from each panorama.
3. Run COLMAP feature extraction, matching, and sparse reconstruction.
4. Train a scene with the official 3D Gaussian Splatting implementation.
5. Run the SuGaR pipeline to extract a surface mesh.
6. Optionally mask trees, grass, plants, and flowers with SegFormer.

### Key Features

- Supports both panoramas and ordinary perspective screenshots
- Resume support with stage logs and `output/status.json`
- Automatic skipping of completed stages
- Shared vegetation masks for COLMAP and 3DGS inputs
- Configurable external paths, training iterations, and processing stages
- Diagnostic command for checking Python, COLMAP, CUDA, and external tools

### Intended Use

This project is intended for research and prototyping in street-level 3D reconstruction, digital twins, and spatial-content production. 3DGS provides visual reconstruction and does not guarantee survey-grade scale or accuracy. Users must also verify their rights to capture, process, and distribute source imagery.

---

## 한국어

아래 문서는 입력 준비부터 COLMAP 정렬, 3D Gaussian Splatting 학습과 SuGaR 메시 추출까지의 상세 사용법을 설명합니다.

# StreetView 3D Builder

2:1 구형 파노라마를 perspective 이미지로 분해하고, COLMAP 카메라 정렬, 3D Gaussian Splatting 학습, SuGaR 메시 추출을 순서대로 실행하는 재시작 가능한 파이프라인입니다.

## 현재 구현 범위

- 파노라마 형식 및 해상도 검사
- 45도 간격 perspective 이미지 생성(설정 가능)
- COLMAP feature extraction / matching / sparse mapping
- 공식 3DGS 저장소의 `train.py` 실행
- SuGaR의 full pipeline 실행
- SegFormer로 나무·잔디·식물·꽃 자동 마스킹(COLMAP 및 3DGS 공통 적용)
- 단계별 로그와 `output/status.json` 기록
- 이미 완료된 단계 자동 건너뛰기

자동 마스킹과 Blender 기반 FBX 후처리는 첫 실제 데이터에서 정렬 품질을 확인한 다음 추가하는 것이 안전합니다. 잘못된 자동 마스크는 건물 특징점까지 제거할 수 있습니다.

## 입력 데이터

입력 방식은 두 가지이며 함께 사용해도 됩니다.

- `input/panoramas`: 2:1 비율의 equirectangular 파노라마 JPG/PNG
- `input/screenshots`: 네이버 지도 등에서 저장한 일반 원근 화면 캡처 JPG/PNG

파일 이름은 촬영 이동 순서와 같도록 `0001.jpg`, `0002.jpg`처럼 지정하는 것을 권장합니다. 일반 캡처는 재투영하지 않고 COLMAP 입력으로 바로 복사됩니다.

권장 조건:

- 서로 다른 촬영 지점 8장 이상
- 인접 촬영 지점에서 동일 외벽이 충분히 겹칠 것
- 뷰어 UI나 검은 여백이 없는 원본 파노라마
- 일반 캡처는 지도 UI를 최대한 숨기고, 같은 건물이 화면의 60% 이상 보이도록 저장
- 일반 캡처는 한 지점에서 회전만 하지 말고 건물 주위를 이동하며 촬영
- 서로 다른 계절/시간대 이미지는 가급적 섞지 않을 것

## 설치

현재 PC의 기본 Python 3.7은 지원 대상이 아닙니다. Python 3.10~3.12와 COLMAP을 설치한 후 PowerShell에서 실행합니다.

```powershell
cd C:\Users\KETI\Downloads\streetview-3d-builder
.\setup.ps1
.\run.ps1 doctor
```

기본 설치는 이미지 전처리와 테스트 환경을 만듭니다. COLMAP, CUDA/PyTorch, 공식 3DGS, PyTorch3D 및 SuGaR는 GPU·Visual Studio 버전에 따라 별도 설치가 필요하며 `doctor`가 누락 항목을 표시합니다. 로컬 완전 설치 환경이 `tools/gs-env`에 있으면 `run.ps1`이 이를 우선 사용하고, 없으면 `.venv`를 사용합니다.

## 첫 사진 테스트

```powershell
# 파노라마는 input\panoramas, 일반 캡처는 input\screenshots에 복사한 후
.\run.ps1 preprocess
```

생성 결과는 `output/frames`에서 바로 확인할 수 있습니다. 기본값은 파노라마 한 장당 수평 8개 뷰입니다.
나무가 제외된 흑백 마스크는 `output/masks`에 저장됩니다. `config.json`의 `mask_vegetation`을 `false`로 바꾸면 끌 수 있습니다.

COLMAP 설치 후:

```powershell
.\run.ps1 align
```

전체 실행:

```powershell
.\run.ps1 all
```

## 외부 복원 저장소

학습 단계는 다음 경로를 기본으로 사용합니다.

```text
external/gaussian-splatting/train.py
external/SuGaR/train_full_pipeline.py
```

경로와 학습 반복 수는 `config.json`에서 바꿀 수 있습니다. 외부 저장소는 각자의 설치 지침과 라이선스를 확인해야 합니다. 공식 3DGS 구현은 기본적으로 비상업 연구/평가 용도입니다.

## 출력

```text
output/
  frames/              perspective 이미지
  frames.json          원본 파노라마와 뷰 방향 대응표
  input_report.json    입력 검사 결과
  colmap/              database 및 sparse model
  dataset/             3DGS 입력 데이터
  3dgs/                학습 checkpoint와 Gaussian PLY
  logs/                 단계별 로그
  status.json           단계 진행 상태
```

## 주의사항

3DGS는 시각적 재현 방법이며 측량 정확도를 자동 보장하지 않습니다. 절대 축척이 필요한 경우 실제 길이를 알고 있는 기준점이나 GPS/측량 좌표를 별도로 제공해야 합니다. 이미지의 저장·가공·배포 권한도 사용자가 확인해야 합니다.
