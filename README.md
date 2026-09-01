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
- `input/target`: `input/screenshots` 사진을 복사해 원하는 건물 둘레에 빨간 사각형을 표시한 JPG/PNG

파일 이름은 촬영 이동 순서와 같도록 `0001.jpg`, `0002.jpg`처럼 지정하는 것을 권장합니다. 일반 캡처는 재투영하지 않고 COLMAP 입력으로 바로 복사됩니다.

권장 조건:

- 서로 다른 촬영 지점 8장 이상
- 인접 촬영 지점에서 동일 외벽이 충분히 겹칠 것
- 뷰어 UI나 검은 여백이 없는 원본 파노라마
- 일반 캡처는 지도 UI를 최대한 숨기고, 같은 건물이 화면의 60% 이상 보이도록 저장
- 일반 캡처는 한 지점에서 회전만 하지 말고 건물 주위를 이동하며 촬영
- 서로 다른 계절/시간대 이미지는 가급적 섞지 않을 것

### 원하는 건물 빨간 박스로 지정하기

주변 건물이나 나무가 함께 찍힌 경우 빨간 박스 지정이 결과를 크게 개선합니다.
원본은 `input/screenshots`에 그대로 두고, 복사본에 **빨간색 테두리 또는 자유곡선 외곽선**을
그려 `input/target`에 넣는 방식이 가장 안전합니다. 간편하게 빨간 선 이미지를
`input/screenshots`에 바로 넣어도 프로그램이 가이드로 자동 복사하고 빨간 선을 제거한 뒤
카메라 입력으로 사용합니다. 파일명은 달라도 됩니다. 프로그램이 특징점을 이용해 원본을
찾고, 다른 시점에서도 같은 건물 영역을 자동 추적합니다.

- 최소 1장으로 동작하지만 정면·왼쪽·오른쪽 3~5장에 각각 박스를 그리는 것을 권장
- 박스 안에는 목표 건물 전체를 넣고 이웃 건물·도로·하늘은 최대한 제외
- 테두리는 선명한 빨간색, 적당히 굵은 선으로 표시
- 가려진 뒤쪽이나 지붕은 사진에 없으면 복원할 수 없음

선택 결과는 `output/building_previews`에서 확인합니다. 초록색은 사용된 시점,
빨간색은 제외된 시점입니다.

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

### 빠른 포인트 클라우드 모드

긴 COLMAP/3DGS/SuGaR 과정을 건너뛰고 VGGT로 표준 XYZRGB PLY를 생성합니다.

```powershell
.\setup-fast.ps1  # 최초 한 번만 실행
.\run.ps1 -Stage fast -Force
```

결과는 `output\pointcloud\fast_building_points.ply`에 저장됩니다. 처음 한 번은
약 5 GB의 VGGT 모델 다운로드가 필요하며, 이후에는 로컬 캐시를 사용합니다.
기본 설정은 최대 16장의 입력을 균등 선택하고 신뢰도가 낮은 점을 제거합니다.
서로 다른 거리뷰 촬영 지점 사이의 이동 시차가 있어야 온전한 형상이 나옵니다.

빠른 모드는 파일명의 촬영 시각을 이용해 30분 이내에 찍은 가장 큰 사진 묶음을
한 번의 촬영 세션으로 선택합니다. 과거 테스트 사진이 같은 폴더에 남아 있어도
자동 제외합니다. 각 사진에서는 화면의 모든 건물이 아니라 주 대상 건물 한 동의
실루엣을 선택하며, 결과 마스크는 `output\building_masks`에서 확인할 수 있습니다.

추가 결과:

- `output\pointcloud\fast_building_points_clean.ply`: 작은 분리 조각과 통계 이상점을 제거한 점군
- `output\mesh\fast_building_mesh.ply`: 빠른 확인용 Poisson 메시
- `output\mesh\fast_building_proxy_mesh.ply`: 관측되지 않은 뒤·지붕을 직육면체로 보완한 프록시 메시
- `output\pointcloud\fast_building_points_preview.png`: 정면·상단·측면 점군 미리보기
- `output\pointcloud\sfm_building_points_clean.ply`: 여러 사진에서 일치한 COLMAP 특징점만 남긴 보수적 기준 점군

메시는 형태 확인용 초안입니다. 정밀 편집에는 clean PLY를 CloudCompare/Blender에서
정리한 뒤 별도로 메시화하는 것을 권장합니다.
정제 점군과 두 메시는 자동으로 Z-up 좌표계에 정렬되고 바닥이 Z=0에 배치됩니다.

### 목표 건물 3D Gaussian Splatting

COLMAP 정렬 뒤, 빨간 박스로 추적된 유효 사진과 건물 점만으로 3DGS를 학습합니다.

```powershell
.\run.ps1 -Stage align -Force
.\run.ps1 -Stage fast -Force
.\run.ps1 -Stage splat -Force
.\run.ps1 -Stage splat-mesh -Force
```

Gaussian PLY는
`output\3dgs_building\point_cloud\iteration_7000\point_cloud.ply`에 저장됩니다.
기본 7,000회 학습은 이 PC에서 약 2~3분이며 `config.json`의
`splat_iterations`로 조절할 수 있습니다. 스크린샷 사이 시차가 작거나 건물의 한 면만
관측되면 입력 화면에서는 잘 보이더라도 자유 시점의 기하가 늘어질 수 있습니다.

`splat-mesh`는 COLMAP 건물 경계와 Gaussian 불투명도·크기로 노이즈를 제거한 뒤
Z-up Poisson 메시를 생성합니다. 결과는 `output/mesh/gaussian_building_mesh.ply`와
`output/mesh/gaussian_building_mesh.obj`에 저장됩니다.

기본 `facebook/VGGT-1B` 체크포인트는 CC-BY-NC-4.0 연구용입니다. 회사 제품이나
상업 배포에는 Meta의 승인을 받은 `VGGT-1B-Commercial` 체크포인트로 설정을
변경해야 합니다.

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
  dataset_building/    빨간 박스로 선택된 건물 전용 3DGS 데이터
  3dgs_building/       목표 건물 Gaussian checkpoint와 PLY
  building_previews/   시점별 목표 건물 선택 확인 이미지
  pointcloud/          빠른 점군, 정제 점군, SfM 기준 점군
  mesh/                Poisson 초안 메시와 프록시 메시
  logs/                 단계별 로그
  status.json           단계 진행 상태
```

## 주의사항

3DGS는 시각적 재현 방법이며 측량 정확도를 자동 보장하지 않습니다. 절대 축척이 필요한 경우 실제 길이를 알고 있는 기준점이나 GPS/측량 좌표를 별도로 제공해야 합니다. 이미지의 저장·가공·배포 권한도 사용자가 확인해야 합니다.
