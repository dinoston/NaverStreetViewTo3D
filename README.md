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

