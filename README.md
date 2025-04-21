# Convert_BBOX2Mask_usingSAM

바운딩 박스(Bounding Box) 정보를 이용하여 [SAM(Segment Anything Model)](https://github.com/facebookresearch/segment-anything)을 활용한 세그멘테이션 마스크로 변환하는 도구입니다.

## 주요 기능

- CSV 파일에서 바운딩 박스 정보를 읽어와 이미지별 세그멘테이션 마스크 생성
- Apple Silicon(M1/M2) GPU 가속(MPS) 및 NVIDIA GPU(CUDA) 지원
- 메모리 최적화를 위한 배치 처리 및 단일 마스크 출력 옵션
- 진행 상태 및 예상 소요 시간 표시

## 동작 방식

1. CSV 파일에서 이미지별 바운딩 박스 정보 로드
2. 각 바운딩 박스에 대해:
   - 중앙점을 양성(positive) 프롬프트로 사용
   - 모서리 4개 지점을 음성(negative) 프롬프트로 사용
   - SAM 모델로 세그멘테이션 마스크 예측
3. 이미지 내 모든 객체의 마스크를 하나로 결합
4. 결과 마스크를 PNG 파일로 저장

## 요구사항

- Python 3.8 이상
- PyTorch 2.0 이상
- OpenCV
- NumPy
- [segment-anything](https://github.com/facebookresearch/segment-anything) 패키지
- SAM 모델 체크포인트 파일 (`sam_vit_b_01ec64.pth`)

## 설치 방법

```bash
# 1. 저장소 복제
git clone https://github.com/bemoregt/Convert_BBOX2Mask_usingSAM.git
cd Convert_BBOX2Mask_usingSAM

# 2. 필요한 패키지 설치
pip install torch opencv-python numpy
pip install git+https://github.com/facebookresearch/segment-anything.git

# 3. SAM 모델 다운로드
# https://github.com/facebookresearch/segment-anything#model-checkpoints 에서 
# 'ViT-B SAM model' 다운로드
```

## 사용 방법

1. `bbox_to_mask.py` 파일을 열어 아래 경로를 자신의 환경에 맞게 수정:
   ```python
   base_dir = "/path/to/your/data"  # 데이터 디렉토리
   csv_path = os.path.join(base_dir, "your_bounding_boxes.csv")  # 바운딩 박스 CSV 파일
   image_dir = os.path.join(base_dir, "images")  # 이미지 디렉토리
   output_dir = os.path.join(base_dir, "segment")  # 결과 저장 디렉토리
   
   # SAM 모델 파일 경로
   sam_checkpoint = "path/to/sam_vit_b_01ec64.pth"
   ```

2. 스크립트 실행:
   ```bash
   python bbox_to_mask.py
   ```

## CSV 파일 형식

CSV 파일은 다음과 같은 형식이어야 합니다:
```
filename,xmin,ymin,xmax,ymax
image1.jpg,100,150,300,400
image2.jpg,200,250,500,600
...
```

## 성능 최적화

- `batch_size` 변수를 조정하여 메모리 사용량과 처리 속도를 조절할 수 있습니다.
- 메모리가 충분하다면 `batch_size`를 증가시켜 처리 속도를 향상시킬 수 있습니다.
- Apple Silicon GPU(MPS) 또는 NVIDIA GPU(CUDA)를 자동으로 감지하여 활용합니다.

## 라이센스

MIT License

## 저자

[bemoregt](https://github.com/bemoregt)