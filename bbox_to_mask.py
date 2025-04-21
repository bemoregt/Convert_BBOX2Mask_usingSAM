import os
import csv
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import time

# 시작 시간 기록
start_time = time.time()

# 경로 설정
base_dir = "/Users/m1_4k/Pictures/car_bbox_1001"  # 이미지 경로에 맞게 수정
csv_path = os.path.join(base_dir, "train_solution_bounding_boxes (1).csv")
image_dir = os.path.join(base_dir, "training_images")
output_dir = os.path.join(base_dir, "segment")

# segment 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# 디바이스 설정: MPS 사용 (Apple Silicon GPU 가속)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS 디바이스 사용 (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA 디바이스 사용 (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("CPU 디바이스 사용")

# 더 가벼운 vit_b 모델 사용
model_type = "vit_b"  # vit_h 대신 가벼운 모델 사용
sam_checkpoint = "sam_vit_b_01ec64.pth"  # 모델 경로를 vit_b 체크포인트로 수정

print(f"SAM 모델 로드 중: {model_type} - {sam_checkpoint}")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# CSV 파일 읽기
print("CSV 파일 읽는 중...")
bbox_data = {}
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 헤더 건너뛰기
    for row in reader:
        image_name, xmin, ymin, xmax, ymax = row[0], float(row[1]), float(row[2]), float(row[3]), float(row[4])
        if image_name not in bbox_data:
            bbox_data[image_name] = []
        bbox_data[image_name].append((xmin, ymin, xmax, ymax))

print(f"총 {len(bbox_data)} 개의 이미지 처리 예정")

# 배치 크기 - 메모리 용량에 따라 조정
batch_size = 1  # 메모리가 충분하면 증가 가능

# 처리된 이미지 수
processed_count = 0

# 이미지 처리
for image_name, bboxes in bbox_data.items():
    # 이미지 파일 경로
    image_path = os.path.join(image_dir, image_name)
    
    # 진행 상황 표시
    processed_count += 1
    print(f"[{processed_count}/{len(bbox_data)}] {image_name} 처리 중...")
    
    # 이미지 파일이 존재하는지 확인
    if not os.path.exists(image_path):
        print(f"경고: {image_path} 파일을 찾을 수 없습니다. 건너뜁니다.")
        continue
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"경고: {image_path} 이미지를 로드할 수 없습니다. 건너뜁니다.")
        continue
    
    # RGB로 변환 (SAM은 RGB 이미지를 사용)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # SAM 예측을 위해 이미지 설정
    predictor.set_image(image_rgb)
    
    # 이미지와 동일한 크기의 빈 마스크 생성 (결과 마스크 누적용)
    final_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # 메모리 최적화: 여러 바운딩 박스를 배치로 처리
    for i in range(0, len(bboxes), batch_size):
        batch_bboxes = bboxes[i:i+batch_size]
        
        for bbox in batch_bboxes:
            xmin, ymin, xmax, ymax = bbox
            
            # 중앙점 계산 (positive prompt)
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            
            # 모서리 계산 (negative prompt)
            corners = [
                (xmin, ymin),  # 좌상단
                (xmax, ymin),  # 우상단
                (xmin, ymax),  # 좌하단
                (xmax, ymax)   # 우하단
            ]
            
            # Prompt 포인트 및 레이블 준비
            input_points = np.array([[center_x, center_y]] + corners)
            input_labels = np.array([1, 0, 0, 0, 0])  # 중앙점은 positive(1), 모서리는 negative(0)
            
            # SAM으로 마스크 예측 - 단일 마스크만 반환하도록 설정하여 메모리 절약
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False  # 단일 마스크만 출력하여 속도 향상
            )
            
            # 최종 마스크에 현재 객체의 마스크 추가 (OR 연산)
            mask = masks[0]  # multimask_output=False이므로 첫 번째 마스크만 사용
            final_mask = np.logical_or(final_mask, mask).astype(np.uint8) * 255
        
        # 배치 처리 후 메모리 정리
        if device.type == "cuda" or device.type == "mps":
            torch.cuda.empty_cache() if device.type == "cuda" else torch.mps.empty_cache()
    
    # 마스크 저장
    output_filename = os.path.splitext(image_name)[0] + "_mask.png"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, final_mask)
    
    # 진행 시간 표시
    elapsed_time = time.time() - start_time
    avg_time_per_image = elapsed_time / processed_count
    remaining_images = len(bbox_data) - processed_count
    estimated_time_remaining = avg_time_per_image * remaining_images
    
    print(f"{image_name} 처리 완료: {output_path}에 마스크 저장됨")
    print(f"진행 상황: {processed_count}/{len(bbox_data)} 이미지 완료")
    print(f"예상 남은 시간: {estimated_time_remaining:.2f}초 ({estimated_time_remaining/60:.2f}분)")

total_time = time.time() - start_time
print(f"모든 이미지 처리 완료! 총 소요 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")