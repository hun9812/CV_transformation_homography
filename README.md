# 영상 변환과 호모그래피

컴퓨터 비전 수업에서 작성한 코드입니다.

- [A2_2d_transformations.py](A2_2d_transformations.py): 이동·회전·크기 변환과 bilinear interpolation
- [A2_homography.py](A2_homography.py): ORB 특징점 매칭, SVD 기반 호모그래피 추정, 좌표 정규화와 RANSAC

호모그래피를 이용해 책 표지를 다른 이미지로 바꾸는 예제를 포함합니다.

## 실행

Python과 `numpy`, `opencv-python`이 필요합니다. 저장소 루트에서 실행합니다.

```bash
python A2_2d_transformations.py
python A2_homography.py
```

입력 이미지는 저장소에 포함되어 있습니다. 결과 확인에는 OpenCV 창을 표시할 수 있는 환경이 필요합니다.
