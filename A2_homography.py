# img = cv2.imread( IMAGE_FILE_PATH , cv2.IMREAD_GRAYSCALE )
import cv2
import numpy as np
import sys
import os
import time

# SVD이용 Homography 계산
# x' = Hx,  4points을 받아서 3x3 행렬 H 구하기
def compute_homography(srcP, destP):
    arr = []
    num_points = srcP.shape[0]

    for i in range(num_points):
        # 기존 좌표와 변환 좌표
        x,y = srcP[i]
        xp, yp = destP[i]
        
        # homography 교안 25p
        # [-x, -y, -1, 0, 0, 0 ,x*X', y*x', x']
        # [0, 0, 0, -x, -y, -1, x*y', y*y', y']
        row1 = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        row2 = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]

        arr.append(row1)
        arr.append(row2)
    
    arr = np.array(arr, dtype = np.float64)
    
    # SVD 분해
    U, E, Vt = np.linalg.svd(arr)

    # Ah = 0의 해는 Vt의 마지막 행
    h_vector = Vt[-1]

    # 1 x 9 벡터 h를 3x3 행렬 H로 변환
    H = h_vector.reshape((3,3))
    # homography 행렬의 마지막 원소가 1이 되도록 정규화
    if abs(H[2,2]) < 1e-10:
        return None
    H = H/H[2,2]

    return H

# srcP나 destP가 매우 크거나 작을 경우 수치적으로 불안하여 정규화 필요
def normalization(points):
    # 2-2 정규화를 위한 3x3 변환행렬 T 구하기, 최대 거리 루트2
    # 중심 계산
    cx, cy = np.mean(points, axis = 0)

    # 중심으로부터 최대거리 계산
    offsets = points - np.array([cx, cy])
    distances = np.sqrt(np.sum(offsets**2, axis = 1))
    # 가장 긴거리
    max_distance = np.max(distances)

    # 다 같은 값이라 max_distance가 0이 될 수 있으므로 예외처리
    if max_distance == 0:
        scale = 1.0
    else:
        scale = np.sqrt(2) / max_distance
    
    # 변환행렬 T 생성
    T = np.array([
        [scale, 0, -scale *cx], # x축 스케일링, s만큼 축소, 중심만큼 이동
        [0, scale, -scale *cy], # y축 스케일링, s만큼 축소, 중심만큼 이동
        [0, 0, 1]
    ], dtype = np.float64)

    return T

def compute_homography_normalized(srcP, destP):
    # 정규화를 적용, 호모그래피 행렬 h 구하기
    T_src = normalization(srcP)
    T_dest = normalization(destP)

    # (n ,2) -> (n, 3) 동차좌표계로 변환
    n = srcP.shape[0]
    srcP_h = np.hstack((srcP, np.ones((n,1))))
    destP_h = np.hstack((destP, np.ones((n,1))))
    # 정규화된 좌표 계산, 다시 2차원 좌표로 변환
    srcP_norm = (T_src @ srcP_h.T).T
    srcP_norm = srcP_norm[:, :2]
    destP_norm = (T_dest @ destP_h.T).T
    destP_norm = destP_norm[:, :2]

    # 정규화된 좌표로 호모그래피 행렬 계산
    H_norm = compute_homography(srcP_norm, destP_norm)
    if H_norm is None:
        return None

    # 원래 좌표계로 변환
    # H = inverse(T_dest) @ H_norm @ T_src
    T_dest_inv = np.linalg.inv(T_dest)
    H = T_dest_inv @ H_norm @ T_src
    # H[2,2] = 1로 정규화
    if H[2,2] != 0:
        H = H /H[2,2]

    return H
    
# 무작위로 4개를 뽑아 호모그래피 계산, RANSAC 적용
def compute_homography_ransac(srcP, destP, th):
    # srcP, destP는 (n,2) 배열, 매칭된 점들의 좌표
    # th는 inlier 판정 임계값
    n = srcP.shape[0]
    
    # 매칭된 점이 4개 미만이라면 호모그래피 계산 불가
    if n < 4:
        print("매칭된 점이 4개 미만입니다.")
        return np.eye(3)
    
    best_inliers_mask = None
    max_inliers = 0

    TIME_LIMIT = 2.9 # 과제 요구사항, 3초
    start_time = time.time()

    while True:
        # 시간 제한을 넘었다면 break
        if time.time() - start_time > TIME_LIMIT:
            break

        # 랜덤 4개 선택
        cur_4 = np.random.choice(n, 4, replace = False)
        src_4 = srcP[cur_4]
        dest_4 = destP[cur_4]

        # H 계산
        try:
            H_cur = compute_homography_normalized(src_4, dest_4)
        except np.linalg.LinAlgError:
            continue

        # 계산실패하면 루프 skip
        if H_cur is None:
            continue

        # 행렬곱 위해 srcP에 1차원 추가 (x,y) -> (x,y,1), transpose
        srcP_h = np.hstack((srcP, np.ones((n,1)))).T

        projected_h = (H_cur @ srcP_h)

        # 다시 마지막 좌표를 1로 만들어줌
        for_two = projected_h[2,:]
        # 0으로 나누기 방지
        for_two[np.abs(for_two) < 1e-10] = 1e-10

        projected_x = projected_h[0, :] / for_two
        projected_y = projected_h[1,:] / for_two
        projected_points = np.vstack((projected_x, projected_y)).T

        # 거리 구해주기 (매칭돼야하는 점과 project한 점의 차이)
        distances = np.sqrt(np.sum((destP - projected_points)**2, axis = 1))

        # inlier 개수 계산
        cur_mask = distances < th
        cur_inliers = np.sum(cur_mask)

        if cur_inliers > max_inliers:
            max_inliers = cur_inliers
            best_inliers_mask = cur_mask

    # 루프 종료 후 찾은 inlier들을 모두 모아 H다시 계산, return
    if best_inliers_mask is not None and max_inliers >= 4:
        final_src = srcP[best_inliers_mask]
        final_dest = destP[best_inliers_mask]
        best_H = compute_homography_normalized(final_src, final_dest) 
        print(f"debug: inlier 개수: {max_inliers}")
    else:
        print("RANSAC 실패")
        best_H = np.eye(3)

    return best_H

def warp_and_stitch(img_obj, img_bg, H):
    h_bg, w_bg = img_bg.shape

    # warping
    warped = cv2.warpPerspective(img_obj, H, (w_bg, h_bg))
    # mask 생성 (검은색 아닌 영역)
    _, mask = cv2.threshold(warped, 1, 255, cv2.THRESH_BINARY)

    # 합성 (배경 복사, 덮어쓰기)
    stitched = img_bg.copy()
    stitched[mask == 255] = warped[mask == 255]

    return warped, stitched


if __name__ == "__main__":
    img_desk = cv2.imread("cv_desk.png", cv2.IMREAD_GRAYSCALE)
    img_cover = cv2.imread("cv_cover.jpg", cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    
    # image에서 keypoint와 descriptor 계산
    # kp는 흥미로운 지점 리스트, des는 kp가 어떻게 생겼나 숫자로 묘사
    # hamming distance는 같은 위치에 다른 비트(0,1) 갯수
    kp1 = orb.detect(img_desk, None)
    kp1, des1 = orb.compute(img_desk, kp1)

    kp2 = orb.detect(img_cover, None)
    kp2, des2 = orb.compute(img_cover, kp2)
    
    
    good_matches = []
    # 좋은 match를 1순위가 2순위 거리의 75%보다 짧을 경우로 정했습니다.
    # good_match가 부족하거나 너무 많다면 조절
    ratio_thresh = 0.75
    # des1의 모든 흥미로운 지점  i에 대하여
    for i in range(len(des1)):
        best_dist = float('inf')
        second_best_dist = float('inf')
        best_pair = -1
        # des2의 모든 흥미로운 지점 j에 대하여
        for j in range(len(des2)):
            # hamming distance로 계산
            cur_dist = cv2.norm(des1[i], des2[j], cv2.NORM_HAMMING)

            if cur_dist < best_dist:
                # cur_dist가 지금까지 본 거리 중 가장 짧으면
                second_best_dist = best_dist
                best_dist = cur_dist
                best_pair = j
            elif cur_dist < second_best_dist:
                # cur_dist가 두번째로 짧으면
                second_best_dist = cur_dist
        
        # 1순위 거리가 2순위 거리의 ratio_thresh배 보다 작으면 good match로 선정
        if best_dist < ratio_thresh * second_best_dist and best_pair != -1 and second_best_dist != float('inf'):
            # DMatch는 des1의 i와 des2의 best_pari와 distance 정보를 가짐
            match = cv2.DMatch(_queryIdx = i, _trainIdx = best_pair, _distance = best_dist)
            good_matches.append(match)

    # good_matches 에서 distance로 오름차순 정렬
    print("")
    print("--------------------------------------------")
    good_matches.sort(key = lambda x: x.distance)
    print("debug len(good matches):", len(good_matches))

    # 상위 10개
    top_10_matches = good_matches[:10]

    # 상위 10개 매칭 시각화
    img_top10 = cv2.drawMatches(
        img_desk, kp1,
        img_cover, kp2,
        top_10_matches,
        None,
        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    print("Top 10 Matches 시각화")
    print("창을 닫으시려면 엔터 키를 누르세요")
    print("")
    print("------------------------------------------------")
    cv2.imshow("Top 10 Matches", img_top10)
    cv2.waitKey(0)
    cv2.destroyWindow("Top 10 Matches")

    if len(good_matches) < 15:
        print("매칭 점이 15개 미만입니다.")
        sys.exit()

    '''
    # good_match가 너무 많다면 조절해서 사용 (어차피 31개라 안씀)
    num_use = 30
    use_matches = good_matches[:num_use]
    '''
    use_matches = good_matches

    # match = cv2.DMatch(_queryIdx = i, _trainIdx = best_pair, _distance = best_dist)
    # train, query Idx 로 해당 번호 키포인트 찾고, 좌표 (x,y) 가져오기
    pts_src = np.float32([kp2[i.trainIdx].pt for i in use_matches])
    pts_dst = np.float32([kp1[i.queryIdx].pt for i in use_matches])
    print("결과 비교")

    # homography with nomalization, 모든 good match 이용 계산
    H_norm_only= compute_homography_normalized(pts_src, pts_dst)
    print("Homography with Normalization (All Points)\n", H_norm_only)
    print("")

    warp_norm, stitch_norm = warp_and_stitch(img_cover, img_desk, H_norm_only)

    H_ransac = compute_homography_ransac(pts_src, pts_dst, 3.0)
    print("Homography with RANSAC\n", H_ransac)
    print("")

    warp_ransac, stitch_ransac = warp_and_stitch(img_cover, img_desk, H_ransac)

    row1 = np.hstack((warp_norm, stitch_norm))
    row2 = np.hstack((warp_ransac, stitch_ransac))
    
    # 이미지를 2개 붙여 띄워서 너무 크기에 scale factor 넣었습니다
    scale_factor = 0.7
    row1_resized = cv2.resize(row1, (0,0), fx = scale_factor, fy = scale_factor)
    row2_resized = cv2.resize(row2, (0,0), fx = scale_factor, fy = scale_factor)

    cv2.imshow("1. Homography with Normalization", row1_resized)
    cv2.imshow("2. Homography with RANSAC", row2_resized)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # hp_cover과 cv_cover은 크기가 달라, 과제의 요구사항에 나온 그림과
    # 같게 하기 위해 resize하고 H_ransak을 이용합니다.
    img_hp_cover = cv2.imread("hp_cover.jpg", cv2.IMREAD_GRAYSCALE)
    h_cover, w_cover = img_cover.shape
    
    img_hp_resized = cv2.resize(img_hp_cover, (w_cover, h_cover))
    
    # cv2.imshow("debug_hp_cover", img_hp_resized)
    # cv2.waitKey(0)
    # resize된 이미지에 H_ransac 적용하여 warp, stitch
    warp_hp, stitch_hp = warp_and_stitch(img_hp_resized, img_desk, H_ransac)

    row3 = np.hstack((warp_hp, stitch_hp))
    row3_resized = cv2.resize(row3, (0,0), fx = scale_factor, fy = scale_factor)
    cv2.imshow("3. hp_cover warped using 2) ransak result", row3_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
