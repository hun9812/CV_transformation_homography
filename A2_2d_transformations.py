# img = cv2.imread( IMAGE_FILE_PATH , cv2.IMREAD_GRAYSCALE )
import cv2
import numpy as np
import sys
import os


# 소숫점일 경우 주변 픽셀값에 거리로 가중치를 주어 보간하는 함수
def linear_interpolation(img, x, y):
    # 801, 801 원본 이미지 크기
    img_h, img_w = img.shape

    # (x, y)는 실수 좌표, 주변 4픽셀의 좌표
    x_left = np.floor(x).astype(int) # x 좌
    x_right = x_left+1 # x 우
    y_up = np.floor(y).astype(int) # y 위
    y_down = y_up+1 # y 아래

    # 가중치 계산, 실제 표와 정수 좌표 차이
    x_diff = x - x_left
    y_diff = y - y_up

    # 원본 이미지 경계 (0 <= x < img_w, 0<= y <img_h)
    mask = (x_left >= 0) & (x_right < img_w) & (y_up >= 0) & (y_down < img_h)

    # 최종 결과 배열 초기화 (배경 흰색)
    output = np.full_like(x, 255.0)

    # valid_count = np.sum(mask)
    # total_count = mask.size
    # print(f"valid pixels = {valid_count} / {total_count}")

    # 유효한 픽셀이 하나라도 있으면 보간 수행
    if np.any(mask):
        # 각 픽셀에서 인덱스 생성, 원본이미지 4픽셀 추출
        left_up = img[y_up[mask], x_left[mask]] # 좌상단
        right_up = img[y_up[mask], x_right[mask]] # 우상단
        left_down = img[y_down[mask], x_left[mask]] # 좌하단
        right_down = img[y_down[mask], x_right[mask]] # 우하단
        
        weight_x = x_diff[mask]
        weight_y = y_diff[mask]
        # 보간 계산 (x축 2*2, y로 합침)
        x1 = left_up * (1-weight_x) + right_up * weight_x
        x2 = left_down * (1-weight_x) + right_down * weight_x
        # y축
        value = x1 * (1 - weight_y) + x2 * weight_y

        output[mask] = value

    return output


def get_transformed_image(img, M):
    # 출력 평면 (801, 801)
    output_size = 801
    center_offset = 400 # 평면 중심점

    # 0부터 800까지 좌표 배열 생성
    range_arr = np.arange(output_size)
    # grid_x, grid_y = 801x801
    grid_x, grid_y = np.meshgrid(range_arr, range_arr)
    
    # 중심점 이동 (400,400) -> (0,0)
    # 평면의 중심을 원점으로, u,v는 중심이 0,0인 좌표계
    u = grid_x - center_offset
    v = grid_y - center_offset

    # homogeneous coordinates
    # [u,v,1]
    ones_vector = np.ones_like(u).flatten()
    # (3, N)
    dest_coordinates = np.stack([u.flatten(), v.flatten(), ones_vector])

    # 역변환
    # destination에서 원본 좌표(src)를 위해 역행렬을 곱해서 알아냄
    # src = M_inv @ destination
    M_inv = np.linalg.inv(M)
    # print(M_inv)
    src_coordinates = M_inv @ dest_coordinates
    
    # 실제 이미지와 맞추기 위해 이미지 중심점만큼 다시 이동
    src_h, src_w = img.shape
    # x좌표에는 width/2를 y좌표에는 height/2를 더함
    raw_src_x = src_coordinates[0,:] + (src_w/2)
    raw_src_y = src_coordinates[1,:] + (src_h/2)
    # print(f"min_x = {raw_src_x.min()}, max_x = {raw_src_x.max()}")
    # print(f"min_y = {raw_src_y.min()}, max_y = {raw_src_y.max()}")

    # 2d 형태로 reshape
    s_x = raw_src_x.reshape((output_size, output_size))
    s_y = raw_src_y.reshape((output_size, output_size))

    transformed_plane = linear_interpolation(img, s_x, s_y)
    
    # float -> int8
    res = transformed_plane.astype(np.uint8)


    # # backward warping
    # for y in range(plane_h):
    #     for x in range(plane_w):
    #         # 현재 평면 좌표 (x,y)를 (0,0) 기준 좌표로 변환
    #         x_plane = x - plane_center_x
    #         y_plane = y - plane_center_y

    #         # 역행렬 M은 3X3 행렬
    #         xy1 = np.array([x_plane, y_plane, 1])

    #         # 역행렬 M을 곱하여 원본 이미지 좌표 계산
    #         pos_xy1 = M_inverse @ xy1

    #         # 2d 좌표로 변환, homogeneous 좌표계 고려
    #         pos_x = pos_xy1[0] / pos_xy1[2]
    #         pos_y = pos_xy1[1] / pos_xy1[2]

    #         # 원본 이미지 좌표를 (0,0) 기준 좌표에서 이미지 기준 좌표로 변환
    #         img_x = pos_x + img_center_x
    #         img_y = pos_y + img_center_y

    #         # 원본 이미지 좌표가 이미지 범위 내에 있는 경우에만 픽셀 값을 복사
    #         if (0 <= img_x < img_w -1) and (0 <= img_y < img_h -1):
    #             value = linear_interpolation(img, img_x, img_y)
    #             plane[y, x] = np.clip(value, 0, 255).astype(np.uint8)

    color = (0,0,0)
    thickness = 3
    # x축: 0-> 801, y축 801->0
    cv2.arrowedLine(res, (0, center_offset), (output_size, center_offset), color, thickness, tipLength = 0.02)
    cv2.arrowedLine(res, (center_offset,output_size), (center_offset,0), color, thickness, tipLength = 0.02)

    return res

def get_M_function(M_prev, key_char):

    move = 5
    theta = np.deg2rad(5)
    scale_enlarge = 1.05
    scale_shrink = 0.95

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    '''
    이동              회전 (theta, -theta)  반전
    [ * * x축 move]  [ cos -sin 0 ]        [ -1 0 0 ] [ 1 0 0 ]
    [ * * y축 move]  [ sin  cos 0 ]        [  0 1 0 ] [ 0 -1 0] 
    [ 0 0 1       ]  [ 0    0   1 ]        [  0 0 1 ] [ 0 0 1 ]
    스케일 
    [ s  0  0 ] [ 1 0 0 ] 
    [ 0  1  0 ] [ 0 s 0 ]
    [ 0  0  1 ] [ 0 0 1 ]
    '''

    key_dict = {
        # 이동
        'a': np.array([[1, 0, -move], [0, 1, 0], [0, 0, 1]], dtype = float),  # 좌
        'd': np.array([[1, 0, move], [0, 1, 0], [0, 0, 1]], dtype = float),   # 우
        'w': np.array([[1, 0, 0], [0, 1, -move], [0, 0, 1]], dtype = float),  # 상
        's': np.array([[1, 0, 0], [0, 1, move], [0, 0, 1]], dtype = float),   # 하
        # 회전
        'r': np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]], dtype = float),  # 반시계
        't': np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]], dtype = float),  # 시계
        # 반전
        'f': np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = float),  # y축
        'g': np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype = float),  # x축
        # x축 스케일
        'x': np.array([[scale_shrink, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = float),  # 축소
        'c': np.array([[scale_enlarge, 0, 0], [0, 1, 0], [0, 0, 1]], dtype = float),  # 확대
        # y축 스케일
        'y': np.array([[1, 0, 0], [0, scale_shrink, 0], [0, 0, 1]], dtype = float),  # 축소
        'u': np.array([[1, 0, 0], [0, scale_enlarge, 0], [0, 0, 1]], dtype = float),  # 확대
    }
    
    # key_dict에서 해당 행렬 A를 꺼내 왼쪽에 곱해고 return
    if key_char in key_dict:
        A = key_dict[key_char]
        # print(A @ M_prev)
        return A @ M_prev
    else:
        return M_prev


if __name__ == "__main__":
    
    img = cv2.imread( "smile.png" , cv2.IMREAD_GRAYSCALE )
    
    if img is None:
        print("이미지 파일을 찾을 수 없습니다.")
        sys.exit()

    print("------------------------------------------------")
    print(" ---Interactive 2D Transformations--- ")
    print("")
    print(" [이동 (5pixel)] 'a': 좌, 'd': 우, 'w': 상, 's': 하 ")
    print(" [회전 (5')]     'r': 반시계, 't': 시계")
    print(" [반전]          'f': y축, 'g': x축")
    print(" [x축 스케일]     'x': 축소, 'c': 확대")
    print(" [y축 스케일]     'y': 축소, 'u': 확대")
    print("")
    print(" [종료] 'q', [초기화] 'h'")
    print("------------------------------------------------")
    print(" 이미지를 클릭하고 키를 눌러주세요 ")
    M_current = np.eye(3)

    while True:
        # 현재 M으로 변환된 이미지 계산
        transformed_plane = get_transformed_image(img, M_current)

        # 화면에 출력
        cv2.imshow("Transformed Image Interative Window", transformed_plane)

        # 키 입력 대기
        key = cv2.waitKey(0)
        key_char = chr(key)

        if key == ord('q'):
            print("프로그램 종료")
            break
        elif key == ord('h'):
            print("초기화")
            # M_current를 단위 행렬로 초기화
            M_current = np.eye(3)
        # 유효한 변환 키인지
        elif key_char in ['a', 'd', 'w', 's', 'r', 't', 'f', 'g', 'x', 'c', 'y', 'u']:
            M_current = get_M_function(M_current, key_char)
            # print(M_current)
        else:
            print(f"'{key_char}' 키는 정의되지 않은 키입니다.")

    # transformed_plane = get_transformed_image(img, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    # cv2.imshow("Transformed Image - Identity", transformed_plane)

    # cv2.waitKey(0)
    cv2.destroyAllWindows()   