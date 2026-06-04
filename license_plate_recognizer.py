# ============================================================
# license_plate_recognizer.py
# 车牌识别：形态学特征 + 汉明距离 融合识别汉字
# 字母数字：NCC + HOG 特征
# ============================================================

import cv2
import numpy as np
import os
from pathlib import Path
from skimage.feature import hog

# ======================== 1. 全局参数配置 ========================

MIN_PLATE_HEIGHT = 15
MIN_PLATE_WIDTH = 30

BLUE_H_MIN = 90
BLUE_H_MAX = 125
BLUE_S_MIN = 30
BLUE_S_MAX = 153
BLUE_V_MIN = 200
BLUE_V_MAX = 255

DILATE_KERNEL_SIZE = 7
DILATE_ITERATIONS = 2

CHAR_WIDTH = 20
CHAR_HEIGHT = 40

NCC_THRESHOLD = 0.7
HOG_CHECK_THRESH = 0.75
AMBIGUOUS_GAP = 0.15

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# 汉字特征权重（形态学）
CHINESE_WEIGHT_HOLE = 5.0
CHINESE_WEIGHT_ENDPOINT = 3.0
CHINESE_WEIGHT_GRID = 2.0
CHINESE_WEIGHT_LINE = 1.5
CHINESE_WEIGHT_PROJ = 1.0

# 融合权重
MORPH_WEIGHT = 0.5   # 形态学特征距离权重
HAMMING_WEIGHT = 0.5 # 汉明距离权重

# ======================== 各位置允许的字符集 ========================

PROVINCE_CHARS = list('京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁')
CITY_LETTERS = list('ABCDEFGHJKLMNPQRSTUVWXYZ')
LETTERS_DIGITS = list('0123456789ABCDEFGHJKLMNPQRSTUVWXYZ')

AMBIGUOUS_PAIRS = [
    ('8', 'B'), ('B', '8'),
    ('8', '6'), ('6', '8'),
    ('0', 'D'), ('D', '0'),
    ('0', 'Q'), ('Q', '0'),
    ('B', 'D'), ('D', 'B'),
    ('S', '5'), ('5', 'S'),
    ('3', '8'), ('8', '3'),
    ('7', 'Z'), ('Z', '7'),
    ('Z', '2'), ('2', 'Z'),
    ('E', 'F'), ('F', 'E'),
]

# ======================== 2. 车牌定位 ========================
def find_plate_by_blue(img_bgr):
    """
    蓝色车牌定位（严格基于HSV，排除黑白灰，无多余约束）
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # ========== 1. 蓝色范围（可根据实际微调） ==========
    # 标准蓝色：H 100~125，S 50~255，V 80~255
    lower = np.array([100, 50, 80])
    upper = np.array([125, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # ========== 2. 形态学处理 ==========
    # 闭运算：填充字符内部的孔洞（如“8”、“0”中的白色空洞）
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # 开运算：去除孤立的小蓝点（噪点）
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # 轻微膨胀：连接蓝色区域可能存在的细小断裂（注意：不要过大，否则会吞没黑色边框）
    kernel_dilate = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    # ========== 3. 查找轮廓 ==========
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    best_rect = None
    best_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_PLATE_HEIGHT * MIN_PLATE_WIDTH:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        # 蓝色占比检查：区域内蓝色像素比例必须很高（排除边缘误检）
        roi_mask = mask[y:y+h, x:x+w]
        blue_ratio = np.sum(roi_mask > 0) / (w * h)
        if blue_ratio < 0.5:   # 要求一半以上是蓝色，可调至0.6更严格
            continue
        
        if area > best_area:
            best_area = area
            best_rect = (x, y, w, h)
    
    if best_rect is None:
        return None
    
    x, y, w, h = best_rect
    
    # ========== 4. 精确内缩到蓝色区域（投影裁剪） ==========
    roi_mask = mask[y:y+h, x:x+w]
    rows = np.any(roi_mask > 0, axis=1)
    cols = np.any(roi_mask > 0, axis=0)
    
    if np.any(rows) and np.any(cols):
        y1 = y + np.where(rows)[0][0]
        y2 = y + np.where(rows)[0][-1]
        x1 = x + np.where(cols)[0][0]
        x2 = x + np.where(cols)[0][-1]
        
        # 确保内缩后的尺寸仍满足最小车牌要求
        if (y2 - y1) >= MIN_PLATE_HEIGHT and (x2 - x1) >= MIN_PLATE_WIDTH:
            top = max(0, y1)
            bottom = min(img_bgr.shape[0] - 1, y2)
            left = max(0, x1)
            right = min(img_bgr.shape[1] - 1, x2)
            return (top, bottom, left, right)
    
    # 回退到原始矩形（内缩失败时）
    top = max(0, y)
    bottom = min(img_bgr.shape[0] - 1, y + h)
    left = max(0, x)
    right = min(img_bgr.shape[1] - 1, x + w)
    return (top, bottom, left, right)

# ======================== 3. 二值化 ========================
def binarize_plate_cropped(plate_img):
    """
    对已经裁剪好的车牌图像进行二值化（不包含裁剪逻辑）
    plate_img: 已经裁剪好的彩色图像（车牌区域）
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio > 0.5:
        binary = 255 - binary
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    # 去除上下边框（保留原逻辑）
    h_img, w_img = binary.shape
    if h_img >= 5:
        row_sum = np.sum(binary == 255, axis=1).astype(np.float32)
        mid_start = h_img // 4
        mid_end = h_img * 3 // 4
        mid_rows = row_sum[mid_start:mid_end]
        median_val = np.median(mid_rows) if len(mid_rows) > 0 else 0
        line_thresh = max(median_val * 1.5, w_img * 0.1)
        for top in range(h_img):
            if row_sum[top] > line_thresh:
                binary[top, :] = 0
            else:
                break
        for bottom in range(h_img-1, -1, -1):
            if row_sum[bottom] > line_thresh:
                binary[bottom, :] = 0
            else:
                break
    return binary

def binarize_plate_roi(img_bgr, roi_rect):
    x, y, w, h = roi_rect
    roi = img_bgr[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio > 0.5:
        binary = 255 - binary

    # 形态学操作
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # 去除上下边框（保留字符主体）
    h_img, w_img = binary.shape
    if h_img >= 5:
        row_sum = np.sum(binary == 255, axis=1).astype(np.float32)
        # 动态计算阈值：取中间部分中位数的1.5倍
        mid_start = h_img // 4
        mid_end = h_img * 3 // 4
        mid_rows = row_sum[mid_start:mid_end]
        median_val = np.median(mid_rows) if len(mid_rows) > 0 else 0
        line_thresh = max(median_val * 1.5, w_img * 0.1)  # 至少10%宽度

        # 从顶部开始清除连续的行
        for top in range(h_img):
            if row_sum[top] > line_thresh:
                binary[top, :] = 0
            else:
                break
        # 从底部开始清除
        for bottom in range(h_img-1, -1, -1):
            if row_sum[bottom] > line_thresh:
                binary[bottom, :] = 0
            else:
                break

    return binary

#  ======================== 4. 字符分割 ========================

def vertical_projection(binary_img):
    return np.sum(binary_img == 255, axis=0)

def segment_chars_by_gap(binary_img, proj):
    h, w = binary_img.shape
    if w == 0 or h == 0:
        return []

    max_val = np.max(proj)
    if max_val == 0:
        return []

    expected_w = max(w // 9, 5)

    gap_thresh = max(max_val * 0.10, 2)

    chars = []
    in_char = False
    start = 0
    for x in range(w):
        if proj[x] > gap_thresh and not in_char:
            start = x
            in_char = True
        elif proj[x] <= gap_thresh and in_char:
            end = x
            if end - start > 2:
                chars.append([start, end - 1])
            in_char = False
    if in_char:
        chars.append([start, w - 1])

    if len(chars) == 0:
        return []

    while len(chars) > 7:
        best_i = -1
        best_score = float('inf')
        for i in range(len(chars) - 1):
            combined_w = chars[i + 1][1] - chars[i][0] + 1
            gap = max(0, chars[i + 1][0] - chars[i][1] - 1)
            score = abs(combined_w - expected_w) + gap * 3
            if score < best_score:
                best_score = score
                best_i = i
        if best_i >= 0:
            chars[best_i] = [chars[best_i][0], chars[best_i + 1][1]]
            chars.pop(best_i + 1)
        else:
            break

    max_w = expected_w * 1.8
    changed = True
    while changed:
        changed = False
        new_chars = []
        for c in chars:
            cw = c[1] - c[0] + 1
            if cw > max_w and len(chars) < 8:
                region_proj = proj[c[0]:c[1] + 1]
                if len(region_proj) > 6:
                    search_start = 2
                    search_end = len(region_proj) - 2
                    sub = region_proj[search_start:search_end]
                    if np.min(sub) < max_val * 0.20:
                        split_local = search_start + int(np.argmin(sub))
                        split_x = c[0] + split_local
                        left_w = split_x - c[0]
                        right_w = c[1] - split_x
                        if left_w > expected_w * 0.3 and right_w > expected_w * 0.3:
                            new_chars.append([c[0], split_x - 1])
                            new_chars.append([split_x, c[1]])
                            changed = True
                            continue
                new_chars.append(c)
            else:
                new_chars.append(c)
        chars = new_chars

    if len(chars) > 7:
        chars_with_score = [(abs((c[1] - c[0]) - expected_w), c) for c in chars]
        chars_with_score.sort(key=lambda x: x[0])
        chars = [c for _, c in chars_with_score[:7]]
        chars.sort(key=lambda c: c[0])

    return [tuple(c) for c in chars]

# ======================== 5. 字符归一化 ========================

def normalize_char(char_roi):
    if char_roi.size == 0:
        return None

    rows = np.any(char_roi == 255, axis=1)
    if not np.any(rows):
        return None
    top_r = int(np.argmax(rows))
    bot_r = int(len(rows) - 1 - np.argmax(rows[::-1]))
    char_roi = char_roi[top_r:bot_r + 1, :]

    cols = np.any(char_roi == 255, axis=0)
    if not np.any(cols):
        return None
    left_c = int(np.argmax(cols))
    right_c = int(len(cols) - 1 - np.argmax(cols[::-1]))
    char_roi = char_roi[:, left_c:right_c + 1]

    h, w = char_roi.shape
    if h >= 8 and w >= 5:
        row_white = np.sum(char_roi == 255, axis=1)
        wide = w * 0.6
        for y in range(min(5, h - 4)):
            if row_white[y] >= wide:
                below = row_white[y + 1: min(y + 4, h)]
                if np.any(below < wide * 0.3):
                    char_roi[y, :] = 0
                else:
                    break
            else:
                break
        h = char_roi.shape[0]
        for y in range(h - 1, max(h - 6, 3), -1):
            if row_white[y] >= wide:
                above = row_white[max(y - 3, 0): y]
                if np.any(above < wide * 0.3):
                    char_roi[y, :] = 0
                else:
                    break
            else:
                break

    rows = np.any(char_roi == 255, axis=1)
    if not np.any(rows):
        return None
    top_r = int(np.argmax(rows))
    bot_r = int(len(rows) - 1 - np.argmax(rows[::-1]))
    char_roi = char_roi[top_r:bot_r + 1, :]

    cols = np.any(char_roi == 255, axis=0)
    if not np.any(cols):
        return None
    left_c = int(np.argmax(cols))
    right_c = int(len(cols) - 1 - np.argmax(cols[::-1]))
    char_roi = char_roi[:, left_c:right_c + 1]

    if char_roi.size == 0:
        return None

    src_h, src_w = char_roi.shape
    if src_h == 0 or src_w == 0:
        return None

    scale = min(CHAR_WIDTH / src_w, CHAR_HEIGHT / src_h)
    new_w = max(int(src_w * scale), 1)
    new_h = max(int(src_h * scale), 1)
    char_scaled = cv2.resize(char_roi, (new_w, new_h),
                             interpolation=cv2.INTER_AREA)

    canvas = np.zeros((CHAR_HEIGHT, CHAR_WIDTH), dtype=np.uint8)
    x_off = (CHAR_WIDTH - new_w) // 2
    y_off = (CHAR_HEIGHT - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = char_scaled

    _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
    return canvas

# ======================== 6. HOG特征（字母数字用） ========================

def extract_hog_features(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fd = hog(img, orientations=HOG_ORIENTATIONS,
             pixels_per_cell=HOG_PIXELS_PER_CELL,
             cells_per_block=HOG_CELLS_PER_BLOCK,
             visualize=False, block_norm='L2-Hys')

    h, w = img.shape
    aspect_ratio = np.array([w / h], dtype=np.float32)

    combined = np.append(fd.astype(np.float32), aspect_ratio)
    return combined

# ======================== 7. 汉字结构特征（形态学） ========================

def count_holes(binary_img):
    """计算孔洞数（黑底白字，RETR_TREE子轮廓）"""
    contours, hierarchy = cv2.findContours(binary_img.copy(),
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    count = 0
    for h in hierarchy[0]:
        if h[2] != -1:
            count += 1
    return count

def grid_features(binary_img, grid_r=5, grid_c=5):
    """5×5网格特征：每格白色像素占比"""
    h, w = binary_img.shape
    features = []
    for r in range(grid_r):
        for c in range(grid_c):
            y1 = h * r // grid_r
            y2 = h * (r + 1) // grid_r
            x1 = w * c // grid_c
            x2 = w * (c + 1) // grid_c
            cell = binary_img[y1:y2, x1:x2]
            ratio = float(np.sum(cell == 255)) / cell.size if cell.size > 0 else 0.0
            features.append(ratio)
    return np.array(features, dtype=np.float32)

def direction_line_histogram(binary_img):
    """
    方向线素直方图（4维）
    """
    img = binary_img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(np.abs(gy), np.abs(gx)) * 180 / np.pi
    hist = np.zeros(4, dtype=np.float32)
    hist[0] = np.sum(mag[(angle >= 0) & (angle < 22.5)]) + np.sum(mag[(angle >= 157.5) & (angle <= 180)])
    hist[1] = np.sum(mag[(angle >= 22.5) & (angle < 67.5)])
    hist[2] = np.sum(mag[(angle >= 67.5) & (angle < 112.5)])
    hist[3] = np.sum(mag[(angle >= 112.5) & (angle < 157.5)])
    total = np.sum(hist) if np.sum(hist) > 0 else 1.0
    return hist / total

def projection_features(binary_img, n_bins=8):
    """水平+垂直投影降采样到n_bins段"""
    h, w = binary_img.shape
    h_proj = np.sum(binary_img == 255, axis=1).astype(np.float32)
    v_proj = np.sum(binary_img == 255, axis=0).astype(np.float32)
    h_bins = np.array_split(h_proj, n_bins)
    v_bins = np.array_split(v_proj, n_bins)
    h_feat = np.array([np.mean(b) for b in h_bins], dtype=np.float32)
    v_feat = np.array([np.mean(b) for b in v_bins], dtype=np.float32)
    h_max = np.max(h_feat) if np.max(h_feat) > 0 else 1.0
    v_max = np.max(v_feat) if np.max(v_feat) > 0 else 1.0
    h_feat = h_feat / h_max
    v_feat = v_feat / v_max
    return np.concatenate([h_feat, v_feat])

def zhang_suen_thinning(img):
    """Zhang-Suen细化算法"""
    img = (img > 0).astype(np.uint8)
    rows, cols = img.shape
    while True:
        changed = False
        markers = np.zeros_like(img, dtype=bool)
        # 子迭代1
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if img[i,j] == 0: continue
                p2 = img[i-1,j]
                p3 = img[i-1,j+1]
                p4 = img[i,j+1]
                p5 = img[i+1,j+1]
                p6 = img[i+1,j]
                p7 = img[i+1,j-1]
                p8 = img[i,j-1]
                p9 = img[i-1,j-1]
                neighbours = [p2,p3,p4,p5,p6,p7,p8,p9]
                non_zero = sum(neighbours)
                if not (2 <= non_zero <= 6): continue
                transitions = sum((neighbours[k]==0 and neighbours[(k+1)%8]==1) for k in range(8))
                if transitions != 1: continue
                if (p2 * p4 * p6 == 0) and (p4 * p6 * p8 == 0):
                    markers[i,j] = True
                    changed = True
        img[markers] = 0
        # 子迭代2
        markers.fill(False)
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if img[i,j] == 0: continue
                p2 = img[i-1,j]
                p3 = img[i-1,j+1]
                p4 = img[i,j+1]
                p5 = img[i+1,j+1]
                p6 = img[i+1,j]
                p7 = img[i+1,j-1]
                p8 = img[i,j-1]
                p9 = img[i-1,j-1]
                neighbours = [p2,p3,p4,p5,p6,p7,p8,p9]
                non_zero = sum(neighbours)
                if not (2 <= non_zero <= 6): continue
                transitions = sum((neighbours[k]==0 and neighbours[(k+1)%8]==1) for k in range(8))
                if transitions != 1: continue
                if (p2 * p4 * p8 == 0) and (p2 * p6 * p8 == 0):
                    markers[i,j] = True
                    changed = True
        img[markers] = 0
        if not changed: break
    return img * 255

def count_endpoints_and_branchpoints(skeleton):
    skeleton_bin = (skeleton > 0).astype(np.uint8)
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbours_count = cv2.filter2D(skeleton_bin, -1, kernel)
    endpoints = float(np.sum((skeleton_bin == 1) & (neighbours_count == 1)))
    branchpoints = float(np.sum((skeleton_bin == 1) & (neighbours_count >= 3)))
    return endpoints, branchpoints

def extract_chinese_features(img):
    """提取汉字完整形态学特征向量（字典形式）"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    holes = float(count_holes(img))
    skeleton = zhang_suen_thinning(img)
    endpoints, branchpoints = count_endpoints_and_branchpoints(skeleton)
    grid = grid_features(img, 5, 5)
    line_hist = direction_line_histogram(img)
    proj = projection_features(img, 8)
    return {
        'holes': holes,
        'endpoints': endpoints,
        'crossings': branchpoints,
        'grid': grid,
        'line_hist': line_hist,
        'proj': proj,
    }

def chinese_weighted_distance(feat1, feat2):
    """形态学特征加权欧氏距离"""
    hole_diff = abs(feat1['holes'] - feat2['holes']) * CHINESE_WEIGHT_HOLE
    ep_dist = (abs(feat1['endpoints'] - feat2['endpoints']) +
               abs(feat1['crossings'] - feat2['crossings'])) * CHINESE_WEIGHT_ENDPOINT
    grid_dist = np.linalg.norm(feat1['grid'] - feat2['grid']) * CHINESE_WEIGHT_GRID
    line_dist = np.linalg.norm(feat1['line_hist'] - feat2['line_hist']) * CHINESE_WEIGHT_LINE
    proj_dist = np.linalg.norm(feat1['proj'] - feat2['proj']) * CHINESE_WEIGHT_PROJ
    return hole_diff + ep_dist + grid_dist + line_dist + proj_dist

# ======================== 8. 汉明距离 ========================

def hamming_distance(img1, img2):
    """计算两幅二值图像的汉明距离（不同像素个数）"""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    diff = cv2.bitwise_xor(img1, img2)
    return np.sum(diff == 255)

# ======================== 9. 汉字识别（融合） ========================

def recognize_province_char(char_img, province_templates, template_chinese_features,
                            w_morph=MORPH_WEIGHT, w_hamming=HAMMING_WEIGHT):
    """
    汉字识别：融合形态学加权距离 + 汉明距离
    char_img: 归一化后的二值图 (40x20)
    province_templates: 省份模板图像字典
    template_chinese_features: 省份模板的形态学特征字典
    """
    if not province_templates:
        return '?'

    # 提取待识别汉字的形态学特征
    char_feat = extract_chinese_features(char_img)

    # 收集所有候选的形态学距离和汉明距离
    chars = []
    morph_dists = []
    hamming_dists = []

    for ch, tmpl_img in province_templates.items():
        if ch not in template_chinese_features:
            continue
        # 形态学距离
        morph_dist = chinese_weighted_distance(char_feat, template_chinese_features[ch])
        # 汉明距离（确保尺寸一致）
        if tmpl_img.shape != char_img.shape:
            tmpl_resized = cv2.resize(tmpl_img, (char_img.shape[1], char_img.shape[0]))
        else:
            tmpl_resized = tmpl_img
        ham_dist = hamming_distance(char_img, tmpl_resized)

        chars.append(ch)
        morph_dists.append(morph_dist)
        hamming_dists.append(ham_dist)

    if not chars:
        return '?'

    # 归一化（Min-Max）
    min_morph, max_morph = min(morph_dists), max(morph_dists)
    min_hamming, max_hamming = min(hamming_dists), max(hamming_dists)

    best_char = '?'
    best_total = float('inf')

    for i, ch in enumerate(chars):
        norm_morph = (morph_dists[i] - min_morph) / (max_morph - min_morph + 1e-8)
        norm_hamming = (hamming_dists[i] - min_hamming) / (max_hamming - min_hamming + 1e-8)
        total = w_morph * norm_morph + w_hamming * norm_hamming
        if total < best_total:
            best_total = total
            best_char = ch

    return best_char

# ======================== 10. 模板加载 ========================

def load_templates(dir_path):
    """加载模板，预计算特征，自动分类"""
    templates_all = {}
    templates_province = {}
    templates_city = {}
    templates_normal = {}
    template_features = {}           # HOG特征（字母数字用）
    template_chinese_features = {}   # 形态学特征（汉字用）

    if not os.path.isdir(dir_path):
        print(f"模板目录不存在: {dir_path}")
        return (templates_all, templates_province, templates_city,
                templates_normal, template_features, template_chinese_features)

    for file in Path(dir_path).iterdir():
        if file.suffix.lower() in ('.pgm', '.png', '.jpg', '.jpeg', '.bmp'):
            char = file.stem

            with open(file, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"无法解码: {file.name}")
                continue

            _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            white_ratio = np.sum(img_bin == 255) / img_bin.size
            if white_ratio > 0.5:
                img_bin = 255 - img_bin

            src_h, src_w = img_bin.shape
            if src_h == 0 or src_w == 0:
                continue

            scale = min(CHAR_WIDTH / src_w, CHAR_HEIGHT / src_h)
            new_w = max(int(src_w * scale), 1)
            new_h = max(int(src_h * scale), 1)
            scaled = cv2.resize(img_bin, (new_w, new_h),
                                interpolation=cv2.INTER_AREA)

            canvas = np.zeros((CHAR_HEIGHT, CHAR_WIDTH), dtype=np.uint8)
            x_off = (CHAR_WIDTH - new_w) // 2
            y_off = (CHAR_HEIGHT - new_h) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = scaled

            templates_all[char] = canvas

    # 预计算特征 + 分类
    for ch, img in templates_all.items():
        template_features[ch] = extract_hog_features(img)

        if ch in PROVINCE_CHARS:
            template_chinese_features[ch] = extract_chinese_features(img)
            templates_province[ch] = img

        if ch in CITY_LETTERS:
            templates_city[ch] = img
            templates_normal[ch] = img
        if ch in LETTERS_DIGITS:
            templates_normal[ch] = img

    print(f"共加载 {len(templates_all)} 个模板")
    print(f"  省份模板: {len(templates_province)} 个: {list(templates_province.keys())}")
    print(f"  城市代号模板: {len(templates_city)} 个: {list(templates_city.keys())}")
    print(f"  字母数字模板: {len(templates_normal)} 个")
    print(f"  汉字形态学特征: {len(template_chinese_features)} 个")

    return (templates_all, templates_province, templates_city,
            templates_normal, template_features, template_chinese_features)

# ======================== 11. 字母数字识别 ========================

def recognize_char(char_img, templates, template_features):
    if not templates:
        return '?'

    scores = []
    for ch, tmpl in templates.items():
        if char_img.shape != tmpl.shape:
            tmpl_resized = cv2.resize(tmpl,
                                      (char_img.shape[1], char_img.shape[0]),
                                      interpolation=cv2.INTER_AREA)
        else:
            tmpl_resized = tmpl
        res = cv2.matchTemplate(char_img, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        score = float(res[0][0])
        scores.append((ch, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    best_char, best_score = scores[0]

    need_hog_check = False

    if best_score < HOG_CHECK_THRESH:
        need_hog_check = True
    elif len(scores) > 1:
        second_char, second_score = scores[1]
        pair = (best_char, second_char)
        if pair in AMBIGUOUS_PAIRS and (best_score - second_score) < AMBIGUOUS_GAP:
            need_hog_check = True

    if not need_hog_check:
        if best_score < NCC_THRESHOLD:
            return '?'
        return best_char

    char_feature = extract_hog_features(char_img)

    best_by_hog = best_char
    best_hog_dist = float('inf')

    top_n = max(len(scores) // 2, 3)
    for ch, ncc_score in scores[:top_n]:
        if ch not in template_features:
            continue
        dist = np.linalg.norm(char_feature - template_features[ch])
        if dist < best_hog_dist:
            best_hog_dist = dist
            best_by_hog = ch

    return best_by_hog


# ======================== 12. 车牌识别主入口 ========================
def recognize_plate(img_bgr, templates_all, templates_province,
                    templates_city, templates_normal,
                    template_features, template_chinese_features):
    """
    车牌识别主入口（固定比例裁剪在二值化之前）
    返回: (plate_number, plate_bounds, char_images_list)
    """
    plate_bounds = find_plate_by_blue(img_bgr)
    if plate_bounds is None:
        return None, None, []
    top, bottom, left, right = plate_bounds

    # 提取车牌区域彩色图像
    plate_color = img_bgr[top:bottom+1, left:right+1]
    h, w = plate_color.shape[:2]

    # ========== 无条件固定比例裁剪（在二值化之前） ==========
    # 修改以下数值即可调整裁剪区域
    LEFT_CROP = 0.03    # 左裁剪比例
    RIGHT_CROP = 0.97   # 右裁剪比例
    TOP_CROP = 0.15     # 顶部裁剪比例
    BOTTOM_CROP = 0.85  # 底部裁剪比例

    crop_left = int(w * LEFT_CROP)
    crop_right = int(w * RIGHT_CROP)
    crop_top = int(h * TOP_CROP)
    crop_bottom = int(h * BOTTOM_CROP)
    crop_left = max(0, crop_left)
    crop_right = min(w-1, crop_right)
    crop_top = max(0, crop_top)
    crop_bottom = min(h-1, crop_bottom)

    cropped_color = plate_color[crop_top:crop_bottom+1, crop_left:crop_right+1]

    # 对裁剪后的图像进行二值化
    bin_plate = binarize_plate_cropped(cropped_color)

    # 字符分割
    proj = vertical_projection(bin_plate)
    char_boundaries = segment_chars_by_gap(bin_plate, proj)

    if len(char_boundaries) < 5:
        return None, (top, bottom, left, right), []

    char_boundaries = char_boundaries[:7]

    position_templates = [
        templates_province,
        templates_city,
        templates_normal,
        templates_normal,
        templates_normal,
        templates_normal,
        templates_normal,
    ]

    plate_number = ""
    char_images = []
    for idx, (left_c, right_c) in enumerate(char_boundaries):
        char_roi = bin_plate[:, left_c:right_c + 1]
        char_norm = normalize_char(char_roi)
        if char_norm is None:
            plate_number += '?'
            char_images.append(None)
            continue
        char_images.append(char_norm)
        if idx == 0:
            ch = recognize_province_char(char_norm, templates_province,
                                         template_chinese_features)
        else:
            if idx < len(position_templates):
                tmpl_set = position_templates[idx]
                if not tmpl_set:
                    tmpl_set = templates_all
            else:
                tmpl_set = templates_all
            ch = recognize_char(char_norm, tmpl_set, template_features)
        plate_number += ch

    return plate_number, (top, bottom, left, right), char_images
