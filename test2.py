# ============================================================
# test_webcam_debug.py
# 多线程版：按键最高优先级，识别不阻塞
# ============================================================

import cv2
import sys
import numpy as np
import time
from collections import Counter
import threading
from queue import Queue

from license_plate_recognizer import (
    load_templates,
    CHAR_WIDTH,
    CHAR_HEIGHT,
    recognize_plate
)

# ======================== 配置 ========================
PGM_TEMPLATE_FOLDER = "./pgm_output"
VOTE_FRAMES = 20
VOTE_CONFIDENCE = 0.6
SHOW_SEG_CHARS = True
DEBUG_IMAGE_PATH = None   # 单张图片调试路径，设为 None 则使用摄像头

print("正在加载字符模板...")
(templates_all, templates_province, templates_city,
 templates_normal, template_features, template_chinese_features) = load_templates(PGM_TEMPLATE_FOLDER)
if not templates_all:
    print(f"错误：未能从 {PGM_TEMPLATE_FOLDER} 加载任何模板。")
    sys.exit(1)


def draw_text(img, text, pos, color=(0, 255, 0), scale=0.6):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def safe_destroy_window(window_name):
    try:
        cv2.getWindowImageRect(window_name)
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass


def show_segmented_chars(char_images, window_name="Segmented Characters"):
    if not char_images:
        return
    h, w = CHAR_HEIGHT, CHAR_WIDTH
    canvas = np.ones((h, len(char_images) * w), dtype=np.uint8) * 255
    for i, img in enumerate(char_images):
        if img is not None:
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            canvas[0:h, i*w:(i+1)*w] = img
    cv2.imshow(window_name, canvas)
    cv2.waitKey(1)


def vote_characters(results_list):
    if not results_list:
        return "", []
    min_len = min(len(r) for r in results_list)
    if min_len == 0:
        return "", []
    final = ""
    confidences = []
    for pos in range(min_len):
        chars_at_pos = [r[pos] for r in results_list if len(r) > pos]
        counter = Counter(chars_at_pos)
        best_char, best_count = counter.most_common(1)[0]
        confidence = best_count / len(chars_at_pos)
        final += best_char
        confidences.append(confidence)
    return final, confidences


def recognize_and_collect(frame):
    """识别车牌并返回 (plate_number, plate_bounds, char_images)"""
    return recognize_plate(frame, templates_all, templates_province,
                           templates_city, templates_normal,
                           template_features, template_chinese_features)


def process_image(image_path, output_path=None):
    """处理单张图片（调试用）"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None
    plate_number, plate_bounds, char_images = recognize_and_collect(img)
    if plate_number is None:
        print("未检测到车牌")
        return None
    print(f"车牌号码: {plate_number}")
    if SHOW_SEG_CHARS and char_images:
        show_segmented_chars(char_images)
        cv2.waitKey(0)
        safe_destroy_window("Segmented Characters")
    if plate_bounds:
        top, bottom, left, right = plate_bounds
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, plate_number, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"结果已保存至: {output_path}")
    else:
        cv2.imshow("License Plate Recognition", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return plate_number


def camera_mode():
    """摄像头模式：多线程，按键优先"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头。")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 线程间共享数据
    result_queue = Queue(maxsize=1)   # 存储最新识别结果
    stop_thread = threading.Event()
    current_frame = None
    frame_lock = threading.Lock()

    def recognition_worker():
        """后台识别线程"""
        while not stop_thread.is_set():
            with frame_lock:
                if current_frame is not None:
                    frame_copy = current_frame.copy()
                else:
                    frame_copy = None
            if frame_copy is not None:
                res = recognize_and_collect(frame_copy)
                # 只保留最新结果
                if result_queue.empty():
                    result_queue.put(res)
                else:
                    # 覆盖旧结果
                    try:
                        result_queue.get_nowait()
                    except:
                        pass
                    result_queue.put(res)
            time.sleep(0.01)   # 避免空转

    worker = threading.Thread(target=recognition_worker, daemon=True)
    worker.start()

    print("摄像头已启动，按 'q' 退出，按 'd' 切换调试窗口")
    print("按 'r' 开始投票识别，按 'a' 切换自动模式，按 'c' 切换分割显示")
    show_debug = True
    auto_mode = False
    prev_time = time.time()
    vote_results = []
    vote_bounds = []
    final_plate = ""
    final_confidences = []
    collecting = False
    show_seg = SHOW_SEG_CHARS

    # 显示缓存
    last_result = None
    last_bounds = None
    last_chars = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 更新当前帧（供工作线程使用）
        with frame_lock:
            current_frame = frame

        # 主线程优先处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_thread.set()
            break
        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyAllWindows()
        elif key == ord('c'):
            show_seg = not show_seg
            if not show_seg:
                safe_destroy_window("Segmented Characters")
            print(f"分割字符显示: {'开启' if show_seg else '关闭'}")
        elif key == ord('r'):
            # 立即重置投票，开始收集
            vote_results = []
            vote_bounds = []
            final_plate = ""
            final_confidences = []
            collecting = True
            print(f"开始收集 {VOTE_FRAMES} 帧...")
        elif key == ord('a'):
            auto_mode = not auto_mode
            vote_results = []
            vote_bounds = []
            collecting = False
            print(f"自动模式: {'开启' if auto_mode else '关闭'}")
        elif key == ord('s'):
            cv2.imwrite("debug_frame.jpg", frame)
            print("已保存")

        # 获取最新识别结果（非阻塞）
        if not result_queue.empty():
            last_result, last_bounds, last_chars = result_queue.get()
        single_result, single_bounds, char_images = last_result, last_bounds, last_chars

        # 显示分割字符
        if show_seg and char_images:
            show_segmented_chars(char_images)
        elif show_seg and not char_images:
            safe_destroy_window("Segmented Characters")

        # 投票逻辑
        trigger_vote = False
        if auto_mode:
            trigger_vote = True
        if collecting:
            trigger_vote = True

        display = frame.copy()
        if trigger_vote and single_result is not None:
            vote_results.append(single_result)
            vote_bounds.append(single_bounds)
            collecting = True
            if len(vote_results) >= VOTE_FRAMES:
                final_plate, final_confidences = vote_characters(vote_results)
                if show_seg and char_images:
                    show_segmented_chars(char_images)
                if vote_bounds[-1] is not None:
                    top, bottom, left, right = vote_bounds[-1]
                    cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
                    avg_conf = np.mean(final_confidences) if final_confidences else 0
                    color = (0, 255, 0) if avg_conf >= VOTE_CONFIDENCE else (0, 165, 255)
                    draw_text(display, f"Plate: {final_plate}", (left, top - 20), color)
                    draw_text(display, f"Conf: {avg_conf:.0%} ({VOTE_FRAMES} frames)",
                              (left, top - 45), color, 0.45)
                    if show_debug and final_confidences:
                        for i, c in enumerate(final_confidences):
                            cx = left + i * 30
                            conf_color = (0, 255, 0) if c >= 0.6 else (0, 0, 255)
                            cv2.putText(display, f"{c:.0%}", (cx, bottom + 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, conf_color, 1)
                print(f"投票结果: {final_plate}  置信度: {final_confidences}")
                if auto_mode:
                    vote_results = []
                    vote_bounds = []
                    collecting = False
                    time.sleep(1)
                else:
                    collecting = False
        elif single_bounds is not None and not collecting:
            top, bottom, left, right = single_bounds
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)
            if single_result:
                draw_text(display, f"Single: {single_result}", (left, top - 20), (0, 165, 255), 0.5)
                draw_text(display, "Press 'r' to vote", (left, top - 45), (200, 200, 200), 0.4)
        elif not collecting:
            draw_text(display, "No blue plate", (10, 30), (0, 0, 255))

        # 状态栏
        mode_str = "AUTO" if auto_mode else "MANUAL"
        collect_str = f"Collecting {len(vote_results)}/{VOTE_FRAMES}" if collecting else ""
        status = f"Mode: {mode_str}  {collect_str}"
        draw_text(display, status, (10, 20), (255, 255, 255), 0.45)
        if show_debug and final_plate:
            draw_text(display, f"Result: {final_plate}", (10, display.shape[0] - 35), (0, 255, 0))
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        draw_text(display, f"FPS: {fps:.1f}", (10, display.shape[0] - 10), (255, 255, 255), 0.45)

        cv2.imshow("LPR Debug", display)

    stop_thread.set()
    cap.release()
    cv2.destroyAllWindows()


def main():
    if DEBUG_IMAGE_PATH is not None:
        process_image(DEBUG_IMAGE_PATH, None)
    else:
        camera_mode()


if __name__ == "__main__":
    main()