import os
import time
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import Counter, deque

# 避免 OpenMP 冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from config import MODEL_DIR
from models.lenet import LeNet5, LeNet5Pruned

def _load(tag: str, device):
    if tag == "baseline":
        path = os.path.join(MODEL_DIR, "baseline.pth")
    elif tag == "woa":
        path = os.path.join(MODEL_DIR, "pruned_woa.pth")
    else:
        path = os.path.join(MODEL_DIR, f"pruned_{tag}.pth")
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("arch") == "lenet5":
        model = LeNet5().to(device)
    else:
        sd = ckpt["state_dict"]
        c1 = sd["conv1.weight"].shape[0]
        c2 = sd["conv2.weight"].shape[0]
        model = LeNet5Pruned(c1, c2).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

def _open_camera(index: int):
    backend_map = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "gstreamer": cv2.CAP_GSTREAMER,
    }
    tried = []
    order = ["dshow", "msmf", "gstreamer", "default"]

    cap = None
    for b in order:
        if b == "default":
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, backend_map[b])
        tried.append(b)
        if cap.isOpened():
            return cap, b
        if cap:
            cap.release()
            cap = None
    return None, ",".join(tried)

def select_model():
    """交互式选择模型"""
    models = ["baseline", "ratio20", "ratio30", "ratio40", "ratio50", "woa"]
    model_descriptions = {
        "baseline": "原始 LeNet-5 模型 (未剪枝)",
        "ratio20": "剪枝 20% 的模型",
        "ratio30": "剪枝 30% 的模型",
        "ratio40": "剪枝 40% 的模型",
        "ratio50": "剪枝 50% 的模型",
        "woa": "使用鲸鱼优化算法优化的剪枝模型"
    }

    print("=" * 60)
    print("          手写数字识别系统 - 模型选择")
    print("=" * 60)

    for i, model in enumerate(models, 1):
        print(f"{i}. {model:10} - {model_descriptions[model]}")

    print("-" * 60)

    while True:
        try:
            choice = input("请选择模型 (1-6): ").strip()
            if not choice:
                print("使用默认模型: baseline")
                return "baseline"

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models):
                selected_model = models[choice_idx]
                print(f"已选择模型: {selected_model} - {model_descriptions[selected_model]}")
                return selected_model
            else:
                print("无效选择，请输入 1-6 之间的数字")
        except ValueError:
            print("请输入有效的数字")
        except KeyboardInterrupt:
            print("\n用户取消选择，使用默认模型")
            return "baseline"

def select_camera():
    """选择摄像头设备"""
    print("\n" + "=" * 40)
    print("摄像头选择")
    print("=" * 40)

    # 测试可用的摄像头
    available_cameras = []
    print("扫描可用摄像头...")

    for i in range(3):  # 检查前 3个摄像头
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()

    if not available_cameras:
        print("未找到可用的摄像头!")
        return None

    print(f"找到 {len(available_cameras)} 个可用摄像头:")
    for cam_idx in available_cameras:
        print(f"  摄像头 {cam_idx}")

    if len(available_cameras) == 1:
        print(f"自动选择摄像头 {available_cameras[0]}")
        return available_cameras[0]

    while True:
        try:
            choice = input(f"请选择摄像头 ({', '.join(map(str, available_cameras))}): ").strip()
            if not choice:
                print(f"使用默认摄像头 {available_cameras[0]}")
                return available_cameras[0]

            cam_idx = int(choice)
            if cam_idx in available_cameras:
                print(f"已选择摄像头 {cam_idx}")
                return cam_idx
            else:
                print(f"无效选择，请从 {available_cameras} 中选择")
        except ValueError:
            print("请输入有效的摄像头编号")
        except KeyboardInterrupt:
            print("\n用户取消选择，使用默认摄像头")
            return available_cameras[0]

def show_instructions():
    """显示使用说明"""
    image_style = "黑底白字 (MNIST风格，适合白纸黑字)"

    print("\n" + "=" * 60)
    print("                   使用说明")
    print("=" * 60)
    print(f"图像样式: {image_style}")
    print("1. 将手写数字对准摄像头中心")
    print("2. 确保数字清晰可见，背景简洁")
    print("3. 保持手部稳定以获得最佳识别效果")
    print("4. 在窗口中按以下按键:")
    print("   - 'q': 退出程序")
    print("   - 'c': 切换摄像头")
    print("   - 'm': 重新选择模型")
    print("   - 's': 显示/隐藏统计信息")
    print("=" * 60)
    input("按回车键开始识别...")

def extract_digit_roi(bw_img, out_size=28, margin=4):
    # 反色使前景白底为黑
    if np.mean(bw_img) < 125:
        thresh = bw_img.copy()
    else:
        thresh = cv2.bitwise_not(bw_img)

    # 1. 膨胀：让细线变粗，提高连通性
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    # 2. 找所有满足面积要求的轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 8]
    if not valid_contours:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    # 合并所有轮廓的整体包围框
    xs, ys, xe, ye = [], [], [], []
    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        xs.append(x)
        ys.append(y)
        xe.append(x + w)
        ye.append(y + h)
    x1, y1, x2, y2 = min(xs), min(ys), max(xe), max(ye)
    digit_roi = dilated[y1:y2, x1:x2]

    # 检查区域大小
    h, w = digit_roi.shape
    if h < 5 or w < 5 or h * w < 30:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    # 3. 缩放居中
    target_size = out_size - 2 * margin
    if w > h:
        new_w = target_size
        new_h = max(1, int(h * (target_size / w)))
    else:
        new_h = target_size
        new_w = max(1, int(w * (target_size / h)))
    digit_resized = cv2.resize(digit_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_size, out_size), dtype=np.uint8)
    start_x = (out_size - new_w) // 2
    start_y = (out_size - new_h) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = digit_resized
    return canvas

def main():
    # 交互式选择模型
    model_tag = select_model()

    # 选择摄像头
    camera_index = select_camera()
    if camera_index is None:
        print("无法找到可用的摄像头，程序退出。")
        return

    # 使用说明（默认黑底白字）
    show_instructions()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    print("加载模型中...")
    model = _load(model_tag, device)
    print(f"模型加载完成: {model_tag}")

    # 图像预处理（直接采用ToTensor、Normalize）
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 打开摄像头
    cap, backend = _open_camera(camera_index)
    if cap is None:
        print(f"无法打开摄像头 index={camera_index}")
        return

    # 获取摄像头信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"摄像头信息: {width}x{height}, 后端: {backend}")

    # 初始化变量
    frame_count = 0
    start_time = time.time()
    prediction_history = deque(maxlen=30)  # 保存最近30次预测用于统计
    show_stats = True

    # 创建窗口
    cv2.namedWindow("Real-time Digit Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-time Digit Recognition", 800, 600)

    print("\n开始实时识别...")
    print("按 'q' 退出, 'c' 切换摄像头, 'm' 重新选择模型, 's' 显示/隐藏统计信息")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("读取帧失败")
                break

            # frame = cv2.flip(frame, 1)  # 水平镜像（左变右，右变左）

            frame_count += 1

            # 转灰度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 轮廓处理区域：中心正方形区域，仅分析中心区域
            center_x, center_y = width // 2, height // 2
            box_size = min(width, height) // 7
            rx1, ry1, rx2, ry2 = (
                max(0, center_x - box_size),
                max(0, center_y - box_size),
                min(width, center_x + box_size),
                min(height, center_y + box_size)
            )
            roi_gray = gray[ry1:ry2, rx1:rx2]

            # 二值化
            _, bw = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 黑底白字，保证背景为黑、字为白，所以如果背景是白需要反色
            bg_mean = np.mean(bw)
            if bg_mean > 127:  # 背景偏白则反色
                bw = cv2.bitwise_not(bw)

            # 提取数字模拟MNIST中心化居中
            processed28 = extract_digit_roi(bw, out_size=28, margin=4)

            # 送入模型前转换为PIL图片
            pil_img = Image.fromarray(processed28)
            x = tfm(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x)
                pred = int(logits.argmax(1).item())
                prob = float(torch.softmax(logits, dim=1)[0, pred].item())

                # 获取所有类别的概率
                all_probs = torch.softmax(logits, dim=1)[0]

                # 保存到历史记录
                prediction_history.append(pred)

            # 在画面上添加信息
            color = (0, 255, 0) if prob > 0.7 else (0, 165, 255) if prob > 0.4 else (0, 0, 255)
            cv2.putText(frame, f"Prediction: {pred}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(frame, f"Confidence: {prob * 100:.1f}%", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # 2. 模型信息
            cv2.putText(frame, f"Model: {model_tag}", (20, height - 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 3. 图像样式信息
            image_style = "Black-on-White for network (已黑底白字)"
            cv2.putText(frame, f"Style: {image_style}", (20, height - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 4. FPS信息
            current_time = time.time()
            elapsed = current_time - start_time
            fps_current = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps_current:.1f}", (20, height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 5. 统计信息（可选显示）
            if show_stats and prediction_history:
                pred_counter = Counter(prediction_history)
                most_common = pred_counter.most_common(3)
                stats_text = f"Recent: {', '.join([f'{num}({count})' for num, count in most_common])}"
                cv2.putText(frame, stats_text, (20, height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # 6. 操作提示
            cv2.putText(frame, "Q:Quit  C:Camera  M:Model  S:Stats", (20, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 7. 绘制识别区域框
            cv2.rectangle(frame,
                          (rx1, ry1),
                          (rx2, ry2),
                          color, 2)
            cv2.putText(frame, "Place digit here",
                        (rx1, ry1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 8. 显示处理后的二值图像（小窗口，左上角）
            bw_vis = cv2.resize(processed28, (100, 100))
            bw_vis_color = cv2.cvtColor(bw_vis, cv2.COLOR_GRAY2BGR)
            frame[10:110, width - 110:width - 10] = bw_vis_color
            cv2.putText(frame, "Processed28x28", (width - 110, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 显示画面
            cv2.imshow("Real-time Digit Recognition", frame)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户退出")
                break
            elif key == ord('c'):
                # 切换摄像头
                cv2.destroyAllWindows()
                cap.release()
                print("切换摄像头...")
                camera_index = select_camera()
                if camera_index is None:
                    break
                cap, backend = _open_camera(camera_index)
                if cap is None:
                    print("无法打开新摄像头")
                    break
                # 重置统计
                frame_count = 0
                start_time = time.time()
                prediction_history.clear()
            elif key == ord('m'):
                # 重新选择模型
                cv2.destroyAllWindows()
                cap.release()
                print("重新选择模型...")
                model_tag = select_model()
                model = _load(model_tag, device)
                print(f"模型已切换为: {model_tag}")
                cap, backend = _open_camera(camera_index)
                if cap is None:
                    print("无法重新打开摄像头")
                    break
                # 重置统计
                frame_count = 0
                start_time = time.time()
                prediction_history.clear()
            elif key == ord('s'):
                # 显示/隐藏统计信息
                show_stats = not show_stats
                print(f"统计信息: {'显示' if show_stats else '隐藏'}")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 计算最终统计信息
        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print(f"\n运行统计:")
        print(f"  总帧数: {frame_count}")
        print(f"  总时间: {total_time:.1f}秒")
        print(f"  平均FPS: {avg_fps:.1f}")

        # if prediction_history:
        #     pred_counter = Counter(prediction_history)
        #     most_common = pred_counter.most_common(5)
        #     print(f"  预测统计: {[f'{num}({count}次)' for num, count in most_common]}")

        # 释放资源
        try:
            cap.release()
            cv2.destroyAllWindows()
            print("资源已释放")
        except Exception as e:
            print(f"释放资源时出错: {e}")

if __name__ == "__main__":
    main()