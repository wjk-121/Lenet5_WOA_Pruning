import os
import csv
import time

# 防止 OpenMP 冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from config import MODEL_DIR, REPORT_DIR, TEST_IMG_DIR
from models.lenet import LeNet5, LeNet5Pruned

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

def _available_models():
    """扫描可用模型，按固定顺序展示；只展示存在的权重文件"""
    candidates = [
        ("baseline", os.path.join(MODEL_DIR, "baseline.pth")),
        ("ratio20", os.path.join(MODEL_DIR, "pruned_ratio20.pth")),
        ("ratio30", os.path.join(MODEL_DIR, "pruned_ratio30.pth")),
        ("ratio40", os.path.join(MODEL_DIR, "pruned_ratio40.pth")),
        ("ratio50", os.path.join(MODEL_DIR, "pruned_ratio50.pth")),
        ("woa",     os.path.join(MODEL_DIR, "pruned_woa.pth")),
    ]
    return [(tag, p) for tag, p in candidates if os.path.isfile(p)]

def _prompt_choice(title, options, default_idx=1):
    """
    通用选择器：options 为 [(key,label)] 或 [label]
    返回选择的 key（或 label）
    """
    print(title)
    for i, opt in enumerate(options, 1):
        if isinstance(opt, (tuple, list)) and len(opt) == 2:
            print(f"  {i}) {opt[1]}")
        else:
            print(f"  {i}) {opt}")
    raw = input(f"请输入序号(默认 {default_idx}): ").strip()
    idx = default_idx
    if raw.isdigit():
        v = int(raw)
        if 1 <= v <= len(options):
            idx = v
    chosen = options[idx - 1]
    if isinstance(chosen, (tuple, list)) and len(chosen) == 2:
        return chosen[0]
    return chosen

def _load_checkpoint(tag: str, device):
    """按标签加载模型到指定设备"""
    if tag == "baseline":
        path = os.path.join(MODEL_DIR, "baseline.pth")
    elif tag == "woa":
        path = os.path.join(MODEL_DIR, "pruned_woa.pth")
    elif tag.startswith("ratio"):
        path = os.path.join(MODEL_DIR, f"pruned_{tag}.pth")
    else:
        raise ValueError(f"未知模型标签: {tag}")

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

def _pick_first_image(root_dir: str) -> str:
    """在 root_dir 下选择第一张图片（不递归）"""
    if not os.path.isdir(root_dir):
        return None
    for fn in sorted(os.listdir(root_dir)):
        fp = os.path.join(root_dir, fn)
        if os.path.isfile(fp) and fn.lower().endswith(IMG_EXTS):
            return fp
    return None

def _list_images(img_dir: str):
    """列出目录内全部图片文件（不递归）"""
    if not os.path.isdir(img_dir):
        return []
    files = []
    for fn in sorted(os.listdir(img_dir)):
        fp = os.path.join(img_dir, fn)
        if os.path.isfile(fp) and fn.lower().endswith(IMG_EXTS):
            files.append(fp)
    return files

def run_single(tag: str, invert: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_checkpoint(tag, device)

    img_path = _pick_first_image(str(TEST_IMG_DIR))
    if img_path is None:
        print(f"未在 {TEST_IMG_DIR} 找到图片（支持: {IMG_EXTS}）。请放入至少一张图片。")
        return

    print(f"\n[单图] 选择图片: {img_path}")
    print(f"[单图] 使用模型: {tag}  设备: {device}")

    tfm = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img = Image.open(img_path).convert("L")
    if invert:
        img = ImageOps.invert(img)
        print("[单图] 已启用反相(黑底白字->白底黑字)")

    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(1).item())
        prob = float(torch.softmax(logits, dim=1)[0, pred].item())

    print(f"[单图] 预测结果: 识别结果={pred}, 置信度={prob:.4f}")

def run_batch(tag: str, invert: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_checkpoint(tag, device)

    img_dir = os.path.join(str(TEST_IMG_DIR), "text")
    files = _list_images(img_dir)
    if not files:
        print(f"未在 {img_dir} 找到图片（支持: {IMG_EXTS}）。请创建 test_imgs\\text 并放入图片。")
        return

    print(f"\n[批量] 读取目录: {img_dir}  共{len(files)}张")
    print(f"[批量] 使用模型: {tag}  设备: {device}")
    if invert:
        print("[批量] 已启用反相(黑底白字->白底黑字)")

    tfm = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    rows = []
    t0 = time.time()
    for i, fp in enumerate(files, 1):
        try:
            img = Image.open(fp).convert("L")
        except Exception as e:
            print(f"[批量] 跳过无法读取的文件: {fp}  原因: {e}")
            continue
        if invert:
            img = ImageOps.invert(img)
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(1).item())
            prob = float(torch.softmax(logits, dim=1)[0, pred].item())
        rows.append([os.path.basename(fp), pred, f"{prob:.4f}"])
        if i % 20 == 0 or i == len(files):
            print(f"[批量] 进度: {i}/{len(files)}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_csv = os.path.join(REPORT_DIR, f"batch_pred_{tag}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "pred", "prob"])
        w.writerows(rows)

    dt = time.time() - t0
    print(f"[批量] 完成: {out_csv}  样本数={len(rows)}  用时={dt:.1f}s")

def main():
    print("==== 数字预测 ====")
    # 1) 选择模式
    mode = _prompt_choice(
        "请选择模式：",
        [("single", "单图预测（从 test_imgs/ 取第一张）"),
         ("batch",  "批量预测（从 test_imgs/text/ 读取全部）")],
        default_idx=1
    )

    # 2) 选择模型（仅展示已存在的模型）
    avail = _available_models()
    if not avail:
        print("未找到可用模型权重，请先运行训练/剪枝：")
        print("  python main.py --step all   或   分步运行 train / prune")
        return
    options = [(tag, f"{tag}  ({os.path.basename(path)})") for tag, path in avail]
    model_tag = _prompt_choice("请选择模型：", options, default_idx=1)

    # 3) 是否反相
    inv_raw = input("是否反相（黑底白字->白底黑字）? [y/N]: ").strip().lower()
    invert = inv_raw in ("y", "yes", "1", "true")

    # 4) 执行
    if mode == "single":
        while True:
            run_single(model_tag, invert)
            key = input('按Enter键继续...(输入任意键退出)：')
            if key:
                break
    else:
        run_batch(model_tag, invert)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断。")