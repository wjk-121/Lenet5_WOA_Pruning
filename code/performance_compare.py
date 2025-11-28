# 汇总基线模型与剪枝模型的测试准确率、参数量与推理速度，输出对比图与报告。

import os
# 设置环境变量，解决可能的库冲突问题
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 设置OpenMP和MKL的线程数为1，确保推理速度测量的一致性
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # 用于处理JSON格式文件
import torch  # 导入PyTorch深度学习框架

# 从配置文件导入模型、图像、报告的保存目录路径
from config import MODEL_DIR, FIG_DIR, REPORT_DIR
# 从工具模块导入日志设置函数
from utils.logger import setup_logger
# 从可视化工具模块导入三个对比图绘制函数
from utils.visualize import plot_accuracy_compare, plot_params_compare, plot_speed_compare


def _load_metric(path: str):
    """加载模型的指标数据（测试准确率、参数量、推理速度）"""
    # 检查文件是否存在，不存在则返回None
    if not os.path.exists(path):
        return None
    # 加载模型文件（使用CPU设备映射，避免GPU依赖）
    ckpt = torch.load(path, map_location="cpu")
    # 提取并返回需要的指标，转换为合适的数据类型
    return {
        "test_acc": float(ckpt.get("test_acc", 0.0)),  # 测试准确率
        "params": int(ckpt.get("params", 0)),  # 参数量
        "speed": float(ckpt.get("speed", 0.0)),  # 推理速度
    }


def run_performance_compare():
    """
    汇总基线模型与剪枝模型的测试指标，绘制对比图并导出报告。
    - 若基线模型缺少测试准确率/参数量/推理速度，将现场评估并补写到baseline.pth中
    - 输出报告格式包括：JSON + CSV + Markdown（Excel格式需要pandas/openpyxl库，若缺失则跳过）
    """
    import csv  # 在本函数内局部导入，避免修改文件顶部的导入内容
    # 仅在本函数内使用的数据目录和训练配置参数
    from config import DATA_DIR, TRAIN
    # 从工具模块导入数据加载器、指标计算和模型相关函数
    from utils.dataset import get_dataloaders
    from utils.metrics import evaluate_accuracy, measure_inference_speed, count_parameters
    from models.lenet import LeNet5  # 导入LeNet5模型

    # 设置日志器，日志文件保存到图像目录的上级目录下的logs文件夹，日志名称为"compare"
    logger, log_path = setup_logger("compare", str(os.path.join(os.path.dirname(FIG_DIR), "logs")), "compare")
    # 选择计算设备（优先使用GPU，若无则使用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化存储模型方法、准确率、参数量、每秒处理图像数的列表
    methods, accs, params, ips = [], [], [], []

    # 1) 读取基线模型指标 + 若缺失则补算
    base_path = os.path.join(MODEL_DIR, "baseline.pth")  # 基线模型文件路径
    base = _load_metric(base_path)  # 加载基线模型指标

    # 若基线模型存在但指标不完整（准确率、参数量或速度为0）
    if base is not None and (base["test_acc"] == 0 or base["params"] == 0 or base["speed"] == 0):
        logger.info("baseline.ckpt 缺少完整指标，正在评估并补写...")
        # 加载原始模型文件（使用指定设备）
        raw = torch.load(base_path, map_location=device)
        # 获取模型状态字典（存储模型权重等参数）
        state = raw.get("state_dict", None)

        # 若没有状态字典，则无法补写指标
        if state is None:
            logger.info("baseline.ckpt 缺少 state_dict，无法补写，跳过基线模型对比。")
            base = None
        else:
            # 初始化LeNet5模型并加载状态字典
            model = LeNet5().to(device)
            model.load_state_dict(state)
            # 获取数据加载器（训练、验证、测试集）
            _, _, test_loader = get_dataloaders(DATA_DIR,
                                                batch_size=TRAIN["batch_size"],
                                                num_workers=TRAIN["num_workers"],
                                                pin_memory=TRAIN["pin_memory"],
                                                seed=TRAIN["seed"])
            # 计算测试准确率
            test_acc = evaluate_accuracy(model, test_loader, device)
            # 计算模型参数量
            p = count_parameters(model)
            # 测量推理速度（每秒处理图像数）
            sp = measure_inference_speed(model, test_loader, device)["images_per_sec"]

            # 将计算得到的指标补写到原始模型文件中
            raw["test_acc"] = float(test_acc)
            raw["params"] = int(p)
            raw["speed"] = float(sp)
            torch.save(raw, base_path)
            # 日志记录补写的指标
            logger.info(f"基线模型指标已补写: 测试准确率={test_acc:.4f}, 参数量={p}, 速度={sp:.1f}")
            # 更新基线模型指标字典
            base = {"test_acc": test_acc, "params": p, "speed": sp}

    # 若基线模型指标有效，则添加到列表中
    if base is not None:
        methods.append("baseline")
        accs.append(float(base["test_acc"]))
        params.append(int(base["params"]))
        ips.append(float(base["speed"]))
    else:
        logger.info("未找到 baseline.pth 或无法读取，跳过基线模型对比。")

    # 2) 处理固定比例剪枝模型
    for r in [20, 30, 40, 50]:  # 剪枝比例为20%、30%、40%、50%
        # 加载对应比例剪枝模型的指标
        ck = _load_metric(os.path.join(MODEL_DIR, f"pruned_ratio{r}.pth"))
        if ck is not None:  # 若模型存在
            methods.append(f"ratio{r}")  # 添加方法名称（如ratio20）
            accs.append(float(ck["test_acc"]))  # 添加准确率
            params.append(int(ck["params"]))  # 添加参数量
            ips.append(float(ck["speed"]))  # 添加推理速度
        else:
            logger.info(f"未找到 pruned_ratio{r}.pth，跳过该模型。")

    # 3) 处理WOA（鲸鱼优化算法）剪枝模型
    ck = _load_metric(os.path.join(MODEL_DIR, "pruned_woa.pth"))
    if ck is not None:  # 若模型存在
        methods.append("woa")  # 添加方法名称
        accs.append(float(ck["test_acc"]))
        params.append(int(ck["params"]))
        ips.append(float(ck["speed"]))

    # 4) 绘制对比图
    if methods:  # 若存在有效模型数据
        # 绘制准确率对比图并保存
        plot_accuracy_compare(methods, accs, os.path.join(FIG_DIR, "acc_compare.png"))
        # 绘制参数量对比图并保存
        plot_params_compare(methods, params, os.path.join(FIG_DIR, "params_compare.png"))
        # 绘制推理速度对比图并保存
        plot_speed_compare(methods, ips, os.path.join(FIG_DIR, "speed_compare.png"))
        logger.info("对比图已保存到 figures/ 目录下")
    else:
        logger.info("未找到任何模型指标，跳过绘图步骤。")

    # 5) 导出报告（JSON + CSV + Markdown + Excel[可选]）
    # 构建报告数据列表
    report = [
        {"method": m, "test_acc": float(a), "params": int(p), "images_per_sec": float(s)}
        for m, a, p, s in zip(methods, accs, params, ips)
    ]
    # 确保报告目录存在，不存在则创建
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 导出JSON格式报告
    rep_path = os.path.join(REPORT_DIR, "performance_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)  # 缩进 2 空格，确保中文正常显示
    logger.info(f"JSON 报告已保存: {rep_path}")

    # 导出CSV格式报告
    csv_path = os.path.join(REPORT_DIR, "performance_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # 写入表头
        w.writerow(["moxing", "test_acc", "params", "images_per_sec"])
        # 写入每行数据（格式化准确率和速度的小数位数）
        for row in report:
            w.writerow([row["method"], f"{row['test_acc']:.4f}", row["params"], f"{row['images_per_sec']:.2f}"])
    logger.info(f"CSV 报告已保存: {csv_path}")

    # 导出Markdown格式报告
    md_path = os.path.join(REPORT_DIR, "performance_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        # 写入Markdown表格表头
        f.write("| 方法 | 准确率 | 参数量 | 每秒处理图像数 |\n")
        f.write("|---|---:|---:|---:|\n")  # 表格对齐方式（准确率、参数量、速度右对齐）
        # 写入每行数据
        for row in report:
            f.write(f"| {row['method']} | {row['test_acc']:.4f} | {row['params']} | {row['images_per_sec']:.2f} |\n")
    logger.info(f"Markdown 报告已保存: {md_path}")

    # 导出Excel格式报告（可选，依赖pandas和openpyxl库）
    try:
        import pandas as pd  # 尝试导入pandas库
        xlsx_path = os.path.join(REPORT_DIR, "performance_report.xlsx")
        # 将报告数据转换为DataFrame并导出为Excel
        pd.DataFrame(report).to_excel(xlsx_path, index=False)
        logger.info(f"Excel 报告已保存: {xlsx_path}")
    except Exception as e:  # 若导入失败或导出出错
        logger.info(f"Excel 导出跳过（需要安装pandas和openpyxl库）: {e}")

    # 返回包含方法列表、报告数据和日志路径的字典
    return {"methods": methods, "report": report, "log": log_path}