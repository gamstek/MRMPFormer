# utils/predict_utils.py

import sys
import os
import re
import io
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from natsort import natsorted
import torchvision.transforms as T
from PIL import Image
import glob
import matplotlib.pyplot as plt

from utils.torch_device import resolve_torch_device, load_torch_checkpoint

# 图像预处理：只做 ToTensor + Normalize，不做 Resize
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def rescale_bboxes(out_bbox, size, device):
    """
    将归一化的 bbox 坐标 (cx, cy, w, h) 转换为原始图像像素坐标 (x1, y1, x2, y2)
    """
    img_w, img_h = size
    b = out_bbox.to(device)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    b[:, :2] -= b[:, 2:] / 2  # cx,cy -> x1,y1
    b[:, 2:] += b[:, :2]      # w,h -> x2,y2
    return b


def predict(images_path, model, transform, threshold=0.9, device='cpu', verbose=False, return_all=False):
    """
    对 ROI 图像列表进行预测。verbose=False 时不打印逐图 DEBUG，加快运行。
    return_all=False: 仅当某张图存在置信度 > threshold 的检测时才追加结果（与 newtest 兼容）
    return_all=True:  每张图都返回结果，无检测时 boxes/scores 为空（用于 plot 时生成全部图像）
    """
    predict_results = []

    for img_path in images_path:
        with Image.open(img_path).convert('RGB') as im:
            if verbose:
                print(f"[DEBUG] Processing: {os.path.basename(img_path)}")
                print(f"[DEBUG] Image size (W,H): {im.size}")
            
            # 预处理：转 tensor + 归一化
            img_tensor = transform(im).unsqueeze(0).to(device)  # [1, C, H, W]
            
            # 模型推理
            with torch.no_grad():
                outputs = model(img_tensor)
            
            # 解析输出
            pred_logits = outputs['pred_logits']  # [1, num_queries, 2]
            pred_boxes = outputs['pred_boxes']    # [1, num_queries, 4]
            
            # 计算置信度（排除背景类；logits 布局为 [背景, 峰1, 峰2, ...]，
            # 取 [..., 1:] 得到所有真实类别，不要用 :-1 —— 那会取到背景列导致检测恒为空）
            probas = pred_logits.softmax(-1)[0, :, 1:]  # [num_queries, num_classes]
            keep = probas.max(-1).values > threshold     # [num_queries]
            
            if verbose:
                max_conf = probas.max().item()
                print(f"[DEBUG] Max confidence: {max_conf:.6f}, detections: {keep.sum().item()}")
            
            if keep.any():
                # 获取有效检测
                boxes = pred_boxes[0, keep].cpu()
                scores = probas[keep].cpu().squeeze(-1)
                boxes = rescale_bboxes(boxes, im.size, device='cpu')
                result = {
                    'boxes': boxes.numpy(),
                    'scores': scores.numpy(),
                    'image_path': img_path
                }
                predict_results.append(result)
            elif return_all:
                result = {
                    'boxes': np.empty((0, 4)),
                    'scores': np.empty((0, 1)),
                    'image_path': img_path
                }
                predict_results.append(result)
    
    return predict_results


def plot_results(
    results,
    save_dir="predicted_plots",
    show_no_detection_label=True,
    out_filenames=None,
):
    """
    可视化预测结果。无检测时也保存图像，并标注 "No detection"。

    out_filenames: 可选，与 results 等长；每项为仅文件名（如 chrom_0001_pred.png），
    保存到 save_dir。用于批量任务时按源文件命名、统一目录。
    """
    os.makedirs(save_dir, exist_ok=True)
    if out_filenames is not None and len(out_filenames) != len(results):
        raise ValueError("out_filenames length must match results")
    for i, res in enumerate(results):
        img_path = res['image_path']
        img = Image.open(img_path).convert('RGB')
        plt.figure(figsize=(6, 4))
        plt.imshow(img)
        ax = plt.gca()
        boxes = res.get('boxes', np.empty((0, 4)))
        scores = res.get('scores', np.empty((0, 1)))
        if len(boxes) > 0:
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, color='red', linewidth=2
                )
                ax.add_patch(rect)
                plt.text(x1, y1, f"{score:.2f}", color='white', backgroundcolor='red')
        elif show_no_detection_label:
            plt.text(10, 20, "No detection", color='orange', fontsize=12, weight='bold',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        if out_filenames is not None:
            fn = out_filenames[i]
            if not fn.lower().endswith(".png"):
                fn = fn + ".png"
            out_path = os.path.join(save_dir, fn)
        else:
            base_name = os.path.basename(img_path)
            out_path = os.path.join(
                save_dir,
                base_name.replace(".jpeg", "_pred.png").replace(".jpg", "_pred.png"),
            )
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
        buf.seek(0)
        with open(out_path, 'wb') as f:
            f.write(buf.read())
        print(f"[INFO] Saved prediction plot: {out_path}")


# utils/predict_utils.py (build_predictor 函数部分)

# utils/predict_utils.py (build_predictor 函数部分)

def build_predictor(
    model_path,
    images_path,
    threshold=0.99,
    plot=False,
    plot_dir="predicted_plots",
    verbose=False,
    plot_out_filenames=None,
):
    device = resolve_torch_device(verbose=True)

    # ========== 加载 checkpoint ==========
    checkpoint = load_torch_checkpoint(model_path, map_location=device)
    print(f"[INFO] Checkpoint keys: {list(checkpoint.keys())}")
    
    # 提取模型权重和参数
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint)
        train_args = checkpoint.get("args", None)
    else:
        state_dict = checkpoint
        train_args = None
    
    # ========== 创建模型结构 ==========
    if train_args is not None:
        train_args.device = str(device)
        model = None
        
        try:
            from models.quanformer.detr import build
            result = build(train_args)  # 返回 (model, criterion, postprocessors)
            
            # ✅ 关键：从元组中提取模型
            if isinstance(result, tuple):
                model = result[0]  # 第一个元素是模型
                print(f"[INFO] Model extracted from tuple (length: {len(result)})")
            else:
                model = result
                
        except Exception as e:
            print(f"[WARN] Failed to build model: {e}")
    else:
        print("[ERROR] No args in checkpoint!")
        return []
    
    # ========== 加载权重 ==========
    if model is not None and hasattr(model, 'load_state_dict'):
        try:
            model.load_state_dict(state_dict, strict=False)
            print("[INFO] Model weights loaded successfully!")
        except Exception as e:
            print(f"[WARN] Failed to load weights: {e}")
    else:
        print("[ERROR] Model is None or doesn't have load_state_dict!")
        return []
    
    # 设置评估模式
    try:
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"[WARN] Failed to set eval mode: {e}")
    
    print(f"[INFO] Model ready from: {model_path}")
    
    # ========== 查找图像文件（按无扩展名基名去重，同一 ROI 只推理一次，避免 .jpg/.jpeg 重复） ==========
    jpg_images = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
    jpeg_images = sorted(glob.glob(os.path.join(images_path, "*.jpeg")))
    jpeg_images_sub = sorted(glob.glob(os.path.join(images_path, "**/*.jpg"), recursive=True))
    jpeg_images_sub2 = sorted(glob.glob(os.path.join(images_path, "**/*.jpeg"), recursive=True))
    
    all_paths = list(set(jpg_images + jpeg_images + jpeg_images_sub + jpeg_images_sub2))
    # 同一 base stem 只保留一个：优先 .jpeg（与原项目一致），否则 .jpg
    by_stem = {}
    for p in all_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in by_stem or p.lower().endswith(".jpeg"):
            by_stem[stem] = p
    # N_mz 命名：同一 N 只保留一张图（避免 1_mz142 与 1_mz163 等重复 N 导致对齐错乱）
    n_mz_pattern = re.compile(r"^(\d+)_mz", re.IGNORECASE)
    by_n = {}
    other_paths = []
    for p in natsorted(by_stem.values()):
        stem = os.path.splitext(os.path.basename(p))[0]
        m = n_mz_pattern.match(stem)
        if m:
            n = int(m.group(1))
            if n not in by_n:
                by_n[n] = p
        else:
            other_paths.append(p)
    image_paths = [by_n[k] for k in sorted(by_n.keys())] + other_paths

    print(f"[INFO] Found {len(image_paths)} ROI images (by stem + N_mz dedup, {len(all_paths)} files on disk).")
    if len(image_paths) == 0:
        print("[WARN] No valid ROI images found.")
        return []
    
    # ========== 执行预测 ==========
    return_all = plot
    results = predict(image_paths, model, transform, threshold, device, verbose=verbose, return_all=return_all)
    n_with_det = sum(1 for r in results if len(r.get('boxes', [])) > 0)
    print(f"[INFO] Detected peaks in {n_with_det} images (total {len(results)}).")
    
    # ========== 可视化（plot=True 时对所有图像生成标注图，无检测的标 "No detection"） ==========
    if plot and len(results) > 0:
        plot_results(results, save_dir=plot_dir, out_filenames=plot_out_filenames)
    
    return results