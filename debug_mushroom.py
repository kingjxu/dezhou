"""调试蘑菇数量识别"""
import base64
import cv2
import numpy as np
from pathlib import Path

# 添加项目根目录到 Python 路径
import sys
sys.path.insert(0, str(Path(__file__).parent))

from app.engines import get_recognizer


def debug_mushroom(image_path: str):
    # 加载图片
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"图片尺寸: {w} x {h}")
    
    # 获取识别器
    recognizer = get_recognizer("poler")
    
    # 获取当前蘑菇 ROI 配置
    roi = recognizer._layout.mushroom_roi
    print(f"\n当前 mushroom_roi: {roi}")
    x1, y1 = int(roi[0] * w), int(roi[1] * h)
    x2, y2 = int(roi[2] * w), int(roi[3] * h)
    print(f"实际像素范围: ({x1}, {y1}) ~ ({x2}, {y2})")
    
    # 裁剪并显示 ROI 区域
    crop = img[y1:y2, x1:x2]
    print(f"\nROI 区域大小: {crop.shape[1]} x {crop.shape[0]}")
    
    # 运行 OCR 并查看结果
    print("\n左上角区域 OCR 结果:")
    items = recognizer._ocr_full(img)
    for it in items:
        if roi[0] <= it.cx <= roi[2] and roi[1] <= it.cy <= roi[3]:
            px = int(it.cx * w)
            py = int(it.cy * h)
            print(f"  位置: ({px}, {py}), 文本: {it.text!r}, 置信度: {it.score:.4f}")
    
    # 尝试不同的 ROI 范围
    print("\n尝试扩大 ROI 范围:")
    test_rois = [
        (0.02, 0.02, 0.25, 0.12),   # 扩大范围
        (0.00, 0.00, 0.20, 0.10),   # 更宽的范围
        (0.01, 0.01, 0.15, 0.06),   # 缩小范围
    ]
    
    for i, test_roi in enumerate(test_rois):
        x1t, y1t = int(test_roi[0] * w), int(test_roi[1] * h)
        x2t, y2t = int(test_roi[2] * w), int(test_roi[3] * h)
        print(f"\n  ROI[{i}]: {test_roi} ({x1t},{y1t})~({x2t},{y2t})")
        for it in items:
            if test_roi[0] <= it.cx <= test_roi[2] and test_roi[1] <= it.cy <= test_roi[3]:
                px = int(it.cx * w)
                py = int(it.cy * h)
                print(f"    位置: ({px}, {py}), 文本: {it.text!r}, 置信度: {it.score:.4f}")


if __name__ == "__main__":
    debug_mushroom("/home/ubuntu/poker/dezhou/16.jpg")