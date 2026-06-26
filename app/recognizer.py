from __future__ import annotations

import base64
import os
import re
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field

import cv2
import numpy as np

# OCR 引擎：默认 RapidOCR（部署轻、CPU 快）。
# 想切 PaddleOCR 自己装 paddle + 设 DEZHOU_OCR=paddle（见 docs/train_rank.md）。
_FORCED_OCR = os.environ.get("DEZHOU_OCR", "").lower().strip() or None
_OCR_BACKEND: str = "none"   # 会在 _get_ocr() 里被设置

# RapidOCR 三种后端任选其一（API 完全一致），按优先级探测：
#   1. rapidocr_onnxruntime — ONNX Runtime，跨平台默认
#   2. rapidocr_openvino    — Intel OpenVINO，Intel CPU 上 1.5-2.5× 快
#   3. rapidocr_paddle      — PaddlePaddle 后端（最慢，仅作兜底）
# 想强制使用某后端，把其他两个 pip uninstall 即可。
_RAPID_PKG: str | None = None
RapidOCR = None  # type: ignore
for _pkg in ("rapidocr_onnxruntime", "rapidocr_openvino", "rapidocr_paddle"):
    try:
        _mod = __import__(_pkg, fromlist=["RapidOCR"])
        RapidOCR = _mod.RapidOCR
        _RAPID_PKG = _pkg
        break
    except Exception:
        continue

# 模板匹配尺寸（所有模板 & 候选 patch 都归一化到这）
_TMPL_SIZE = (32, 48)  # (w, h)
_TMPL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rank_templates")
_RANKS_ALL = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

# CNN rank/suit 分类器（按 app 分别加载）
# 文件名约定：
#   dpzx:  rank_classifier.onnx / suit_classifier.onnx（默认，兼容旧版）
#   poler: rank_classifier_poler.onnx / suit_classifier_poler.onnx
_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_CNN_IMG_SIZE  = (32, 48)  # rank：(w, h)
_SUIT_IMG_SIZE = (32, 32)  # suit：正方形
_SUITS_ALL = ["红桃", "方块", "梅花", "黑桃"]  # = hearts, diamonds, clubs, spades

_rank_sess_cache: dict[str, object] = {}   # app_name -> session | False
_suit_sess_cache: dict[str, object] = {}

def _rank_model_path(app: str = "dpzx") -> str:
    return os.path.join(_MODEL_DIR, f"rank_classifier_{app}.onnx")

def _suit_model_path(app: str = "dpzx") -> str:
    return os.path.join(_MODEL_DIR, f"suit_classifier_{app}.onnx")

# ── GPU/CPU provider 选择 ─────────────────────────────────────────
# 默认 CPU；设 DEZHOU_GPU=1（或 cuda/gpu）启用 NVIDIA CUDA。
# 任何 NVIDIA 显卡（T1000/GTX/RTX/A 系列）走 CUDAExecutionProvider，
# 失败自动回退 CPU（onnxruntime-gpu 没装、没驱动、CUDA 版本不对都不会崩）。
def _ort_providers() -> list:
    use_gpu = os.environ.get("DEZHOU_GPU", "").lower().strip() in ("1", "cuda", "gpu", "true", "yes")
    if use_gpu:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _get_rank_classifier(app: str = "dpzx"):
    """按 app 加载 ONNX rank 分类器；未找到则返回 None。"""
    if app in _rank_sess_cache:
        v = _rank_sess_cache[app]
        return v if v is not False else None
    path = _rank_model_path(app)
    if not os.path.isfile(path):
        _rank_sess_cache[app] = False
        return None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=_ort_providers())
        _rank_sess_cache[app] = sess
        return sess
    except Exception as e:
        print(f"[rank_classifier/{app}] 加载失败: {e}")
        _rank_sess_cache[app] = False
        return None

def _get_suit_classifier(app: str = "dpzx"):
    if app in _suit_sess_cache:
        v = _suit_sess_cache[app]
        return v if v is not False else None
    path = _suit_model_path(app)
    if not os.path.isfile(path):
        _suit_sess_cache[app] = False
        return None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=_ort_providers())
        _suit_sess_cache[app] = sess
        return sess
    except Exception as e:
        print(f"[suit_classifier/{app}] 加载失败: {e}")
        _suit_sess_cache[app] = False
        return None


def _classify_suit_cnn(patch_bgr: np.ndarray, app: str = "dpzx") -> tuple[str, float] | None:
    """suit 分类器：输入 3 通道 BGR patch（保留颜色让模型区分红/黑）。"""
    sess = _get_suit_classifier(app)
    if sess is None: return None
    if patch_bgr is None or patch_bgr.size == 0: return None
    img = cv2.resize(patch_bgr, _SUIT_IMG_SIZE, interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, :, :, :]  # (1, 3, H, W)
    try:
        logits = sess.run(None, {"input": x})[0][0]
    except Exception:
        return None
    ex = np.exp(logits - logits.max())
    probs = ex / ex.sum()
    idx = int(np.argmax(probs))
    return _SUITS_ALL[idx], float(probs[idx])


def _smart_rank_grayscale(bgr: np.ndarray) -> np.ndarray:
    """智能灰度转换：红色文字用蓝通道（红色在 B 通道最暗→对比度最高），
    与训练时 _smart_grayscale 保持一致。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 70, 70]), np.array([180, 255, 255]))
    red_ratio = (cv2.countNonZero(m1) + cv2.countNonZero(m2)) / max(1, bgr.shape[0] * bgr.shape[1])
    if red_ratio > 0.02:
        return bgr[:, :, 0]
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _classify_rank_cnn(patch_bgr: np.ndarray, app: str = "dpzx") -> tuple[str, float] | None:
    """用 CNN 分类 rank；返回 (rank, confidence) 或 None。"""
    sess = _get_rank_classifier(app)
    if sess is None: return None
    if patch_bgr is None or patch_bgr.size == 0: return None
    gray = _smart_rank_grayscale(patch_bgr)
    gray = cv2.resize(gray, _CNN_IMG_SIZE, interpolation=cv2.INTER_AREA)
    x = gray.astype(np.float32) / 255.0
    x = x[None, None, :, :]  # (1, 1, H, W)
    try:
        logits = sess.run(None, {"input": x})[0][0]
    except Exception:
        return None
    ex = np.exp(logits - logits.max())
    probs = ex / ex.sum()
    idx = int(np.argmax(probs))
    return _RANKS_ALL[idx], float(probs[idx])


def _binarize_rank_upper(patch_bgr: np.ndarray):
    """rank patch 上部二值化，返回 (bw, is_hero) 或 (None, False)。"""
    if patch_bgr is None or patch_bgr.size == 0:
        return None, False
    h_full, w_full = patch_bgr.shape[:2]
    is_hero = w_full < 60
    crop_ratio = 0.75 if is_hero else 0.60
    upper = patch_bgr[:int(h_full * crop_ratio), :]
    if upper.size == 0:
        return None, is_hero
    gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
    if is_hero:
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw, is_hero


def _find_rank_component(bw: np.ndarray):
    """在二值化 rank patch 中用连通组件分离 rank 字符（排除花色符号和边框伪影）。
    返回 (rank_mask, x, y, w, h) 或 None。"""
    h, w = bw.shape

    # ── 预处理：清除左右边框伪影（高密度垂直条带） ──
    bw_clean = bw.copy()
    col_fg = np.sum(bw > 0, axis=0).astype(float) / max(1, h)
    # 左边框
    for c in range(min(w // 4, 15)):
        if col_fg[c] > 0.70:
            bw_clean[:, c] = 0
        else:
            break
    # 右边框
    for c in range(w - 1, max(w * 3 // 4, w - 15) - 1, -1):
        if col_fg[c] > 0.70:
            bw_clean[:, c] = 0
        else:
            break

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw_clean)
    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30:
            continue
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        # 跳过残余窄条伪影
        if ch > h * 0.85 and cw < w * 0.15:
            continue
        candidates.append(i)
    if not candidates:
        return None
    # 最上面的显著组件 = rank 字符
    candidates.sort(key=lambda i: stats[i, cv2.CC_STAT_TOP])
    ri = candidates[0]
    mask = (labels == ri).astype(np.uint8) * 255
    return (mask,
            stats[ri, cv2.CC_STAT_LEFT],
            stats[ri, cv2.CC_STAT_TOP],
            stats[ri, cv2.CC_STAT_WIDTH],
            stats[ri, cv2.CC_STAT_HEIGHT])


def _check_Q_vs_9(bw: np.ndarray, contours, hierarchy) -> str:
    """当孔洞比例指向 Q 时，进一步区分 Q 和 9。
    用连通组件把 rank 字符与花色符号分离后，检查字符底部宽度：
      9 底部窄（竖笔 descender），Q 底部宽（圆形 body）。
    返回 'Q' 或 '9'。"""
    res = _find_rank_component(bw)
    if res is None:
        return "Q"
    mask, rx, ry, rw, rh = res

    # 字符底部 20% 的最大前景跨度
    slice_h = max(2, int(rh * 0.2))
    bot_y = ry + rh
    bottom = mask[bot_y - slice_h:bot_y, :]
    if bottom.size == 0:
        return "Q"

    bot_span = 0
    for row in bottom:
        fg = np.where(row > 0)[0]
        if len(fg) > 0:
            bot_span = max(bot_span, int(fg[-1] - fg[0] + 1))

    # "9"：底部是窄竖笔 → span < 50% 字符宽度
    # "Q"：底部是圆弧 → span ≥ 50%
    if bot_span < rw * 0.5:
        return "9"
    return "Q"


def _disambiguate_AKQ(patch_bgr: np.ndarray,
                      hero_thresholds: tuple = (0.25, 0.08, 0.03),
                      board_thresholds: tuple = (0.07, 0.015, 0.008),
                      ) -> str | None:
    """A/K/Q/9 混淆消歧：利用拓扑特征（孔洞大小 + 形状）区分。
    Q 字母内有大椭圆孔洞，A 横杠形成中等三角孔洞，K 无封闭区域。
    9 也有大孔洞但其下方是窄竖笔，与 Q 的宽圆体不同。

    关键：原始 rank patch（ry 0.02~0.46）包含花色符号（♣ 有孔洞会误判为 A），
    因此只取 patch 上部 60%（仅含字母，排除花色图标）。

    hero_thresholds / board_thresholds: (Q下限, A下限, K上限)
      hole_ratio > Q下限 → Q/9, > A下限 → A, < K上限 → K
      设为负数可禁用某项判断（信任 CNN）。

    返回 'A' / 'K' / 'Q' / '9' / None（边界情况不做判断）。"""
    bw, is_hero = _binarize_rank_upper(patch_bgr)
    if bw is None:
        return None

    total_fg = cv2.countNonZero(bw)
    if total_fg < 15:
        return None

    # ── 孔洞检测（RETR_CCOMP 检测内轮廓） ──
    contours, hierarchy = cv2.findContours(
        bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hole_area = 0
    if hierarchy is not None:
        for i in range(len(contours)):
            if hierarchy[0][i][3] >= 0:  # 有父轮廓 → 孔洞
                hole_area += cv2.contourArea(contours[i])

    hole_ratio = hole_area / total_fg if total_fg > 0 else 0

    # 根据 hero/board 选择对应阈值
    th_q, th_a, th_k = hero_thresholds if is_hero else board_thresholds

    if th_q >= 0 and hole_ratio > th_q:
        return _check_Q_vs_9(bw, contours, hierarchy)
    if th_a >= 0 and hole_ratio > th_a:
        return "A"
    if th_k >= 0 and hole_ratio < th_k:
        return "K"
    return None  # 边界情况不做判断


def _binarize_rank_upper_smart(patch_bgr: np.ndarray):
    """专为 6/8 消歧优化的二值化：对红色文字自动切换 B 通道（红色 → 深色），
    避免普通 grayscale 把红色数字和绿色牌桌背景混在一起导致笔画断裂。
    返回 (bw, is_hero)。"""
    if patch_bgr is None or patch_bgr.size == 0:
        return None, False
    h_full, w_full = patch_bgr.shape[:2]
    is_hero = w_full < 60
    crop_ratio = 0.75 if is_hero else 0.60
    upper = patch_bgr[:int(h_full * crop_ratio), :]
    if upper.size == 0:
        return None, is_hero
    # 红色判断
    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 70, 70]), np.array([180, 255, 255]))
    red_ratio = (cv2.countNonZero(m1) + cv2.countNonZero(m2)) / max(1, upper.shape[0] * upper.shape[1])
    if red_ratio > 0.02:
        # 红字：用 B 通道取反（红色在 B 上最暗 → 取反后红字变亮，白背景变暗）
        gray = upper[:, :, 0]  # B channel
    else:
        gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
    if is_hero:
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    else:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw, is_hero


def _disambiguate_6_vs_8(patch_bgr: np.ndarray) -> str | None:
    """当 rank 疑似 8 时，进一步区分 6 vs 8。
    6 只有**下半封闭环**（上半是钩形开口）；8 是**上下双闭环**。

    使用保守策略（宁可 CNN 原判断对，也不错判）：
      - 上半闭合环面积足够大 → 肯定是 8
      - 上半毫无闭合环 AND 下半有明显闭环 → 高度疑似 6
      - 其他情况返回 None（让 CNN 原结果胜出）

    返回 '6' / '8' / None。"""
    bw, _ = _binarize_rank_upper_smart(patch_bgr)
    if bw is None:
        return None
    res = _find_rank_component(bw)
    if res is None:
        return None
    mask, rx, ry, rw, rh = res
    if rh < 14 or rw < 6:
        return None

    # 上半 45% / 下半 55%（8 的上下环大小相近，6 的下环大、上开口）
    split_h = int(rh * 0.45)
    # padding 便于 findContours 正确识别边界闭环
    pad = 1
    def count_holes(sub_mask):
        if sub_mask.size == 0:
            return 0, 0
        cnts, hier = cv2.findContours(sub_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnt, area = 0, 0
        if hier is not None:
            for i in range(len(cnts)):
                if hier[0][i][3] >= 0:
                    a = cv2.contourArea(cnts[i])
                    if a >= 4:
                        cnt += 1
                        area += a
        return cnt, area

    y1u, x1u = max(0, ry - pad), max(0, rx - pad)
    y2u, x2u = min(mask.shape[0], ry + split_h + pad), min(mask.shape[1], rx + rw + pad)
    upper = mask[y1u:y2u, x1u:x2u]

    y1l = max(0, ry + split_h - pad)
    y2l = min(mask.shape[0], ry + rh + pad)
    lower = mask[y1l:y2l, x1u:x2u]

    uh_cnt, uh_area = count_holes(upper)
    lh_cnt, lh_area = count_holes(lower)
    upper_px = max(1, upper.shape[0] * upper.shape[1])
    lower_px = max(1, lower.shape[0] * lower.shape[1])
    upper_ratio = uh_area / upper_px
    lower_ratio = lh_area / lower_px

    # 强信号：8 → 两个环都较大
    if upper_ratio > 0.05 and lower_ratio > 0.03:
        return "8"
    # 强信号：6 → 上半几乎无环 AND 下半有清晰环
    if uh_cnt == 0 and upper_ratio < 0.005 and lh_cnt >= 1 and lower_ratio > 0.04:
        return "6"
    return None


def _disambiguate_7_vs_2(patch_bgr: np.ndarray) -> str:
    """当 rank 为 '7' 时，检查是否实际是 '2'。
    用连通组件分离 rank 字符（排除边框伪影和花色符号），
    然后比较字符顶部和底部的宽度：
      "2" 底部有宽横笔 > 顶部弧线；"7" 顶部有宽横笔 > 底部斜笔尖。
    返回 '2' 或 '7'。"""
    bw, _ = _binarize_rank_upper(patch_bgr)
    if bw is None:
        return "7"

    res = _find_rank_component(bw)
    if res is None:
        return "7"
    mask, rx, ry, rw, rh = res
    if rh < 10:
        return "7"

    slice_h = max(3, int(rh * 0.2))

    # 顶部 20%：最大前景宽度跨度
    top_region = mask[ry:ry + slice_h, :]
    top_max = 0
    for row in top_region:
        fg = np.where(row > 0)[0]
        if len(fg) > 0:
            top_max = max(top_max, int(fg[-1] - fg[0] + 1))

    # 底部 20%：最大前景宽度跨度
    bot_y = ry + rh
    bot_region = mask[bot_y - slice_h:bot_y, :]
    bot_max = 0
    for row in bot_region:
        fg = np.where(row > 0)[0]
        if len(fg) > 0:
            bot_max = max(bot_max, int(fg[-1] - fg[0] + 1))

    # "2": 底部宽 > 顶部宽（底横 vs 弧顶）
    # "7": 顶部宽 > 底部宽（顶横 vs 斜笔尖）
    if bot_max > top_max * 1.15:
        return "2"
    return "7"


def _has_red_text(patch_bgr: np.ndarray) -> bool:
    """检测 rank patch 中的文字是否为红色（红桃/方块牌）。"""
    if patch_bgr is None or patch_bgr.size == 0:
        return False
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 70, 70]), np.array([180, 255, 255]))
    red_ratio = (cv2.countNonZero(mask1) + cv2.countNonZero(mask2)) / max(1, patch_bgr.shape[0] * patch_bgr.shape[1])
    return red_ratio > 0.03


def _enhance_red_for_ocr(patch_bgr: np.ndarray) -> np.ndarray:
    """将红色文字 patch 转换为灰度增强版（红色→黑色），供 OCR 使用。
    用蓝色通道：红色文字在 B 通道值极低，白色背景 B 值高 → 对比最大。"""
    blue = patch_bgr[:, :, 0]  # BGR → B channel
    # 反转后红色文字变白、背景变黑，再反转回来做成标准黑底白字
    enhanced = cv2.merge([blue, blue, blue])
    return enhanced


_DATA_URL_RE = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)
# FIX: 支持小数点作为千位分隔符（OCR常把逗号识别成小数点）
_INT_RE       = re.compile(r"\d[\d,，.]*")
_POT_RE       = re.compile(r"[底][池][:：]?\s*(\d[\d,.，]*)", re.IGNORECASE)
_CHINESE_RE   = re.compile(r'[\u4e00-\u9fff]')
_UI_NOISE_RE  = re.compile(r'[%％\[\]()（）◆★●]')
# FIX: 支持全角括号 （25）
_BLIND_RE = re.compile(r"(\d+)/(\d+)(?:[（(](\d+)[）)])?")

# SPEED：原 720 改 640——主图 OCR 工作量约降 20% （像素从 1.15M → 0.92M），
# 实测对 dpzx/poler 渲染截图准确率几乎无影响。可以通过环境变量覆盖，例如调试时
# 拉回 720 看精度差：set DEZHOU_OCR_WIDTH=720
_OCR_WIDTH = int(os.environ.get("DEZHOU_OCR_WIDTH", "640"))
# 线程池：1+N 架构里 N 部分需要跑 ~15 个 ROI OCR。OCR 引擎内部可能会串行化推理，
# 但 Python 侧开销 + 预处理能并行；加上两个 CV 检测，总共 ~20 任务。
_executor  = ThreadPoolExecutor(max_workers=8)


def _b64_to_bgr(image_base64: str) -> np.ndarray:
    b64 = _DATA_URL_RE.sub("", image_base64.strip())
    raw = base64.b64decode(b64, validate=False)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    return img


def _parse_int(text: str) -> int | None:
    """解析整数，支持半角逗号、全角逗号、小数点作分隔符。"""
    m = _INT_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", "").replace("，", "").replace(".", ""))
    except ValueError:
        return None


@dataclass
class OcrItem:
    text:  str
    cx:    float
    cy:    float
    score: float


# 导入布局基类
from app.layouts.base import Anchor, LayoutBase
from app.layouts.dpzx import LayoutDpzx
from app.layouts.poler import LayoutPoler


_ocr_engine = None
# 探测一次后缓存：rapid 调用时的最佳 kwargs（如 {"use_cls": False}）。
# None = 还没探测；{} = 探测了但默认 OK / 任何 kwarg 都不接受。
_RAPID_CALL_KWARGS: dict | None = None

def _get_ocr():
    """返回 OCR 引擎对象（封装 PaddleOCR 或 RapidOCR，API 不同）。
    _OCR_BACKEND 会被设置为 'paddle' 或 'rapid'。"""
    global _ocr_engine, _OCR_BACKEND
    if _ocr_engine is not None:
        return _ocr_engine

    # 默认 RapidOCR；DEZHOU_OCR=paddle 走 CPU，DEZHOU_OCR=paddle_gpu 走 GPU
    want_paddle = _FORCED_OCR in ("paddle", "paddle_gpu")
    use_gpu = (_FORCED_OCR == "paddle_gpu")
    want_rapid  = not want_paddle or (_FORCED_OCR is None) or (_FORCED_OCR == "rapid")

    # 尝试 PaddleOCR
    if want_paddle:
        try:
            # 禁用 Paddle 3.x 新 PIR 执行器 + oneDNN（Windows 崩溃根源）
            os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
            os.environ.setdefault("FLAGS_enable_pir_api", "0")
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            from paddleocr import PaddleOCR
            # PaddleOCR 3.x / 2.x 参数不同，依次尝试
            _ocr_engine = None
            _last_err = None
            # 关键：enable_mkldnn=False 规避 Paddle 3.x Windows 上 oneDNN 新 PIR 执行器崩溃
            for kwargs in (
                # 3.x 风格（关掉文档方向/去畸变/文本行方向，纯检测+识别）
                dict(lang="ch",
                     use_doc_orientation_classify=False,
                     use_doc_unwarping=False,
                     use_textline_orientation=False,
                     enable_mkldnn=False,
                     use_gpu=use_gpu,
                     ocr_version="PP-OCRv4"),
                dict(lang="ch",
                     use_doc_orientation_classify=False,
                     use_doc_unwarping=False,
                     use_textline_orientation=False,
                     enable_mkldnn=False,
                     use_gpu=use_gpu),
                dict(lang="ch",
                     use_doc_orientation_classify=False,
                     use_doc_unwarping=False,
                     use_textline_orientation=False,
                     use_gpu=use_gpu),
                # 2.x 风格
                dict(lang="ch", use_angle_cls=False, enable_mkldnn=False,
                     use_gpu=use_gpu,
                     ocr_version="PP-OCRv4", show_log=False),
                dict(lang="ch", use_angle_cls=False, use_gpu=use_gpu,
                     show_log=False),
                dict(lang="ch", use_gpu=use_gpu),
            ):
                try:
                    _ocr_engine = PaddleOCR(**kwargs)
                    break
                except (TypeError, ValueError) as e:
                    _last_err = e
                    continue
            if _ocr_engine is None:
                raise _last_err or RuntimeError("PaddleOCR 初始化失败")
            _OCR_BACKEND = "paddle"
            gpu_tag = " (GPU)" if use_gpu else " (CPU)"
            print(f"[OCR] 使用 PaddleOCR PP-OCRv4{gpu_tag}")
            return _ocr_engine
        except Exception as e:
            if _FORCED_OCR in ("paddle", "paddle_gpu"):
                raise RuntimeError(f"强制使用 paddle 但加载失败: {e}")
            print(f"[OCR] PaddleOCR 未加载 ({e.__class__.__name__})，回退 RapidOCR")

    # 回退 RapidOCR
    if want_rapid:
        if RapidOCR is None:
            raise RuntimeError(
                "RapidOCR 也没装；任选其一安装：\n"
                "  pip install rapidocr-onnxruntime>=1.4.4   (默认，跨平台)\n"
                "  pip install rapidocr-openvino             (Intel CPU 1.5-2.5x 快)\n"
                "  pip install rapidocr-paddle               (PaddlePaddle 后端，最慢)"
            )
        # 在 _OCR_BACKEND 里追加具体哪个后端，方便观察
        print(f"[OCR] RapidOCR 后端包: {_RAPID_PKG}")

        # ── 模型档位选择：默认 PP-OCRv4 server（rapidocr 1.4.x 自带）；
        # 设 DEZHOU_OCR_MODEL=mobile → 加载 PP-OCRv3 mobile（约 2-3x 快，
        # 准确率低 1-2%；适合渲染截图）。
        # 模型放在 app/ocr_models/ 下；先用 tools/download_ocr_mobile.py 拉好。
        model_kind = os.environ.get("DEZHOU_OCR_MODEL", "").lower().strip()
        custom_kwargs: dict = {}
        if model_kind in ("mobile", "v3", "ppocrv3"):
            here = os.path.dirname(os.path.abspath(__file__))
            mdir = os.path.join(here, "ocr_models")
            det = os.path.join(mdir, "ch_PP-OCRv3_det_infer.onnx")
            rec = os.path.join(mdir, "ch_PP-OCRv3_rec_infer.onnx")
            missing = [p for p in (det, rec) if not os.path.isfile(p)]
            if missing:
                print(f"[OCR] 警告：DEZHOU_OCR_MODEL={model_kind} 但找不到模型文件：")
                for p in missing:
                    print(f"        {p}")
                print(f"     → 退回默认 server 模型。先执行：python tools/download_ocr_mobile.py")
            else:
                # rapidocr 1.4.x 接受这两个 kwarg；不同小版本 key 名可能略不同，
                # 这里给出多种兼容候选。
                custom_kwargs = dict(det_model_path=det, rec_model_path=rec)

        # GPU 支持（v0.7+）：DEZHOU_GPU=1 时让 OCR 走 CUDA。
        # rapidocr-onnxruntime 1.4.x 接受 det_use_cuda / rec_use_cuda / cls_use_cuda 三个 kwarg；
        # 加进去后 RapidOCR 内部会用 onnxruntime-gpu 的 CUDAExecutionProvider。
        # 注意：onnxruntime-gpu 必须装好且 CUDA 驱动可用，否则 RapidOCR 回退 CPU。
        _use_gpu = os.environ.get("DEZHOU_GPU", "").lower().strip() in ("1", "cuda", "gpu", "true", "yes")
        if _use_gpu:
            custom_kwargs.setdefault("det_use_cuda", True)
            custom_kwargs.setdefault("rec_use_cuda", True)
            custom_kwargs.setdefault("cls_use_cuda", True)

        # 多档兼容：1.4.x 用 *_model_path；旧版本可能用 det/rec 模型路径不同 kwarg
        _last_err: Exception | None = None
        if custom_kwargs:
            # 兜底链：原 kwargs → 不带 cuda 标志 → 全小写 → 空 dict
            kwargs_no_cuda = {k: v for k, v in custom_kwargs.items() if not k.endswith("_use_cuda")}
            for kwargs in (
                custom_kwargs,
                kwargs_no_cuda,
                {f.lower(): v for f, v in kwargs_no_cuda.items()},
                {},
            ):
                try:
                    _ocr_engine = RapidOCR(**kwargs)
                    gpu_tag = " (GPU)" if any(k.endswith("_use_cuda") for k in kwargs) else ""
                    if "det_model_path" in kwargs or "det_model_path" in {k.lower() for k in kwargs}:
                        print(f"[OCR] 使用 RapidOCR (mobile, PP-OCRv3){gpu_tag}")
                    else:
                        print(f"[OCR] 使用 RapidOCR (默认 server, PP-OCRv4){gpu_tag}")
                    break
                except (TypeError, FileNotFoundError) as e:
                    _last_err = e
                    continue
            if _ocr_engine is None:
                raise RuntimeError(f"RapidOCR 初始化失败: {_last_err}")
        else:
            _ocr_engine = RapidOCR()
            print("[OCR] 使用 RapidOCR (默认 server, PP-OCRv4)")

        _OCR_BACKEND = "rapid"
        return _ocr_engine

    raise RuntimeError("没有可用的 OCR 引擎")


def _ocr_run(engine, bgr: np.ndarray):
    """跨引擎的统一调用接口，返回 list[(box, text, score)]。"""
    if _OCR_BACKEND == "paddle":
        # PaddleOCR 3.x: 优先 .predict()，返回 list[OCRResult(dict-like)]；
        # 2.x: .ocr(img, cls=False)，返回 [[[box,(text,score)], ...]]。
        res = None
        try:
            res = engine.predict(bgr)
        except Exception:
            try:
                res = engine.ocr(bgr, cls=False)
            except TypeError:
                res = engine.ocr(bgr)
        if not res:
            return []
        out = []
        first = res[0]
        # --- 3.x 格式：dict-like，含 rec_texts / rec_scores / rec_polys ---
        if isinstance(first, dict) or hasattr(first, "get"):
            try:
                texts  = first.get("rec_texts") or []
                scores = first.get("rec_scores") or []
                polys  = first.get("rec_polys") or first.get("dt_polys") or []
            except Exception:
                texts, scores, polys = [], [], []
            n = min(len(texts), len(scores)) if scores else len(texts)
            for i in range(n):
                box = polys[i] if i < len(polys) else None
                if box is not None and hasattr(box, "tolist"):
                    box = box.tolist()
                out.append((box, str(texts[i]), float(scores[i]) if scores else 0.9))
            return out
        # --- 2.x 格式：[[box, (text, score)], ...] ---
        if first is None:
            return []
        for line in first:
            try:
                box, (text, score) = line
                out.append((box, str(text), float(score)))
            except Exception:
                continue
        return out
    else:  # rapid
        # SPEED: rapidocr 1.4.x 的 __call__ 接受 use_cls/use_det/use_rec 关键字。
        # 截图永远是正向，关掉 cls 可省 20-30%。一次性探测，结果缓存到全局，
        # 后续直接用最佳路径，避免每次 try/except 抖动。
        global _RAPID_CALL_KWARGS
        if _RAPID_CALL_KWARGS is None:
            for kw in ({"use_cls": False}, {}):
                try:
                    res, _ = engine(bgr, **kw)
                    _RAPID_CALL_KWARGS = kw
                    if kw:
                        print(f"[OCR] RapidOCR 调用参数生效: {kw}")
                    break
                except TypeError:
                    continue
            else:
                _RAPID_CALL_KWARGS = {}
                res, _ = engine(bgr)
        else:
            res, _ = engine(bgr, **_RAPID_CALL_KWARGS)
        if not res:
            return []
        # FIX: rapidocr-onnxruntime 1.2.x（Python 3.13/3.14 上唯一能装的版本）
        # 偶尔会把 score 返回成 str（"0.9876"）而不是 float，下游 `score > best_score`
        # 会抛 "'>' not supported between instances of 'str' and 'float'"。
        # 在这里统一把每条结果归一成 (box, str(text), float(score))。
        out = []
        for item in res:
            try:
                box, text, score = item[0], item[1], item[2]
            except (TypeError, IndexError, ValueError):
                continue
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = 0.0
            out.append((box, str(text) if text is not None else "", score_f))
        return out


def _load_rank_templates() -> dict[str, np.ndarray]:
    """加载 rank_templates/ 下所有模板。文件名 = rank，如 A.png、10.png。
    所有模板已二值化（白字黑底）+ resize 到 _TMPL_SIZE。返回 {rank: gray_img}。
    目录不存在或空文件夹 → 返回空 dict（模板匹配被跳过，走纯 OCR）。"""
    out: dict[str, np.ndarray] = {}
    if not os.path.isdir(_TMPL_DIR):
        return out
    for rank in _RANKS_ALL:
        p = os.path.join(_TMPL_DIR, f"{rank}.png")
        if not os.path.isfile(p): continue
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        if im.shape[::-1] != _TMPL_SIZE:
            im = cv2.resize(im, _TMPL_SIZE, interpolation=cv2.INTER_AREA)
        out[rank] = im
    return out


def _prepare_rank_patch(patch_bgr: np.ndarray) -> np.ndarray | None:
    """把候选 rank 区域二值化 + 归一化到 _TMPL_SIZE。自动处理红字/黑字。"""
    if patch_bgr is None or patch_bgr.size == 0: return None
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    # 如果平均值偏白（卡片背景），字是暗的 → THRESH_BINARY_INV；
    # 偏暗的情况少见但用 Otsu 自适应
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 找到前景外包矩形，只保留字符区域再归一化
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 8]
    if not cnts: return None
    xs, ys, ws, hs = [], [], [], []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        xs.append(x); ys.append(y); ws.append(x+w); hs.append(y+h)
    x1, y1, x2, y2 = min(xs), min(ys), max(ws), max(hs)
    cropped = th[y1:y2, x1:x2]
    if cropped.size == 0: return None
    return cv2.resize(cropped, _TMPL_SIZE, interpolation=cv2.INTER_AREA)


def _template_match_ranks(patch_bgr: np.ndarray,
                          templates: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    """对候选 patch 做归一化模板匹配，返回按分数降序的 (rank, score) 列表。"""
    if not templates: return []
    q = _prepare_rank_patch(patch_bgr)
    if q is None: return []
    scores: list[tuple[str, float]] = []
    for rank, tmpl in templates.items():
        r = cv2.matchTemplate(q, tmpl, cv2.TM_CCOEFF_NORMED)
        scores.append((rank, float(r[0, 0])))
    scores.sort(key=lambda kv: kv[1], reverse=True)
    return scores


class TableRecognizer:
    def __init__(self, layout: LayoutBase | None = None) -> None:
        self._ocr       = _get_ocr()
        self._layout    = layout or LayoutDpzx()
        self._clahe     = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        self._templates = _load_rank_templates()

    # ==================================================================
    # 主入口
    # ==================================================================
    def recognize(self, image_base64: str, parse_all: bool = False) -> dict:
        img  = _b64_to_bgr(image_base64)
        h, w = img.shape[:2]

        # ── 轮次门控（由请求参数 parse_all 控制）──
        # parse_all=False(默认)：先廉价判断"是否轮到自己"，没轮到 → 直接返回置空结构
        #   (is_hero_turn=False)，跳过最贵的全图 OCR / 牌面 / 座位识别。
        # parse_all=True：不论是否轮到自己都完整解析。
        if not parse_all and not self._detect_turn_fast(img):
            return self._empty_result()

        scale = _OCR_WIDTH / w if w > _OCR_WIDTH else 1.0
        img_g = cv2.resize(img, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
        img_g_raw = img_g.copy()   # 保留未增强副本，用于 stack 兜底
        self._enhance_inplace(img_g, self._layout.hero_stack_roi)
        for roi in self._layout.status_enhance_rois:
            self._enhance_inplace(img_g, roi)

        # ============================================================
        # 混合架构：
        #  - 全图 OCR：负责数字文本（筹码/底池/盲注/状态），准确率高
        #  - 白色文字提取 + ROI OCR：负责下注金额（DPZX专用，更精准）
        #  - CNN 分类器（可选）：负责卡牌 rank/suit
        #  - 纯 CV 辅助：庄位 D 检测
        # ============================================================
        use_white_bet = bool(self._layout.seat_bet_rois)
        use_countdown = bool(self._layout.hero_countdown_roi)

        future_button: Future = _executor.submit(self._detect_button_seat, img)
        # 倒计时模式下不并发跑红色按钮（倒计时检测依赖 OCR，主线程顺序执行）
        future_turn: Future | None = None
        if not use_countdown:
            future_turn = _executor.submit(self._detect_hero_turn, img)

        items = self._ocr_full(img_g)

        button_seat = future_button.result()

        # Hero 轮次检测
        if use_countdown:
            is_hero_turn = self._detect_hero_turn_countdown(img, items)
        elif future_turn is not None:
            is_hero_turn = future_turn.result()
        else:
            is_hero_turn = None

        pot        = self._find_pot(items, img_g=img_g)
        table_current_bet = self._find_current_bet(items)
        blind_size = self._find_blind_size(items)
        hero_stack = self._pick_stack(items, self._layout.hero_stack_anchor)
        if hero_stack is None:
            hero_stack = self._pick_stack(items, self._layout.hero_stack_anchor, min_value=0)
        # hero 筹码兜底：在座但读不到 → 0
        if hero_stack is None:
            hero_stack = 0

        hero_status = self._hero_status(items)

        # ── 收集所有 stack（先于 bet 检测，避免 stack 被误读为 bet） ──
        max_stk = getattr(self._layout, "max_stack", None)
        seat_info: list = []
        stack_values: set = set()
        if hero_stack > 0:
            stack_values.add(hero_stack)
        for seat, anchor in self._layout.seat_anchors:
            status = self._seat_status(items, seat)
            if status == "empty":
                stack = None        # 空座无筹码
            elif status == "all_in":
                stack = 0
            else:
                stack = self._pick_stack(items, anchor, max_value=max_stk)
                if stack is None:
                    stack = self._pick_stack(items, anchor, min_value=0, max_value=max_stk)
                if stack is None:
                    # 有玩家但读不到筹码 → 0（不再返回 None）
                    stack = 0
            if stack is not None and stack > 0:
                stack_values.add(stack)
            seat_info.append((seat, status, stack))

        # ── 兜底：CLAHE 可能破坏边缘文字，对 stack 异常低的在局座位用未增强全图 OCR ──
        # stack < 5 视为可疑（1/2 桌极少出现如此低的筹码，多为 OCR 误读如 "1/4"→1）
        need_raw_fallback = any(
            stack is not None and stack < 5
            and status not in ("empty", "all_in", "folded", "waiting")
            for _, status, stack in seat_info
        )
        if need_raw_fallback:
            # SPEED(v21): 旧逻辑在此对整张未增强图再跑一次全图 OCR（~2000ms）。
            # 实测仅 ~15% 图触发，且每次只需修正 1-3 个座位的 stack——
            # 改为只对可疑座位的小块 ROI 在原图上补 OCR（复用 _ocr_strip），
            # 单块 ~50-150ms，替代整图重跑，零精度损失（小块仍是原始未增强像素）。
            seat_anchor_map_raw = dict(self._layout.seat_anchors)
            for i, (seat, status, stack) in enumerate(seat_info):
                if (stack is not None and stack < 5
                        and status not in ("empty", "all_in", "folded", "waiting")):
                    anchor = seat_anchor_map_raw.get(seat)
                    if anchor is None:
                        continue
                    # 围绕座位锚点裁一小块（与 _build_strips 的座位条同尺寸）
                    roi = (max(0.0, anchor.cx - 0.095), max(0.0, anchor.cy - 0.055),
                           min(1.0, anchor.cx + 0.095), min(1.0, anchor.cy + 0.085))
                    raw_items = self._ocr_strip(img_g_raw, roi)
                    raw_stack = self._pick_stack(raw_items, anchor, max_value=max_stk)
                    if raw_stack is None:
                        raw_stack = self._pick_stack(raw_items, anchor, min_value=0, max_value=max_stk)
                    if raw_stack is not None and raw_stack > stack:
                        seat_info[i] = (seat, status, raw_stack)
                        stack_values.add(raw_stack)

        # ── 判断公共牌是否可见（用于下注检测的 board ROI 过滤开关）──
        # 翻牌前公共牌区域无亮卡，此时 board ROI 内的 OCR 数字是下注筹码而非牌面 rank，
        # 不应被 board ROI 过滤器误杀。快速检查：取所有 board ROI 的平均亮度，
        # 超过阈值说明有白色卡牌可见。
        _board_bright_thr = getattr(self._layout, 'card_brightness_threshold', 90.0)
        _has_board_cards = False
        for broi in self._layout.board_rois:
            bcrop = self._roi(img, broi)
            if bcrop.size > 0:
                b_gray = cv2.cvtColor(bcrop, cv2.COLOR_BGR2GRAY) if len(bcrop.shape) == 3 else bcrop
                if float(np.mean(b_gray)) >= _board_bright_thr:
                    _has_board_cards = True
                    break

        # ── 下注检测 ──
        # 白色文字模式（DPZX）：并行裁剪各座位 bet ROI + 白色文字 OCR
        # 传统模式（Poler 等）：从全图 OCR items 按锚点搜索
        bet_futures: dict[int, Future] = {}
        if use_white_bet:
            for seat, status, stack in seat_info:
                if status in ("folded", "empty", "waiting"):
                    continue
                bet_roi = self._layout.seat_bet_rois.get(seat)
                if bet_roi:
                    bet_futures[seat] = _executor.submit(
                        self._detect_bet_white_text, img, bet_roi)

        # SPEED(v21): 传统模式（poler）的逐座位下注 fallback OCR 原本在下面的
        # villain 循环里【串行】执行——每个非弃牌座位一次 5x 放大 OCR（~225ms），
        # 6 座位累计 ~1350ms，几乎与全图 OCR 等价。这里提前把它们【并发】提交到
        # 线程池（与上面 DPZX 白字下注同款做法），循环内直接取 future 结果。
        # 逻辑完全不变（同样的 ROI / chip_colors / 覆盖规则），仅并行化。
        enh_bet_futures: dict[int, Future] = {}
        _seat_fb_rois = getattr(self._layout, 'seat_bet_fallback_rois', {})
        _fb_chip_map = getattr(self._layout, 'seat_bet_fallback_chip_colors', {})
        # FIX(下注被别座筹码误排除)：下注兜底的 exclude_values 只用【本座自己的 stack】，
        # 不用全局所有座位的 stack。原因：下注 ROI(cx≈0.2~0.78) 不含任何座位的筹码 pill
        # (筹码在画面两侧 cx≈0.05/0.92)，全局排除基本无用却会误伤——例如某座筹码被 CLAHE
        # 误读成 16，会把另一座真实下注 16 也排除掉（实测 20260625 图座7/8 加注/跟注16 全丢）。
        if _seat_fb_rois and not use_white_bet:
            for seat, status, stack in seat_info:
                if status in ("folded", "empty", "waiting"):
                    continue
                bet_roi = _seat_fb_rois.get(seat)
                if not bet_roi:
                    continue
                own_excl = {stack} if stack else set()
                enh_bet_futures[seat] = _executor.submit(
                    self._detect_bet_roi_enhanced, img, bet_roi,
                    exclude_values=own_excl, pot_value=pot,
                    has_board_cards=_has_board_cards,
                    chip_colors=_fb_chip_map.get(seat, ("red",)))
        # hero 自己的下注 fallback 也并发提交（仅 1 次，但同样在串行关键路径上）
        _hero_fb_roi = getattr(self._layout, 'hero_bet_fallback_roi', None)
        hero_enh_future: Future | None = None
        if _hero_fb_roi and not use_white_bet:
            hero_enh_future = _executor.submit(
                self._detect_bet_roi_enhanced, img, _hero_fb_roi,
                exclude_values=({hero_stack} if hero_stack else set()), pot_value=pot,
                has_board_cards=_has_board_cards, chip_colors=("red", "green"))

        villains = []
        seat_anchor_map = dict(self._layout.seat_anchors)
        for seat, status, stack in seat_info:
            if status in ("folded", "empty", "waiting"):
                current_bet = None
            elif use_white_bet:
                # 白色文字检测（DPZX）
                ft = bet_futures.get(seat)
                current_bet = ft.result() if ft else None
            elif status == "all_in":
                # all-in 下注筹码通常在 bet anchor 附近，而非 stack anchor
                # 优先用 bet anchor 搜索（距离更近），找不到再用 stack anchor
                bet_anchor_ai = self._layout.seat_bet_anchors.get(seat)
                sa = seat_anchor_map.get(seat)
                current_bet = None
                if bet_anchor_ai:
                    current_bet = self._pick_all_in_bet(
                        items, bet_anchor_ai, has_board_cards=_has_board_cards,
                        exclude_values=stack_values)
                if current_bet is None and sa:
                    current_bet = self._pick_all_in_bet(
                        items, sa, has_board_cards=_has_board_cards,
                        exclude_values=stack_values)
                # FIX(721 s8 2020 vs 202)：CLAHE 增强后右侧座位的 stack "0" 会
                # 与相邻 bet "202" 在 OCR 视觉上粘连，被读成 "2020"。
                # 用 _detect_bet_roi_enhanced 在 ROI 内做红芯片门控 + 紧裁剪 OCR，
                # 比锚点搜索可靠得多——它直接基于红色 chip 轮廓，避开任何相邻文字。
                # 仅当 fallback 给出 != current_bet 时覆盖，避免误伤 anchor 已对的值。
                # SPEED(v21): 取并发提交的 fallback 结果（逻辑同前：找到才覆盖）
                if seat in enh_bet_futures:
                    fb_bet = enh_bet_futures[seat].result()
                    if fb_bet is not None:
                        current_bet = fb_bet
            else:
                bet_anchor = self._layout.seat_bet_anchors.get(seat)
                if bet_anchor is None:
                    current_bet = None
                else:
                    sa = seat_anchor_map.get(seat)
                    current_bet = self._pick_current_bet(
                        items, bet_anchor, exclude_values=stack_values,
                        seat_anchor=sa, has_board_cards=_has_board_cards)
                # 红色牌背门控兜底：尝试在固定 ROI 内做红芯片检测 + OCR
                # · 座位 6/7/8（右侧）：fallback 是权威——找不到红芯片即清除锚点误报
                #   （右侧锚点经常把绿色 BB 标记误读为 "1"，必须靠红芯门控否决）
                # · 座位 5（顶部）：anchor 已经收紧到上层红芯位置 (cy≈0.255)，
                #   fallback 用紧裁剪+5x 放大补 OCR 对单字 "1" 的漏读。
                #   此时**只在 fallback 找到值时覆盖**，找不到不要清除 anchor 结果。
                # SPEED(v21): 取并发提交的 fallback 结果（覆盖规则同前）
                if seat in enh_bet_futures:
                    fb_bet = enh_bet_futures[seat].result()
                    # 座位 5 / 左侧 (2/3/4)：fallback 仅在找到值时覆盖（避免误清空 anchor 命中）
                    # 右侧 (6/7/8)：fallback 是权威——找不到红芯即清除 anchor 误报
                    if seat in (2, 3, 4, 5):
                        if fb_bet is not None:
                            current_bet = fb_bet
                    else:
                        current_bet = fb_bet
            villains.append({
                "seat": seat, "status": status,
                "stack": stack, "current_bet": current_bet,
            })

        all_rois = list(self._layout.board_rois) + [
            self._layout.hero_card1_roi, self._layout.hero_card2_roi,
        ]
        rank_map        = self._find_ranks(img, items, all_rois)
        community_cards = self._build_card_list(img, rank_map, list(self._layout.board_rois))
        hero_cards      = self._build_card_list(
            img, rank_map,
            [self._layout.hero_card1_roi, self._layout.hero_card2_roi],
        )

        stage = self._infer_stage(community_cards)
        # 操作按钮可见（is_hero_turn=True）时，hero 不可能已弃牌——
        # hero_status_anchor 区域重叠了 action_btn_y，"弃牌"按钮文字会被误读为状态。
        # FIX(poler face-up dim)：layout.keep_folded_when_cards_visible=True 时
        # 不再因检到 hero_cards 而把 fold 翻成 active —— poler 弃牌后保留正面 dim
        # 卡牌显示，cards != [] 是正常状态而非"误判"。
        keep_folded = getattr(
            self._layout, 'keep_folded_when_cards_visible', False)
        if hero_status == "folded" and is_hero_turn:
            hero_status = "active"
        elif hero_status == "folded" and not hero_cards:
            # 真弃牌：状态 folded 且未检到手牌 → 保持 []
            pass
        elif hero_status == "folded" and hero_cards:
            if keep_folded:
                # poler：dim face-up 弃牌牌仍可读，但状态保持 folded
                pass
            else:
                # 状态误判或 show hand 阶段：实际检到了牌 → 保留手牌，修正状态
                hero_status = "active"

        # Hero 下注检测
        if hero_status == "folded":
            hero_current_bet = None
        elif use_white_bet and self._layout.hero_bet_roi:
            hero_current_bet = self._detect_bet_white_text(
                img, self._layout.hero_bet_roi)
        else:
            hero_current_bet = self._pick_current_bet(
                items, self._layout.hero_bet_anchor,
                exclude_values=stack_values)
            # FIX(v0.7)：hero 自己的小芯（SB/BB/STR=1/2/4）在 640px OCR 下经常漏读。
            # 用 hero_bet_fallback_roi 紧裁剪 + 5x 放大 OCR 兜底。
            if hero_enh_future is not None:
                fb_bet = hero_enh_future.result()
                if fb_bet is not None:
                    # 仅在锚点没找到时使用 fallback 值；锚点找到的话信任锚点
                    if hero_current_bet is None:
                        hero_current_bet = fb_bet

        # ── Post-inference: 空座视觉兜底（必须在 all-in 推断之前）──
        # 状态被识别成 active 但 stack=0/None 时，极有可能是空座（"+" 占位但
        # 旁边有残留筹码/底池数字 → 状态识别误判 active，bet 检测被残留数字命中）。
        # 用 _is_seat_avatar_empty 视觉检查覆盖。仅在 layout 显式 opt-in 时启用，
        # 避免影响 dpzx 这种"刚入场玩家也是 stack=0/active"的合法情况。
        # 注意：不能用 `not current_bet` 作为门控——poler_004/005 正是 chip 残留
        # 被错读成 bet=12/446，要无视 bet 强制视觉判定。
        if getattr(self._layout, 'enable_empty_seat_visual_check', False) \
                and getattr(self._layout, 'seat_positions', None):
            for v in villains:
                # 视觉检查触发条件：
                # (a) status=active 且 stack 为 0/None（旧逻辑）
                # (b) status=waiting 且 stack 为 0/None — 处理 471 s8 这种被相邻
                #     座位的 "补盲或过庄" 文字串扰、其实是真空座的情况。
                #     视觉返回 empty 时强制覆盖 waiting → empty。
                trigger = (
                    (v["status"] == "active"
                     and (v.get("stack") == 0 or v.get("stack") is None))
                    or
                    (v["status"] == "waiting"
                     and (v.get("stack") == 0 or v.get("stack") is None))
                )
                if not trigger:
                    continue
                visual = self._seat_visual_inactivity(img, v["seat"])
                if visual == "empty":
                    v["status"] = "empty"
                    v["stack"] = None
                    v["current_bet"] = None
                elif visual == "waiting" and v["status"] == "active":
                    # 暂离/断线：玩家不参与本手，bet 必为 None；
                    # stack 保持 0（实际是离线显示，不是空位）。
                    v["status"] = "waiting"
                    v["current_bet"] = None

        # ── Post-inference: all-in 推断（在空座兜底之后）──
        # 状态识别有时漏掉 "All in" 标签（被花色/数字遮挡或字体太小），
        # 用 stack=0 + bet>0 作为补充信号。
        # FIX(v21, 生产图2/6/10 座位4 加注168/剩105 被误判全押)：
        #   current_bet 是本轮下注额，stack 是下注【之后】的剩余筹码。
        #   "加注到 N、还剩 M"时 bet(N) > stack(M) 是正常加注，不是全押。
        #   真全押时 stack 显示 0（或带 "All in" 标签，已由状态检测捕获）。
        #   因此去掉旧的 `bet > stack → all_in` 误判条件，只认 stack==0。
        # 只动 active 状态，不动 folded/empty/waiting/all_in。
        for v in villains:
            if v["status"] == "active":
                bt = v.get("current_bet")
                sk = v.get("stack")
                if isinstance(bt, int) and bt > 0 and isinstance(sk, int):
                    if sk == 0:
                        v["status"] = "all_in"
        if hero_status == "active" and isinstance(hero_current_bet, int) \
                and hero_current_bet > 0 and isinstance(hero_stack, int):
            if hero_stack == 0:
                hero_status = "all_in"

        # FIX(table.current_bet)：poler 在底池正上方有累计/底注小芯（如 "16"），
        # 旧 OCR 逻辑会把它误读为 current_bet。改用 max(seat bet) 计算。
        # dpzx 走默认 "ocr" 策略不受影响。
        strategy = getattr(self._layout, 'table_current_bet_strategy', 'ocr')
        if strategy == "max_seat_bet":
            seat_bets = []
            for v in villains:
                bt = v.get("current_bet")
                if isinstance(bt, int) and bt > 0:
                    seat_bets.append(bt)
            if isinstance(hero_current_bet, int) and hero_current_bet > 0:
                seat_bets.append(hero_current_bet)
            table_current_bet = max(seat_bets) if seat_bets else None

        return {
            "table_info": {
                "stage": stage, "community_cards": community_cards,
                "main_pot": pot, "current_bet": table_current_bet,
                "button_seat": button_seat, "blind_size": blind_size,
            },
            "hero_info": {
                "seat": 1, "status": hero_status, "stack": hero_stack,
                "current_bet": hero_current_bet, "is_hero_turn": is_hero_turn,
                "hero_cards": hero_cards,
            },
            "villains_info": villains,
        }

    # ==================================================================
    # 庄家按钮（纯色检测）
    # ==================================================================
    # 庄家按钮位置从 layout 读取

    def _detect_button_seat(self, img: np.ndarray) -> int | None:
        h, w = img.shape[:2]
        hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lo = getattr(self._layout, 'button_hsv_lower', (15, 120, 100))
        hi = getattr(self._layout, 'button_hsv_upper', (40, 255, 255))
        min_area = getattr(self._layout, 'button_min_area', 80)
        mask = cv2.inRange(hsv, lo, hi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 收集所有候选圆形轮廓
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_area < area < 3000): continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx = M["m10"] / M["m00"] / w
            cy = M["m01"] / M["m00"] / h
            if cy >= 0.90: continue           # 排除底部UI
            # 排除中央区域（加分按钮等UI元素），但允许靠近已知 button_position 的候选
            if 0.35 < cx < 0.65 and cy > 0.55:
                near_known = any(
                    ((cx - px)**2 + (cy - py)**2)**0.5 < 0.10
                    for positions in self._layout.button_positions.values()
                    for px, py in positions
                )
                if not near_known:
                    continue
            # 庄家按钮是圆形，排除矩形高亮框
            peri = cv2.arcLength(c, True)
            if peri == 0: continue
            if 4 * np.pi * area / (peri * peri) < 0.40: continue
            candidates.append((cx, cy, area))
        if not candidates: return None
        # 在所有候选中找最接近某个 button_position 的
        best_dist, best_seat = 999.0, None
        for bcx, bcy, _ in candidates:
            for seat, positions in self._layout.button_positions.items():
                for px, py in positions:
                    d = ((bcx-px)**2+(bcy-py)**2)**0.5
                    if d < best_dist:
                        best_dist = d; best_seat = seat
        return best_seat if best_dist < 0.15 else None

    # ==================================================================
    # 盲注（支持全角括号）
    # ==================================================================
    def _find_blind_size(self, items: list[OcrItem]) -> str | None:
        x1, y1, x2, y2 = self._layout.blind_roi
        candidates = [it for it in items if x1 <= it.cx <= x2 and y1 <= it.cy <= y2]
        candidates += [it for it in items if 0.55 < it.cy < 0.62]
        for it in candidates:
            # 预处理：去掉常见前缀（"德州"等），OCR 常把它们和数字粘连
            # 例如 "德州1/2/4（2）" → OCR读为 "德州172/4(2)" 导致误匹配
            text = it.text
            for prefix in ("德州", "德扑", "DPZX", "dpzx"):
                if prefix in text:
                    text = text[text.index(prefix) + len(prefix):]
                    break

            # 尝试匹配 "sb/bb/ante(buyin)" 或 "sb/bb(ante)" 格式
            # 先尝试三段格式: 1/2/4(2) → sb=1, bb=2, 后面忽略
            m3 = re.search(r"(\d+)/(\d+)/(\d+)(?:[（(](\d+)[）)])?", text)
            if m3:
                sb, bb = m3.group(1), m3.group(2)
                ante = m3.group(4)
                return f"{sb}/{bb}({ante})" if ante else f"{sb}/{bb}"

            # FIX: OCR 经常把 "/" 读成 "7"，导致 "1/2/4(2)" → "172/4(2)"
            # 尝试把数字间的 "7" 替换回 "/" 再匹配
            text_fixed = re.sub(r"(\d)7(\d)", r"\1/\2", text)
            if text_fixed != text:
                m3f = re.search(r"(\d+)/(\d+)/(\d+)(?:[（(](\d+)[）)])?", text_fixed)
                if m3f:
                    sb, bb = m3f.group(1), m3f.group(2)
                    ante = m3f.group(4)
                    return f"{sb}/{bb}({ante})" if ante else f"{sb}/{bb}"

            m = _BLIND_RE.search(text)
            if m:
                sb, bb, ante = m.group(1), m.group(2), m.group(3)
                return f"{sb}/{bb}({ante})" if ante else f"{sb}/{bb}"
        return None

    # ==================================================================
    # current_bet
    # ==================================================================
    # 广告/UI噪声关键词（出现在这些文本里的小数字不是下注）
    # 噪声关键词从 layout 读取

    def _pick_all_in_bet(self, items: list[OcrItem], seat_anchor: Anchor,
                         radius: float = 0.13,
                         has_board_cards: bool = True,
                         exclude_values: set | None = None) -> int | None:
        """all-in 专用：不管 UI 文字里是否含 'All in'、括号，取座位头像周围
        半径内**最近的合法数字**。过滤掉：底池、盲注格式 N/N、倒计时 Ns、
        明显过大的筹码栈（> 100000）、公共牌 ROI 内的文字（仅当公共牌可见时）、
        已知筹码值（避免误读相邻座位筹码）。"""
        # 收集底池值用于排除
        pot_val = None
        for it in items:
            if '底池' in it.text or '池' in it.text:
                v = _parse_int(it.text)
                if v and v > 0:
                    pot_val = v
                    break
        card_excl_rois = list(getattr(self._layout, 'board_rois', ())) if has_board_cards else []
        best_val, best_dist = None, 999.0
        for it in items:
            dx = it.cx - seat_anchor.cx
            dy = it.cy - seat_anchor.cy
            d = (dx*dx + dy*dy) ** 0.5
            if d > radius: continue
            t = it.text.strip()
            if re.match(r'^\d{1,3}[Ss]$', t): continue          # 倒计时
            if any(kw in t for kw in ('秒', '留座', '倒计时')): continue  # 留座倒计时
            if '/' in t: continue                                # 盲注 50/100
            if '底池' in t or '池' in t: continue
            if t.startswith('+'): continue                       # YOU WIN +N 不算下注
            # 排除落在牌面 ROI 内的 OCR item（仅公共牌可见时）
            in_card = False
            for roi in card_excl_rois:
                if roi[0] <= it.cx <= roi[2] and roi[1] <= it.cy <= roi[3]:
                    in_card = True
                    break
            if in_card: continue
            val = _parse_int(t)
            if val is None or val < 1: continue
            if pot_val and val == pot_val: continue
            if exclude_values and val in exclude_values: continue
            if val > 100000: continue
            if d < best_dist:
                best_dist = d
                best_val = val
        return best_val

    def _pick_current_bet(self, items: list[OcrItem], anchor: Anchor,
                          exclude_values: set | None = None,
                          seat_anchor: Anchor | None = None,
                          has_board_cards: bool = True) -> int | None:
        # 先在扩展范围内找底池值 + 底池文本的精确位置（用于位置敏感的过滤）
        # FIX(座位5)：旧逻辑无差别地把"任何 val == pot 的候选"全部丢弃，
        # 但 seat 5 紧贴底池上方，下注金额经常 == 底池金额（盲注/跨注），
        # 误把真实的下注芯片"70"过滤掉，结果该位下注永远抓不到。
        # 改为只在候选**与底池文本同一水平条带**(|dy|<0.025) 时才排除。
        pot_val: int | None = None
        pot_cy:  float | None = None
        for it in items:
            d = ((it.cx-anchor.cx)**2+(it.cy-anchor.cy)**2)**0.5
            if d < 0.18 and any(kw in it.text for kw in ('底池', '池')):
                v = _parse_int(it.text)
                if v and v > 0:
                    pot_val = v
                    pot_cy  = it.cy
                    break

        # 预计算牌面 ROI 列表，用于排除 rank 文字泄漏到下注区域。
        # 仅在公共牌可见时启用公共牌 ROI 过滤——翻牌前该区域无卡牌，
        # OCR 数字是下注筹码（如盲注/跨注），不应被过滤。
        # hero 手牌 ROI 始终启用（手牌始终可能存在）。
        card_excl_rois = []
        if has_board_cards:
            card_excl_rois.extend(getattr(self._layout, 'board_rois', ()))
        hero_c1 = getattr(self._layout, 'hero_card1_roi', None)
        hero_c2 = getattr(self._layout, 'hero_card2_roi', None)
        if hero_c1: card_excl_rois.append(hero_c1)
        if hero_c2: card_excl_rois.append(hero_c2)

        # FIX: 使用Y-加权距离并强制Y带限制，避免相邻座位(如seat7/seat8)互相干扰。
        # 当两个候选数字都落在圆内时，几何距离可能让"错座位"胜出。
        # 通过 (1) 硬Y带：|dy| 必须 < max_dist*0.85； (2) Y权重2倍，让竖向错位强烈惩罚。
        best_val, best_score = None, 1e9
        y_band = anchor.max_dist * 0.85
        for it in items:
            dx = it.cx - anchor.cx
            dy = it.cy - anchor.cy
            if abs(dy) > y_band: continue                          # 硬Y带过滤��邻座位
            d_weighted = (dx * dx + (dy * 2.0) ** 2) ** 0.5        # Y权重2x
            if d_weighted > anchor.max_dist * 2.0: continue
            # FIX: 排除更靠近头像(seat_anchor)的 OCR 数字——
            # 头像里的倒计时纯数字(如 "10")无关键词可过滤，
            # 但它一定比下注数字更靠近头像中心。
            if seat_anchor is not None:
                d_to_seat = ((it.cx - seat_anchor.cx)**2
                             + (it.cy - seat_anchor.cy)**2) ** 0.5
                d_to_bet  = ((it.cx - anchor.cx)**2
                             + (it.cy - anchor.cy)**2) ** 0.5
                if d_to_seat < d_to_bet:
                    continue
            # FIX: 排除落在牌面 ROI 内的 OCR item——
            # 公共牌可见时，其 rank 文字(如 "8"、"4"、"2") 会被误读为下注；
            # 翻牌前不启用公共牌 ROI 过滤（此时该区域的数字是下注筹码）。
            # hero 手牌 ROI 始终过滤。
            in_card = False
            for roi in card_excl_rois:
                if roi[0] <= it.cx <= roi[2] and roi[1] <= it.cy <= roi[3]:
                    in_card = True
                    break
            if in_card:
                continue
            t = it.text.strip()
            if re.match(r'^\d{1,3}[Ss]$', t): continue          # 倒计时
            if any(kw in t for kw in ('秒', '留座', '倒计时')): continue  # 留座倒计时
            if any(kw in t for kw in ('底池', '池')): continue   # 底池标签
            # FIX: "+166" / "+1,200" 是 YOU WIN / 赢得金额指示器，不是下注。
            # poler 终局图 hero 头像上方会浮一个 +N 提示，紧贴 hero_bet_anchor，
            # 若不过滤会被错读成 hero 本轮下注。
            if t.startswith('+'): continue
            is_noisy = any(kw in t for kw in self._layout.noise_keywords)
            val = _parse_int(t)
            if val is None or val < 1: continue
            if is_noisy and val < 100: continue                    # 广告小数字
            # 位置敏感的底池排除：仅当候选 OCR 与底池文本同一行 (|dy|<0.025)
            # 才视为"底池旁孤立数字"。否则即使 val==pot_val 也保留——
            # seat5 下注芯片在底池正上方 ~0.05，几乎贴着底池但仍是独立元素。
            if pot_val and val == pot_val and pot_cy is not None:
                if abs(it.cy - pot_cy) < 0.025:
                    continue
            if exclude_values and val in exclude_values: continue  # 排除筹码栈值
            if d_weighted < best_score:
                best_score = d_weighted
                best_val = val
        return best_val

    # ==================================================================
    # 局部增强 OCR 下注检测（兜底：全图 OCR 漏检小数字时启用）
    # 前置门控：先检测 ROI 内是否存在红色牌背图标（筹码标志），
    #   只有确认存在筹码时才跑 OCR，否则直接返回 None。
    # ==================================================================
    _RED_CHIP_MIN_AREA = 500   # 红色轮廓最小面积阈值（区分筹码 vs 公共牌花色）

    # 绿色筹码检测：饱和绿（hue 70~100, sat>140）区分桌面深绿（sat<100）
    _GREEN_CHIP_MIN_AREA = 200    # 小面值绿芯尺寸通常较小，阈值放宽

    def _detect_bet_roi_enhanced(self, img: np.ndarray, roi: tuple,
                                  exclude_values: set | None = None,
                                  pot_value: int | None = None,
                                  has_board_cards: bool = True,
                                  chip_colors: tuple = ("red",)) -> int | None:
        """色彩门控 + 紧凑裁剪 + 放大 OCR 识别下注值。

        chip_colors: 一个或多个颜色标识 ("red", "green")。
          red:  AA Poker 中 raise/call/all_in 的红色筹码 + 白色牌图标（默认）。
          green: SB/BB/STR straddle 等小面值绿色筹码。

        FIX(471 s7=9 vs None)：右侧座位 (s6/s7/s8) 的 fallback ROI Y 范围
        与公共牌 ROI 在 y=0.48~0.56 重叠。当公共牌可见且最右一张是红色花色
        （♥/♦），其 rank 数字会作为最大红色轮廓被命中，导致下注错读为牌面 rank。
        加一层"主轮廓重心是否落在某张公共牌内"的过滤，是即返 None。

        FIX(SB/BB/STR 小芯)：默认 OCR 宽度 640 太低，小绿芯上 "1"/"2"/"4"
        在全图 OCR 中读不到。此函数用 5x 紧裁剪 OCR 救回。
        """
        H, W = img.shape[:2]
        x0, y0, x1, y1 = roi
        xa, ya = max(0, int(x0 * W)), max(0, int(y0 * H))
        xb, yb = min(W, int(x1 * W)), min(H, int(y1 * H))
        if xb <= xa or yb <= ya:
            return None
        crop = img[ya:yb, xa:xb]
        if crop.size == 0:
            return None

        # ---- 色彩门控：根据 chip_colors 选择 mask ----
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        masks = []
        if "red" in chip_colors:
            r1 = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255))
            r2 = cv2.inRange(hsv, (155, 60, 60), (180, 255, 255))
            masks.append((cv2.bitwise_or(r1, r2), self._RED_CHIP_MIN_AREA, "red"))
        if "green" in chip_colors:
            # 饱和绿（chip）vs 桌面绿：chip sat ≥ 150；桌面绿 sat ≤ 110
            g = cv2.inRange(hsv, (70, 140, 80), (100, 255, 255))
            masks.append((g, self._GREEN_CHIP_MIN_AREA, "green"))
        if not masks:
            return None
        # 选每个 mask 内最大轮廓，再在所有候选中取面积最大者
        best_c = None
        best_area = 0
        for mask, min_area, _color in masks:
            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            a = cv2.contourArea(c)
            if a < min_area:
                continue
            if a > best_area:
                best_area = a
                best_c = c
        if best_c is None:
            return None
        max_c = best_c

        # ---- 公共牌占用过滤：主轮廓在公共牌 ROI 内 → 不是筹码，是牌面 rank ----
        if has_board_cards:
            mr_x, mr_y, mr_w, mr_h = cv2.boundingRect(max_c)
            # 轮廓重心（裁剪图坐标 → 全图分数坐标）
            cx_full = (xa + mr_x + mr_w / 2) / W
            cy_full = (ya + mr_y + mr_h / 2) / H
            for broi in getattr(self._layout, 'board_rois', ()):
                if broi[0] <= cx_full <= broi[2] and broi[1] <= cy_full <= broi[3]:
                    return None

        # ---- 紧凑裁剪：围绕红色区域 + padding 包含白色数字文字 ----
        rx, ry, rw, rh = cv2.boundingRect(max_c)
        pad = max(rw, rh)
        ch, cw = crop.shape[:2]
        tx0 = max(0, rx - pad)
        ty0 = max(0, ry - pad)
        tx1 = min(cw, rx + rw + pad)
        ty1 = min(ch, ry + rh + pad)
        tight = crop[ty0:ty1, tx0:tx1]
        if tight.size == 0:
            return None

        # 放大 5x（小数字如 "2" 在 3x 下仍被 OCR 漏检，5x 可识别）
        tight_big = cv2.resize(tight, None, fx=5.0, fy=5.0,
                                interpolation=cv2.INTER_CUBIC)

        # 第一遍：直接用彩色图 OCR
        res = _ocr_run(self._ocr, tight_big) or []

        # FIX(v0.7 小芯单字 1/2)：白色数字写在红芯/红卡上时 RapidOCR 经常读不出
        # （白字红底对比对 OCR 不友好）。第二遍：抽白色像素 + 反色 → 黑字白底，
        # OCR 对此最友好，能稳定读出 "1"、"2" 等小单字。
        #
        # FIX(v0.7.1 661/781 s2 误读 1→6)：白色反转有时会把"1"读成"6. 1"
        # （pill 边缘的 "6" 字形伪影），分数还更高 → 错读 6。
        # 改为门控：只在第一遍 OCR 没拿到任何 *单字数字* 结果时才跑第二遍。
        # 第一遍能读到 "1" / "2" / "4" 这类干净单字时直接相信它。
        def _has_clean_digit(ocr_res):
            for box, text, score in ocr_res or []:
                t = (text or "").strip()
                if not t: continue
                # 同时排除噪声关键词（避免被 "POKER" 等 partial 命中）
                if any(kw in t for kw in self._layout.noise_keywords):
                    continue
                if any(kw in t for kw in ('底池', '池', '秒', '留座', '倒计时')):
                    continue
                if re.search(r'[()（）/]', t):
                    continue
                # 干净单字数字 (例如 "1"、"4"、"22")，无 "."、空格、其他字符干扰
                if re.fullmatch(r'\d{1,4}', t):
                    return True
            return False

        if not _has_clean_digit(res):
            hsv_t = cv2.cvtColor(tight_big, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(hsv_t, (0, 0, 200), (180, 80, 255))
            white_mask = cv2.morphologyEx(
                white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            if cv2.countNonZero(white_mask) > 50:
                inv = cv2.bitwise_not(white_mask)
                inv_bgr = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
                res2 = _ocr_run(self._ocr, inv_bgr) or []
                res = list(res) + list(res2)

        if not res:
            return None

        best_val = None
        best_score = -1.0
        for box, text, score in res:
            if not text:
                continue
            t = text.strip()
            # 过滤噪声关键词
            if any(kw in t for kw in self._layout.noise_keywords):
                continue
            if any(kw in t for kw in ('底池', '池', '秒', '留座', '倒计时')):
                continue
            if re.match(r'^\d{1,3}[Ss]$', t):
                continue
            # 过滤盲注格式文字（如 "4(2)"、"1/2/4（2）" 的残片）
            if re.search(r'[()（）/]', t):
                continue
            val = _parse_int(t)
            if val is None or val < 1:
                continue
            if pot_value and val == pot_value:
                continue
            if exclude_values and val in exclude_values:
                continue
            # max_stack 过滤
            max_stk = getattr(self._layout, 'max_stack', None)
            if max_stk and val > max_stk:
                continue
            # 选置信度最高的（OCR score）
            # FIX: 防御性转 float —— 旧版 rapidocr 偶尔返回 str/None
            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = 0.0
            if score_f > best_score or best_val is None:
                best_score = score_f
                best_val = val
        return best_val

    # ==================================================================
    # 白色文字下注检测（DPZX 用）
    # ==================================================================
    def _detect_bet_white_text(self, img: np.ndarray, roi: tuple) -> int | None:
        """在指定 ROI 区域直接 OCR 识别下注金额。

        DPZX 的下注金额为白色数字（⊕图标+数值），位于座位头像与牌桌中心之间。
        ROI 已精确标定到数字文本位置（排除筹码图标），直接放大裁剪区域跑 OCR。
        不使用 HSV 白色掩码（绿色桌面与白色文字 HSV 重叠，掩码会破坏文字形状）。
        通过文本区域亮度过滤剔除深色水印文字（如 "No.14"、"桌号:36"）。
        """
        crop = self._roi(img, roi)
        if crop.size == 0:
            return None
        ch, cw = crop.shape[:2]
        if ch < 5 or cw < 5:
            return None

        # 灰度图 → 二值化：仅保留亮度 > 130 的像素（白色下注数字）
        # 暗色水印（"No.14"、"桌号:36" 等，灰度 < 100）被置黑，
        # 防止 OCR 将水印数字与下注数字合并（如 "36"+"777" → "3777"）。
        # 绿色桌面灰度 ≈ 125-130，阈值 130 可在保留白字的同时过滤水印。
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_crop, 130, 255, cv2.THRESH_BINARY)

        # SPEED 前置门控：白色数字大概覆盖 40-150 px²；少于 30 px 的亮区
        # 一定不是有效下注文本（绿桌底亮度 ~125-130 阈值后是 0，没有杂讯）。
        # 跳过 OCR 可省掉一次重型推理调用。dpzx 8 个座位有 6-7 个不下注是常态，
        # 实测把 dpzx 平均耗时从 ~4400ms 降到 ~3200ms。
        bright_count = int(cv2.countNonZero(binary))
        if bright_count < 30:
            return None

        ocr_input = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # 放大 3 倍提高 OCR 准确度
        ocr_bgr = cv2.resize(ocr_input, (cw * 3, ch * 3),
                              interpolation=cv2.INTER_CUBIC)

        res = _ocr_run(self._ocr, ocr_bgr)
        if not res:
            return None

        # 逐条 OCR 结果过滤：
        # 0) 亮度过滤：白色下注数字 P90 亮度 > 140，深色水印 < 100
        # 1) 含中文/UI 符号 → 直接跳过（"跟分"、"100%底池" 等）
        # 2) 数字占比 ≥ 40% → 提取数字（允许 OCR 噪声字母如 "50 K" → 50）
        candidates: list[tuple[int, float]] = []
        scaled_h = ch * 3
        for box, text, _ in res:
            t = (text or "").strip()
            if not t:
                continue

            # ---- 亮度过滤：剔除深色水印/背景文字 ----
            if box is not None:
                try:
                    pts = np.array(box) / 3.0  # 缩回原始 crop 坐标
                    y_min = max(0, int(min(p[1] for p in pts)))
                    y_max = min(ch, int(max(p[1] for p in pts)) + 1)
                    x_min = max(0, int(min(p[0] for p in pts)))
                    x_max = min(cw, int(max(p[0] for p in pts)) + 1)
                    if y_max > y_min and x_max > x_min:
                        region = gray_crop[y_min:y_max, x_min:x_max]
                        # 用 90 百分位亮度：白色文字顶部像素 > 200，水印 < 120
                        if np.percentile(region, 90) < 140:
                            continue
                except Exception:
                    pass

            # 先剥离动作按钮标签（"下分"、"加分"、"跟分"），
            # 这些白色文字经二值化后可能与下注数字粘连，
            # 例如 OCR 读出 "下分1,020"，剥离后保留 "1,020"。
            t = re.sub(r'[下加跟]分', '', t).strip()
            # YOU WIN/赢得提示 "+166" 不是下注
            if t.startswith('+'):
                continue
            if not t:
                continue
            # 含中文 → UI 文字，不是下注数字
            if _CHINESE_RE.search(t):
                continue
            # 含 UI 符号 → 跳过（如 "100%底池" OCR 有时只读到 "100%"）
            if _UI_NOISE_RE.search(t):
                continue
            # 数字占比 < 40% → 大概率是纯文字噪声，跳过
            digit_cnt = sum(c.isdigit() for c in t)
            if digit_cnt == 0 or digit_cnt / len(t) < 0.4:
                continue
            # 排除倒计时（如 "15s"）
            t = re.sub(r'\d{1,2}[Ss秒]', '', t)
            # 排除盲注格式（如 "50/100"）
            t = re.sub(r'\d+/\d+', '', t)
            # 计算此 OCR 文本在 ROI 内的归一化 Y 位置
            y_norm = 0.0
            if box is not None:
                try:
                    y_center = sum(p[1] for p in box) / len(box)
                    y_norm = y_center / scaled_h
                except Exception:
                    pass
            for m in _INT_RE.finditer(t):
                try:
                    val = int(m.group(0).replace(",", "").replace("，", "").replace(".", ""))
                    # 最小下注阈值 10：过滤 OCR 噪声（"1"、"9" 等杂点）
                    if val >= 10:
                        candidates.append((val, y_norm))
                except ValueError:
                    continue
        if not candidates:
            return None
        # 优先取最靠近顶部（头像位置）的数字。
        # 真实下注在 ROI 上方（y_norm < 0.35），"100%底池" 等按钮在下方。
        # 如果顶部有数字就只用顶部的；否则用全部。
        top = [(v, y) for v, y in candidates if y < 0.35]
        pool = top if top else candidates
        return max(v for v, _ in pool)

    # ==================================================================
    # is_hero_turn — 倒计时检测 (DPZX) / 红色按钮 (其他 app)
    # ==================================================================
    # ==================================================================
    # 空座视觉判定（poler 专用兜底）
    # ==================================================================
    def _is_seat_avatar_empty(self, img: np.ndarray, seat: int) -> bool:
        """向后兼容包装：仅当头像为"绿桌+号占位"时返回 True。"""
        return self._seat_visual_inactivity(img, seat) == "empty"

    def _seat_visual_inactivity(self, img: np.ndarray, seat: int) -> str | None:
        """头像区视觉判定：返回 'empty' / 'waiting' / None。

        实测（poler，1080x2400）：
          - 真空座 "+ 占位"        : green > 60%, colorful < 15%, white < 10%
          - 真实玩家头像           : green < 5%,  colorful > 10%, white < 10%
          - 离线/暂离 (白色药丸+话筒): green < 2%,  colorful < 12%, white > 50%
        三类在小样本上 100% 区分。'waiting' 既包含"留座 N 秒"也包含
        断线/电话图标的整圈白色 pill 占位（951776571512 seat6）。

        仅在 layout.enable_empty_seat_visual_check=True 且 layout 提供
        seat_positions 时调用。dpzx 默认关闭。
        """
        seat_pos = (self._layout.seat_positions or {}).get(seat)
        if not seat_pos:
            return None
        H, W = img.shape[:2]
        cx, cy = int(seat_pos[0] * W), int(seat_pos[1] * H)
        # 头像半径约占短边 4%（1080x2400 上 ~40px）
        rad = max(20, int(min(W, H) * 0.04))
        y0, y1 = max(0, cy - rad), min(H, cy + rad)
        x0, x1 = max(0, cx - rad), min(W, cx + rad)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        green = (hue >= 75) & (hue <= 100) & (sat > 50)
        colorful_non_green = (sat > 80) & ~green
        white = (sat < 40) & (val > 200)
        green_ratio = float(green.mean())
        color_ratio = float(colorful_non_green.mean())
        white_ratio = float(white.mean())
        if green_ratio > 0.60 and color_ratio < 0.15:
            return "empty"
        # 整圈白色 pill（断线/电话图标 / 留座占位）：白色 > 50%，
        # 同时 green 和 colorful 都很低（不是真实玩家、也不是绿桌）。
        if white_ratio > 0.50 and color_ratio < 0.18 and green_ratio < 0.10:
            return "waiting"
        return None

    def _detect_turn_fast(self, img: np.ndarray) -> bool:
        """轮次快速预判（turn-gated 模式专用）：尽量不跑全图 OCR。
        dpzx 等倒计时 app：只对 hero 倒计时小块做局部 OCR；
        其他(poler)：纯 CV 红色操作按钮检测。
        宁可偏向"是"（误真→后续完整识别再纠正），避免误"否"漏掉真轮次。"""
        if getattr(self._layout, 'hero_countdown_roi', None):
            return self._detect_turn_countdown_local(img)
        return bool(self._detect_hero_turn(img))

    def _detect_turn_countdown_local(self, img: np.ndarray) -> bool:
        """dpzx：仅对 hero 倒计时 ROI 做白字提取+局部 OCR（不依赖全图 OCR）。
        逻辑同 _detect_hero_turn_countdown 的 Step2。"""
        roi = getattr(self._layout, 'hero_countdown_roi', None)
        if not roi:
            return False
        crop = self._roi(img, roi)
        if crop.size == 0:
            return False
        ch, cw = crop.shape[:2]
        if ch < 5 or cw < 5:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 80, 255))
        if np.count_nonzero(white_mask) / (ch * cw) < 0.005:
            return False
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        ocr_bgr = cv2.cvtColor(cv2.bitwise_not(white_mask), cv2.COLOR_GRAY2BGR)
        ocr_bgr = cv2.resize(ocr_bgr, (cw * 3, ch * 3), interpolation=cv2.INTER_CUBIC)
        res = _ocr_run(self._ocr, ocr_bgr)
        if not res:
            return False
        _cd = re.compile(r'^(\d{1,2})[Ss秒]$')
        for _, text, _s in res:
            m = _cd.match((text or "").strip())
            if m and 1 <= int(m.group(1)) <= 15:
                return True
        return False

    def _empty_result(self) -> dict:
        """turn-gated 非轮次时的置空返回（保留完整结构，is_hero_turn=False）。"""
        return {
            "table_info": {
                "stage": "unknown", "community_cards": [],
                "main_pot": None, "current_bet": None,
                "button_seat": None, "blind_size": None,
            },
            "hero_info": {
                "seat": 1, "status": "unknown", "stack": None,
                "current_bet": None, "is_hero_turn": False, "hero_cards": [],
            },
            "villains_info": [],
        }

    def _detect_hero_turn(self, img: np.ndarray) -> bool | None:
        """Fallback：红色操作按钮检测（非 DPZX app 用）。

        FIX(poler)：旧实现存在两个独立缺陷——
          1) y 取错：原 (0.79, 0.88) 实际指向按钮下方的标签行 + hero 手牌区，
             与红心♥/方块♦严重重叠，假阳性超 80%。
          2) HSV 太宽 (hue 0-15)：会捕获橙色 "偷偷看牌" 标签，
             造成 hand-end 阶段也误判为 hero 轮次。
        新实现：
          - y 锁定按钮圆本身（layout.action_btn_y 由各 layout 标定）。
          - x 可选限制（layout.action_btn_x 默认全宽），弃牌按钮独占左半即可避开
            hero 卡片红心干扰。
          - hue 收紧到 0-8 + 172-180 (深红)，刚好跨过 X 弃牌按钮但跳过橙色标签。
        """
        h, w   = img.shape[:2]
        y1, y2 = self._layout.action_btn_y
        x1, x2 = getattr(self._layout, 'action_btn_x', (0.0, 1.0))
        region = img[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
        if region.size == 0: return None
        hsv  = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # 深红：hue 0-8 + 172-180; saturation > 140; value > 100
        red1 = cv2.inRange(hsv, (0,   140, 100), (8,   255, 255))
        red2 = cv2.inRange(hsv, (172, 140, 100), (180, 255, 255))
        ratio = cv2.countNonZero(red1|red2) / (region.shape[0]*region.shape[1])
        return ratio >= self._layout.action_red_threshold

    def _detect_hero_turn_countdown(self, img: np.ndarray,
                                     items: list[OcrItem]) -> bool:
        """DPZX 专用：从 1 号位头像区域检测 ≤15s 倒计时白色数字。

        只匹配带 "s" / "秒" 后缀的数字（如 "12s"、"8秒"），
        纯数字不算（避免和筹码/下注数字混淆）。

        两步策略：
          1) 先在全图 OCR 结果中搜索 hero 头像区域的倒计时文字
          2) 全图 OCR 没找到 → 对头像区域做白色文字提取 + 单独 OCR
        """
        roi = self._layout.hero_countdown_roi
        if not roi:
            return False
        x1, y1, x2, y2 = roi

        # 倒计时正则：必须有 s/S/秒 后缀
        _CD_RE = re.compile(r'^(\d{1,2})[Ss秒]$')

        # ── Step 1: 从已有 OCR items 中搜索 ──
        for it in items:
            if x1 <= it.cx <= x2 and y1 <= it.cy <= y2:
                t = it.text.strip()
                m = _CD_RE.match(t)
                if m and 1 <= int(m.group(1)) <= 15:
                    return True

        # ── Step 2: 白色文字提取 + 单独 OCR ──
        crop = self._roi(img, roi)
        if crop.size == 0:
            return False
        ch, cw = crop.shape[:2]
        if ch < 5 or cw < 5:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 80, 255))
        if np.count_nonzero(white_mask) / (ch * cw) < 0.005:
            return False

        kernel = np.ones((2, 2), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        ocr_gray = cv2.bitwise_not(white_mask)
        ocr_bgr = cv2.cvtColor(ocr_gray, cv2.COLOR_GRAY2BGR)
        ocr_bgr = cv2.resize(ocr_bgr, (cw * 3, ch * 3),
                              interpolation=cv2.INTER_CUBIC)

        res = _ocr_run(self._ocr, ocr_bgr)
        if not res:
            return False

        for _, text, _ in res:
            t = (text or "").strip()
            m = _CD_RE.match(t)
            if m and 1 <= int(m.group(1)) <= 15:
                return True
        return False

    # ==================================================================
    # CLAHE
    # ==================================================================
    def _enhance_inplace(self, img_small: np.ndarray, roi: tuple) -> None:
        hs, ws = img_small.shape[:2]
        x1, y1, x2, y2 = roi
        xa, ya = int(x1*ws), int(y1*hs)
        xb, yb = int(x2*ws), int(y2*hs)
        region = img_small[ya:yb, xa:xb]
        if region.size == 0: return
        gray     = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        img_small[ya:yb, xa:xb] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # ==================================================================
    # OCR
    # ==================================================================
    def _ocr_full(self, bgr: np.ndarray) -> list[OcrItem]:
        h, w = bgr.shape[:2]
        res = _ocr_run(self._ocr, bgr)
        if not res: return []
        out = []
        for box, text, score in res:
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            out.append(OcrItem(
                text=text or "", cx=sum(xs)/len(xs)/w,
                cy=sum(ys)/len(ys)/h, score=float(score),
            ))
        return out

    # ------------------------------------------------------------------
    # ROI-OCR：1+N 架构的"N"部分。对小 ROI 并发跑 OCR，返回全图归一化坐标。
    # 替换全图 OCR，速度通常快 5-10 倍，准确率因 CLAHE 只作用于小图而更高。
    # ------------------------------------------------------------------
    def _ocr_strip(self, bgr: np.ndarray, frac_roi: tuple) -> list[OcrItem]:
        """OCR 一个小矩形 ROI，返回 OcrItem 列表（坐标已映射回全图 fractional）。"""
        H, W = bgr.shape[:2]
        x0, y0, x1, y1 = frac_roi
        xa, ya = max(0, int(x0 * W)), max(0, int(y0 * H))
        xb, yb = min(W, int(x1 * W)), min(H, int(y1 * H))
        if xb <= xa or yb <= ya: return []
        crop = bgr[ya:yb, xa:xb]
        if crop.size == 0: return []
        res = _ocr_run(self._ocr, crop)
        if not res: return []
        out = []
        for box, text, score in res:
            if box is None: continue
            xs = [p[0] for p in box]; ys = [p[1] for p in box]
            cx_px = (sum(xs) / len(xs)) + xa
            cy_px = (sum(ys) / len(ys)) + ya
            out.append(OcrItem(
                text=text or "", cx=cx_px / W, cy=cy_px / H,
                score=float(score),
            ))
        return out

    def _build_strips(self) -> list[tuple]:
        """构造覆盖所有信息的小 ROI 列表（全图 fractional 坐标）。
        重叠没关系——_merge_items 会去重。"""
        strips = [
            # 顶部横条：底池 / 盲注 / 当前轮次 / 跟注提示
            (0.00, 0.00, 1.00, 0.13),
            # 底池周边（跨度稍宽，兼容庄位图标遮挡）
            (0.30, 0.23, 0.70, 0.40),
            # 底部 hero 行：筹码 / 状态 / 下注金额 / 操作按钮文字
            (0.00, 0.78, 1.00, 1.00),
            # 中央（盲注 50/100 这种）
            (0.20, 0.56, 0.80, 0.62),
        ]
        # 每个座位：头像中心 ±0.08 宽、上方 0.04 到下方 0.08
        for seat, ac in self._layout.seat_anchors:
            x0 = max(0.0, ac.cx - 0.095); x1 = min(1.0, ac.cx + 0.095)
            y0 = max(0.0, ac.cy - 0.055); y1 = min(1.0, ac.cy + 0.085)
            strips.append((x0, y0, x1, y1))
        # 每个座位的 bet_anchor（下注金额可能飘出座位框，加一层保险）
        for seat, ac in self._layout.seat_bet_anchors.items():
            x0 = max(0.0, ac.cx - 0.06); x1 = min(1.0, ac.cx + 0.06)
            y0 = max(0.0, ac.cy - 0.035); y1 = min(1.0, ac.cy + 0.035)
            strips.append((x0, y0, x1, y1))
        return strips

    @staticmethod
    def _merge_items(lists: list[list[OcrItem]]) -> list[OcrItem]:
        """合并多个 ROI 的 OCR 结果，按 (文本, cx/cy 近似) 去重（重叠 ROI 会重复读）。"""
        seen = set()
        out: list[OcrItem] = []
        for lst in lists:
            for it in lst:
                key = (it.text.strip(), round(it.cx, 3), round(it.cy, 3))
                if key in seen: continue
                seen.add(key); out.append(it)
        return out

    def _ocr_all_strips(self, bgr: np.ndarray,
                         strips: list[tuple]) -> list[OcrItem]:
        """并发跑所有 ROI OCR，合并+去重返回。"""
        futures = [_executor.submit(self._ocr_strip, bgr, r) for r in strips]
        results = [f.result() for f in futures]
        return self._merge_items(results)

    # ==================================================================
    # 卡牌白矩形检测（手牌关键：ROI 里真卡只占约 68%）
    # ==================================================================
    @staticmethod
    def _find_card_bbox(crop: np.ndarray) -> tuple | None:
        """在 crop 里定位卡牌矩形，返回 crop 内的像素坐标 (x,y,w,h)，或 None。
        两级阈值：先找白色卡（在局时）；找不到再放宽到灰色（弃牌置灰时）。"""
        ch, cw = crop.shape[:2]
        if ch < 20 or cw < 20: return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        for thr in (180, 120, 90):
            _, th = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if cv2.contourArea(c) > ch*cw*0.10]
            if cnts:
                return cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return None

    @staticmethod
    def _dim_trim_bbox(crop: np.ndarray, bb: tuple) -> tuple:
        """暗牌横向/纵向收紧（poler 专属）。
        find_card_bbox 的白阈值对暗牌（最亮才~130）裁不动，常返回带头像暗边的全宽框，
        导致 rank/suit patch 抓到暗区+残缺字符（如 5♣ 被读成 Q）。
        用按列/行的高百分位亮度（忽略暗字形）定位卡面真实边界，只在确有暗边时收紧；
        亮牌满幅时为 no-op。"""
        x, y, w, h = bb
        sub = crop[y:y+h, x:x+w]
        if sub.size == 0 or w < 10 or h < 10:
            return bb
        g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        ref = float(np.percentile(g, 75))      # 卡面参考亮度（暗字形被高百分位忽略）
        thr = max(ref * 0.6, 55.0)
        col = np.percentile(g, 75, axis=0)
        row = np.percentile(g, 75, axis=1)
        cols = np.where(col > thr)[0]
        rows = np.where(row > thr)[0]
        if len(cols) < 3 or len(rows) < 3:
            return bb
        nx, nw = x + int(cols.min()), int(cols.max() - cols.min() + 1)
        ny, nh = y + int(rows.min()), int(rows.max() - rows.min() + 1)
        # 防过度收紧：保留至少原框 40% 宽高，否则视为异常、保持原框
        if nw < w * 0.4 or nh < h * 0.4:
            return bb
        return (nx, ny, nw, nh)

    def _is_dim_faceup_hero(self, raw_crop: np.ndarray) -> bool:
        """dpzx 暗结算正面手牌检测（layout 开 hero_dim_faceup 时生效）。
        摊牌/弃牌后 hero 正面牌整体变暗（亮度~99-110），低于正常阈值会被跳过；
        但同区间还有【蓝色牌背】(未看牌, 亮度~101)。实测 dim 正面牌 blue≈1%、
        牌背 blue≈42%，用 blue 占比即可干净区分。再要求有浅色卡面排除纯牌桌。"""
        if not getattr(self._layout, 'hero_dim_faceup', False):
            return False
        if raw_crop is None or raw_crop.size == 0:
            return False
        g = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
        if float(np.mean(g)) < getattr(self._layout, 'hero_dim_min_bright', 90.0):
            return False
        hsv = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2HSV)
        tot = max(1, raw_crop.shape[0] * raw_crop.shape[1])
        blue = cv2.inRange(hsv, (95, 80, 40), (130, 255, 255))
        if cv2.countNonZero(blue) / tot > getattr(self._layout, 'hero_dim_blue_reject', 0.20):
            return False   # 蓝色牌背 → 未看牌，非正面牌
        white = cv2.inRange(hsv, (0, 0, 135), (180, 70, 255))
        if cv2.countNonZero(white) / tot < 0.015:
            return False   # 无浅色卡面 → 纯牌桌/空
        return True

    def _hero_card_crop(self, img: np.ndarray, roi: tuple) -> tuple:
        """返回 (card_crop, card_roi_frac) —— card_roi_frac 是真卡在整图的分数坐标。
        如果检测失败，返回 ROI 本身。"""
        crop = self._roi(img, roi)
        bb = self._find_card_bbox(crop)
        # poler 暗牌：白阈值 bbox 常含头像暗边，按亮度剖面再收紧（dpzx 不开）
        if bb is not None and getattr(self._layout, 'hero_dim_localize', False):
            bb = self._dim_trim_bbox(crop, bb)
        x1f, y1f, x2f, y2f = roi
        cw_f, ch_f = x2f-x1f, y2f-y1f
        if bb is None or crop.size == 0:
            return crop, roi
        x, y, bw, bh = bb
        ch_px, cw_px = crop.shape[:2]
        # 换算回整图分数坐标
        card_x1 = x1f + x / cw_px * cw_f
        card_y1 = y1f + y / ch_px * ch_f
        card_x2 = x1f + (x+bw) / cw_px * cw_f
        card_y2 = y1f + (y+bh) / ch_px * ch_f
        return crop[y:y+bh, x:x+bw], (card_x1, card_y1, card_x2, card_y2)

    # ==================================================================
    # rank 识别（手牌先定位真卡，再在真卡内部取角标；公共牌沿用旧方案）
    # ==================================================================
    def _find_ranks(self, img, items, rois):
        rank_map: dict[int, str] = {}
        # 保存每张卡的 OCR 置信度（用于判断是否需要模板校验）
        ocr_score: dict[int, float] = {}
        # 保存每张卡裁下的 rank-patch（CNN/模板匹配）
        rank_patches: dict[int, np.ndarray] = {}
        missing: list = []  # (idx, search_roi, search_crop, is_hero)
        board_count = len(self._layout.board_rois)
        rx_board_lo, rx_board_hi = self._layout.card_rank_rx
        rx_hero_lo,  rx_hero_hi  = self._layout.hero_card_rank_rx
        ry_board_lo, ry_board_hi = self._layout.board_card_rank_ry
        ry_hero_lo,  ry_hero_hi  = self._layout.hero_card_rank_ry
        # 若已训练 CNN，直接走 CNN 路径
        _app = self._layout.app_name
        cnn_sess = _get_rank_classifier(_app)

        for idx, roi in enumerate(rois):
            is_hero = idx >= board_count
            # 手牌：先检查原始 ROI 亮度，弃牌牌面整体变暗（~75-85），
            # 直接跳过，避免 _hero_card_crop 在暗区找到局部亮点而误检。
            #
            # FIX(face-up dimmed)：AA Poker 在 hero 弃牌后保留正面但灰暗的卡牌（亮度
            # 70-90）—— 与 preflop 未看牌的红色 AA POKER 牌背（亮度 84-95）区间重叠，
            # 单纯靠亮度无法区分。
            # 用红色像素比例作"face-down 门控"：
            #   - 红色背 >50%（红 AA POKER 牌背）→ face-down，跳过
            #   - 红色背 <50% AND 亮度足够 → 视作 face-up（含 dim 状态），交给后续解析
            # 这样 poler 在 status=folded 时能把灰暗的明牌 rank/suit 也读出来。
            # dpzx 通过 layout 默认禁用此分支（face_down_red_ratio_threshold=0.0），
            # 行为完全与旧版一致。
            _fd_thr = getattr(self._layout, 'face_down_red_ratio_threshold', 0.0)
            dim_faceup = False
            if is_hero:
                raw_crop = self._roi(img, roi)
                if raw_crop.size > 0:
                    # 先做 face-down 红背门控（仅当 layout 启用）
                    if _fd_thr > 0.0:
                        hsv = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2HSV)
                        red_mask = (cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
                                    | cv2.inRange(hsv, (155, 80, 60), (180, 255, 255)))
                        red_r = cv2.countNonZero(red_mask) / max(1, raw_crop.shape[0]*raw_crop.shape[1])
                        if red_r > _fd_thr:
                            continue
                    raw_bright = float(np.mean(cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)))
                    if raw_bright < self._layout.hero_card_brightness_threshold:
                        # dpzx 暗结算正面牌：亮度低于阈值但非蓝牌背 → 仍识别
                        if self._is_dim_faceup_hero(raw_crop):
                            dim_faceup = True
                        else:
                            continue
                crop, search_roi = self._hero_card_crop(img, roi)
            else:
                crop = self._roi(img, roi)
                search_roi = roi
            if crop.size == 0: continue
            # 亮度二次检查（crop 后的精细检查）；dim 正面牌已通过专门门控，跳过此检查
            bright_thr = self._layout.hero_card_brightness_threshold if is_hero \
                         else self._layout.card_brightness_threshold
            if not dim_faceup and np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)) < bright_thr:
                continue

            x1, y1, x2, y2 = search_roi
            cw, ch = x2-x1, y2-y1
            ry_lo, ry_hi = (ry_hero_lo, ry_hero_hi) if is_hero else (ry_board_lo, ry_board_hi)
            rx_lo, rx_hi = (rx_hero_lo, rx_hero_hi) if is_hero else (rx_board_lo, rx_board_hi)
            rx1, rx2 = x1+rx_lo*cw, x1+rx_hi*cw
            ry1, ry2 = y1+ry_lo*ch, y1+ry_hi*ch
            # 切 patch（给 CNN 和模板用）
            cch, ccw = crop.shape[:2]
            rp = crop[int(cch*ry_lo):int(cch*ry_hi), int(ccw*rx_lo):int(ccw*rx_hi)]
            if rp.size > 0:
                rank_patches[idx] = rp

            # 1) 如果有 CNN 分类器：跑 CNN，再用 OCR 交叉验证
            if cnn_sess is not None and rp.size > 0:
                res = _classify_rank_cnn(rp, _app)
                # 阈值从 layout 读取：dpzx 默认 0.40，poler 降到 0.25
                _rank_min_conf = getattr(self._layout, 'rank_cnn_min_conf', 0.40)
                if res is not None and res[1] >= _rank_min_conf:
                    cnn_rank, cnn_conf = res
                    # OCR 交叉验证阈值从 layout 读取。
                    # dpzx 默认 0.70；poler 设 0.0 → 完全跳过 OCR 校验
                    # （OCR 对 AA Poker 小角标识别不稳，实测反而降准确度）。
                    _ocr_verify_thr = getattr(
                        self._layout, 'rank_ocr_verify_threshold', 0.70)
                    _ocr_override_min = getattr(
                        self._layout, 'rank_ocr_override_min_score', 0.70)
                    if cnn_conf < _ocr_verify_thr:
                        rp_ocr = _enhance_red_for_ocr(rp) if _has_red_text(rp) else rp
                        rp_big = cv2.resize(rp_ocr, None, fx=4.0, fy=4.0,
                                            interpolation=cv2.INTER_CUBIC)
                        ocr_items = self._ocr_full(rp_big)
                        for oit in ocr_items:
                            orank = self._parse_rank(oit.text)
                            if orank and orank != cnn_rank and oit.score > _ocr_override_min:
                                cnn_rank = orank
                                break
                    # A/K/Q/9 消歧：CNN+OCR 容易混淆 A↔K↔Q，
                    # 也会把 9 误判为 Q（两者都有圆形孔洞）。
                    # 用拓扑特征（孔洞大小 + 形状）做最终判断。
                    _akq_ht = self._layout.akq_hero_thresholds
                    _akq_bt = self._layout.akq_board_thresholds
                    if cnn_rank in ("A", "K"):
                        akq = _disambiguate_AKQ(rp, _akq_ht, _akq_bt)
                        if akq is not None:
                            cnn_rank = akq
                    elif cnn_rank == "Q":
                        akq = _disambiguate_AKQ(rp, _akq_ht, _akq_bt)
                        if akq == "A":
                            cnn_rank = akq
                        # Q↔8 消歧：CNN 把 8 误判为 Q（两者都有圆形孔洞），
                        # 但 8 有上下双闭环、Q 只有单环+尾巴。
                        # 复用 _disambiguate_6_vs_8 的双孔检测：返回 "8" 说明上下都有环。
                        elif cnn_conf < 0.70:
                            d_q8 = _disambiguate_6_vs_8(rp)
                            if d_q8 == "8":
                                cnn_rank = "8"
                    # 2/7 消歧：CNN 容易混淆 2↔7
                    if cnn_rank == "7":
                        d27 = _disambiguate_7_vs_2(rp)
                        if d27 == "2":
                            cnn_rank = "2"
                    # 6/8 消歧（poler 专属）：CNN 对小号字 6/8 上半闭合判别不稳。
                    # 仅在 cnn_rank=="8" 且 CNN conf 偏低 (<0.70) 时校验。
                    if (cnn_rank == "8"
                            and cnn_conf < 0.70
                            and getattr(self._layout, 'enable_disambiguate_6_vs_8', False)):
                        d68 = _disambiguate_6_vs_8(rp)
                        if d68 == "6":
                            cnn_rank = "6"
                    rank_map[idx] = cnn_rank
                    ocr_score[idx] = cnn_conf
                    continue

            # 2) 没 CNN 或 CNN 置信度低 → 走 OCR
            rank, best = None, -1.0
            for it in items:
                if it.cx < rx1 or it.cx > rx2 or it.cy < ry1 or it.cy > ry2: continue
                r = self._parse_rank(it.text)
                if r and it.score > best: best = it.score; rank = r
            if rank is not None:
                # A/K/Q/9 消歧（同 CNN 路径）
                _akq_ht = self._layout.akq_hero_thresholds
                _akq_bt = self._layout.akq_board_thresholds
                if rank in ("A", "K") and rp.size > 0:
                    akq = _disambiguate_AKQ(rp, _akq_ht, _akq_bt)
                    if akq is not None:
                        rank = akq
                elif rank == "Q" and rp.size > 0:
                    akq = _disambiguate_AKQ(rp, _akq_ht, _akq_bt)
                    if akq == "A":
                        rank = akq
                # 2/7 消歧
                if rank == "7" and rp.size > 0:
                    d27 = _disambiguate_7_vs_2(rp)
                    if d27 == "2":
                        rank = "2"
                rank_map[idx] = rank
                ocr_score[idx] = best
            else:
                missing.append((idx, search_roi, crop, is_hero))

        if missing:
            patches, meta = [], []
            for idx, _roi, crop, is_hero in missing:
                ch, cw = crop.shape[:2]
                ry_lo, ry_hi = (ry_hero_lo, ry_hero_hi) if is_hero else (ry_board_lo, ry_board_hi)
                rx_lo, rx_hi = (rx_hero_lo, rx_hero_hi) if is_hero else (rx_board_lo, rx_board_hi)
                patch = crop[int(ch*ry_lo):int(ch*ry_hi), int(cw*rx_lo):int(cw*rx_hi)]
                if patch.size == 0: continue
                patch = cv2.resize(patch, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
                patches.append(patch); meta.append((idx, patch.shape[1]))
            if patches:
                th = max(p.shape[0] for p in patches)
                pieces, x_cur, slots = [], 0, []
                for (idx, pw), patch in zip(meta, patches):
                    pad   = th - patch.shape[0]
                    piece = cv2.copyMakeBorder(patch, 0, pad, 0, 10,
                                               cv2.BORDER_CONSTANT, value=(180,180,180))
                    slots.append((idx, x_cur, x_cur+pw)); x_cur += piece.shape[1]
                    pieces.append(piece)
                strip = np.hstack(pieces); sw = strip.shape[1]
                for it in self._ocr_full(strip):
                    for idx, xs, xe in slots:
                        if idx in rank_map: continue
                        if it.cx < xs/sw or it.cx > xe/sw: continue
                        r = self._parse_rank(it.text)
                        if r: rank_map[idx] = r

        # 第三轮的多预处理投票已删除——耗时太高（5-8s）。
        # 缺的 rank 全部交给模板匹配。

        # ── 第三轮：模板匹配 ──
        # 策略：
        #   1. OCR 完全没结果 → 取模板 top-1
        #   2. OCR 给了 "8"（常见 Q 误读为 8 的场景）且模板 Q 分数 > "8" 分数 + 0.10 → 改 Q
        #   其他情况一律信 OCR。
        if self._templates:
            for idx, patch in rank_patches.items():
                scores = _template_match_ranks(patch, self._templates)
                if not scores: continue
                top_rank, top_score = scores[0]
                existing = rank_map.get(idx)
                if existing is None:
                    if top_score > 0.3:
                        rank_map[idx] = top_rank
                elif existing == "8":
                    q_score = next((s for r, s in scores if r == "Q"), -1.0)
                    e_score = next((s for r, s in scores if r == "8"), -1.0)
                    if q_score >= 0 and e_score >= 0 and (q_score - e_score) > 0.10:
                        rank_map[idx] = "Q"
        return rank_map

    def _build_card_list(self, img, rank_map, rois):
        all_rois = list(self._layout.board_rois) + [
            self._layout.hero_card1_roi, self._layout.hero_card2_roi,
        ]
        board_count = len(self._layout.board_rois)
        cards = []
        for roi in rois:
            try: idx = all_rois.index(roi)
            except ValueError: continue
            rank = rank_map.get(idx)
            if rank is None: continue
            is_hero = idx >= board_count
            if is_hero:
                card_crop, _ = self._hero_card_crop(img, roi)
            else:
                card_crop = self._roi(img, roi)
            suit = self._resolve_suit(card_crop, is_hero, rank)
            if suit == "?": continue
            # 点数和花色间加空格，和 labels.txt 保持一致，方便对比。
            cards.append(f"{rank} {suit}")
        return cards

    def _crop_suit_patch(self, card_crop: np.ndarray, is_hero: bool) -> np.ndarray:
        """切花色 patch（放宽后的独立区域）。"""
        if card_crop.size == 0: return card_crop
        ch, cw = card_crop.shape[:2]
        if is_hero:
            ry_lo, ry_hi = self._layout.hero_card_suit_ry
            rx_lo, rx_hi = self._layout.hero_card_suit_rx
        else:
            ry_lo, ry_hi = self._layout.board_card_suit_ry
            rx_lo, rx_hi = self._layout.board_card_suit_rx
        return card_crop[int(ch*ry_lo):int(ch*ry_hi),
                         int(cw*rx_lo):int(cw*rx_hi)]

    def _classify_suit_by_color(self, card_crop: np.ndarray, is_hero: bool) -> str:
        """颜色优先花色识别（适用于 poler 等使用彩色花色的 app）。

        Poler 花色配色：
          红桃 hearts   = RED
          方块 diamonds  = BLUE
          梅花 clubs     = GREEN
          黑桃 spades    = BLACK

        关键：board card 的 suit patch 会采到绿色牌桌背景，
        需要通过白色卡牌区域 mask 区分牌桌绿 vs 梅花绿。
        """
        suit_patch = self._crop_suit_patch(card_crop, is_hero)
        if suit_patch.size == 0:
            return "?"

        hsv = cv2.cvtColor(suit_patch, cv2.COLOR_BGR2HSV)
        total_px = suit_patch.shape[0] * suit_patch.shape[1]
        if total_px == 0:
            return "?"

        # ── Step 1: 找到白色卡牌区域，用它构建 card_mask ──
        white_m = cv2.inRange(hsv, (0, 0, 170), (180, 75, 255))
        white_r = np.count_nonzero(white_m) / total_px

        if white_r < 0.08:
            # 几乎没有白色卡背景 → 空位或纯牌桌
            return "?"

        # 膨胀白色区域 → "卡牌区域"（花色符号紧贴白色背景）
        kernel = np.ones((5, 5), np.uint8)
        card_mask = cv2.dilate(white_m, kernel, iterations=3)
        card_px = max(np.count_nonzero(card_mask), 1)

        # ── Step 2: 在卡牌区域内检测各颜色 ──
        # RED: H in [0,12] or [168,180], S>50, V>50
        red_m = (cv2.inRange(hsv, (0, 50, 50), (12, 255, 255))
                 | cv2.inRange(hsv, (168, 50, 50), (180, 255, 255)))
        red_m = cv2.bitwise_and(red_m, card_mask)

        # BLUE: H in [95,135], S>40, V>40
        blue_m = cv2.inRange(hsv, (95, 40, 40), (135, 255, 255))
        blue_m = cv2.bitwise_and(blue_m, card_mask)

        # GREEN suit symbol: H [35,85], S>40, V>80
        green_m = cv2.inRange(hsv, (35, 40, 80), (85, 255, 255))
        green_m = cv2.bitwise_and(green_m, card_mask)

        # BLACK: V < 80 (dark pixels on card)
        dark_m = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
        dark_m = cv2.bitwise_and(dark_m, card_mask)

        # 占比基于卡牌区域像素数（而不是整个 patch）
        red_r = np.count_nonzero(red_m) / card_px
        blue_r = np.count_nonzero(blue_m) / card_px
        green_r = np.count_nonzero(green_m) / card_px
        dark_r = np.count_nonzero(dark_m) / card_px

        # ── Step 3: 颜色优先决策 ──
        # Blue → 方块（最独特，误判率极低）
        if blue_r > 0.03:
            return "方块"

        # Green on card → 梅花（已通过 card_mask 排除牌桌绿）
        if green_r > 0.03:
            return "梅花"

        # Red → 红桃
        if red_r > 0.03:
            return "红桃"

        # Black on card → 黑桃
        if dark_r > 0.05:
            return "黑桃"

        return "?"

    def _classify_corner_suit_morpho(self, card_crop: np.ndarray, is_hero: bool) -> str:
        """用卡牌左上角小花色符号做形态学分析，避免 J/Q/K 中央插画干扰。"""
        ch, cw = card_crop.shape[:2]
        if ch < 10 or cw < 10:
            return "?"
        if is_hero:
            ry_lo, ry_hi = self._layout.hero_card_suit_ry
            rx_lo, rx_hi = self._layout.hero_card_suit_rx
        else:
            ry_lo, ry_hi = self._layout.board_card_suit_ry
            rx_lo, rx_hi = self._layout.board_card_suit_rx
        region = card_crop[int(ch * ry_lo):int(ch * ry_hi),
                           int(cw * rx_lo):int(cw * rx_hi)]
        if region.size == 0:
            return "?"
        region = cv2.resize(region, (128, 128), interpolation=cv2.INTER_CUBIC)

        # ── 红色判断 ──
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        rm = (cv2.inRange(hsv, (0, 70, 70), (12, 255, 255))
              | cv2.inRange(hsv, (155, 70, 70), (179, 255, 255)))
        red_ratio = cv2.countNonZero(rm) / (128 * 128)
        if red_ratio > 0.03:
            rm = cv2.morphologyEx(rm, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            cnts, _ = cv2.findContours(rm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if cv2.contourArea(c) > 50]
            if not cnts:
                return "?"
            c = max(cnts, key=cv2.contourArea)
            _, _, bw, bh = cv2.boundingRect(c)
            aspect = bw / bh if bh > 0 else 1.0
            hull_idx = cv2.convexHull(c, returnPoints=False)
            try:
                defects = cv2.convexityDefects(c, hull_idx) if hull_idx is not None else None
                max_depth = max(d[0][3] for d in defects) / 256.0 if defects is not None else 0.0
            except Exception:
                max_depth = 0.0
            return "红桃" if (aspect > 0.95 or max_depth > 6.0) else "方块"

        # ── 黑色判断 ──
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        dk = cv2.inRange(gray, 0, 110)
        dk = cv2.morphologyEx(dk, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        total_dk = cv2.countNonZero(dk)
        if total_dk < 100:
            return "?"
        n_lab, _, stats, _ = cv2.connectedComponentsWithStats(dk)
        areas = sorted([stats[i, cv2.CC_STAT_AREA] for i in range(1, n_lab)
                         if stats[i, cv2.CC_STAT_AREA] > 30], reverse=True)
        n_comp = len(areas)
        dk_er = cv2.erode(dk, np.ones((3, 3), np.uint8), iterations=1)
        nl2, _, st2, _ = cv2.connectedComponentsWithStats(dk_er)
        areas_e = sorted([st2[i, cv2.CC_STAT_AREA] for i in range(1, nl2)
                           if st2[i, cv2.CC_STAT_AREA] > 15], reverse=True)
        n_comp_e = len(areas_e)

        # ♣ 有三瓣，侵蚀后分裂为 3 个大小相近的分量。
        # ♠ 可能因 "10" 笔画渗入或细柄断裂产生 3+ 分量，
        # 但最大分量远大于其余（主导比 > 0.65）。
        # 仅当分量个数 >= 3 且最大分量不占绝对主导时才判定为 ♣。
        is_clubs_by_comp = False
        if n_comp >= 3:
            dominant_ratio = areas[0] / sum(areas) if areas else 0
            is_clubs_by_comp = dominant_ratio < 0.65
        if not is_clubs_by_comp and n_comp_e >= 3:
            dominant_ratio_e = areas_e[0] / sum(areas_e) if areas_e else 0
            is_clubs_by_comp = dominant_ratio_e < 0.65
        if is_clubs_by_comp:
            return "梅花"

        cnts, _ = cv2.findContours(dk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_big = [c for c in cnts if cv2.contourArea(c) > 30]
        if cnts_big:
            c_big = max(cnts_big, key=cv2.contourArea)
            ha = cv2.contourArea(cv2.convexHull(c_big))
            sol = cv2.contourArea(c_big) / ha if ha > 0 else 0
            if sol >= 0.88:
                return "黑桃"
        return "梅花"

    def _suit_upper_crop(self, card_crop: np.ndarray) -> np.ndarray:
        """裁剪卡牌上半部分（含 rank 文字 + 花色符号，排除中央面牌插画）。
        用于 CNN 花色推理，减少面牌插画和数字变化对分类的干扰。
        """
        ch, cw = card_crop.shape[:2]
        return card_crop[:int(ch * 0.65), :int(cw * 0.55)]

    def _crop_suit_patch_for(self, card_crop: np.ndarray, is_hero: bool,
                             rank: str | None) -> np.ndarray:
        """rank-aware 花色 patch。
        face card (J/Q/K)：中央是人物插画，常规花色区会抓到画而非花色，
        因此改用【左上角角标小花色】(layout.face_suit_ry/rx，相对裁紧后的真卡)。
        其余 rank：沿用常规花色区。
        仅当对应 layout 定义了 face_suit_ry 时启用 face 分支（poler/dpzx 各自独立）。"""
        if rank in ("J", "Q", "K"):
            if is_hero:
                ry = getattr(self._layout, "face_hero_suit_ry", None)
                rx = getattr(self._layout, "face_hero_suit_rx", None)
            else:
                ry = getattr(self._layout, "face_board_suit_ry", None)
                rx = getattr(self._layout, "face_board_suit_rx", None)
            if ry is not None and rx is not None:
                tight = card_crop
                # board 卡传入的是原始 ROI（带边距），需 bbox 裁紧再取角标；
                # hero 卡已由 _hero_card_crop 裁紧，不可再 bbox——face card 的白底被
                # 中央人物画切割，二次 bbox 会误取顶部白条、漏掉角标花色。
                if not is_hero:
                    bb = self._find_card_bbox(card_crop)
                    if bb is not None:
                        x, y, bw, bh = bb
                        cand = card_crop[y:y+bh, x:x+bw]
                        if cand.size > 0:
                            tight = cand
                ch, cw = tight.shape[:2]
                return tight[int(ch*ry[0]):int(ch*ry[1]), int(cw*rx[0]):int(cw*rx[1])]
        return self._crop_suit_patch(card_crop, is_hero)

    def _resolve_suit(self, card_crop: np.ndarray, is_hero: bool,
                      rank: str | None = None) -> str:
        """花色识别：CNN + 形态学多信号投票。"""
        if card_crop.size == 0: return "?"

        # 0) Poler 彩色花色（仅特定版本，当前绿桌版使用标准黑/红花色）
        # 跳过，由 CNN + 形态学识别
        # if self._layout.app_name == "poler":
        #     color_suit = self._classify_suit_by_color(card_crop, is_hero)
        #     if color_suit != "?":
        #         return color_suit

        # 1) CNN 分类器（用花色 patch，和训练数据一致）
        _app = self._layout.app_name
        cnn_suit, cnn_conf = None, 0.0
        if _get_suit_classifier(_app) is not None:
            suit_patch = self._crop_suit_patch_for(card_crop, is_hero, rank)
            if suit_patch.size > 0:
                res = _classify_suit_cnn(suit_patch, _app)
                if res is not None:
                    cnn_suit, cnn_conf = res

        # 2) 形态学角标分析
        morpho_suit = self._classify_corner_suit_morpho(card_crop, is_hero)

        # 2.5) 黑色系低置信度保护（poler 专属）
        # CNN 对小角标 梅花↔黑桃 在 conf 0.5~0.65 区间不稳；形态学通过孔洞/连通块
        # 数量区分更可靠。只要 CNN、morpho 都是黑色系且 CNN conf 低于阈值，
        # 让 morpho 投票主导（黑色系专用，红色系保持原 CNN 优先策略）。
        _black_trust = getattr(self._layout, 'suit_black_cnn_trust_threshold', 0.0)
        if (_black_trust > 0.0
                and cnn_suit in ("梅花", "黑桃")
                and morpho_suit in ("梅花", "黑桃")
                and cnn_conf < _black_trust):
            return morpho_suit

        # 3) 投票决策 — CNN 重训后以 CNN 为主力
        # CNN 有结果且置信度足够时，直接采用
        if cnn_suit and cnn_conf >= 0.7:
            return cnn_suit

        # CNN 置信度较低时，参考形态学
        if cnn_suit and cnn_conf >= 0.5:
            # 同色系一致 → 取 CNN
            if morpho_suit:
                cnn_is_red = cnn_suit in ("红桃", "方块")
                morpho_is_red = morpho_suit in ("红桃", "方块")
                if cnn_is_red == morpho_is_red:
                    return cnn_suit   # 同色系内 CNN 更准
            return cnn_suit

        # CNN 无结果或极低置信度 → 用形态学
        if morpho_suit and morpho_suit != "?":
            return morpho_suit

        # 形态学无结果时用 CNN
        if cnn_suit and cnn_conf >= 0.6:
            return cnn_suit

        # 最终回退
        return self._classify_suit_from_card(card_crop, is_hero=is_hero)

    # ==================================================================
    # 花色识别
    # 红色系:
    #   手牌(is_hero=True):  resize到48x48，eps=0.03，top3加权投票
    #   公共牌(is_hero=False): 原始尺寸，convexity defects判心形凹口
    # 黑色系:
    #   公共牌: 最大轮廓solidity>=0.88 -> 黑桃
    #   手牌:   多特征投票
    # ==================================================================
    def _classify_hero_suit_by_corner(self, card_crop: np.ndarray) -> str:
        """手牌花色：在**真实白色卡牌**内部、点数字母正下方的小花色图。
        采样位置：y 33%-62%，x 0%-42% of 真卡（点数字母占 0-33%，花色在其正下方）。
        调用方必须保证 card_crop 是 _hero_card_crop 返回的真卡裁剪（不带绿背景）。

        分类逻辑：
          红色 (red_ratio > 0.03):
            宽高比 > 0.95 或顶凹口深度 > 6 → 红桃；否则 → 方块
          黑色:
            连通块数 ≥ 3（或侵蚀后 ≥ 3）→ 梅花
            单块且 solidity ≥ 0.88 → 黑桃
            否则 → 梅花
        """
        ch, cw = card_crop.shape[:2]
        if ch < 10 or cw < 10: return "?"
        # 用 layout 里放宽后的 suit ROI（不再硬编码 0.33-0.62 / 0-0.42）
        ry_lo, ry_hi = self._layout.hero_card_suit_ry
        rx_lo, rx_hi = self._layout.hero_card_suit_rx
        region = card_crop[int(ch*ry_lo):int(ch*ry_hi),
                           int(cw*rx_lo):int(cw*rx_hi)]
        if region.size == 0: return "?"
        region = cv2.resize(region, (128, 128), interpolation=cv2.INTER_CUBIC)
        area_total = 128 * 128

        # ── 红色判断 ──
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        rm  = (cv2.inRange(hsv, (0,   70, 70), (12,  255, 255))
             | cv2.inRange(hsv, (155, 70, 70), (179, 255, 255)))
        red_ratio = cv2.countNonZero(rm) / area_total

        if red_ratio > 0.03:
            rm = cv2.morphologyEx(rm, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
            cnts, _ = cv2.findContours(rm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [c for c in cnts if cv2.contourArea(c) > 50]
            if not cnts: return "?"
            c = max(cnts, key=cv2.contourArea)
            _, _, bw, bh = cv2.boundingRect(c)
            aspect = bw / bh if bh > 0 else 1.0
            hull_idx = cv2.convexHull(c, returnPoints=False)
            try:
                defects = cv2.convexityDefects(c, hull_idx) if hull_idx is not None else None
                max_depth = max(d[0][3] for d in defects) / 256.0 if defects is not None else 0.0
            except Exception:
                max_depth = 0.0
            # 心形：典型两耸起使 w > h（aspect≈1.1~1.4）；凹口深度>6 是另一条强信号
            if aspect > 0.95 or max_depth > 6.0:
                return "红桃"
            return "方块"

        # ── 黑色判断 ──
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        dk   = cv2.inRange(gray, 0, 110)
        dk   = cv2.morphologyEx(dk, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        if cv2.countNonZero(dk) < 100: return "?"

        # 原始连通块（♣ 三叶一般直接分离；♠ 单块）
        n_lab, _, stats, _ = cv2.connectedComponentsWithStats(dk)
        n_comp = sum(1 for i in range(1, n_lab) if stats[i, cv2.CC_STAT_AREA] > 30)

        # 一次侵蚀后再数（处理粘连的 ♣）
        dk_er = cv2.erode(dk, np.ones((3,3), np.uint8), iterations=1)
        nl2, _, st2, _ = cv2.connectedComponentsWithStats(dk_er)
        n_comp_e = sum(1 for i in range(1, nl2) if st2[i, cv2.CC_STAT_AREA] > 15)

        if n_comp >= 3 or n_comp_e >= 3:
            return "梅花"

        # 主轮廓 solidity：♠ 一般 ≥ 0.9（整体凸），♣ 单瓣包络后也凸但整体三瓣合起来不凸
        cnts, _ = cv2.findContours(dk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_big = [c for c in cnts if cv2.contourArea(c) > 30]
        if not cnts_big: return "?"
        c_big = max(cnts_big, key=cv2.contourArea)
        ha = cv2.contourArea(cv2.convexHull(c_big))
        sol_main = cv2.contourArea(c_big) / ha if ha > 0 else 0
        if sol_main >= 0.88:
            return "黑桃"
        return "梅花"

    def _classify_suit_from_card(self, card_crop: np.ndarray, is_hero: bool = False) -> str:
        if card_crop.size == 0: return "?"
        # FIX: 手牌用左上角小花色图（在点数字母下方）判花色——J/Q/K 中央是插画，
        # 老逻辑取中央区域会把插画的颜色当花色，导致 Q♠ 被误判为 ♣。
        if is_hero:
            suit = self._classify_hero_suit_by_corner(card_crop)
            if suit != "?":
                return suit
            # fallback to center-region logic
        ch, cw = card_crop.shape[:2]
        m = 0.06
        trimmed = card_crop[int(ch*m):int(ch*(1-m)), int(cw*m):int(cw*(1-m))]
        th, tw  = trimmed.shape[:2]
        region  = trimmed[int(th*0.38):int(th*0.92), int(tw*0.08):int(tw*0.92)]
        if region.size == 0: return "?"

        # 红色检测（在原始region上做）
        hsv_r = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        r1 = cv2.inRange(hsv_r, (0,   80, 80), (12,  255, 255))
        r2 = cv2.inRange(hsv_r, (155, 80, 80), (179, 255, 255))
        red_mask  = r1 | r2
        red_ratio = cv2.countNonZero(red_mask) / (region.shape[0] * region.shape[1])

        # ── 红色系 ──
        if red_ratio > 0.04:
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
            cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: return "?"

            if is_hero:
                # 手牌：resize到48x48后，eps=0.03，top3加权投票
                resized = cv2.resize(region, (48,48), interpolation=cv2.INTER_AREA)
                hsv_48  = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
                rm48    = cv2.inRange(hsv_48,(0,80,80),(12,255,255)) | cv2.inRange(hsv_48,(155,80,80),(179,255,255))
                rm48    = cv2.morphologyEx(rm48, cv2.MORPH_OPEN,  np.ones((3,3),np.uint8))
                rm48    = cv2.morphologyEx(rm48, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
                cnts48, _ = cv2.findContours(rm48, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                top3 = sorted(cnts48, key=cv2.contourArea, reverse=True)[:3]
                dw = hw = 0.0
                for c in top3:
                    if cv2.contourArea(c) < 20: continue
                    hull = cv2.convexHull(c); peri = cv2.arcLength(hull, True)
                    if peri == 0: continue
                    av = len(cv2.approxPolyDP(hull, 0.03*peri, True))
                    w  = cv2.contourArea(c)
                    if av <= 4: dw += w
                    else:       hw += w
                if dw == 0 and hw == 0: return "?"
                return "方块" if dw >= hw else "红桃"
            else:
                # 公共牌：用convexity defects判心形凹口（depth>5 -> 心形 -> 红桃）
                c_big = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(c_big) < 15: return "?"
                hull_idx = cv2.convexHull(c_big, returnPoints=False)
                if len(hull_idx) <= 3: return "方块"
                try:
                    defects = cv2.convexityDefects(c_big, hull_idx)
                    if defects is None: return "方块"
                    max_depth = max(d[0][3] for d in defects) / 256.0
                except Exception:
                    max_depth = 0.0
                return "红桃" if max_depth > 5.0 else "方块"

        # ── 黑色系 ──
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        dk   = cv2.inRange(gray, 0, 90)
        dk   = cv2.morphologyEx(dk, cv2.MORPH_OPEN,  np.ones((2,2), np.uint8))
        dk   = cv2.morphologyEx(dk, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

        if not is_hero:
            # 公共牌：最大轮廓solidity>=0.88 -> 黑桃
            cnts_b, _ = cv2.findContours(dk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts_b: return "?"
            cb = max(cnts_b, key=cv2.contourArea)
            if cv2.contourArea(cb) < 15: return "?"
            ha_b = cv2.contourArea(cv2.convexHull(cb))
            sol_b = cv2.contourArea(cb) / ha_b if ha_b > 0 else 0
            return "黑桃" if sol_b >= 0.88 else "梅花"

        # 手牌黑色：多特征投票
        resized2 = cv2.resize(region, (48,48), interpolation=cv2.INTER_AREA)
        gray2 = cv2.cvtColor(resized2, cv2.COLOR_BGR2GRAY)
        dk2   = cv2.inRange(gray2, 0, 90)
        dk2   = cv2.morphologyEx(dk2, cv2.MORPH_OPEN,  np.ones((2,2), np.uint8))
        dk2   = cv2.morphologyEx(dk2, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        cnts2, _ = cv2.findContours(dk2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid2 = [c for c in cnts2 if cv2.contourArea(c) > 8]
        if not valid2: return "?"
        all_pts = np.vstack([c.reshape(-1,2) for c in valid2])
        combined_area = sum(cv2.contourArea(c) for c in valid2)
        hull_area = cv2.contourArea(cv2.convexHull(all_pts))
        combined_sol = combined_area / hull_area if hull_area > 0 else 0
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(dk2)
        n_comp = sum(1 for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] > 12)
        top_dark = cv2.countNonZero(dk2[:24, :]) / (24*48)
        bot_dark = cv2.countNonZero(dk2[24:, :]) / (24*48)
        spade_score = club_score = 0
        if combined_sol > 0.55:   spade_score += 2
        elif combined_sol < 0.40: club_score  += 2
        else:                     spade_score += 1
        if n_comp >= 4:   club_score  += 2
        elif n_comp <= 2: spade_score += 2
        else:             club_score  += 1
        if top_dark - bot_dark > 0.05:    spade_score += 1
        elif top_dark - bot_dark < -0.03: club_score  += 1
        return "黑桃" if spade_score >= club_score else "梅花"


    # ==================================================================
    # 阶段 / 状态
    # ==================================================================
    @staticmethod
    def _infer_stage(cc): return {0:"preflop",3:"flop",4:"turn",5:"river"}.get(len(cc),"unknown")

    def _hero_status(self, items):
        # 收集 hero_status_anchor 附近的 OCR item，单条匹配状态关键词。
        # FIX(711 hero=fold 误判)：preflop 未看牌时动作按钮显示 "看牌/弃牌"，
        # 之前 join 后整体含 "弃牌" 直接判 folded。改为按单条匹配，
        # 含 "看牌" 的条目（动作按钮 / "看牌" 状态）按 active 处理，
        # 避免被 "弃牌" 子串污染。
        anchor = self._layout.hero_status_anchor
        nearby = []
        for it in items:
            if not it.text: continue
            d = ((it.cx-anchor.cx)**2 + (it.cy-anchor.cy)**2) ** 0.5
            if d <= anchor.max_dist:
                nearby.append(it.text)
        for t in nearby:
            tl = t.lower()
            # 排除 "看牌/弃牌" 这种 preflop 未看牌动作按钮
            if "看牌" in t and "弃牌" in t:
                continue
            if "all" in tl and "in" in tl:
                return "all_in"
            if "弃牌" in t:
                return "folded"
        # "看牌" = check (poler 术语), 视为 active
        return "active"

    # 状态关键词 → 状态映射（按优先级排列）
    # "留座" = AA Poker 玩家暂离时显示的"留座 N秒"提示（坐位保留 N 秒后自动释放），
    # 玩家不参与本手牌，按 waiting 处理。OCR 在 491776571458 等图上稳定输出 "留座N秒"。
    # "补盲" / "过庄" = 玩家筹码归零（all_in 失败/重新入座）后下一手前的选择按钮，
    # 当前手牌已结束，按 waiting 处理。在 471776571456 hand-end 图上稳定出现。
    _STATUS_KEYWORDS = (
        ("空座", "empty"),
        ("弃牌", "folded"),
        ("留座", "waiting"),
        ("补盲", "waiting"),
        ("过庄", "waiting"),
        ("等待", "waiting"),
        ("看牌", "active"),
        ("跟注", "active"),
        ("加注", "active"),
    )

    def _seat_status(self, items, seat):
        anchor = self._layout.seat_status_anchors.get(seat)
        if anchor is None: return "active"

        # 收集两个锚点附近的 OCR items（status anchor + seat/头像 anchor）
        seat_anchor_item = dict(self._layout.seat_anchors).get(seat)
        search_anchors = [anchor]
        if seat_anchor_item is not None:
            search_anchors.append(seat_anchor_item)

        # FIX: 按距离找最近的状态关键词，而非在拼接文本中按固定顺序匹配。
        # 之前的问题：seat 7 的"加注"和 seat 8 的"弃牌"都在 seat 7 搜索半径内，
        # "弃牌"排在前面导致 seat 7 被误判为 folded。
        # 现在：找到离锚点最近的含状态关键词的 OCR item，用该关键词决定状态。
        best_status = None
        best_dist = 999.0
        all_texts = []  # 用于后续空座判断

        # Y-band 限制：只接受 |dy| < max_dist*0.75 的 OCR item，
        # 防止相邻座位（Y间距≈0.12）的状态文字串扰。
        # 0.75 系数使 Y-band = 0.09（< 座位间距 0.12），杜绝越界。
        y_band = anchor.max_dist * 0.75

        for it in items:
            if not it.text:
                continue
            # 计算到最近搜索锚点的距离
            min_d = min(
                ((it.cx - a.cx)**2 + (it.cy - a.cy)**2)**0.5
                for a in search_anchors
            )
            if min_d > anchor.max_dist:
                continue
            # Y-band 过滤：状态文字不应该来自上下相邻座位
            dy_to_anchor = abs(it.cy - anchor.cy)
            if dy_to_anchor > y_band:
                # 仍然收集到 all_texts（用于后续空座判断），但不做状态匹配
                all_texts.append(it.text)
                continue
            all_texts.append(it.text)
            t = it.text
            lower_t = t.lower()
            # 检查 all-in（特殊处理，两个单词）
            if "all" in lower_t and "in" in lower_t:
                if min_d < best_dist:
                    best_dist = min_d
                    best_status = "all_in"
                continue
            # 检查中文状态关键词
            for kw, status in self._STATUS_KEYWORDS:
                if kw in t:
                    if min_d < best_dist:
                        best_dist = min_d
                        best_status = status
                    break  # 一个 item 只匹配一个关键词

        if best_status is not None:
            return best_status

        # FIX: 如果座位附近完全没有找到任何文本（无玩家名、无筹码、无状态标签），
        # 视为空座而非默认 active。真实在座玩家一定有可见的文字信息。
        # 过滤掉噪声关键词后检查
        noise = getattr(self._layout, "noise_keywords", frozenset())
        meaningful = [t for t in all_texts
                      if t.strip() and t.strip() not in noise
                      and not all(c in '.|,，' for c in t.strip())]
        if not meaningful:
            return "empty"

        return "active"

    # ==================================================================
    # 全桌跟注线（底池上方的最大下注额）
    # ==================================================================
    def _find_current_bet(self, items: list) -> int | None:
        """找底池正上方的跟注线数字（本轮最大下注额）。
        选取最靠近底池正上方的数字（距离优先），避免误抢附近的筹码/全押数。"""
        pot_cy = None
        pot_val = None
        for it in items:
            if '底池' in it.text:
                pot_cy = it.cy
                pot_val = _parse_int(it.text)
                break
        if pot_cy is None:
            return None
        best_val, best_dist = None, 999.0
        for it in items:
            cy_lo, cy_hi = self._layout.current_bet_cy_offset
            if not (pot_cy + cy_lo < it.cy < pot_cy + cy_hi): continue
            cx_lo, cx_hi = self._layout.current_bet_cx_range
            if not (cx_lo < it.cx < cx_hi): continue
            if any(kw in it.text for kw in ('底池', '池')): continue
            if any(kw in it.text for kw in self._layout.noise_keywords): continue
            val = _parse_int(it.text)
            if not val or val < 1: continue
            # 跟注线不太可能大于底池的 10 倍（防止误抢筹码栈）
            if pot_val and val > pot_val * 10: continue
            # 排除等于底池值（过牌轮底池来源显示）
            if pot_val and val == pot_val: continue
            dist = abs(it.cy - (pot_cy - 0.025)) + abs(it.cx - 0.50) * 0.5
            if dist < best_dist:
                best_dist = dist
                best_val = val
        return best_val

    # ==================================================================
    # 底池
    # ==================================================================
    def _find_pot(self, items, img_g=None):
        pot_re = None
        for it in items:
            m = _POT_RE.search(it.text)
            if m:
                try:
                    val = int(m.group(1).replace(",","").replace("，","").replace(".",""))
                    if val >= 100: return val
                    pot_re = val
                except ValueError: pass
        # 如果正则已匹配到"底池:XX"（即使 <100），优先返回，避免被误读覆盖
        if pot_re is not None:
            return pot_re
        fb = self._pick_stack(items, self._layout.pot_anchor, min_value=100)
        if fb is not None:
            return fb

        # ── 回退：专用底池区域重度增强 + 单独 OCR ──
        # 底池文字在深色圆角框上，全图 OCR + CLAHE 仍可能漏读。
        # 裁剪后用多种预处理尝试：CLAHE、二值化反转、自适应阈值。
        if img_g is not None:
            pot_val = self._ocr_pot_enhanced(img_g)
            if pot_val is not None:
                return pot_val
        return None

    def _ocr_pot_enhanced(self, img_g: np.ndarray) -> int | None:
        """专用底池区域 OCR：裁剪底池位置，多种增强方式尝试读取数字。

        SPEED：旧实现是把 4 种预处理结果都跑一遍 OCR 再投票。
        新策略：按"成本/命中率"排好顺序，第一个能拿到合法数字就返回，省掉 3 次 OCR。
        实测命中通常发生在方法 1（CLAHE 强力增强），4 种全跑约 4× 单次 OCR 耗时。
        最坏情况（4 种全失败）等价于旧实现 4 次都跑——退化为相同。
        """
        h, w = img_g.shape[:2]
        # 底池区域大致范围（比 pot_anchor 稍宽）
        x1, y1, x2, y2 = 0.30, 0.26, 0.65, 0.40
        xa, ya = int(x1 * w), int(y1 * h)
        xb, yb = int(x2 * w), int(y2 * h)
        crop = img_g[ya:yb, xa:xb]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))

        def _pick_first_valid(bgr_input):
            for val in self._extract_pot_numbers(bgr_input):
                if val >= 100:
                    return val
            return None

        # 方法1: CLAHE 强力增强（最高命中率，先试）
        enh_bgr = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        v = _pick_first_valid(enh_bgr)
        if v is not None:
            return v

        # 方法2: 二值化反转（白字黑底 → 黑字白底）
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv_bgr = cv2.cvtColor(cv2.bitwise_not(bw), cv2.COLOR_GRAY2BGR)
        v = _pick_first_valid(inv_bgr)
        if v is not None:
            return v

        # 方法3: 自适应阈值
        ada = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5)
        v = _pick_first_valid(cv2.cvtColor(ada, cv2.COLOR_GRAY2BGR))
        if v is not None:
            return v

        # 方法4: 放大 2x + CLAHE（小字更容易被 OCR 读到）— 最贵留最后
        scaled = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                            interpolation=cv2.INTER_CUBIC)
        gray_s = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        v = _pick_first_valid(cv2.cvtColor(clahe.apply(gray_s), cv2.COLOR_GRAY2BGR))
        return v

    def _extract_pot_numbers(self, bgr_crop: np.ndarray) -> list[int]:
        """对底池区域裁剪图跑 OCR，提取所有 ≥100 的数字（排除盲注格式）。"""
        res = _ocr_run(self._ocr, bgr_crop)
        if not res:
            return []
        values = []
        for box, text, score in res:
            if not text:
                continue
            # 跳过盲注格式 50/100
            if '/' in text:
                continue
            # 优先匹配 "底池:数字" 格式
            m = _POT_RE.search(text)
            if m:
                try:
                    val = int(m.group(1).replace(",", "").replace("，", "").replace(".", ""))
                    if val >= 100:
                        values.append(val)
                    continue
                except ValueError:
                    pass
            # 普通数字
            val = _parse_int(text)
            if val is not None and val >= 100:
                values.append(val)
        return values

    # ==================================================================
    # 筹码
    # FIX: slen 用匹配到的数字串长度，不含中文字符
    # ==================================================================
    # OCR 常见误读: "7" → "/"（如 "1473" → "14/3"）。
    # 用于 stack 解析时，将 digit/digit 模式中的 "/" 替换为 "7" 再合并。
    _SLASH_DIGIT_RE = re.compile(r'(\d+)/(\d+)')

    @staticmethod
    def _pick_stack(items, anchor, min_value=100, max_value=None):
        best_val, best_len, best_dist = None, -1, 999.0
        # 倒计时/留座文字含数字但不是筹码
        _stack_noise = ("秒", "留座", "倒计时", "延时")
        for it in items:
            d = ((it.cx-anchor.cx)**2+(it.cy-anchor.cy)**2)**0.5
            if d > anchor.max_dist: continue
            if any(kw in it.text for kw in _stack_noise): continue
            # ── OCR "7→/" 修补：如 "14/3" → "1473" ──
            # 盲注格式 "1/2/4" 含多个 "/"，筹码最多一个，用 count 区分
            txt = it.text
            if '/' in txt and txt.count('/') == 1:
                m_slash = TableRecognizer._SLASH_DIGIT_RE.search(txt)
                if m_slash:
                    # 把 "14/3" 拼成 "1473"（"/" 替换为 "7"）
                    txt = txt[:m_slash.start()] + m_slash.group(1) + '7' + m_slash.group(2) + txt[m_slash.end():]
            m = _INT_RE.search(txt)
            if not m: continue
            raw = m.group(0)
            try:
                val = int(raw.replace(",","").replace("，","").replace(".",""))
            except ValueError:
                continue
            if val < min_value: continue
            if max_value is not None and val > max_value: continue
            # FIX: 只用数字串长度，不含中文
            slen = len(raw.replace(",","").replace("，","").replace(".",""))
            if slen > best_len or (slen == best_len and d < best_dist):
                best_val = val; best_len = slen; best_dist = d
        return best_val

    # ==================================================================
    # 点数解析
    # ==================================================================
    @staticmethod
    def _parse_rank(text):
        text = (text or "").replace("O","0").replace("I","1").strip().upper()
        # 常见OCR误读映射
        if text in {"N","NN","N,","Q)","(Q","B","8B","B8"}: return "Q"
        if text in {"K<","<K","KC","CK"}: return "K"
        # FIX: "9" 常见误读（带括号、标点、小写g）
        if text in {"G","9,","(9","9)",".9","9.","Q9","9Q"}: return "9"
        for t in ["10","K","Q","J","A","9","8","7","6","5","4","3","2"]:
            if t in text: return t
        m = re.search(r"(10|[2-9])", text)
        return m.group(1) if m else None

    # ==================================================================
    # 辅助
    # ==================================================================
    @staticmethod
    def _texts_near(items, anchor):
        return [it.text for it in items
                if it.text and ((it.cx-anchor.cx)**2+(it.cy-anchor.cy)**2)**0.5<=anchor.max_dist]

    @staticmethod
    def _roi(img, box):
        h, w = img.shape[:2]; x1,y1,x2,y2 = box
        return img[max(0,int(round(y1*h))):min(h,int(round(y2*h))),
                   max(0,int(round(x1*w))):min(w,int(round(x2*w)))]
