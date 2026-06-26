"""DPZX (德扑之星) 专用布局配置。
所有坐标为归一化分数 (0~1)，基于 720px 宽图。
"""
from __future__ import annotations
from dataclasses import dataclass, field

from app.layouts.base import LayoutBase, Anchor


@dataclass(frozen=True)
class LayoutDpzx(LayoutBase):
    """德扑之星 app 的座位/ROI 布局。"""

    app_name: str = "dpzx"

    pot_anchor: Anchor = field(default_factory=lambda: Anchor(0.46, 0.32, 0.14))
    blind_roi:  tuple  = (0.25, 0.582, 0.75, 0.600)

    hero_stack_anchor: Anchor = field(default_factory=lambda: Anchor(0.36, 0.890, 0.10))
    hero_stack_roi:    tuple  = (0.22, 0.870, 0.50, 0.915)

    seat_anchors: tuple = (
        (2, Anchor(0.10, 0.618, 0.10)),
        (3, Anchor(0.10, 0.430, 0.10)),
        (4, Anchor(0.10, 0.255, 0.10)),
        (5, Anchor(0.46, 0.175, 0.12)),
        (6, Anchor(0.89, 0.270, 0.10)),
        (7, Anchor(0.89, 0.441, 0.10)),
        (8, Anchor(0.89, 0.668, 0.10)),
    )

    seat_bet_anchors: dict = field(default_factory=lambda: {
        2: Anchor(0.280, 0.618, 0.055),
        3: Anchor(0.280, 0.430, 0.055),
        4: Anchor(0.280, 0.255, 0.055),
        5: Anchor(0.460, 0.230, 0.07),
        6: Anchor(0.718, 0.270, 0.07),
        7: Anchor(0.718, 0.450, 0.07),
        8: Anchor(0.700, 0.670, 0.07),
    })

    # 白色文字下注检测 ROI (x1, y1, x2, y2)
    # 二值化(>130)已将绿色⊕图标过滤为黑色，无需靠 ROI 排除⊕。
    # 左侧 x_min=0.15：覆盖⊕区域无妨（二值化后消失），确保 "1,020" 等宽数字完整。
    # 右侧 x_min=0.62：覆盖 4-5 位数（文字向左延伸），避免过宽读到中心区域。
    seat_bet_rois: dict = field(default_factory=lambda: {
        2: (0.15, 0.59, 0.33, 0.66),   # 左下：座位(0.10,0.62)右侧
        3: (0.15, 0.41, 0.33, 0.49),   # 左中：座位(0.10,0.43)右侧，y扩展覆盖大额下注
        4: (0.15, 0.25, 0.33, 0.31),   # 左上：座位(0.10,0.26)右侧
        5: (0.40, 0.19, 0.60, 0.26),   # 顶部：座位(0.35,0.14)下方
        6: (0.62, 0.25, 0.80, 0.31),   # 右上：座位(0.89,0.27)左侧
        7: (0.62, 0.41, 0.80, 0.49),   # 右中：座位(0.89,0.44)左侧
        8: (0.61, 0.59, 0.80, 0.67),   # 右下：座位(0.89,0.65)左侧
    })

    # Hero 下注白色文字检测 ROI
    hero_bet_roi: tuple = (0.28, 0.825, 0.47, 0.885)

    # Hero 头像倒计时检测 ROI（仅覆盖头像中心区域，避免和筹码/下注重叠）
    hero_countdown_roi: tuple = (0.34, 0.80, 0.52, 0.88)

    seat_status_anchors: dict = field(default_factory=lambda: {
        2: Anchor(0.22, 0.631, 0.08),
        3: Anchor(0.22, 0.433, 0.08),
        4: Anchor(0.22, 0.238, 0.08),
        5: Anchor(0.58, 0.130, 0.12),
        6: Anchor(0.78, 0.238, 0.08),
        7: Anchor(0.78, 0.433, 0.08),
        8: Anchor(0.78, 0.631, 0.08),
    })

    hero_status_anchor: Anchor = field(default_factory=lambda: Anchor(0.22, 0.858, 0.08))
    hero_bet_anchor: Anchor = field(default_factory=lambda: Anchor(0.37, 0.855, 0.05))

    hero_card1_roi: tuple = (0.470, 0.828, 0.582, 0.912)
    hero_card2_roi: tuple = (0.584, 0.828, 0.722, 0.912)
    board_rois: tuple = (
        (0.195, 0.500, 0.318, 0.582),
        (0.312, 0.500, 0.435, 0.582),
        (0.429, 0.500, 0.552, 0.582),
        (0.546, 0.500, 0.669, 0.582),
        (0.663, 0.500, 0.786, 0.582),
    )

    card_brightness_threshold: float = 93.0
    hero_card_brightness_threshold: float = 115.0

    # 暗结算/弃牌正面手牌检测：摊牌后 hero 正面牌变暗(亮度~99-110)，低于上面阈值
    # 会被跳过；但同区间还有【蓝色牌背】(未看牌)。实测 dim 正面牌 blue≈1%、牌背 blue≈42%，
    # 用 blue 占比干净区分。开启后这些暗正面牌也能识别 rank/suit（dpzx 专属）。
    hero_dim_faceup: bool = True
    hero_dim_min_bright: float = 90.0
    hero_dim_blue_reject: float = 0.20

    action_btn_y:         tuple = (0.76, 0.88)
    action_red_threshold: float = 0.012

    status_enhance_rois: tuple = (
        (0.18, 0.615, 0.30, 0.650),
        (0.18, 0.415, 0.30, 0.452),
        (0.18, 0.220, 0.30, 0.257),
        (0.47, 0.110, 0.62, 0.175),
        (0.70, 0.220, 0.83, 0.257),
        (0.70, 0.415, 0.83, 0.452),
        (0.70, 0.615, 0.83, 0.650),
        (0.14, 0.838, 0.30, 0.878),
        (0.80, 0.248, 0.99, 0.295),
        (0.80, 0.418, 0.99, 0.465),
        (0.80, 0.645, 0.99, 0.695),
        # 底池区域
        (0.32, 0.28, 0.62, 0.38),
    )

    seat_positions: dict = field(default_factory=lambda: {
        1: (0.37, 0.93), 2: (0.10, 0.62), 3: (0.10, 0.43),
        4: (0.10, 0.26), 5: (0.35, 0.14), 6: (0.89, 0.27),
        7: (0.89, 0.44), 8: (0.89, 0.65),
    })

    hero_card_rank_ry:  tuple = (0.02, 0.45)
    board_card_rank_ry: tuple = (0.02, 0.46)
    card_rank_rx:       tuple = (0.03, 0.62)
    hero_card_rank_rx:  tuple = (0.00, 0.45)

    hero_card_suit_ry:  tuple = (0.38, 0.78)
    hero_card_suit_rx:  tuple = (0.00, 0.50)
    board_card_suit_ry: tuple = (0.50, 0.85)
    board_card_suit_rx: tuple = (0.00, 0.55)

    # face card (J/Q/K) 专用花色区：中央是人物插画，必须用左上角角标小花色。
    # 分数相对裁紧后的真卡（board 经 bbox，hero 用 _hero_card_crop 的紧裁）。
    # rank 字母在上，花色符号紧贴其下、偏左。hero 卡比例略不同，单列一套。
    face_board_suit_ry: tuple = (0.26, 0.55)
    face_board_suit_rx: tuple = (0.02, 0.34)
    face_hero_suit_ry:  tuple = (0.26, 0.58)
    face_hero_suit_rx:  tuple = (0.02, 0.42)

    # 庄家按钮候选位置
    button_positions: dict = field(default_factory=lambda: {
        1: [(0.15, 0.93), (0.58, 0.93), (0.30, 0.87)],
        2: [(0.20, 0.62)],
        3: [(0.20, 0.43)],
        4: [(0.20, 0.31)],
        5: [(0.60, 0.14)],
        6: [(0.79, 0.27)],
        7: [(0.79, 0.44)],
        8: [(0.79, 0.65)],
    })

    # 底池增强 OCR 区域
    pot_enhanced_roi: tuple = (0.30, 0.26, 0.65, 0.40)

    # 跟注线搜索区域（相对底池位置）
    current_bet_cx_range: tuple = (0.38, 0.65)
    current_bet_cy_offset: tuple = (-0.07, -0.005)

    # 噪声关键词
    noise_keywords: frozenset = frozenset({
        '骐骥', '掘金赛', 'DPT', '德扑之星', '桌号', '欢乐',
        '级别', '距离', '训练赛', '手', '底池', '池', '倍',
    })
