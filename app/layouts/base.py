"""布局基类：定义所有 app 共用的字段名。
各 app 实现自己的 Layout 子类来覆盖坐标值。"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Anchor:
    cx:       float
    cy:       float
    max_dist: float = 0.10


@dataclass(frozen=True)
class LayoutBase:
    """所有 app 布局的公共接口。子类必须覆盖全部字段。"""
    app_name: str = "base"

    # ── 底池 ──
    pot_anchor: Anchor = field(default_factory=lambda: Anchor(0.50, 0.30, 0.14))
    pot_enhanced_roi: tuple = (0.30, 0.24, 0.70, 0.40)

    # ── 盲注 ──
    blind_roi: tuple = (0.25, 0.55, 0.75, 0.62)

    # ── 蘑菇数量（保险图标）──
    mushroom_roi: tuple = ()  # (x1, y1, x2, y2) 蘑菇图标旁数字区域

    # ── Hero ──
    hero_stack_anchor: Anchor = field(default_factory=lambda: Anchor(0.50, 0.85, 0.10))
    hero_stack_roi:    tuple  = (0.30, 0.83, 0.70, 0.90)
    hero_status_anchor: Anchor = field(default_factory=lambda: Anchor(0.30, 0.85, 0.08))
    hero_bet_anchor: Anchor = field(default_factory=lambda: Anchor(0.37, 0.855, 0.05))

    hero_card1_roi: tuple = (0.40, 0.85, 0.50, 0.93)
    hero_card2_roi: tuple = (0.52, 0.85, 0.62, 0.93)

    # ── 公共牌 ──
    board_rois: tuple = ()

    # ── 对手座位 ──
    seat_anchors: tuple = ()  # ((seat_num, Anchor), ...)
    seat_bet_anchors: dict = field(default_factory=dict)
    seat_bet_rois: dict = field(default_factory=dict)       # {seat: (x1,y1,x2,y2)} 白色文字下注检测区域
    # 兜底 ROI 颜色门控（v0.6 新增）：{seat: ("red",) 或 ("red","green")}。
    # 不在 dict 内的 seat 默认走 ("red",)，与原版兼容。
    seat_bet_fallback_chip_colors: dict = field(default_factory=dict)
    seat_status_anchors: dict = field(default_factory=dict)
    seat_positions: dict = field(default_factory=dict)

    # ── Hero 下注 / 倒计时 ──
    hero_bet_roi: tuple = ()          # (x1,y1,x2,y2) hero 下注白色文字检测区域
    hero_countdown_roi: tuple = ()    # (x1,y1,x2,y2) hero 头像倒计时检测区域
    # hero 下注 fallback ROI（v0.7）：紧裁剪 + 5x 放大 OCR 兜底小芯片数字
    hero_bet_fallback_roi: tuple = ()

    # ── 卡牌 ──
    card_brightness_threshold: float = 93.0
    hero_card_brightness_threshold: float = 140.0

    hero_card_rank_ry:  tuple = (0.02, 0.45)
    board_card_rank_ry: tuple = (0.02, 0.46)
    card_rank_rx:       tuple = (0.03, 0.62)
    hero_card_rank_rx:  tuple = (0.00, 0.45)

    hero_card_suit_ry:  tuple = (0.38, 0.78)
    hero_card_suit_rx:  tuple = (0.00, 0.50)
    board_card_suit_ry: tuple = (0.40, 0.85)
    board_card_suit_rx: tuple = (0.00, 0.55)

    # ── 同色花色仲裁策略（v21）──
    # True：CNN 定红/黑（可靠），形态学定同色形状（♥/♦、♠/♣，权威）。
    #   用于无独立训练数据、CNN 同色形状弱的 app（如 dpzx）。
    # False（默认）：CNN 优先（适合已重训好 suit 模型的 app，如 poler）。
    suit_samecolor_use_morpho: bool = False

    # ── AKQ 消歧阈值 (hole_ratio) ──
    # hero: (Q下限, A下限, K上限)  即 >Q下限→Q, >A下限→A, <K上限→K
    # board: 同理
    akq_hero_thresholds: tuple = (0.25, 0.08, 0.03)   # dpzx 默认
    akq_board_thresholds: tuple = (0.07, 0.015, 0.008)  # dpzx 默认

    # ── 操作按钮 / Hero 轮次检测 ──
    action_btn_y:         tuple = (0.76, 0.88)
    # 红色按钮检测的 X 范围（fractional）。默认全宽，poler 等可缩到左半侧避免 hero 红心/方块卡误触发。
    action_btn_x:         tuple = (0.0, 1.0)
    action_red_threshold: float = 0.012

    # ── CLAHE 增强 ROI ──
    status_enhance_rois: tuple = ()

    # ── 庄家按钮候选位置 ──
    button_positions: dict = field(default_factory=dict)

    # ── 庄家按钮颜色 HSV 范围 ──
    # dpzx 默认黄色; poler 是绿色 "D" 圆圈
    button_hsv_lower: tuple = (15, 120, 100)   # 黄色下界
    button_hsv_upper: tuple = (40, 255, 255)   # 黄色上界

    # ── 跟注线搜索参数 ──
    current_bet_cx_range: tuple = (0.35, 0.65)
    current_bet_cy_offset: tuple = (-0.07, -0.005)

    # 计算 table.current_bet 的策略：
    #   "ocr"  → 沿用旧逻辑，在底池上方搜索 OCR 数字（dpzx 默认）。
    #   "max_seat_bet" → 取所有 active 玩家 bet 的最大值（poler 用，
    #     避开 AA Poker 底池上方"累计/底注"小芯片误读）。
    table_current_bet_strategy: str = "ocr"

    # face-down 红背门控阈值（v0.6 新增）。> 0 时启用：
    # 当 hero ROI 红色像素比例 > 阈值，视为面朝下的"AA POKER"红背 → 跳过卡牌识别。
    # 0.0 表示禁用（dpzx 默认；背面是黑色/绿色，不需要红色门控）。
    face_down_red_ratio_threshold: float = 0.0

    # 检到 hero_cards 时是否保留 status=folded（v0.6 新增）。
    # poler 弃牌后会保留 dim face-up 卡牌；为 True 时不会把 status flip 回 active。
    # dpzx 默认 False（旧行为：检到牌就视为状态误判，flip 成 active）。
    keep_folded_when_cards_visible: bool = False

    # ── 噪声关键词 ──
    noise_keywords: frozenset = frozenset()

    # ── 识别器可调参数（2026-04-19 新增） ──
    # rank CNN 的最低接受置信度。低于此值视为 CNN 放弃，走 OCR/模板兜底。
    # dpzx 默认 0.40；poler 可下调到 0.25，避免把正确的低 conf 预测丢掉。
    rank_cnn_min_conf: float = 0.40

    # rank CNN 置信度 < 此值时才运行 OCR 交叉校验；设 0.0 则完全禁用 OCR 校验。
    # dpzx OCR 对手牌角标识别稳，保留 0.70 原值；
    # poler 由于小角标 OCR 噪声大，设为 0.0 后可显著减少误覆盖。
    rank_ocr_verify_threshold: float = 0.70

    # OCR 覆盖 CNN 结果所需的 OCR 置信度下限（越大越保守）。
    rank_ocr_override_min_score: float = 0.70

    # 黑色系花色（梅花/黑桃）的 CNN 置信度 < 此值时，让形态学投票主导。
    # 设为 1.0 表示“黑色系永远以形态学为准”，设为 0.0 表示“始终信 CNN”。
    suit_black_cnn_trust_threshold: float = 0.0  # 默认关闭（dpzx 保持原逻辑）

    # 是否启用 6↔8 消歧（依据上半部是否双环闭合）。poler CNN 对小号字 6/8 易混，
    # 开启后可在 CNN 给 8 时额外校验是否其实是 6。
    enable_disambiguate_6_vs_8: bool = False

    # 筹码上限：OCR 有时把附近用户名中的数字拼到筹码里，产生离谱大值。
    # None = 不限制；设正整数则跳过超标值。
    max_stack: int | None = None

    # 是否启用"空座视觉兜底"——对 stack=0 active 的座位用头像区颜色判定是否真空座。
    # 仅适用绿色桌面 + 空座显示 "+" 的 app（poler 是这种）。
    # dpzx 默认关闭：dpzx 等待发牌的玩家也是 stack=0/active，会被误杀。
    enable_empty_seat_visual_check: bool = False
