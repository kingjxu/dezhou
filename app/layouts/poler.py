"""Poler (AA Poker) 专用布局配置。
仅支持绿色桌面主题（1080x2400 分辨率实测）。
8 人桌: hero=1, 对手顺时针 2~8（左上角蘑菇图标不是座位）。
座位编号与 dpzx 一致。

实测坐标 (基于 1080x2400)：
  Seat 2 (left-lower):   stack cx=0.05  cy=0.61
  Seat 3 (left-middle):  stack cx=0.05  cy=0.50
  Seat 4 (left-upper):   stack cx=0.05  cy=0.39
  (top-left = 蘑菇保险图标，不是座位)
  Seat 5 (top-center):   stack cx=0.50  cy=0.20
  Seat 6 (right-upper):  stack cx=0.92  cy=0.38
  Seat 7 (right-middle): stack cx=0.92  cy=0.50
  Seat 8 (right-lower):  stack cx=0.92  cy=0.61
  Hero:                   stack cx=0.47  cy=0.925
  Pot:                    "底池:XX" cx=0.50  cy=0.33
  Board cards:            y=0.48~0.56, x=0.20~0.79
  Hero cards:             y=0.825~0.89, x=0.55~0.76 (右侧，紧靠头像)
"""
from __future__ import annotations
from dataclasses import dataclass, field

from app.layouts.base import LayoutBase, Anchor


@dataclass(frozen=True)
class LayoutPoler(LayoutBase):
    """AA Poker app 的座位/ROI 布局（绿色桌面）。"""

    app_name: str = "poler"

    # ── 底池 ──
    # "底池:XX" 文字在 y≈0.33; 绿色指示器在 y≈0.26
    # 用小半径避免抓到 seat5 筹码 (y≈0.20) 和桌面信息 (y≈0.37)
    pot_anchor: Anchor = field(default_factory=lambda: Anchor(0.50, 0.33, 0.05))
    pot_enhanced_roi: tuple = (0.30, 0.28, 0.70, 0.38)

    # ── 盲注 ──
    # "德州 1/2/4（2）" at y≈0.36
    blind_roi: tuple = (0.20, 0.34, 0.80, 0.42)

    # ── 蘑菇数量（保险图标，左上角）──
    # 蘑菇图标在左上角，数字紧跟其后
    mushroom_roi: tuple = (0.02, 0.02, 0.18, 0.08)

    # ── Hero ──
    # Hero 筹码在头像正下方，实测 cy≈0.925（"197" 等数字）
    hero_stack_anchor: Anchor = field(default_factory=lambda: Anchor(0.47, 0.925, 0.06))
    hero_stack_roi:    tuple  = (0.30, 0.91, 0.65, 0.95)
    # Hero 状态标签("弃牌"/"三条"等)在头像上方，实测 cy≈0.85~0.87
    hero_status_anchor: Anchor = field(default_factory=lambda: Anchor(0.33, 0.86, 0.12))
    # FIX(v0.7)：AA Poker hero 自己的下注小芯在 hero 卡的右上方一点，
    # 紧贴"自动跟牌 N"动作按钮的右下角，chip 中心稳定在 cx≈0.74~0.76, cy≈0.76~0.78。
    # 锚点收紧到这个范围，避开按钮内的"自动跟牌 20" 等文字（文字位于 cx≈0.68）。
    # max_dist=0.04 使 "20" 文字被距离过滤掉（0.685 距离 0.755 已 > 0.04）。
    hero_bet_anchor: Anchor = field(default_factory=lambda: Anchor(0.76, 0.77, 0.04))
    # Hero 下注 fallback ROI（v0.7）：紧裁剪 hero 卡右上方的小芯区域，
    # 避开按钮内文字。x>0.71 确保不包含 "20" 按钮中心。
    hero_bet_fallback_roi: tuple = (0.71, 0.74, 0.83, 0.80)

    # ── Hero 手牌 ROI ──
    # AA Poker 手牌在头像右侧: card1 x≈0.58~0.67, card2 x≈0.67~0.81
    # y≈0.82~0.89; 左边界避开头像（头像右边缘 x≈0.56）
    # card2 右边界扩展到 0.81（实测 10♠/A♦ 等宽牌在 0.77 被裁切）
    hero_card1_roi: tuple = (0.575, 0.820, 0.670, 0.895)
    hero_card2_roi: tuple = (0.660, 0.820, 0.810, 0.895)

    # ── 公共牌 ──
    # 5 张牌在 y=0.485~0.555, x 从 0.20 到 0.79
    # 每张牌宽约 0.115, 间隙约 0.005
    board_rois: tuple = (
        (0.190, 0.480, 0.320, 0.560),
        (0.310, 0.480, 0.440, 0.560),
        (0.430, 0.480, 0.560, 0.560),
        (0.550, 0.480, 0.680, 0.560),
        (0.670, 0.480, 0.800, 0.560),
    )

    # ── 对手座位（筹码锚点）──
    # 用小半径 (0.06) 精确定位筹码数字，避免抓到用户名
    seat_anchors: tuple = (
        (2, Anchor(0.05, 0.61, 0.06)),   # 左-下
        (3, Anchor(0.05, 0.50, 0.06)),   # 左-中
        (4, Anchor(0.05, 0.39, 0.06)),   # 左-上
        # 左上角蘑菇保险图标，不是座位
        (5, Anchor(0.50, 0.20, 0.05)),   # 顶-中（小半径避免抓底池）
        (6, Anchor(0.92, 0.38, 0.06)),   # 右-上
        (7, Anchor(0.92, 0.50, 0.06)),   # 右-中
        (8, Anchor(0.92, 0.61, 0.06)),   # 右-下
    )

    # ── 对手下注锚点 ──
    # 下注筹码在头像内侧（靠近桌面中心）。
    # 右侧座位(6/7/8)的筹码实测位置在 cx≈0.55-0.82 之间浮动：
    #   翻牌前盲注/跨注筹码偏向桌面中心(cx≈0.55)，后续街下注偏向头像(cx≈0.82)。
    # 因此锚点设在中间位置 cx=0.70，半径 0.10 可同时覆盖两个极端。
    # 左侧座位类似：盲注时筹码偏右(cx≈0.40)，常规下注在 cx≈0.18 附近，
    # 锚点调至 cx=0.28 半径 0.10。
    # Y-band (max_dist*0.85=0.085) 小于相邻座位间距(≈0.11)，不会串座。
    seat_bet_anchors: dict = field(default_factory=lambda: {
        2: Anchor(0.18, 0.60, 0.06),
        3: Anchor(0.18, 0.49, 0.06),
        4: Anchor(0.18, 0.38, 0.06),
        # FIX(座位5, v3 2026-04-28)：座位 5 真正的"当前下注"是 **上层小红芯片**
        # （带卡牌图标，cy≈0.250），下层绿色硬币（cy≈0.288）是入池累积/装饰，
        # 不应当作 bet 读。锚点收紧到 cy=0.255，max_dist=0.020 → y_band ≈ 0.017，
        # 实际接受 Y∈[0.238, 0.272]，**完全跳过下层绿币**。
        # 没有上层红芯时（poler_007/008 这类历史样本），返回 None / 0，符合实际语义。
        5: Anchor(0.50, 0.255, 0.020),
        6: Anchor(0.70, 0.38, 0.10),
        7: Anchor(0.70, 0.49, 0.10),
        8: Anchor(0.70, 0.60, 0.10),
    })

    # ── 对手下注兜底 ROI ──
    # 当锚点搜索在全图 OCR 中找不到下注时，裁剪此区域做放大+增强 OCR。
    # 注意：字段名不能叫 seat_bet_rois（那个是 DPZX 白色文字检测用的开关）。
    # 右侧: 筹码在 cx≈0.50~0.84; 左侧: cx≈0.12~0.42; 顶部: cx≈0.42~0.58
    # Y 范围: 座位 cy ± 0.04（避免和相邻座位重叠）
    # 左侧 (s2/s3/s4) v0.6 新增：preflop/常规下注红/绿芯。
    # 实测多张图（731/601/521/981/781）红芯中心稳定在 cx=0.255、
    # cy={0.337, 0.470, 0.603} 这三档，与 stack pill (cx≈0.087) 完全错开。
    # ROI 宽度 0.12 给 chip 周围数字留余量；高度 ±0.035。
    seat_bet_fallback_rois: dict = field(default_factory=lambda: {
        2: (0.20, 0.57, 0.32, 0.64),   # 左-下，chip @ (0.255, 0.603)
        3: (0.20, 0.43, 0.32, 0.50),   # 左-中，chip @ (0.255, 0.470)
        4: (0.20, 0.30, 0.32, 0.37),   # 左-上，chip @ (0.255, 0.337)
        # 座位 5 上层小红芯（带卡牌图标）："1" / "2" 等小字，全图 OCR 经常漏读。
        # ROI 严格锁定 cy=0.235~0.275 这一窄条，避开下层绿币和上方筹码栈。
        # _detect_bet_roi_enhanced 走 红色 HSV 门控 + 紧裁剪 + 5x 放大 + OCR，
        # 对小数字命中率明显高于全图 OCR。
        5: (0.42, 0.235, 0.58, 0.275),
        6: (0.52, 0.30, 0.78, 0.41),
        7: (0.52, 0.46, 0.78, 0.52),
        8: (0.52, 0.57, 0.78, 0.63),
    })

    # ── 兜底 ROI 的颜色门控配置（v0.6 新增） ──
    # 默认 ('red',)（原版仅识别红芯）。左侧 + 顶部座位 SB/BB/STR 同时含小绿芯，
    # 故 s2/s3/s4/s5 加 'green'。s6/s7/s8 保持仅红芯，避免桌面绿误触。
    seat_bet_fallback_chip_colors: dict = field(default_factory=lambda: {
        2: ("red", "green"),
        3: ("red", "green"),
        4: ("red", "green"),
        5: ("red", "green"),
        6: ("red",),
        7: ("red",),
        8: ("red",),
    })

    # ── 对手状态锚点 ──
    # "弃牌"/"跟注"/"加注"/"All in" 标签在头像旁：
    #   左侧座位→标签在右侧(cx偏大)，右侧座位→标签在左侧(cx偏小)
    # 半径从 0.08 扩大到 0.12，确保标签在搜索范围内
    seat_status_anchors: dict = field(default_factory=lambda: {
        2: Anchor(0.12, 0.59, 0.12),   # 左-下，标签在头像右侧
        3: Anchor(0.12, 0.47, 0.12),   # 左-中
        4: Anchor(0.12, 0.35, 0.12),   # 左-上
        5: Anchor(0.55, 0.15, 0.12),   # 顶-中，标签在头像右侧 cx≈0.59
        6: Anchor(0.82, 0.34, 0.12),   # 右-上，标签在头像左侧
        7: Anchor(0.82, 0.47, 0.12),   # 右-中
        8: Anchor(0.82, 0.59, 0.12),   # 右-下
    })

    # ── 卡牌参数 ──
    # FIX(1021 hand-end)：手牌结算时整桌灯光变暗，公共牌亮度从 130+ 降到 80~90，
    # 旧阈值 90 会丢首尾两张。
    # 实测：preflop 空板背景绿桌 max ≈ 78（card3 中心位略亮），
    # 1021 hand-end 真实牌最暗 83.7。80 是稳健分界。
    card_brightness_threshold: float = 80.0
    # AA Poker hero 手牌实测亮度：
    #   - 活跃 face-up: 122-157
    #   - 弃牌后 face-up dim: 70-92（v0.6 起加入 dim patches + dim 增强重训 CNN，可读）
    #   - face-down 红 AA POKER 牌背: 84-95（用 face_down_red_ratio_threshold 过滤）
    # 阈值 65：覆盖 dim face-up 区间；红背门控同时挡住 face-down 红背。
    hero_card_brightness_threshold: float = 65.0
    # face-down 红 AA POKER 牌背的红色像素占比 ≈ 0.67~0.82；
    # face-up（包括 dim）红色 < 0.25。0.40 是稳健阈值。
    face_down_red_ratio_threshold: float = 0.40
    # AA Poker 弃牌后保留 dim face-up 卡牌；不要把 fold flip 成 active。
    keep_folded_when_cards_visible: bool = True

    # rank/suit 切割参数
    hero_card_rank_ry:  tuple = (0.02, 0.45)
    board_card_rank_ry: tuple = (0.02, 0.46)
    card_rank_rx:       tuple = (0.03, 0.62)
    hero_card_rank_rx:  tuple = (0.00, 0.50)

    hero_card_suit_ry:  tuple = (0.38, 0.80)
    hero_card_suit_rx:  tuple = (0.00, 0.55)
    board_card_suit_ry: tuple = (0.38, 0.85)
    board_card_suit_rx: tuple = (0.00, 0.55)

    # face card (J/Q/K) 专用花色区：中央是人物插画，常规花色区会抓到画而非花色
    # （新生产图 J♥ 被误判为 J♦ 的根因）。改用左上角角标小花色。
    # 分数相对裁紧后的真卡：rank 字母在上，花色符号紧贴其下。board 卡较大、符号靠左；
    # hero 卡窄（~80px）符号比例偏右下，故 hero/board 各自一套。
    # 需配合 suit_classifier_poler.onnx 用 corner face patch 重训（train_poler --suit-only）。
    face_board_suit_ry: tuple = (0.28, 0.55)
    face_board_suit_rx: tuple = (0.02, 0.34)
    face_hero_suit_ry:  tuple = (0.30, 0.62)
    face_hero_suit_rx:  tuple = (0.06, 0.50)

    # ── AKQ 消歧阈值 (hole_ratio) ──
    # poler hero 实测: A ≈ 0.29~0.35, Q ≈ 1.33~1.50, K ≈ 0.00
    akq_hero_thresholds: tuple = (0.80, 0.15, 0.05)
    # poler board 实测: A ≈ 0.00 与 K 无法区分，禁用 A/K 消歧（信任 CNN）
    akq_board_thresholds: tuple = (0.07, -1.0, -1.0)

    # ── 操作按钮（hero 轮次检测）──
    # 弃牌(红 X)/自由加注(蓝 +)/让牌或跟注(蓝/绿) 按钮行实测 y≈0.71~0.77。
    # 旧值 y=(0.79, 0.88) 取错了——那里其实是按钮**下方**的中文标签
    # ("弃牌"/"自由加注"/"跟注") 行 + hero 头像/手牌区，与红心♥/方块♦
    # 严重重叠，导致 is_hero_turn 长期假阳性。新值锁定按钮圆本身。
    action_btn_y:         tuple = (0.70, 0.78)
    # X 范围：弃牌（红 X）按钮独占 x≈0.13~0.30；不要扩到 0.50 以避免
    # 偶尔出现的"自由加注 +"按钮上的高亮渐变带来抖动。
    action_btn_x:         tuple = (0.05, 0.45)
    # 阈值 0.015：按钮存在时红色像素占比 ~0.05，没按钮时几乎为 0，安全区间宽。
    action_red_threshold: float = 0.015

    # ── CLAHE 增强 ROI ──
    status_enhance_rois: tuple = (
        # 左侧 3 个座位筹码
        (0.00, 0.36, 0.14, 0.42),
        (0.00, 0.47, 0.14, 0.53),
        (0.00, 0.58, 0.14, 0.64),
        # 右侧 3 个座位筹码
        (0.86, 0.35, 1.00, 0.41),
        (0.86, 0.47, 1.00, 0.53),
        (0.86, 0.58, 1.00, 0.64),
        # 顶部座位筹码（seat 5）
        (0.42, 0.17, 0.58, 0.23),
        # Hero 筹码（下移到实际位置 cy≈0.925）
        (0.30, 0.90, 0.65, 0.96),
        # 底池区域
        (0.30, 0.28, 0.70, 0.38),
    )

    seat_positions: dict = field(default_factory=lambda: {
        1: (0.47, 0.90), 2: (0.08, 0.59), 3: (0.08, 0.47),
        4: (0.08, 0.35), 5: (0.50, 0.15),
        6: (0.88, 0.34), 7: (0.88, 0.47), 8: (0.88, 0.59),
    })

    # ── 庄家按钮颜色 HSV 范围（白色圆圈内黑色 "D"）──
    button_hsv_lower: tuple = (0, 0, 180)
    button_hsv_upper: tuple = (180, 60, 255)
    # 白色检测会匹配很多UI元素，需要面积下限过滤
    button_min_area: int = 1000

    # ── 庄家按钮候选位置 ──
    button_positions: dict = field(default_factory=lambda: {
        1: [(0.32, 0.87), (0.60, 0.87), (0.61, 0.80)],  # hero 左/右/右上
        2: [(0.15, 0.60), (0.15, 0.57)],      # 左-下
        3: [(0.15, 0.48)],                     # 左-中
        4: [(0.15, 0.37)],                     # 左-上
        5: [(0.56, 0.17), (0.44, 0.17)],       # 顶-中
        6: [(0.82, 0.32)],                     # 右-上
        7: [(0.82, 0.45), (0.82, 0.48)],       # 右-中
        8: [(0.82, 0.59), (0.82, 0.57)],       # 右-下
    })

    # ── 跟注线搜索参数 ──
    current_bet_cx_range: tuple = (0.35, 0.65)
    current_bet_cy_offset: tuple = (-0.06, -0.005)
    # FIX(v0.6)：AA Poker 底池正上方常驻一枚累计/底注小芯（如 16 = 8人×2底注），
    # 旧 OCR 逻辑会把它误读为 table.current_bet。改用 max(seat bet) 计算。
    table_current_bet_strategy: str = "max_seat_bet"

    # ── 噪声关键词 ──
    noise_keywords: frozenset = frozenset({
        'AAPOKER', 'aapoker', 'AA', 'POKER', 'KER',
        '桌号', '底池', '池', '倍',
        '保险', 'GPS', '级别', '距离', '精确', '德州',
        '延时', '看牌', 'POT', 'STR', '私局', 'No.',
        'aapk', 'app', 'www', 'com',
    })

    # ── 识别器可调参数（poler 专属调优，2026-04-19）──
    # 把 CNN 放行阈值从 0.40 降到 0.25：能抢回 conf≈0.34-0.39 的正确低置信度预测
    #（如 1001776571516_b1=Q@0.34、691776571489_b1=8@0.39）。
    rank_cnn_min_conf: float = 0.25

    # poler 的手牌/公共牌角标非常小，OCR 容易把 J 衬线读成 3、2 底横读成 3、
    # 8 顶部断笔读成 6 等；实测 OCR 校验在 poler 上导致 4 处 rank 错误（见 FINDINGS）。
    # 关键：设为 0.0 彻底禁用 OCR 校验分支。
    rank_ocr_verify_threshold: float = 0.0

    # 黑色系（梅花/黑桃）CNN 极低置信度 (conf < 0.55) 时由形态学仲裁。
    # 实测：CNN conf 0.52-0.54 区间 A梅花 常被误判为 A黑桃，morpho 凭三连通块特征
    # 判 梅花 更稳。阈值 0.55 控制作用范围仅覆盖 CNN 极低 conf 边界，避免误伤
    # CNN 在 0.55-0.65 区间的正确预测（如 2黑桃@0.50）。
    suit_black_cnn_trust_threshold: float = 0.55

    # 启用 6↔8 消歧（解决 841776571502_h2 的 6→8 误判）。
    enable_disambiguate_6_vs_8: bool = True

    # 筹码上限：1/2 桌合理最大值 ~5000，设 9999 过滤 OCR 拼接的离谱值（如 61502）。
    max_stack: int | None = 9999

    # 启用空座视觉兜底：实测绿色桌面 + "+" 占位时，头像区
    # green_ratio>60% 且 colorful_non_green<15%，与真实玩家头像有清晰分界。
    enable_empty_seat_visual_check: bool = True

    # 启用 hero 暗牌横向/纵向收紧（poler 专属）：弃牌/手牌结算后卡牌变暗（最亮~130），
    # find_card_bbox 白阈值裁不动、返回含头像暗边的全宽框，使 rank patch 抓到
    # 暗区+残缺字符（实测 5♣ 被误判 Q）。按亮度剖面收紧到卡面真实边界。
    # dpzx 不开此项（其暗牌按亮度阈值直接跳过，不受影响）。
    hero_dim_localize: bool = True
