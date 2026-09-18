---
version: "1.0"
last_updated: "2026-09-18"
---
# 业务语义
- hero：自己，seat=1；其余 seat=2..8 对应左下、左中、左上、顶部、右上、右中、右下。
- hero_info.current_bet：本轮已经投入的下注，不能读取动作按钮中的待跟注金额。
- table_info.current_bet：本轮全桌最大已下注额。
- stack：剩余筹码；空座与 polerv2 等待入座返回 null。现有兼容逻辑对其他读不到的在座筹码仍可能返回 0，因此 0 不保证是真实余额，本轮用正确 hero ROI 修复已观测的短筹码漏读。
- button_seat：白色 D 庄家标记所在座位（polerv2）。
- parse_all=true：即使不是自己行动轮次也完整识别；false 可在轮次门控时返回空结构。
- is_hero_turn：是否轮到自己，不等同画面中存在红色预操作按钮。
- stage：0/3/4/5 张有效公共牌对应 preflop/flop/turn/river；不能把无牌桌背景识别成牌。
- 抽样连续截图不是独立牌局；准确率必须说明所用字段、样本和人工标注口径。
