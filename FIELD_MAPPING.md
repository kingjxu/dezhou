## 字段对照表（`/recognize` 响应）

> 说明：接口响应为纯字段值，不含中文备注。本文件仅用于字段含义对照。

### 顶层字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `received_at` | string (ISO8601 UTC) | 服务器收到请求的时间 |
| `parsed_at` | string (ISO8601 UTC) | 完成识别并生成响应的时间 |
| `elapsed_ms` | int | 识别耗时（毫秒） |

### table_info

| 字段 | 类型 | 含义 |
|---|---|---|
| `stage` | string | 当前阶段：`preflop` / `flop` / `turn` / `river` / `unknown` |
| `community_cards` | array\<string\> | 公共牌列表，从左到右。例如 `["8梅花","6黑桃","9红桃"]` |
| `main_pot` | int\|null | 主底池筹码数 |
| `button_seat` | int\|null | 庄家按钮所在座位号（纯颜色检测，识别黄色圆形图标） |
| `blind_size` | string\|null | 盲注结构，格式 `"小盲/大盲(Ante)"` 例如 `"50/100(25)"` 或 `"100/200"` |

### hero_info（自己，seat=1）

| 字段 | 类型 | 含义 |
|---|---|---|
| `seat` | int | 固定为 `1` |
| `status` | string | `active` / `folded` / `all_in` |
| `stack` | int\|null | 剩余筹码 |
| `current_bet` | int\|null | 本轮当前已下注金额 |
| `is_hero_turn` | bool\|null | `true` = 当前轮到自己行动；`null` = 非行动轮 |
| `hero_cards` | array\<string\> | 自己手牌，最多 2 张。弃牌后为 `[]` |

### villains_info（其他玩家，seat 2~8）

数组按**顺时针**顺序排列，seat 1（自己）之后依次：

| seat | 屏幕位置 |
|---|---|
| 2 | 左下 |
| 3 | 左中 |
| 4 | 左上 |
| 5 | 顶部 |
| 6 | 右上 |
| 7 | 右中 |
| 8 | 右下 |

每个对象字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `seat` | int | 座位号 2~8 |
| `status` | string | `active` / `folded` / `all_in` |
| `stack` | int\|null | 剩余筹码。`all_in` 时为 `0` |
| `current_bet` | int\|null | 本轮当前已下注金额 |

### 花色表示

| 符号 | 中文 | 颜色 |
|---|---|---|
| ♠ | `黑桃` | 黑 |
| ♥ | `红桃` | 红 |
| ♣ | `梅花` | 黑 |
| ♦ | `方块` | 红 |

### 典型响应示例

```json
{
  "received_at": "2026-04-14T17:00:00.000Z",
  "parsed_at":   "2026-04-14T17:00:02.300Z",
  "elapsed_ms":  2300,
  "table_info": {
    "stage": "preflop",
    "community_cards": [],
    "main_pot": 350,
    "button_seat": 8,
    "blind_size": "50/100(25)"
  },
  "hero_info": {
    "seat": 1,
    "status": "active",
    "stack": 7234,
    "current_bet": 100,
    "is_hero_turn": true,
    "hero_cards": ["Q黑桃", "10梅花"]
  },
  "villains_info": [
    {"seat": 2, "status": "active",  "stack": 5288,  "current_bet": 200},
    {"seat": 3, "status": "active",  "stack": 2576,  "current_bet": null},
    {"seat": 4, "status": "folded",  "stack": 4900,  "current_bet": null},
    {"seat": 5, "status": "active",  "stack": null,   "current_bet": null},
    {"seat": 6, "status": "active",  "stack": 5087,  "current_bet": null},
    {"seat": 7, "status": "active",  "stack": 4675,  "current_bet": null},
    {"seat": 8, "status": "active",  "stack": 11530, "current_bet": 50}
  ]
}
```
