# Hero 无 marker 下注识别修复

## 问题

2026-10-13 23:03:29 的 polerv2 decided 帧中，Hero 翻前已加注到 28。画面上可见本轮新投入 24、剩余筹码 115，但该下注只以黑色牌桌边缘上的白色数字显示，没有彩色筹码 marker。原逻辑强制要求 marker，因此输出 `hero.current_bet=null`。下游跨街重建只保留了 Hero 先前的 straddle 4，在 flop 决策请求中误生成 `call 4`。

decided 帧的一次性保存/全量识别机制仍然存在：`1f56020` 恢复了曾短暂删除的 post-Hero recognition，本次现场帧也已保存为 `-decided`。回归来源是 Hero 金额门控：`bc3905f` 为避免吸收操作按钮禁用通用锚点，后续专用路径只接受带 marker 金额，因而一并过滤了合法的无 marker decided 金额。

## 修复

- 保留 Hero 现有 ROI 和数字识别链，不扩大搜索范围，不修改 action history 算法。
- 只有 Hero 手牌可见且当前不是 Hero 行动轮时，主 ROI 才允许无 marker 的白色数字。
- Hero 正在行动时继续要求 marker，避免将“跟注 4”按钮误认为 Hero 已投入 4。

## 证据与验证

- 真实行动帧 `hero-bet-call-4-action.jpg` SHA256: `5eeb4251cc2d590279da5fcf79a07cf5ca801c748d8ce79842e8ea89d0e18cd4`。
- 真实 decided 帧 `hero-bet-24-chipless.jpg` SHA256: `2852431ca6eb7338dfd190794c8860b0e12681a76d8fd47853ad81b41e10b80d`。
- 专项结果：行动帧 `is_hero_turn=true/current_bet=null`；decided 帧 `is_hero_turn=false/current_bet=24`，同时识别 Hero stack=115、pot=49。
- Python 3.12 单测 70/70 通过。
- 本地 504 张正常图完整全字段回归 504/504 通过，零字段差异；均值 1.272 秒，中位数 1.240 秒，P95 1.801 秒，最大 2.508 秒。结果保存于本地 `/tmp/dezhou-chipless-hero-bet-regression-20261014-v2`，未在生产服务器执行全量回归。
- 在服务器上使用独立候选目录回放 2026-10-13 23:00–23:50 的 160 张原始 decided 图；候选代码未覆盖线上文件，服务未重启。与线上基线相比，共 9 张图发生变化，变化只涉及 Hero `current_bet` 以及由其派生的桌面 `current_bet`，其他字段零变化。
- 9 个新增 Hero 下注识别结果为 22、24、40、22、31、26、38、28、21。逐张检查服务器原图，画面均存在对应的白色下注数字，没有按钮或状态文字误吸收。其中 23:03:29 的目标图由 `hero.current_bet=null/table.current_bet=4` 修复为 `24/24`。
- 部署后又使用部署前备份和当前版本，在服务器隔离目录对 2026-10-13 07:00–08:59 的 273 张原始 decided 图做完整字段 A/B；两侧图片 SHA256 全部一致。共 7 张、12 个字段变化，其他字段零变化。
- 本次 Hero decided 修复在该批次恢复 3 个真实下注：87、54、408。上一轮普通下注修复同时纠正 4 个真实金额：627、1221、396、123。7 张变化图均已逐张检查，识别结果与画面一致。
- `py_compile` 和 `git diff --check` 通过。

## 发布状态

`7b98ae4` 和服务器回归证据 `641e713` 均已通过 CR。2026-10-14 00:27（PRC）已部署到服务器：

- 线上 `app/recognizer.py` SHA256 为 `515978e5ed693de8c5b53b3053703daf4070db20d9d764e8a0ad50c23452d844`。
- 运行进程由 PID 3743084 更新为 PID 4085210，PPID 为 1；`/health` 返回 200，启动日志无错误。
- 回滚文件位于 `backups/deploy-20261014-002734-641e713/recognizer.py`。
- 线上 HTTP 回放 23:03:27 行动帧返回 `is_hero_turn=true/hero.current_bet=null/table.current_bet=4`；23:03:29 decided 帧返回 `is_hero_turn=false/hero.current_bet=24/table.current_bet=24`。两次请求均返回 200。
- 本地 `main` 已快进到本报告提交；推送 GitHub 远端因当前环境没有可用 HTTPS 凭据而失败，远端 `main` 暂未更新。
