# 22:17 A♥J♦ 识别修复报告

## 结论

2026-10-12 22:17:03 的真实 Hero 图同时触发了两个独立的 polerv2 识别缺陷：seat 3 的“等待下一手”被错误归给相邻 seat 2，以及 seat 6 的下注 `4` 被 marker 图案干扰后误读为 `7`。错误输入使下游把七人桌构造成六人桌，并虚构了一次 `raise to 7`；生产 v1 因而返回 `call`。

候选修复后，同一原图恢复为七人桌：seat 2=`active/stack 550/bet 1`，seat 3=`waiting/null/null`，seat 6=`active/stack 45/bet 4`，桌面最大下注为 `4`。按修正输入隔离回放旧 `/gto/decision`，结果为 `raise to 28`、概率 `1.0`。该回放用于证明错误输入会改变动作，不把单个响应声明为 GTO 真值。

本报告先记录候选代码和隔离验证，随后按用户明确指示于2026-10-13部署识别服务；部署事实与最终本地完整回归见末节。本次不修改 ddbaby。

## 证据与根因

- 原图：`2026-10-12 22:17:03-hero.jpg`，SHA256 `aa4c569e137ed9ec3f7d905351ab650523df1e5ad8380efe1ea34394b30621c8`。
- 状态 OCR 中“等待下一手”中心约为 `(0.0859, 0.5113)`；它距 seat 3 更近，但也落入 seat 2 原有半径。旧实现只检查范围，两个座位都消费该状态，随后 waiting 规则清空 seat 2 可见的余额 `550` 和下注 `1`。
- seat 6 下注裁剪内，数字 `4` 的白色组件为 `(24,41,20,25)`；marker 图案另产生 `(66,44,18,19)` 的行外白色组件。旧实现以数字基线计算可见位数，却仍用所有组件生成 OCR 行和完整候选重试，最终把 `4` 接受为 `7`。
- 新增脱敏下注夹具 `wager-4-off-row-marker-crop.jpg`，SHA256 `b209b7d26edc87ed1855d6615a3cd3a815147293d6837ec3d05c31b0322e50d4`；只包含下注与 marker，不包含昵称、身份或牌局凭证。
- 首轮全量回归在三位下注 `570/238` 发现 marker 边框比数字高 1–2 像素的边界，候选随即停止；选行改为优先同基线组件数量、单组件再按高度后，5 张相关图聚焦回归通过。新增脱敏夹具 `wager-570-tall-marker-border-crop.jpg`，SHA256 `1660f4c74ffe0722037969ed2e0878ee172759a911a7edecedb1553dfe8a484a`。
- 后续全量回归复核第299张原图时，画面下注、底池120和Hero跟注按钮均证明正确值为`60`；旧标签`600`把相邻marker圆环当成尾零。manifest按原图更正为`60`并记录原因，新增脱敏反例 `wager-60-overlapping-marker-crop.jpg`，SHA256 `9bd20b3eb2ad2864a53026dda889a9111c948932fcadef3a9106b239d0c4b95f`。实现不做猜测补零。
- 首次最终回归在第432张真实图发现单数字 `4` 比常见字形矮约2像素，且裁剪左边缘含有相邻昵称残片；候选立即停止。最终选行允许边缘组件继续参与合法多位数字链，但当候选数量相同时优先完整的内侧组件，避免边缘残片单独抢占基线。新增现场裁剪 `wager-4-clipped-neighbor-crop.jpg`，SHA256 `36ddbfabcca5ae0bb03452b8617a4dd6f165b39694480351c3132733aaf6a056`。
- 第452张旧 manifest 标为无下注，但原图 seat 8 明确显示 `8`，Hero 跟注按钮同为 `8`；按两处独立画面证据将 table/seat8 纠正为 `8` 并记录原因。代码没有为旧错误标签增加特殊兼容。

## 修复

- polerv2 状态关键词按状态锚点和头像锚点的最近距离唯一归属；普通昵称、余额文字仍用于座位存在性判断。旧 poler 布局不进入该规则。
- 下注识别优先选择同基线组件数量最多的数字行，单组件候选再以高度区分；以相近高度和 y 对齐得到 `row_boxes` 后，OCR 行、完整候选重试和几何兜底均不再使用行外组件。
- 新增真实下注夹具回归和 22:17:03 实际 OCR 坐标的相邻状态回归，并同步 `.ai-context` 架构与测试约束。

## 输入与决策对照

生产错误请求的关键差异：`num_players=6`、缺少 seat 2、seat 6 被映射为 straddle 且 `bet=7`，并生成虚假的 `raise amount=7` 历史；响应为 `call`、概率 `1.0`。

修正请求为 `num_players=7`，位置依次为 BTN-Hero、SB、BB、straddle、UTG、MP、CO；SB=`550/1`，BB=`244/2`，straddle=`408/4`，UTG=`45/4`，仅包含 UTG call 4、MP fold、CO fold。隔离 v1 响应为 `raise`、目标总下注 `28`、概率 `1.0`。

## 验证

- 真实 22:17:03、06、08、10、12 五个连续帧候选复跑：均输出 current_bet=4、seat 2=`active/550/1`、seat 3=`waiting/null/null`、seat 6=`active/45/4`。
- 部署前服务器 Python 单测：61/61 通过；最终本地 Python 3.12 单测：66/66 通过。
- `570/238/60/20` 及第432张边缘 `4` 等8张真实图聚焦回归：8/8通过。
- 固定 manifest v6 506 张真实 Hero 图在本地逐字段回归：506/506通过，0个字段差异；平均1.613秒/张，中位数1.521秒，P95 2.409秒，最大3.958秒。
- `python -m py_compile app/recognizer.py test_polerv2.py`：通过。
- `git diff --check`：提交前执行。

## 风险边界

- 最近锚点归属只处理 polerv2 状态关键词，不修改余额、下注或旧布局的坐标规则。
- 数字基线仍依赖既有高度和 y 对齐阈值；本次用真实 `4` 夹具、既有多位下注用例和固定真实图集覆盖回归。
- 修正后的 `raise to 28` 是当前上游对正确输入的返回值；策略是否最优需要独立 solver 评估。

## 生产部署（2026-10-13）

- 用户明确要求先部署再继续全量回归；部署前最小门禁为 Python 61/61 和8张真实下注边界8/8通过。
- 生产 `/home/ubuntu/poker/dezhou/app/recognizer.py` 原子替换后 SHA256 为 `ea7270cff918141d17de4b3dcd8b2d71d683db194311bbbee2ccc85cc584d7fa`；旧进程 `3206426`，新进程 `3715181`、PPID 1，`/health` 正常。
- 回滚目录 `/home/ubuntu/poker/dezhou-backups/20261013-003535-before-aj-seat-wager` 保存替换前源码、旧进程及新旧 SHA。
- 生产 HTTP 回验第432张为 table/seat4 `current_bet=4`；22:17:03 A♥J♦ 原图为 table `current_bet=4`、seat2=`active/550/1`、seat3=`waiting/null/null`、seat6=`active/45/4`。
- 部署后服务器根盘一度仅余25MB，`/recognize` 日志出现 `No space left on device`，端上无决策。服务进程未退出；清理长期累计的 systemd journal 后释放约1.5GB，健康检查与两张生产HTTP识别均恢复200。本轮服务器隔离回归已按用户要求停止，完整图集复制到本地后继续验证。
- 完整图集复制到本地后从第1张重新运行，最终506/506通过；服务器没有执行该全量回归。代码与生产文件 SHA 一致，后续仅需完成提交与 CR 闭环。
