---
version: "1.5"
last_updated: "2026-10-13"
---
# 架构
`app/main.py` 提供 /health、/recognize。Pydantic schema 在 app/schemas.py。
`app/engines/__init__.py` 按 dpzx/poler/polerv2 选择布局，缓存 TableRecognizer。
`app/recognizer.py` 完成图片解码、画布归一化、轮次门控、OCR、牌面 CNN、下注/状态/庄位识别并组装响应。
`app/layouts/` 是坐标和识别策略配置；polerv2 继承 poler，但应局部覆盖不同 UI 的参数和规则，不改变旧 app 行为。
OCR 使用 RapidOCR/Paddle 适配；牌点/花色通过 ONNX Runtime。polerv2 当前复用 poler 模型文件。
同步入口可被多个 HTTP 请求并发调用，实例和 OCR 引擎有缓存；新增请求级数据不要写入共享实例。

`app/recognition_log.py` 将每次进入识别函数的完成/异常事件写入轮转 JSONL；默认 `logs/recognition.jsonl`，不记录图片载荷。`scripts/watch_recognition.py` 在开发机经 SSH 读新增日志，通过 botmux 在指定会话汇报，生命周期有界，详情见 `docs/recognition-monitoring.md`。

识别响应先在 handler 内验证 schema，再记录成功结果；非法引擎输出记录 500。validation 排除需要服务端非空环境 token 与请求 token 匹配；未配置时所有请求均为 live。通知失败限次重试并写观察器日志，不中止 SSH 消费。

V2归一化后检查顶部两个空座十字，或“左右两条顶部余额且中央余额缺失”的占座形态；命中时浅拷贝识别器并绑定九人桌布局，仅供该请求使用，不修改缓存实例的 layout。顶部两座的动作标签 ROI 隔离后合成一次 OCR detector 调用，再映射回各自状态锚点，避免相邻标签互相覆盖并减少九人桌 CPU；八人桌不进入该分支。未匹配九人桌特征时维持八人桌布局。

polerv2 的Hero turn先验证中央绿色牌桌表面，再组合左侧fold动作面和右侧操作控件；左侧支持普通红色fold与all-in option的灰色白X。该门控拒绝启动页、历史面板和结束页。对手下注先按marker颜色/牌形定位，再从紧邻marker的数字组件开始，按同一基线、相近高度和数字级间距连续追踪；所有后续OCR和几何兜底只使用选中基线的组件，不能再带入marker图案形成的行外白色轮廓。裁剪从连续链最左端开始，遇到大间隔即停止，避免把同排状态文字或另一串数字送入OCR。金额支持与牌桌`max_stack=9999`一致的四位数，RapidOCR仍逐候选独立识别，避免宽数字行改变同批单数字行。该策略只修复绿桌布局，深色布局继续返回不可决策结果。

polerv2 相邻座位的状态搜索范围允许重叠，但状态文本只能归属到状态锚点或头像锚点距离最近的一个座位；不能因为同一句“等待下一手”同时进入两个ROI，就把仍显示余额和下注的相邻玩家清空。该归属规则只用于状态关键词，普通余额/昵称文本仍用于判断座位是否有人。

polerv2 八人桌空座使用专用“+”中心坐标，不复用在座玩家头像中心；右侧中间空座在当前2400高画布位于约`y=0.47`，而在座头像中心为约`y=0.50`。空座命中后输出`status=empty, stack=null, current_bet=null`；该规则只作用于八人桌显式绿色“+”占位，不据`stack=0`删除玩家。

## 2026-10-13 相邻状态与下注基线修复部署

- 用户明确要求在最小门禁通过后先部署、再继续完整回归。生产 `/home/ubuntu/poker/dezhou/app/recognizer.py` SHA256 为 `ea7270cff918141d17de4b3dcd8b2d71d683db194311bbbee2ccc85cc584d7fa`；进程由 `3206426` 更新为 `3715181`，PPID 1，健康检查正常。
- 部署前 Python 61/61、8张真实下注边界8/8通过；部署后第432张单数字4和22:17:03相邻等待状态/下注4均经本机HTTP逐字段回验。
- 回滚目录 `/home/ubuntu/poker/dezhou-backups/20261013-003535-before-aj-seat-wager`；完整506张真实图已在本地从第1张重跑并以506/506、0字段差异通过，最终本地 Python 3.12 单测66/66通过。
- 部署后根盘被历史 systemd journal 占满，`/recognize` 日志报 `No space left on device`；仅清理 `/var/log/journal` 的归档系统日志约1.5GB后，健康检查和真实HTTP识别恢复。Hero图、业务日志、识别记录、数据库和项目文件均未删除。506张回归迁到本地执行，未再占用生产服务器CPU。

## 2026-10-02 下注识别修复部署

- 业务提交：`4e74655`；服务目录 `/home/ubuntu/poker/dezhou`，监听 5555。
- polerv2 villain marker 按红色优先、蓝色兜底逐色定位，再由牌形比例、白色数字轮廓和 OCR 确认金额；Hero 下注使用旧 ROI 优先、新画布 ROI 兜底。
- 部署后进程 PID=129462、PPID=1，本机 `/health` 正常；`app/recognizer.py` SHA256 为 `413f729d85288b09b424965544011d301aceb2860fb719ce638560516c5d86a1`。
- 三张线上问题图复验通过：`18:24:09/12` 均识别 Hero=1、seat2=2、seat3=4、seat6=4；`18:42:29` 识别 Hero=2、seat5=2、seat8=2。
- 部署前验证 33 项相关单测通过，96 张人工标注 Hero 全字段回归为 96/96；回滚目录 `/home/ubuntu/poker/dezhou-backups/20261002-214513-before-4e74655`。

## 2026-10-02 polerv2 OCR CPU 优化候选

- 多个 villain 下注候选在颜色、牌形和数字轮廓门控后批量 OCR；Hero 多个固定 ROI 优先批量识别数字行，仅对未确定候选执行完整 ROI OCR 兜底。业务门控和金额安全规则不变。
- 固定 manifest version=4、SHA256 `197b2d898e77c1e88ba18c5e6b023fad797f1bfae5ecf98640665db23e94e736` 的 96 图全字段结果为 96/96；无下注反例未产生 Hero 或 villain 下注。
- 相同 4 vCPU 机器顺序 A/B：均值 1.830s→1.586s，P95 2.739s→2.046s，进程累计 CPU 674.085s→588.030s。完整证据见 `reports/cpu-optimization-20261002/REPORT.md`。
- 1/2/4 并发只用于验证共享识别器没有字段串扰和比较 CPU 饱和退化，不替代 96 图串行准确性结论。

## 2026-10-02 polerv2 OCR CPU 优化部署

- 业务提交 `8ec92fe` 经 CR LGTM 后部署到 `/home/ubuntu/poker/dezhou`；`app/recognizer.py` SHA256 为 `c913ba43237cf8b3cee70b0dd35617cc39a403ffc64a1f1940444e615e6a1832`。
- 服务 PID=156257、PPID=1，监听 5555；SSH 连接退出后进程仍存活，本机 `/health` 正常，部署后日志未发现 traceback/error/exception。
- 线上入口顺序回放固定 manifest 前 12 张：12/12 全字段通过；外层 wall 均值 1.705s、P95 2.281s，服务内均值 1.696s、P95 2.234s，进程 CPU 累计 77.980s（6.498 CPU-s/图）。
- 回滚目录 `/home/ubuntu/poker/dezhou-backups/20261002-231715-before-8ec92fe`，保存了部署前源码 SHA、旧 PID 和原 `app/recognizer.py`。

## 2026-10-03 全量 Hero 识别修复部署

- 业务提交 `d749eae` 经 CR LGTM 后部署到 `/home/ubuntu/poker/dezhou`。运行时 `app/recognizer.py` SHA256 为 `9d79907fa2b0addd78da2035db69525b2e47392cb62ffb13d85b5ec0e365a089`，`app/layouts/polerv2.py` SHA256 为 `9b957ebf90d8f93df631343a94a22de56ddefd2478eca4a115d5316fa7d64d23`。
- 修复覆盖九人桌占座检测和顶部状态、旧宽屏顶部余额、Hero/folded 余额复核、八人桌空座，以及蓝色牌背稀疏白线被猜成盲注 `1/2`；批量 OCR 路径和颜色/牌形门控保持不变。
- 部署前固定 manifest v5 的506张、20,998字段全量回归为506/506，最后28张分歧聚焦回归为28/28；远端 Python 3.12 全量43项单测、`py_compile` 和 `git diff --check` 通过。
- 部署后进程 PID=217913、PPID=1，监听5555；SSH重连后进程存活，本机 `/health` 正常，新增日志未发现 traceback/error/exception。
- 线上 HTTP 顺序回放12张代表图为12/12全字段通过；外层均值2.352秒、最大3.103秒。代表图覆盖九人桌顶部状态、宽屏/折叠/Hero余额、无下注牌背反例和相邻座位误判。
- 回滚目录 `/home/ubuntu/poker/dezhou-backups/20261003-022621-before-d749eae`，保存部署前两个运行时源码文件、SHA256和旧 PID。

## 2026-10-03 九人桌状态 OCR 优化部署

- 业务提交 `15b956b` 经 CR LGTM 后部署到 `/home/ubuntu/poker/dezhou`；运行时 `app/recognizer.py` SHA256 为 `19b06782aa57f4855f3164e8647786e0ca8a68f0e3ccf1af11f8e281c53ba659`。
- 服务 PID=312035、PPID=1，监听5555；本机 `/health` 正常，SSH重连后进程存活，部署日志未发现 traceback/error/exception。
- 部署后通过本机 HTTP 顺序回放12张九人桌和1张八人桌真实标注图，13/13全字段通过；外层均值1.977秒、最大2.645秒。
- 回滚目录 `/home/ubuntu/poker/dezhou-backups/20261003-085029-before-15b956b`，保存部署前 `app/recognizer.py`、旧PID和源码SHA。

## 2026-10-06 polerv2 轮次与公共牌过渡识别部署

- 业务提交 `c92e6dc`。polerv2 轮到 Hero 必须同时存在左侧红色 fold 控件和右侧操作控件；公共牌 rank CNN 低置信时进入 OCR 复核，Hero 手牌路径不变。
- Python 3.12 完整单测 48/48、52 张当天现场图候选重放、`py_compile` 和 `git diff --check` 通过。
- 运行时 `app/recognizer.py` SHA256 为 `26f42e12844a49560dc159bfc0e1deaa5ea944a783cf690de5fb62f1c83f4b4c`；部署后 PID 1512987、PPID 1，SSH 重连后 `/health` 正常。
- 生产服务顺序回放 52/52 HTTP 200：5 张结束页从 Hero turn true 修正为 false，其余 47 张保持 true；`17:51:12` 公共牌从错误的 `8♥` 修正为原图可见的 `10♥`。回放证据 SHA256 为 `2dbdc3db70868b3ca886f22b20fdf2595ce3f69673b49f71c464fad569f256ea`。
- 回滚目录 `/home/ubuntu/poker/dezhou-backups/20261006-1835-before-c92e6dc`，包含替换前 `app/recognizer.py`、旧 PID 和源码校验值。

## 2026-10-10 Hero 门控与下注首位识别修复部署

- 识别业务提交 `c50ad37` 经 CR LGTM 后部署；部署事实记录提交为 `5ced119`，不改变运行时。生产 `app/recognizer.py` SHA256 为 `eaa084af216082b8cab01c1a40ee4bff4d4c671ef5e53dcff9a46473c8028211`，`app/layouts/polerv2.py` SHA256 为 `9f1b4944e0d211c68da6460bdb94abd162eb4f97dbf0f35524c68ef4bee5fd9c`，与业务提交候选一致。
- 部署后 PID `2938328`、PPID `1`，SSH 重连后 `/health` 正常。5张问题图生产 HTTP 回放均恢复原图金额：3张102、2张264；启动页和历史面板继续拒绝，真实牌桌样本完整识别。
- 完整验证为 Python 48/48、固定manifest 506/506逐字段、当天158张中仅5张预期修正。深色布局仍不在支持范围内。
- 回滚目录 `/home/ubuntu/poker/dezhou-backups/20261010-210905-before-c50ad37`，保存替换前两个运行时源码、SHA256和旧 PID。

## 2026-10-11 polerv2 中右空座识别修复部署

- 识别业务提交 `6ab7fcb` 经 CR LGTM 后部署到 `/home/ubuntu/poker/dezhou`；仅替换运行时 `app/layouts/polerv2.py`，SHA256 为 `fb2580c827fcf776dce2749cbd87a1fe827ec28d5aa1477d0b5cb74a13086f4f`，`app/recognizer.py` 保持 `eaa084af216082b8cab01c1a40ee4bff4d4c671ef5e53dcff9a46473c8028211`。
- 部署后 PID `3206426`、PPID `1`，监听 5555；SSH 重连后 `/health` 正常。生产 HTTP 回放 14/14 张已确认问题图均输出 seat7=`empty/null/null`，A♥6♥ 样本与评审候选逐字段一致，参与人数恢复为 7，下游位置恢复为 `BB-Hero`。
- 真实 all-in 防回归样本中 seat4/seat7 继续输出 `all_in`、`stack=0`，证明本次显式绿色“+”坐标修正没有泛化为 `stack=0 → empty`。部署前验证为 Python 49/49、固定 manifest v6 506/506、150 张窗口新增命中恰好 14 张且无额外命中。
- 回滚目录 `/home/ubuntu/poker/dezhou-backups/20261011-153929-before-6ab7fcb`，保存替换前 `app/layouts/polerv2.py`、源码 SHA256 和旧/新 PID。
