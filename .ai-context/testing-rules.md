---
version: "1.0"
last_updated: "2026-09-18"
---
# 测试与性能验证
- Python 3.12 + `pip install -r requirements-test.txt`（OCR / ONNX / OpenCV 与 HTTP TestClient 依赖）执行 `python -m unittest discover`；原有 polerv2 用例包含 1080、1176、1179 宽截图与画布归一化，必须保留。
- `python scripts/check_hero_images.py --images <图片目录> --output <结果目录>`：按默认 `reports/hero-regression/manifest.json` 的 SHA256 校验 96 张原图（95张八人桌加1张九人桌）；逐字段比较人工真值，保存每次响应、差异、耗时，失败退出非零。
- 图片集外置于本次审计归档；文件名、哈希和标签随代码提交。缺文件或哈希不符直接失败，不跳过。
- 历史2026-09-18结果仅覆盖部分人工标注字段；2026-09-21的manifest扩展为95张八人桌每图41字段。新版默认manifest另含已确认编号的九人桌；全96图通过后才能报告本样本集完整字段通过。
- 基准单进程串行执行，初始化单列，测量 `recognize(parse_all=True)` 包含解码与预处理，不包括 HTTP、网络和排队。P95 使用 nearest-rank。
- 不与其他 OCR 回归并行跑性能测试；保留依赖版本和原始结果，区分首次探索和最终验收。
- 修复说明与结果见 `reports/hero-regression/REPORT.md`。每次代码提交后需交 cr 审查，阻塞意见修复后再提交复审。

- 多位下注单测用真实 CV 轮廓路径和受控 OCR 低置信/异常，保证 33/156 不降成单个小数字；manifest 同时校验可见 24/33/156/30。日志单测覆盖成功、400、500 与图片内容不落日志；监控脚本用 dry-run 演练通知与验证流量排除。

- `test_recognition_log.py` 经 TestClient 验证真实响应状态：成功/400/500、非法模型 500、默认/缺失/错误验证 token 不可排除监控。`test_watch_recognition.py` 启动独立 CLI，用临时 SSH/botmux 替身验证首次失败、三次失败后恢复、持续失败均继续消费事件并正确汇总，不发送实际群消息。

## 2026-09-21 全字段复核（尚未部署）
- 新一轮盘点 96 张 hero 图，其中旧 95 张为八人桌，新增一张为九人桌。默认manifest已补第96图45字段；八人桌A/B仍固定eight-seat-manifest。见 `reports/turn-fix-20260921/REPORT.md`。
- `reports/turn-fix-20260921/eight-seat-manifest.json` 补齐八人桌每图 41 项扑克字段人工断言；不能把旧的部分字段 95/95 当成整份响应准确率。
- 用户确认 folded 的 current_bet 保持 null。九人桌座位编号已由用户确认；默认manifest第96项为重新人工核对后的最终标签，未跟踪探索manifest不用于验收。
- 未完成九人桌验证、CR 与部署，不能宣称本轮全量修复完成。

- 性能修复必须用 `reports/turn-fix-20260921/run_ab.py` 同机顺序A/B；`check_hero_images.py --code-root`选择被测代码，固定同95图、同完整标签、同runner。不能以不同日期的部分字段耗时替代严格A/B。
- active余额按缺失/低置信/位数不符门控并合并OCR；测试覆盖真实191截成19、多行归属、跨行/多候选拒绝，以及真实下注22/21/9/4。完整下注掩码只在快速路径不确定时重试。

- `test_nine_seat.py` 验证两幅真实九人桌截图及尺寸/JPEG/黑边变换、缺少一个十字的反例、并发请求布局隔离（不并发OCR）、实际九人桌后八人桌识别不串位。

- 多位下注即使OCR高置信，也不能少于可见数字行组件数；真实21图被模拟高置信读成2时必须拒绝，避免序列相关的截断结果。
