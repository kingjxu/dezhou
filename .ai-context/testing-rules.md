---
version: "1.0"
last_updated: "2026-09-18"
---
# 测试与性能验证
- Python 3.12 + `pip install -r requirements-test.txt`（OCR / ONNX / OpenCV 与 HTTP TestClient 依赖）执行 `python -m unittest discover`；原有 polerv2 用例包含 1080、1176、1179 宽截图与画布归一化，必须保留。
- `python scripts/check_hero_images.py --images <图片目录> --output <结果目录>`：按 `reports/hero-regression/manifest.json` 的 SHA256 校验 95 张原图；逐字段比较人工真值，保存每次响应、差异、耗时，失败退出非零。
- 图片集外置于本次审计归档；文件名、哈希和标签随代码提交。缺文件或哈希不符直接失败，不跳过。
- 历史2026-09-18结果仅覆盖部分人工标注字段；2026-09-21的manifest扩展为95张八人桌每图41字段。新增九人桌仍待口径确认，不能宣称全部96图整份JSON准确。
- 基准单进程串行执行，初始化单列，测量 `recognize(parse_all=True)` 包含解码与预处理，不包括 HTTP、网络和排队。P95 使用 nearest-rank。
- 不与其他 OCR 回归并行跑性能测试；保留依赖版本和原始结果，区分首次探索和最终验收。
- 修复说明与结果见 `reports/hero-regression/REPORT.md`。每次代码提交后需交 cr 审查，阻塞意见修复后再提交复审。

- 多位下注单测用真实 CV 轮廓路径和受控 OCR 低置信/异常，保证 33/156 不降成单个小数字；manifest 同时校验可见 24/33/156/30。日志单测覆盖成功、400、500 与图片内容不落日志；监控脚本用 dry-run 演练通知与验证流量排除。

- `test_recognition_log.py` 经 TestClient 验证真实响应状态：成功/400/500、非法模型 500、默认/缺失/错误验证 token 不可排除监控。`test_watch_recognition.py` 启动独立 CLI，用临时 SSH/botmux 替身验证首次失败、三次失败后恢复、持续失败均继续消费事件并正确汇总，不发送实际群消息。

## 2026-09-21 全字段复核（尚未部署）
- 新一轮盘点 96 张 hero 图，其中旧 95 张为八人桌，新增一张为九人桌。见 `reports/turn-fix-20260921/REPORT.md`。
- `reports/turn-fix-20260921/eight-seat-manifest.json` 补齐八人桌每图 41 项扑克字段人工断言；不能把旧的部分字段 95/95 当成整份响应准确率。
- 用户确认 folded 的 current_bet 保持 null。九人桌座位编号正在确认，探索 manifest 第 96 项不是最终验收标签。
- 未完成九人桌验证、CR 与部署，不能宣称本轮全量修复完成。
