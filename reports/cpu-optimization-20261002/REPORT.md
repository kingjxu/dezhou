# polerv2 OCR CPU 优化验证报告（2026-10-02）

## 变更范围

本次只调整 `app/recognizer.py` 的 polerv2 下注金额读取：

- 同一帧多个 villain 下注候选先按既有颜色、牌形和数字轮廓规则筛选，再批量执行 OCR，减少重复启动文本检测器。
- Hero 多个固定 ROI 先提取数字行并批量识别；仅在直接识别不确定时执行原有完整 ROI OCR 兜底。
- 颜色仍只定位下注候选，金额仍由白色数字轮廓和 OCR 决定；多位数字不得退化为单个 `1/2/4` 的安全规则不变。

对比源码：

- 基线 `app/recognizer.py` SHA256：`413f729d85288b09b424965544011d301aceb2860fb719ce638560516c5d86a1`
- 候选 `app/recognizer.py` SHA256：`c913ba43237cf8b3cee70b0dd35617cc39a403ffc64a1f1940444e615e6a1832`

## 固定样本与环境

- 人工标注 manifest：`reports/hero-regression/manifest.json`，version=4，SHA256 `197b2d898e77c1e88ba18c5e6b023fad797f1bfae5ecf98640665db23e94e736`。
- 串行准确性与性能使用 manifest 的全部 96 张图；每张图运行前均重新校验图片 SHA256。
- 并发对比固定使用同一 manifest 的前 12 张图，分别以 1、2、4 个线程调用同一缓存识别器。
- 机器：4 vCPU，Intel Xeon Platinum 8255C 2.50GHz；基线和候选依次运行，避免互相争抢 CPU。
- 每组先运行一次不计时的模型预热；wall time 使用 `perf_counter`，进程累计 CPU time 使用 `process_time`，包含 OCR/ONNX 内部线程。

## 96 图串行结果

| 指标 | 基线 | 候选 | 变化 |
|---|---:|---:|---:|
| 全字段准确性 | 96/96 | 96/96 | 无回归 |
| wall time | 175.732s | 152.221s | -13.4% |
| 单图均值 | 1.830s | 1.586s | -13.4% |
| 单图中位数 | 1.812s | 1.616s | -10.8% |
| 单图 P95 | 2.739s | 2.046s | -25.3% |
| 单图最大值 | 3.245s | 2.246s | -30.8% |
| 进程累计 CPU time | 674.085s | 588.030s | -12.8% |
| CPU time / 图 | 7.022s | 6.125s | -12.8% |
| 吞吐 | 0.546 图/s | 0.631 图/s | +15.4% |

## 1/2/4 并发对比

| 并发 | 版本 | 准确性 | wall time | 均值 | P95 | CPU 累计 | CPU/图 | 吞吐 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 基线 | 12/12 | 21.988s | 1.832s | 3.118s | 84.738s | 7.061s | 0.546 图/s |
| 1 | 候选 | 12/12 | 20.634s | 1.719s | 1.994s | 79.541s | 6.628s | 0.582 图/s |
| 2 | 基线 | 12/12 | 27.010s | 4.496s | 6.697s | 106.182s | 8.848s | 0.444 图/s |
| 2 | 候选 | 12/12 | 20.798s | 3.454s | 4.103s | 81.817s | 6.818s | 0.577 图/s |
| 4 | 基线 | 12/12 | 26.156s | 8.357s | 12.243s | 103.225s | 8.602s | 0.459 图/s |
| 4 | 候选 | 12/12 | 20.743s | 6.785s | 7.764s | 81.782s | 6.815s | 0.578 图/s |

候选在并发 2/4 时 CPU 累计分别下降 22.9%/20.8%，吞吐分别提升 29.9%/26.1%，所有响应仍与人工标注一致。4 核机器单请求已使用多个 ONNX 线程，因此增加请求并发不会继续线性提高吞吐；这组数据只验证共享识别器在并发调用时没有字段串扰，并比较 CPU 饱和时的退化程度。准确性结论以 96 图串行回归为准。

## 边界和测试

- `python3 -m unittest test_mushroom test_nine_seat test_ocr_engine_isolation test_polerv2 test_turn_gate_images test_watch_recognition`：33 项通过。
- `fixtures/turn-gate/no-actions.webp`：Hero 下注为 `None`，seat 2～8 的下注均为 `None`，没有因批量 OCR 虚构下注。
- `python3 -m py_compile app/recognizer.py test_polerv2.py`：通过。
- `git diff --check`：通过。

远端 staging 的 `python3 -m unittest discover` 额外收集 `test_recognition_log.py` 时因环境未安装测试专用 `httpx` 无法导入；该测试覆盖 HTTP 日志，与本次 OCR 代码无调用关系。没有为通过测试修改生产依赖，相关的其余 33 项、96 图和边界反例均已单独运行通过。

原始机器汇总数据保存在同目录的 `benchmark-summary.json`；并发数据不替代 96 图串行人工真值回归。
