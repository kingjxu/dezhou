---
version: "1.0"
last_updated: "2026-09-21"
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

V2归一化后检查顶部两个空座十字标记，命中时浅拷贝识别器并绑定九人桌布局，仅供该请求使用；不修改缓存实例的layout，不增加OCR实例或并发OCR。未匹配时维持八人桌布局，顶部有人/遮挡的九人桌尚无验证样本。

## 2026-10-02 下注识别修复部署

- 业务提交：`4e74655`；服务目录 `/home/ubuntu/poker/dezhou`，监听 5555。
- polerv2 villain marker 按红色优先、蓝色兜底逐色定位，再由牌形比例、白色数字轮廓和 OCR 确认金额；Hero 下注使用旧 ROI 优先、新画布 ROI 兜底。
- 部署后进程 PID=129462、PPID=1，本机 `/health` 正常；`app/recognizer.py` SHA256 为 `413f729d85288b09b424965544011d301aceb2860fb719ce638560516c5d86a1`。
- 三张线上问题图复验通过：`18:24:09/12` 均识别 Hero=1、seat2=2、seat3=4、seat6=4；`18:42:29` 识别 Hero=2、seat5=2、seat8=2。
- 部署前验证 33 项相关单测通过，96 张人工标注 Hero 全字段回归为 96/96；回滚目录 `/home/ubuntu/poker/dezhou-backups/20261002-214513-before-4e74655`。
