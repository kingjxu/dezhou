---
version: "1.0"
last_updated: "2026-09-18"
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
