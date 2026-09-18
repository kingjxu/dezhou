# 识别日志与实时监控

`POST /recognize` 进入识别函数后，成功和异常都会记录一条 JSONL：request_id、接收时间、app、parse_all、输入字符数、HTTP 状态、elapsed_ms；成功包含 table_info/hero_info/villains_info，失败包含异常类型。图片、base64 和原始异常文本不落日志。请求模型校验阶段的 422 仍由服务器访问日志记录。

默认路径 `logs/recognition.jsonl`，可用 `DEZHOU_RECOGNITION_LOG` 指定。单文件 10 MiB，保留 3 个备份（共约 40 MiB）；采用当前单 uvicorn 进程的部署方式。多进程共享该文件的轮转不在本次支持范围内，应分别配置日志路径或接集中日志系统。

验证请求加 `X-Recognition-Source: validation`，日志 source=validation；其他请求为 live。该标记只用于排除自测流量，不是鉴权或可信客户端身份。

在装有 botmux 且已经建立 SSH 认证的开发机运行：

```bash
python scripts/watch_recognition.py \
  --ssh ubuntu@211.159.223.91 \
  --control-path /path/to/ssh-control-socket \
  --session-id <当前会话ID> \
  --duration 1800 --summary-interval 300 --slow-ms 3000
```

监控只追踪启动后的新增日志；每条识别完成记录通常在约 1 秒 tail 轮询后被读取。首笔真实流量立即通知；HTTP 错误和达到慢阈值的请求分开限频，每类最多每分钟一条，其余计入汇总。有新增请求时每 5 分钟汇总总请求、错误、慢请求、平均耗时及最近最多 1000 条的 P95。默认持续 30 分钟，到时发送结束汇总并退出。SSH 中断会通知监控停止；需要保持监控进程、botmux 凭据及 SSH 会话可用，不能理解为永久后台服务。

监控显示的是服务识别输出；没有同步的原图人工标注时，不能自动判定真实准确率。迟迟未完成的请求也尚未产生结果日志。原图仍按现有数据路径保管。

本地演练可用 `--input-file <JSONL> --dry-run`，只打印将要发送的通知，不发送消息。不使用密码参数或把密码写入脚本。
