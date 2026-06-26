# Linux 部署指南 —— 持久化后台运行

## 前置：装依赖

```bash
cd /opt/dezhou_v19_cc_filter/dezhou   # 或你实际放代码的目录
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements-linux.txt
```

如果要训练 CNN 就再装：
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnx
```

## 快速启动（一次性）

```bash
cd /opt/dezhou_v19_cc_filter/dezhou
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 5555
```

`Ctrl+C` 就停了 —— **只适合调试**。

---

## 🔥 方案一：一条命令后台跑（nohup，最简单）

```bash
cd /opt/dezhou_v19_cc_filter/dezhou
nohup .venv/bin/python -m uvicorn app.main:app \
    --host 0.0.0.0 --port 5555 \
    > /var/log/dezhou.log 2>&1 &
```

**特点**：
- ✅ SSH 断开后依然在跑
- ✅ 输出全部写到 `/var/log/dezhou.log`
- ❌ 机器重启后要手动再跑
- ❌ 服务崩了不会自动拉起

**管理命令**：
```bash
# 看日志
tail -f /var/log/dezhou.log

# 查进程
ps aux | grep uvicorn

# 健康检查
curl http://127.0.0.1:5555/health

# 停
pkill -f "uvicorn app.main:app"
```

---

## 🏭 方案二：systemd 服务（生产推荐）

自动重启 + 开机自启 + 标准日志管理。

### 1. 创建 service 文件

`/etc/systemd/system/dezhou.service`：

```ini
[Unit]
Description=Dezhou Poker Recognition Service
After=network.target

[Service]
Type=simple
User=ubuntu                                   # 改成跑服务的用户
Group=ubuntu
WorkingDirectory=/opt/dezhou_v19_cc_filter/dezhou
Environment="PATH=/opt/dezhou_v19_cc_filter/dezhou/.venv/bin"
# 如果用 PaddleOCR 在这里加: Environment="DEZHOU_OCR=paddle"
ExecStart=/opt/dezhou_v19_cc_filter/dezhou/.venv/bin/python \
          -m uvicorn app.main:app --host 0.0.0.0 --port 5555 --workers 1
Restart=on-failure
RestartSec=5
# 防止 OCR 模型首次加载超过默认 90s 导致被 kill
TimeoutStartSec=120

# 日志写入 journald（journalctl 可查）
StandardOutput=journal
StandardError=journal
SyslogIdentifier=dezhou

[Install]
WantedBy=multi-user.target
```

> **`--workers 1` 建议理由**：每个 worker 会独立加载一份 OCR + 两个 ONNX 模型（约 500MB 内存）。只有高并发场景才需要 2+ workers。

### 2. 启动并设置开机自启

```bash
sudo systemctl daemon-reload
sudo systemctl enable dezhou          # 开机自启
sudo systemctl start dezhou           # 立即启动

# 查状态
sudo systemctl status dezhou

# 看日志（实时）
sudo journalctl -u dezhou -f

# 看最近 200 行日志
sudo journalctl -u dezhou -n 200
```

### 3. 日常管理

```bash
sudo systemctl stop dezhou             # 停
sudo systemctl restart dezhou          # 重启
sudo systemctl disable dezhou          # 取消开机自启

# 配置改了要重载
sudo systemctl daemon-reload
sudo systemctl restart dezhou
```

---

## 📦 方案三：tmux（开发/运维调试用）

服务在可恢复的交互窗口里跑，登进去就能看实时日志：

```bash
# 创建 session
tmux new -s dezhou

# 在 session 里正常启动
cd /opt/dezhou_v19_cc_filter/dezhou
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 5555

# 按 Ctrl+B 然后 D 退出 session（服务还在跑）
# 重新连：
tmux attach -t dezhou

# 列出 session
tmux ls

# 干掉 session
tmux kill-session -t dezhou
```

---

## 📋 选择建议

| 场景 | 方案 |
|---|---|
| 只想赶紧跑起来看能不能用 | **方案一 nohup**，三行搞定 |
| 正式部署，要稳 | **方案二 systemd**，崩了自动起 + 开机自启 |
| 本地服务器自己用，想随时看日志 | **方案三 tmux** |

---

## 常见问题

### Q1：服务器 2C4G 够吗？
勉强够。RapidOCR + ONNX 模型加载后驻留 ~400MB。建议 4GB+ 内存。

### Q2：端口被占？
改 `--port 5555` 成别的；或者 `lsof -i:5555` 看谁占着。

### Q3：防火墙？
```bash
# Ubuntu/Debian (ufw)
sudo ufw allow 5555/tcp

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-port=5555/tcp
sudo firewall-cmd --reload
```

### Q4：首次启动很慢？
OCR 模型第一次下载/加载大约 10-30 秒。用 systemd 记得把 `TimeoutStartSec` 设到 120 以上。

### Q5：想看详细 HTTP 访问日志？
uvicorn 加 `--access-log` 参数，或者前面套一层 nginx reverse proxy。

### Q6：要 nginx 反代吗？
内部使用不用。如果要公网暴露建议套 nginx：
```nginx
server {
    listen 80;
    server_name your.domain.com;
    location / {
        proxy_pass http://127.0.0.1:5555;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
```
