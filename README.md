# 德州扑克识别服务 — Linux 部署包（CPU 版）

精简版，**纯 CPU 推理，零 GPU 代码**。
要 GPU 加速请用姊妹包 `dezhou_v20_linux_gpu/`。

## 目录结构

```
dezhou_v20_linux_cpu/
├── README.md                    本文件
├── LINUX_DEPLOY.md              完整部署步骤（systemd / nginx / 防火墙）
├── FIELD_MAPPING.md             /recognize 接口字段含义
├── requirements.txt             Linux CPU 依赖（headless OpenCV + 鉴权 OCR）
├── dezhou.service.example       systemd unit 示例
└── app/
    ├── main.py                  FastAPI 入口
    ├── recognizer.py            识别核心（CPU 专用，无任何 CUDA 路径）
    ├── schemas.py               请求/响应 schema
    ├── engines/__init__.py      多 app 工厂
    ├── layouts/                 dpzx + poler 坐标布局
    │   ├── base.py
    │   ├── dpzx.py
    │   └── poler.py
    ├── rank_classifier_dpzx.onnx   dpzx 牌点 CNN
    ├── rank_classifier_poler.onnx  poler 牌点 CNN（v0.7 含 dim 重训）
    ├── suit_classifier_dpzx.onnx   dpzx 花色 CNN
    ├── suit_classifier_poler.onnx  poler 花色 CNN
    ├── rank_templates/             模板匹配兜底
    └── ocr_models/                 PP-OCRv3 mobile 模型（可选，提速 50%）
        ├── ch_PP-OCRv3_det_infer.onnx
        └── ch_PP-OCRv3_rec_infer.onnx
```

**不包含**：训练数据 (`poler_patches*`)、原始截图 (`poler/`、`tests/poler/`)、
训练脚本 (`tools/`)、单元测试 (`tests/`)、调试图 (`_dbg_*`)、`.bak_*` 历史模型，
**也不含 GPU/CUDA 任何代码或依赖**。

## 一分钟启动

```bash
# 1. 建 venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# 2. 装依赖（约 5 分钟）
pip install -r requirements.txt

# 3. 跑起来
python -m uvicorn app.main:app --host 0.0.0.0 --port 5555
```

健康检查：`curl http://127.0.0.1:5555/health` → `{"ok":true,"supported_apps":["dpzx","poler"]}`

## 调用示例

```bash
# 把图片转 base64 → POST /recognize
B64=$(base64 -w0 your_image.jpg)
curl -X POST http://127.0.0.1:5555/recognize \
     -H 'Content-Type: application/json' \
     -d "{\"image_base64\": \"$B64\", \"app\": \"poler\"}"
```

`app` 字段：`"poler"` 或 `"dpzx"`。响应字段含义见 [FIELD_MAPPING.md](FIELD_MAPPING.md)。

## 后台守护（systemd）

把 `dezhou.service.example` 里的 `WorkingDirectory` 改成实际路径，然后：

```bash
sudo cp dezhou.service.example /etc/systemd/system/dezhou.service
sudo systemctl daemon-reload
sudo systemctl enable --now dezhou.service
sudo systemctl status dezhou
```

详细排错见 [LINUX_DEPLOY.md](LINUX_DEPLOY.md)。

## 性能参考（CPU）

| CPU 配置 | 单张耗时 | 备注 |
|---|---|---|
| 1 物理核 | ~16 s | 入门 VPS / 边缘设备 |
| 2 物理核 | ~8 s | 估算 |
| **3 物理核**（i7-9750H 实测） | **~3.4 s** | 笔记本 |
| 4 物理核 | ~2.7 s | 性价比甜区 |
| 8 物理核 | ~1.7 s | 接近 CPU 极限 |

**吞吐**：单 worker 串行约 0.25-0.5 张/秒。多 worker 在物理核数 ≤ 3 的机器上**并发没收益**（CPU 饱和），6+ 物理核机器开 `--workers 2` 可翻倍。

需要 GPU 加速请切到 `dezhou_v20_linux_gpu/` 包（任何 NVIDIA 显卡 3-10x 提速）。

## 配置开关（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEZHOU_OCR` | `rapid` | OCR 后端：`rapid` / `paddle`（**纯 CPU**） |
| `DEZHOU_OCR_MODEL` | （server） | 设 `mobile` 切到 PP-OCRv3 mobile，提速 50% 准确率 -1~2% |
| `DEZHOU_OCR_WIDTH` | `640` | 主图 OCR 缩放宽度（提高到 1080 准确率↑速度↓） |

## 版本

v0.7-cpu（CPU only）— 包含 poler dim 弃牌牌识别、白底反转 OCR、hero 下注位置修正、
留座/补盲/过庄状态、table.current_bet 取 max(seat bet)、以及全部 v0.5/v0.6 修复。
**dpzx 行为与 v0.5 byte-identical**。
