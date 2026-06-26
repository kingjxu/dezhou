"""多 app 识别引擎工厂。

用法：
    from app.engines import get_recognizer
    rec = get_recognizer("dpzx")
    result = rec.recognize(image_base64)
"""
from __future__ import annotations

from app.layouts.dpzx import LayoutDpzx
from app.layouts.poler import LayoutPoler
from app.recognizer import TableRecognizer

# 已创建的识别器缓存（每个 app 只创建一次，线程安全靠 GIL）
_cache: dict[str, TableRecognizer] = {}

_LAYOUTS = {
    "dpzx":  LayoutDpzx,
    "poler": LayoutPoler,
}


def get_recognizer(app_name: str) -> TableRecognizer:
    """根据 app 名称返回对应的识别器实例（带缓存）。

    Args:
        app_name: 支持 "dpzx"、"poler"
    Raises:
        ValueError: 未知的 app 名称
    """
    if app_name in _cache:
        return _cache[app_name]
    layout_cls = _LAYOUTS.get(app_name)
    if layout_cls is None:
        supported = ", ".join(sorted(_LAYOUTS.keys()))
        raise ValueError(f"Unknown app: {app_name!r}. Supported: {supported}")
    rec = TableRecognizer(layout=layout_cls())
    _cache[app_name] = rec
    return rec


def supported_apps() -> list[str]:
    """返回所有支持的 app 名称列表。"""
    return sorted(_LAYOUTS.keys())
