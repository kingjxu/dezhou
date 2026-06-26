from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RecognizeRequest(BaseModel):
    image_base64: str = Field(..., description="图片 base64（允许带 data:image/...;base64, 前缀）")
    app: str = Field(default="dpzx", description="目标 app 名称：dpzx / poler")
    parse_all: bool = Field(
        default=False,
        description="是否完整解析。true=不论是否轮到自己都完整解析；"
                    "false(默认)=先判是否轮到自己，轮到则完整返回，没轮到返回置空(is_hero_turn=false)。",
    )


Stage = Literal["preflop", "flop", "turn", "river", "unknown"]


class TableInfo(BaseModel):
    stage:           Stage       = Field(default="unknown")
    community_cards: list[str]   = Field(default_factory=list)
    main_pot:        int | None  = Field(default=None)
    button_seat:     int | None  = Field(default=None, description="庄家座位号")
    blind_size:      str | None  = Field(default=None, description="盲注结构，如 '50/100' 或 '50/100(25)'")
    current_bet:     int | None  = Field(default=None, description="全桌跟注线（本轮最大下注额）")
    mushroom_count:  int | None  = Field(default=None, description="蘑菇数量（保险图标旁的数字）")


class PlayerInfo(BaseModel):
    seat:        int                                              = Field(...)
    status:      Literal["active","folded","all_in","empty","waiting","unknown"]   = Field(default="unknown")
    stack:       int | None  = Field(default=None)
    current_bet: int | None  = Field(default=None, description="本轮已下注金额")


class HeroInfo(PlayerInfo):
    is_hero_turn: bool | None  = Field(default=None)
    hero_cards:   list[str]    = Field(default_factory=list)


class RecognizeResponse(BaseModel):
    received_at: datetime
    parsed_at:   datetime
    elapsed_ms:  int
    app:         str = Field(default="dpzx", description="识别所用的 app 引擎")
    table_info:  TableInfo
    hero_info:   HeroInfo
    villains_info: list[PlayerInfo]
