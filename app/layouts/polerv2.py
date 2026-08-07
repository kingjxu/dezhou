"""Poler V2 layout.

This layout is intentionally separate from ``poler``. It carries the
fixed-orientation and resolution-tolerant configuration for the newer AA Poker
table UI without changing the existing app's coordinates or OCR behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.layouts.base import Anchor
from app.layouts.poler import LayoutPoler


@dataclass(frozen=True)
class LayoutPolerV2(LayoutPoler):
    app_name: str = "polerv2"
    reference_canvas: tuple = (1080, 2400)
    # V2 hero cards are wider than the legacy crop. Keep enough of the
    # top-left corner for two-character ranks such as "10".
    hero_card_rank_rx: tuple = (0.00, 0.80)
    hero_card_rank_ry: tuple = (0.02, 0.60)

    # The newer UI places the in-table chip stack farther toward the center and
    # slightly above the original Poler anchors. Keep these coordinates local to
    # V2 so old Poler screenshots retain their current interpretation.
    seat_bet_anchors: dict = field(default_factory=lambda: {
        2: Anchor(0.255, 0.610, 0.055),
        3: Anchor(0.255, 0.495, 0.055),
        4: Anchor(0.255, 0.380, 0.055),
        5: Anchor(0.505, 0.240, 0.045),
        6: Anchor(0.760, 0.380, 0.060),
        7: Anchor(0.760, 0.466, 0.060),
        8: Anchor(0.760, 0.610, 0.060),
    })
    seat_bet_fallback_rois: dict = field(default_factory=lambda: {
        2: (0.205, 0.575, 0.305, 0.645),
        3: (0.205, 0.460, 0.305, 0.530),
        4: (0.205, 0.345, 0.305, 0.415),
        5: (0.455, 0.210, 0.555, 0.275),
        6: (0.710, 0.345, 0.810, 0.415),
        7: (0.710, 0.430, 0.810, 0.500),
        8: (0.710, 0.575, 0.810, 0.645),
    })
    seat_bet_fallback_chip_colors: dict = field(default_factory=lambda: {
        seat: ("red",) for seat in range(2, 9)
    })
