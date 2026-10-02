"""Poler V2 layout.

This layout is intentionally separate from ``poler``. It carries the
fixed-orientation and resolution-tolerant configuration for the newer AA Poker
table UI without changing the existing app's coordinates or OCR behavior.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from app.layouts.base import Anchor
from app.layouts.poler import LayoutPoler


@dataclass(frozen=True)
class LayoutPolerV2(LayoutPoler):
    app_name: str = "polerv2"
    reference_canvas: tuple = (1080, 2400)
    # V2 hero cards are wider than the legacy crop. Keep enough of the
    # top-left corner for two-character ranks such as "10". The recognizer may
    # run an additional tight-corner check for K/A when this broad patch is
    # polluted by face art.
    hero_card_rank_rx: tuple = (0.00, 0.80)
    hero_card_rank_ry: tuple = (0.02, 0.60)
    hero_stack_anchor: Anchor = field(default_factory=lambda: Anchor(0.50, 0.888, 0.035))
    hero_stack_roi: tuple = (0.42, 0.875, 0.58, 0.905)
    hero_bet_fallback_roi: tuple = (0.755, 0.765, 0.860, 0.795)
    # A newer V2 canvas places the committed Hero wager below the action
    # buttons. Read this only after the original ROI misses, preserving both UI
    # generations without treating the action-button amount as a wager.
    hero_bet_alternate_rois: tuple = ((0.750, 0.790, 0.870, 0.825),)
    # Face-card suit glyph is below the rank in the upper-left corner; use a
    # larger patch than legacy poler so K♠ is not split into club-like blobs.
    face_hero_suit_ry: tuple = (0.35, 0.70)
    face_hero_suit_rx: tuple = (0.00, 0.50)
    face_board_suit_ry: tuple = (0.35, 0.70)
    face_board_suit_rx: tuple = (0.00, 0.40)
    mushroom_roi: tuple = (0.130, 0.115, 0.200, 0.165)
    seat_stack_rois: dict = field(default_factory=lambda: {
        2: (0.020, 0.630, 0.145, 0.658),
        3: (0.020, 0.498, 0.145, 0.526),
        4: (0.020, 0.366, 0.145, 0.394),
        5: (0.435, 0.210, 0.565, 0.238),
        6: (0.855, 0.366, 0.980, 0.394),
        7: (0.855, 0.498, 0.980, 0.526),
        8: (0.855, 0.630, 0.980, 0.658),
    })
    seat_positions: dict = field(default_factory=lambda: {
        **LayoutPoler().seat_positions,
        5: (0.50, 0.18),
    })
    button_positions: dict = field(default_factory=lambda: {
        **LayoutPoler().button_positions,
        1: [(0.32, 0.87), (0.60, 0.87), (0.61, 0.80), (0.65, 0.78)],
    })

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
        4: (0.120, 0.270, 0.280, 0.380),
        5: (0.500, 0.210, 0.570, 0.285),
        6: (0.710, 0.270, 0.810, 0.370),
        7: (0.710, 0.400, 0.810, 0.500),
        8: (0.710, 0.575, 0.810, 0.645),
    })
    seat_bet_fallback_chip_colors: dict = field(default_factory=lambda: {
        # V2 uses both red and blue marker-card themes. Colour narrows the
        # candidate region; white digit geometry and OCR decide the value.
        seat: ("red", "blue") for seat in range(2, 9)
    })


def nine_seat_layout() -> LayoutPolerV2:
    """Hero=1, then left-bottom to top-left/top-right and right-bottom=9."""
    base = LayoutPolerV2()

    def remap(values):
        return {(seat + 1 if seat >= 6 else seat): value
                for seat, value in values.items() if seat != 5}

    positions = remap(base.seat_positions)
    positions.update({5: (.33, .19), 6: (.67, .19)})
    anchors = remap(dict(base.seat_anchors))
    anchors.update({5: Anchor(.33, .23, .04), 6: Anchor(.67, .23, .04)})
    status = remap(base.seat_status_anchors)
    status.update({5: Anchor(.33, .19, .09), 6: Anchor(.67, .19, .09)})
    stacks = remap(base.seat_stack_rois)
    stacks.update({5: (.265, .216, .395, .244), 6: (.605, .216, .735, .244)})
    bets = remap(base.seat_bet_anchors)
    bets.update({5: Anchor(.37, .26, .04), 6: Anchor(.63, .26, .04)})
    bet_rois = remap(base.seat_bet_fallback_rois)
    bet_rois.update({5: (.31, .225, .42, .295), 6: (.58, .225, .69, .295)})
    buttons = remap(base.button_positions)
    buttons.update({5: [(.40, .20)], 6: [(.60, .20)]})
    return replace(base, seat_positions=positions, seat_anchors=tuple(sorted(anchors.items())),
                   seat_status_anchors=status, seat_stack_rois=stacks,
                   seat_bet_anchors=bets, seat_bet_fallback_rois=bet_rois,
                   button_positions=buttons,
                   seat_bet_fallback_chip_colors={seat: ("red", "blue") for seat in range(2, 10)})
