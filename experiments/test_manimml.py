"""
Purpose: Neural network animation — rebuilt from channel analysis (3B1B/ScienceClic/EfficientEngineer style)
Inputs: None
Outputs: media/videos/test_manimml/480p15/NeuralNetworkTest.mp4
How to run: manim -pql test_manimml.py NeuralNetworkTest
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

from manim import *
from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer

# ── Palette (3B1B-inspired: near-black BG, high contrast) ────────
BG      = "#111111"      # near black — max contrast (3B1B style)
WHITE   = "#F5F5F5"
DIM     = "#555577"      # subtle secondary
CYAN    = "#5BA4F5"      # primary accent (DINOv2 blue)
YELLOW  = "#FFDD44"      # highlight / active
ORANGE  = "#FF9944"
GREEN   = "#44CC88"      # SAFE
RED     = "#EE4444"      # HAZARD
AMBER   = "#FFAA22"      # CAUTION
PURPLE  = "#9966DD"      # STOP

# Terrain colors (semantic, consistent)
TC = {"Soil": "#E67E22", "Bedrock": "#95A5A6", "Sand": "#F1C40F", "Big Rock": "#E74C3C"}
VC = {"Soil": GREEN,     "Bedrock": RED,       "Sand": AMBER,    "Big Rock": PURPLE}
VL = {"Soil": "SAFE",    "Bedrock": "HAZARD",  "Sand": "CAUTION","Big Rock": "STOP"}


def glow_dot(pos, col=YELLOW, r=0.16):
    """Small glowing dot: bright center + soft halo."""
    center = Dot(radius=r, color=col).move_to(pos)
    halo   = Dot(radius=r * 2.6, color=col, fill_opacity=0.18).move_to(pos)
    return VGroup(halo, center)


class NeuralNetworkTest(Scene):
    """
    Layout (lessons applied):
      • Near-black background → maximum contrast
      • One concept at a time → breathe between reveals
      • Network fills center → star of the show
      • Labels float next to their object → spatial memory
      • Output = name only, large font, colour-coded
      • run_time ≥ 0.6 s for every visible move
    """

    def construct(self):
        self.camera.background_color = BG

        # ════════════════════════════════════════════════════════
        # TITLE — minimal, fades after serving purpose
        # ════════════════════════════════════════════════════════
        title = Text("DINOv2  Linear Probe",
                     font_size=40, color=WHITE, weight=BOLD)
        sub = Text("frozen encoder  ·  linear classifier  ·  terrain class",
                   font_size=16, color=DIM)
        sub.next_to(title, DOWN, buff=0.2)
        VGroup(title, sub).to_edge(UP, buff=0.44)

        self.play(Write(title), run_time=1.2, rate_func=smooth)
        self.play(FadeIn(sub, shift=UP * 0.1), run_time=0.6)
        self.wait(0.5)

        # ════════════════════════════════════════════════════════
        # NEURAL NETWORK — large, centered, given room to breathe
        # ════════════════════════════════════════════════════════
        nn = NeuralNetwork([
            FeedForwardLayer(7),
            FeedForwardLayer(5),
            FeedForwardLayer(4),
        ])
        # Grab layer refs BEFORE any transform
        layers = nn.submobjects[1].submobjects   # [0] FF [1] conn [2] FF [3] conn [4] FF
        nn.scale(1.55)
        nn.move_to(LEFT * 0.5 + DOWN * 0.2)     # center-left, room for output right

        # Float in from slight offset — ScienceClic style entry
        nn.shift(UP * 0.4)
        self.play(FadeIn(nn, shift=DOWN * 0.4), run_time=1.5, rate_func=smooth)
        self.wait(0.5)

        # ── Layer labels float below their layer (3B1B: text near object) ──
        lbl_specs = [
            (layers[0], "Input\n384-d", CYAN),
            (layers[2], "Hidden\n128",  DIM),
            (layers[4], "Output\n4",    CYAN),
        ]
        lbls = VGroup()
        for layer, txt, col in lbl_specs:
            l = Text(txt, font_size=13, color=col, line_spacing=1.2)
            l.next_to(layer, DOWN, buff=0.28)
            lbls.add(l)

        self.play(
            LaggedStart(*[FadeIn(l, shift=UP * 0.1) for l in lbls], lag_ratio=0.35),
            run_time=0.9
        )
        self.wait(0.4)

        # ── Input chip — small, slides in from left ──────────────
        chip_box = RoundedRectangle(
            width=1.3, height=0.85, corner_radius=0.13,
            color=CYAN, fill_color="#0A1520", fill_opacity=1.0,
            stroke_width=1.5
        )
        chip_lbl = VGroup(
            Text("CLS", font_size=19, color=CYAN, weight=BOLD),
            Text("token", font_size=11, color=DIM),
        ).arrange(DOWN, buff=0.05)
        chip_lbl.move_to(chip_box)
        chip = VGroup(chip_box, chip_lbl)
        chip.next_to(layers[0], LEFT, buff=1.0)

        arrow_in = Arrow(
            chip.get_right(), layers[0].get_left(),
            buff=0.14, stroke_width=2.0, color=CYAN,
            max_tip_length_to_length_ratio=0.18,
        )

        self.play(FadeIn(chip, shift=RIGHT * 0.2), run_time=0.7, rate_func=smooth)
        self.play(GrowArrow(arrow_in), run_time=0.7, rate_func=smooth)
        self.wait(0.4)

        # ════════════════════════════════════════════════════════
        # FORWARD PASS — glowing dot travels left → right
        # Lesson: make it slow & dramatic, like ScienceClic
        # ════════════════════════════════════════════════════════
        step_txt = Text("Forward pass →", font_size=15, color=DIM)
        step_txt.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(step_txt), run_time=0.5)

        pulse = glow_dot(layers[0].get_center(), col=CYAN)
        self.play(FadeIn(pulse, scale=0.4), run_time=0.5, rate_func=smooth)

        # Pulse travels through: input → connector → hidden → connector → output
        waypoints = [
            (layers[1].get_center(), YELLOW),   # connector
            (layers[2].get_center(), YELLOW),   # hidden layer
            (layers[3].get_center(), AMBER),    # connector
            (layers[4].get_center(), GREEN),    # output layer
        ]
        for pos, col in waypoints:
            self.play(
                pulse.animate.move_to(pos).set_color(col),
                run_time=0.6, rate_func=smooth
            )
            self.wait(0.12)

        self.play(FadeOut(pulse), FadeOut(step_txt), run_time=0.55, rate_func=smooth)
        self.wait(0.3)

        # ════════════════════════════════════════════════════════
        # OUTPUT — right side, clean: large name, small tag
        # Lesson (Efficient Engineer): remove all clutter
        # ════════════════════════════════════════════════════════
        class_order = ["Soil", "Bedrock", "Sand", "Big Rock"]

        out_cards = VGroup()
        for name in class_order:
            t_col = TC[name]
            v_col = VC[name]
            label = VL[name]

            name_txt = Text(name, font_size=22, color=t_col, weight=BOLD)
            trav_txt = Text(label, font_size=13, color=v_col)
            trav_txt.next_to(name_txt, RIGHT, buff=0.22)

            card = VGroup(name_txt, trav_txt)
            # thin underline in terrain color
            line = Line(card.get_left(), card.get_right(),
                        color=t_col, stroke_width=1.5, stroke_opacity=0.4)
            line.next_to(card, DOWN, buff=0.08)
            out_cards.add(VGroup(card, line))

        out_cards.arrange(DOWN, buff=0.38)
        out_cards.move_to(RIGHT * 4.4 + DOWN * 0.2)

        # thin arrows from output layer → cards
        arrows_out = VGroup(*[
            Arrow(
                layers[4].get_right(), c.get_left(),
                buff=0.15, stroke_width=1.4, color=DIM,
                max_tip_length_to_length_ratio=0.12,
            )
            for c in out_cards
        ])

        # Fan out arrows first
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows_out], lag_ratio=0.14),
            run_time=0.85, rate_func=smooth
        )
        # Cards slide in from right
        self.play(
            LaggedStart(
                *[FadeIn(c, shift=LEFT * 0.2) for c in out_cards],
                lag_ratio=0.2
            ),
            run_time=1.0, rate_func=smooth
        )
        self.wait(0.5)

        # ════════════════════════════════════════════════════════
        # CLIMAX — Bedrock highlighted, everything else dims
        # Lesson (3B1B): let the key element be the only thing
        # ════════════════════════════════════════════════════════
        others = VGroup(
            out_cards[0], out_cards[2], out_cards[3],
            arrows_out[0], arrows_out[2], arrows_out[3]
        )

        highlight_line = Line(
            out_cards[1].get_left() + LEFT * 0.1,
            out_cards[1].get_right() + RIGHT * 0.1,
            color=WHITE, stroke_width=1.5
        ).next_to(out_cards[1], DOWN, buff=0.05)

        self.play(
            others.animate.set_opacity(0.15),
            run_time=0.7, rate_func=smooth
        )
        self.play(
            out_cards[1].animate.scale(1.08),
            Create(highlight_line),
            run_time=0.55, rate_func=smooth
        )
        self.wait(0.3)

        # ── Result banner — slides up, simple & bold ─────────────
        result = Text(
            "Bedrock  →  HAZARD  ·  slow to 0.03 m/s",
            font_size=20, color=WHITE
        )
        result_line = Line(LEFT * 3.5, RIGHT * 3.5, color=RED, stroke_width=1.5)
        result_line.next_to(result, DOWN, buff=0.12)
        banner = VGroup(result, result_line)
        banner.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(banner, shift=UP * 0.15), run_time=0.75, rate_func=smooth)
        self.wait(3.0)
