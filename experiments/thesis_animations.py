"""
Purpose: Thesis presentation animations for Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
Inputs: None (uses hardcoded thesis results)
Outputs: media/videos/thesis_animations/480p15/<SceneName>.mp4
How to run:
  manim -pql thesis_animations.py DINOv2Pipeline
  manim -pql thesis_animations.py AccuracyComparison
  manim -pql thesis_animations.py TraversabilityPolicy
  manim -pqh thesis_animations.py DINOv2Pipeline   # high quality
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

from manim import *


# ── shared palette ────────────────────────────────────────────────
MARS_ORANGE  = "#C1440E"
DINO_BLUE    = "#4A90D9"
SAFE_GREEN   = "#27AE60"
CAUTION_AMB  = "#F39C12"
HAZARD_RED   = "#E74C3C"
STOP_DARK    = "#8E44AD"
BG_DARK      = "#1A1A2E"


# ══════════════════════════════════════════════════════════════════
# Scene 1 – DINOv2 Pipeline
# ══════════════════════════════════════════════════════════════════
class DINOv2Pipeline(Scene):
    """Visualise: Mars image → patch tokens → DINOv2 encoder → linear probe → terrain class."""

    def construct(self):
        self.camera.background_color = BG_DARK

        # ── title ─────────────────────────────────────────────────
        title = Text("DINOv2 Terrain Classification Pipeline",
                     font_size=32, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=1.2)
        self.wait(0.3)

        # ── Step 1: Mars image ────────────────────────────────────
        img_box = Rectangle(width=2.0, height=2.0, color=MARS_ORANGE,
                            fill_color=MARS_ORANGE, fill_opacity=0.3)
        img_label = Text("Mars Image\n224×224 px", font_size=16, color=WHITE)
        img_label.move_to(img_box)
        img_group = VGroup(img_box, img_label)
        img_group.move_to(LEFT * 5)

        # ── Step 2: patch grid ────────────────────────────────────
        patch_grid = VGroup()
        for r in range(4):
            for c in range(4):
                sq = Square(side_length=0.38, color=DINO_BLUE,
                            fill_color=DINO_BLUE, fill_opacity=0.25,
                            stroke_width=1.5)
                sq.move_to(LEFT * 2.5 + RIGHT * c * 0.42 + DOWN * r * 0.42)
                patch_grid.add(sq)
        patch_grid.move_to(LEFT * 2.5)
        patch_title = Text("16×16 Patches\n(14×14 px each)", font_size=14, color=DINO_BLUE)
        patch_title.next_to(patch_grid, DOWN, buff=0.2)

        # ── Step 3: transformer block ─────────────────────────────
        enc_box = RoundedRectangle(width=2.0, height=2.4, corner_radius=0.15,
                                   color=DINO_BLUE, fill_color="#1C3A5E",
                                   fill_opacity=0.9)
        enc_label = VGroup(
            Text("DINOv2", font_size=18, color=DINO_BLUE, weight=BOLD),
            Text("ViT-S/14", font_size=15, color=WHITE),
            Text("Transformer", font_size=13, color=GRAY),
            Text("Encoder", font_size=13, color=GRAY),
        ).arrange(DOWN, buff=0.1)
        enc_label.move_to(enc_box)
        enc_group = VGroup(enc_box, enc_label)
        enc_group.move_to(ORIGIN)

        # ── Step 4: CLS token ─────────────────────────────────────
        cls_box = Circle(radius=0.5, color=YELLOW,
                         fill_color="#3D3000", fill_opacity=0.9)
        cls_label = Text("[CLS]\n384-d", font_size=14, color=YELLOW)
        cls_label.move_to(cls_box)
        cls_group = VGroup(cls_box, cls_label)
        cls_group.move_to(RIGHT * 2.8)

        # ── Step 5: linear probe ──────────────────────────────────
        probe_box = RoundedRectangle(width=1.6, height=1.0, corner_radius=0.1,
                                     color=SAFE_GREEN, fill_color="#0D2B1A",
                                     fill_opacity=0.9)
        probe_label = Text("Linear\nProbe", font_size=15, color=SAFE_GREEN)
        probe_label.move_to(probe_box)
        probe_group = VGroup(probe_box, probe_label)
        probe_group.move_to(RIGHT * 4.8)

        # ── Step 6: output classes ────────────────────────────────
        class_colors = [ORANGE, GRAY_C, YELLOW, RED]
        class_names  = ["Soil", "Bedrock", "Sand", "Big Rock"]
        class_boxes  = VGroup()
        for i, (name, col) in enumerate(zip(class_names, class_colors)):
            b = RoundedRectangle(width=1.1, height=0.45, corner_radius=0.08,
                                 color=col, fill_color=col, fill_opacity=0.7)
            t = Text(name, font_size=13, color=WHITE)
            t.move_to(b)
            g = VGroup(b, t)
            g.move_to(RIGHT * 6.5 + UP * (1.5 - i * 1.0))
            class_boxes.add(g)

        # ── arrows ────────────────────────────────────────────────
        a1 = Arrow(img_group.get_right(), patch_grid.get_left(),   buff=0.15, color=GRAY_B)
        a2 = Arrow(patch_grid.get_right(), enc_group.get_left(),   buff=0.15, color=GRAY_B)
        a3 = Arrow(enc_group.get_right(), cls_group.get_left(),    buff=0.15, color=GRAY_B)
        a4 = Arrow(cls_group.get_right(), probe_group.get_left(),  buff=0.15, color=GRAY_B)
        a5 = Arrow(probe_group.get_right(), class_boxes.get_left(), buff=0.15, color=GRAY_B)

        # ── animate ───────────────────────────────────────────────
        self.play(FadeIn(img_group, scale=0.8))
        self.play(GrowArrow(a1))
        self.play(LaggedStart(*[FadeIn(sq, scale=0.5) for sq in patch_grid],
                               lag_ratio=0.03), FadeIn(patch_title))
        self.play(GrowArrow(a2))
        self.play(FadeIn(enc_group, scale=0.8))

        # pulse the encoder
        self.play(enc_box.animate.set_fill(DINO_BLUE, opacity=0.5), run_time=0.4)
        self.play(enc_box.animate.set_fill("#1C3A5E", opacity=0.9), run_time=0.4)

        self.play(GrowArrow(a3))
        self.play(FadeIn(cls_group, scale=0.8))
        self.play(GrowArrow(a4))
        self.play(FadeIn(probe_group, scale=0.8))
        self.play(GrowArrow(a5))
        self.play(LaggedStart(*[FadeIn(b, scale=0.8) for b in class_boxes], lag_ratio=0.15))

        # highlight "Bedrock" as example output
        highlight = SurroundingRectangle(class_boxes[1], color=YELLOW, buff=0.05)
        label_out = Text("Predicted: Bedrock", font_size=18, color=YELLOW)
        label_out.to_edge(DOWN, buff=0.4)
        self.play(Create(highlight), Write(label_out))
        self.wait(2)

        # frozen badge
        frozen = Text("Frozen weights — no fine-tuning", font_size=14, color=GRAY_B)
        frozen.next_to(enc_group, UP, buff=0.15)
        self.play(FadeIn(frozen))
        self.wait(2)


# ══════════════════════════════════════════════════════════════════
# Scene 2 – Accuracy Comparison (paradigm bar chart)
# ══════════════════════════════════════════════════════════════════
class AccuracyComparison(Scene):
    """Animated bar chart: 6 paradigms + supervised ceiling."""

    def construct(self):
        self.camera.background_color = BG_DARK

        title = Text("Foundation Model Accuracy — AI4Mars Test Set",
                     font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # data (paradigm, best accuracy, colour)
        data = [
            ("Depth\nAny. V2", 36.6,  "#5D6D7E"),
            ("RADIO-B",        65.2,  "#884EA0"),
            ("EVA-02",         81.2,  "#2E86C1"),
            ("CLIP\nSigLIP2",  87.8,  "#1ABC9C"),
            ("AIMv2\nFranca",  89.6,  "#F39C12"),
            ("DINO\nEnsemble", 94.4,  DINO_BLUE),
            ("Supervised\nCeiling", 96.7, SAFE_GREEN),
        ]

        BAR_W    = 0.85
        SPACING  = 1.15
        MAX_H    = 4.2
        MAX_ACC  = 100.0
        BASELINE = DOWN * 2.6

        bars   = VGroup()
        labels = VGroup()
        vals   = VGroup()

        for i, (name, acc, col) in enumerate(data):
            h = acc / MAX_ACC * MAX_H
            bar = Rectangle(width=BAR_W, height=h,
                            color=col, fill_color=col, fill_opacity=0.85,
                            stroke_width=1.5)
            bar.move_to(BASELINE + UP * h / 2 + RIGHT * (i - 3) * SPACING)

            lbl = Text(name, font_size=13, color=WHITE)
            lbl.next_to(bar, DOWN, buff=0.15)

            pct = Text(f"{acc}%", font_size=14, color=col, weight=BOLD)
            pct.next_to(bar, UP, buff=0.08)

            bars.add(bar)
            labels.add(lbl)
            vals.add(pct)

        # axis line
        axis = Line(LEFT * 4.2, RIGHT * 4.2, color=GRAY_B, stroke_width=1.5)
        axis.move_to(BASELINE)

        # 90% reference line
        ref_y   = BASELINE + UP * (90 / MAX_ACC * MAX_H)
        ref_line = DashedLine(LEFT * 4.0, RIGHT * 4.0, color=GRAY_C,
                               stroke_width=1, dash_length=0.12)
        ref_line.move_to(ref_y)
        ref_txt = Text("90%", font_size=13, color=GRAY_C)
        ref_txt.next_to(ref_line, LEFT, buff=0.1)

        self.play(Create(axis))
        self.play(Create(ref_line), FadeIn(ref_txt))
        self.play(
            LaggedStart(
                *[GrowFromEdge(b, DOWN) for b in bars],
                lag_ratio=0.12
            )
        )
        self.play(
            LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.08),
            LaggedStart(*[FadeIn(v) for v in vals],   lag_ratio=0.08),
        )
        self.wait(0.5)

        # highlight DINO cluster
        dino_rect = SurroundingRectangle(
            VGroup(bars[5], vals[5], labels[5]),
            color=DINO_BLUE, buff=0.12, corner_radius=0.08
        )
        dino_note = Text("DINO uniquely\n89.9–94.4%", font_size=15, color=DINO_BLUE)
        dino_note.next_to(dino_rect, UP, buff=0.15)
        self.play(Create(dino_rect), Write(dino_note))

        # deployed model callout
        deploy_arrow = Arrow(
            bars[4].get_top() + UP * 0.5 + LEFT * 0.3,
            bars[4].get_top() + UP * 0.08,
            color=CAUTION_AMB, buff=0.0
        )
        deploy_txt = Text("Deployed on\nRPi4 (570ms)", font_size=13, color=CAUTION_AMB)
        deploy_txt.next_to(deploy_arrow.get_start(), UP, buff=0.05)
        self.play(GrowArrow(deploy_arrow), FadeIn(deploy_txt))

        self.wait(2.5)


# ══════════════════════════════════════════════════════════════════
# Scene 3 – Traversability Policy
# ══════════════════════════════════════════════════════════════════
class TraversabilityPolicy(Scene):
    """Terrain class → traversability label → speed limit decision."""

    def construct(self):
        self.camera.background_color = BG_DARK

        title = Text("Traversability Policy", font_size=34, color=WHITE)
        subtitle = Text("Terrain Classification → Rover Speed Decision",
                        font_size=20, color=GRAY_B)
        subtitle.next_to(title, DOWN, buff=0.2)
        self.play(Write(title), FadeIn(subtitle))
        self.wait(0.8)
        self.play(FadeOut(title), FadeOut(subtitle))

        # table rows: (terrain, icon, traversability, speed, terrain_col, trav_col)
        rows = [
            ("Soil",     "🪨", "SAFE",    "0.10 m/s",  ORANGE,    SAFE_GREEN),
            ("Sand",     "🏜️", "CAUTION", "0.05 m/s",  YELLOW,    CAUTION_AMB),
            ("Bedrock",  "🗿", "HAZARD",  "0.03 m/s",  GRAY_C,    HAZARD_RED),
            ("Big Rock", "🪨", "STOP",    "0.00 m/s",  RED,       STOP_DARK),
        ]

        COL_X = [-4.5, -1.8, 1.2, 4.2]   # x positions for 4 columns
        HDR_Y = 2.5

        # column headers
        headers = ["Terrain Class", "Classification", "Traversability", "Speed Limit"]
        hdr_group = VGroup()
        for hx, htxt in zip(COL_X, headers):
            h = Text(htxt, font_size=18, color=GRAY_A, weight=BOLD)
            h.move_to(RIGHT * hx + UP * HDR_Y)
            hdr_group.add(h)

        divider = Line(LEFT * 6, RIGHT * 6, color=GRAY_C, stroke_width=1)
        divider.move_to(UP * (HDR_Y - 0.4))

        self.play(LaggedStart(*[FadeIn(h) for h in hdr_group], lag_ratio=0.1))
        self.play(Create(divider))

        for row_i, (terrain, icon, trav, speed, t_col, v_col) in enumerate(rows):
            y = HDR_Y - 1.2 - row_i * 1.0

            terrain_box = RoundedRectangle(width=2.0, height=0.7, corner_radius=0.08,
                                           color=t_col, fill_color=t_col,
                                           fill_opacity=0.25)
            terrain_box.move_to(RIGHT * COL_X[0] + UP * y)
            terrain_txt = Text(terrain, font_size=18, color=t_col, weight=BOLD)
            terrain_txt.move_to(terrain_box)

            class_txt = Text("✓ Correct", font_size=16, color=SAFE_GREEN)
            class_txt.move_to(RIGHT * COL_X[1] + UP * y)

            trav_box = RoundedRectangle(width=1.8, height=0.65, corner_radius=0.08,
                                        color=v_col, fill_color=v_col,
                                        fill_opacity=0.3)
            trav_box.move_to(RIGHT * COL_X[2] + UP * y)
            trav_txt = Text(trav, font_size=16, color=v_col, weight=BOLD)
            trav_txt.move_to(trav_box)

            speed_txt = Text(speed, font_size=17, color=WHITE)
            speed_txt.move_to(RIGHT * COL_X[3] + UP * y)

            self.play(
                FadeIn(VGroup(terrain_box, terrain_txt)),
                run_time=0.4
            )
            self.play(
                Write(class_txt),
                FadeIn(VGroup(trav_box, trav_txt)),
                FadeIn(speed_txt),
                run_time=0.5
            )

        # summary banner
        self.wait(0.5)
        banner = RoundedRectangle(width=9.0, height=0.75, corner_radius=0.1,
                                  color=SAFE_GREEN, fill_color=SAFE_GREEN,
                                  fill_opacity=0.15)
        banner.to_edge(DOWN, buff=0.35)
        banner_txt = Text(
            "Gazebo Experiment: 3/5 terrain correct (60%)  |  5/5 traversability correct (100%)",
            font_size=17, color=SAFE_GREEN
        )
        banner_txt.move_to(banner)
        self.play(FadeIn(banner), Write(banner_txt))
        self.wait(3)
