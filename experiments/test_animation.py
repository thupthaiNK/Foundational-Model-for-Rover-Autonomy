"""
Purpose: Test Manim installation with a simple thesis-themed animation
Inputs: None
Outputs: media/videos/test_animation/480p15/TestAnimation.mp4
How to run: manim -pql test_animation.py TestAnimation
Project: Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation
"""

from manim import *

class TestAnimation(Scene):
    def construct(self):
        # Title
        title = Text("Onboard Visual Foundation Models for Mars Terrain Perception and Rover Navigation", font_size=36, color=BLUE)
        subtitle = Text("DINOv2 Terrain Classification", font_size=24, color=GRAY)
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle))
        self.wait(1)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Terrain classes as colored boxes
        classes = ["Soil", "Bedrock", "Sand", "Big Rock"]
        colors  = [ORANGE, GRAY, YELLOW, RED]

        boxes = VGroup(*[
            VGroup(
                RoundedRectangle(width=2.2, height=1.0, corner_radius=0.1, color=c, fill_opacity=0.8),
                Text(label, font_size=22, color=WHITE)
            ).arrange(ORIGIN)
            for label, c in zip(classes, colors)
        ]).arrange(RIGHT, buff=0.4)

        label = Text("AI4Mars Terrain Classes", font_size=28, color=WHITE)
        label.to_edge(UP)

        self.play(Write(label))
        self.play(LaggedStart(*[FadeIn(b, scale=0.8) for b in boxes], lag_ratio=0.2))
        self.wait(1)

        # Simple accuracy bar chart
        self.play(FadeOut(label), FadeOut(boxes))

        chart_title = Text("5-Zone Gazebo Experiment", font_size=28, color=WHITE)
        chart_title.to_edge(UP)
        self.play(Write(chart_title))

        bar_data   = [1, 1, 0, 1, 0]   # 3/5 correct
        bar_labels = ["Zone 1", "Zone 2", "Zone 3", "Zone 4", "Zone 5"]
        bar_colors = [GREEN if v else RED for v in bar_data]

        bars = VGroup()
        for i, (val, lbl, col) in enumerate(zip(bar_data, bar_labels, bar_colors)):
            bar  = Rectangle(width=0.8, height=val * 2.0, color=col, fill_opacity=0.9)
            text = Text(lbl, font_size=16).next_to(bar, DOWN, buff=0.15)
            mark = Text("✓" if val else "✗", font_size=20, color=col).next_to(bar, UP, buff=0.1)
            group = VGroup(bar, text, mark).shift(RIGHT * (i - 2) * 1.3)
            bars.add(group)

        bars.shift(DOWN * 0.5)
        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars], lag_ratio=0.15))
        self.play(LaggedStart(*[FadeIn(b[1]) for b in bars], lag_ratio=0.1))
        self.play(LaggedStart(*[FadeIn(b[2]) for b in bars], lag_ratio=0.1))

        result = Text("Result: 3/5 zones correctly classified", font_size=22, color=YELLOW)
        result.to_edge(DOWN)
        self.play(Write(result))
        self.wait(2)
