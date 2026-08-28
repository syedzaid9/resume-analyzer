"""UI and Visualization Helpers for ResumeAI.
Generates Plotly gauges, breakdown charts, and styled HTML badge elements.
"""

from typing import Dict, List, Any
import plotly.graph_objects as go
from src.config import UI_THEME, IMPORTANCE_LEVELS


def create_score_gauge(score: int, title: str = "AI Resume Fit Score", color: str = "#2563EB") -> go.Figure:
    """Generates a modern circular Plotly gauge indicator."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"size": 18, "family": UI_THEME["font_family"], "color": UI_THEME["text_primary"]}},
            number={"suffix": "/100", "font": {"size": 36, "family": UI_THEME["font_family"], "color": UI_THEME["text_primary"], "weight": "bold"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "#F1F5F9",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 60], "color": "#FEE2E2"},
                    {"range": [60, 75], "color": "#FEF3C7"},
                    {"range": [75, 90], "color": "#EDE9FE"},
                    {"range": [90, 100], "color": "#DCFCE7"},
                ],
            },
        )
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": UI_THEME["font_family"]},
    )
    return fig


def create_breakdown_bar(breakdown: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Renders a clean horizontal bar chart displaying earned vs max points per component."""
    categories = []
    earned_pts = []
    max_pts = []

    names_map = {
        "skill_match": "Skill Match",
        "experience": "Experience",
        "projects": "Projects",
        "education": "Education",
        "ats_compatibility": "ATS Compliance",
        "structure": "Structure",
        "formatting": "Formatting",
    }

    for key, data in breakdown.items():
        categories.append(names_map.get(key, key.title()))
        earned_pts.append(data["earned"])
        max_pts.append(data["max"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=categories,
            x=earned_pts,
            name="Earned Score",
            orientation="h",
            marker=dict(color=UI_THEME["primary"], line=dict(width=0)),
            text=[f"{e:.1f}/{m} pts" for e, m in zip(earned_pts, max_pts)],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="#FFFFFF", family=UI_THEME["font_family"], size=12),
        )
    )

    fig.update_layout(
        barmode="stack",
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 35], showgrid=True, gridcolor="#F1F5F9", title="Points"),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        font={"family": UI_THEME["font_family"]},
    )
    return fig


def render_skill_badge(skill_name: str, importance: str = "Medium", matched: bool = True) -> str:
    """Returns styled HTML badge string for matched or missing skills."""
    if matched:
        bg_color = "#DCFCE7"  # Green tint
        text_color = "#15803D"
        border_color = "#86EFAC"
        icon = "✓"
    else:
        imp_color = IMPORTANCE_LEVELS.get(importance, {}).get("badge_color", "#DC2626")
        if importance == "Critical":
            bg_color = "#FEE2E2"
            text_color = "#B91C1C"
            border_color = "#FCA5A5"
        elif importance == "High":
            bg_color = "#FEF3C7"
            text_color = "#B45309"
            border_color = "#FCD34D"
        else:
            bg_color = "#F1F5F9"
            text_color = "#475569"
            border_color = "#CBD5E1"
        icon = "✗"

    return f"""<span style="
        display: inline-block;
        background-color: {bg_color};
        color: {text_color};
        border: 1px solid {border_color};
        padding: 4px 10px;
        margin: 3px 4px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
        font-family: {UI_THEME['font_family']};
    ">{icon} {skill_name} <small style="opacity: 0.8; font-size: 0.75rem;">({importance})</small></span>"""
