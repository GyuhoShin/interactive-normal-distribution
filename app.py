# app.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from math import erf, sqrt, pi


# ---------- Standard Normal ----------
def norm_pdf(x):
    return (1.0 / sqrt(2*pi)) * np.exp(-0.5 * x**2)

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2)))

def area_one_side(mode, a):
    return (1.0 - norm_cdf(a)) if mode == "gt" else norm_cdf(a)

def area_two_side(mode, a, b):
    lo, hi = (a, b) if a <= b else (b, a)
    inside = norm_cdf(hi) - norm_cdf(lo)
    return inside if mode == "inside" else (1.0 - inside)

# ---------- Figure ----------

def make_figure(kind, mode, a, b):
    x = np.linspace(-4, 4, 1200)
    y = norm_pdf(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Normal PDF"))

    if kind == "one":
        mask = (x >= a) if mode == "gt" else (x <= a)
        xs, ys = x[mask], y[mask]
        fig.add_trace(go.Scatter(
            x=np.r_[xs[0], xs, xs[-1]],
            y=np.r_[0, ys, 0],
            fill="toself",
            mode="lines",
            line=dict(width=0),
            name="Shaded"
        ))
        fig.add_vline(x=a, line_width=2, line_dash="dash")
    else:
        lo, hi = (a, b) if a <= b else (b, a)
        if mode == "inside":
            mask = (x >= lo) & (x <= hi)
            xs, ys = x[mask], y[mask]
            fig.add_trace(go.Scatter(
                x=np.r_[xs[0], xs, xs[-1]],
                y=np.r_[0, ys, 0],
                fill="toself",
                mode="lines",
                line=dict(width=0),
                name="Shaded"
            ))
        else:
            for mask in (x <= lo, x >= hi):
                xs, ys = x[mask], y[mask]
                fig.add_trace(go.Scatter(
                    x=np.r_[xs[0], xs, xs[-1]],
                    y=np.r_[0, ys, 0],
                    fill="toself",
                    mode="lines",
                    line=dict(width=0),
                    name="Shaded"
                ))
        fig.add_vline(x=lo, line_width=2, line_dash="dash")
        fig.add_vline(x=hi, line_width=2, line_dash="dash")

    fig.update_layout(
        title="Interactive Normal Probability (Standard Normal)",
        xaxis_title="z",
        yaxis_title="density",
        template="plotly_white",
        showlegend=False,
        margin=dict(t=60),
        width=800,
        height=400,
    )
    fig.update_xaxes(range=[-4, 4])
    fig.update_yaxes(range=[0, 0.45], visible=False)
    return fig

# ---------- Helpers ----------
def clip4(x: float) -> float:
    return float(np.clip(x, -4.0, 4.0))

# callbacks: slider -> inputs
def on_a_slider_change():
    st.session_state["a"] = clip4(st.session_state["a_slider"])

def on_ab_slider_change():
    lo, hi = st.session_state["ab_slider"]
    st.session_state["a"] = clip4(lo)
    st.session_state["b"] = clip4(hi)

# callbacks: inputs -> slider
def on_a_input_change():
    st.session_state["a"] = clip4(st.session_state["a"])
    st.session_state["a_slider"] = st.session_state["a"]

def on_ab_input_change():
    st.session_state["a"] = clip4(st.session_state["a"])
    st.session_state["b"] = clip4(st.session_state["b"])
    lo, hi = (st.session_state["a"], st.session_state["b"])
    if lo > hi:
        lo, hi = hi, lo
        st.session_state["a"], st.session_state["b"] = lo, hi
    st.session_state["ab_slider"] = (lo, hi)

# ---------- Init state ----------
if "a" not in st.session_state:
    st.session_state["a"] = 1.0
if "b" not in st.session_state:
    st.session_state["b"] = 2.0
if "a_slider" not in st.session_state:
    st.session_state["a_slider"] = st.session_state["a"]
if "ab_slider" not in st.session_state:
    lo, hi = (st.session_state["a"], st.session_state["b"])
    if lo > hi:
        lo, hi = hi, lo
    st.session_state["ab_slider"] = (lo, hi)

# ---------- UI ----------
st.set_page_config(page_title="Normal Probability", layout="centered")
st.markdown(
    """
    <style>
    /* Margin Below Plotly chart */
    div[data-testid="stPlotlyChart"] {
        margin-bottom: 0rem;
    }

    /* Move slider up */
    div[data-testid="stSlider"] {
        margin-top: -3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Normal Distribution Probability")

col1, col2 = st.columns([1, 3])
with col1:
    st.write("**Test type**")
with col2:
    kind = st.selectbox("Test type", ["One-sided", "Two-sided"], label_visibility="collapsed")

if kind == "One-sided":
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("**Mode**")
    with col2:
        mode_label = st.selectbox("Mode", ["Larger than  P(X > a)", "Lower than  P(X < a)"], label_visibility="collapsed")
    mode = "gt" if ">" in mode_label else "lt"

    # number_input (a)  -> slider sync
    st.number_input(
        "a",
        key="a",
        step=0.01,
        format="%.2f",
        on_change=on_a_input_change
    )

    a = clip4(st.session_state["a"])
    p = area_one_side(mode, a)
    fig = make_figure("one", mode, a, 0.0)
    st.plotly_chart(fig, use_container_width=True)

    st.slider(
        "",
        min_value=-4.0,
        max_value=4.0,
        key="a_slider",
        step=0.01,
        on_change=on_a_slider_change
    )

    st.subheader(f"Probability = {p:.4f}")

else:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("**Mode**")
    with col2:
        mode_label = st.selectbox("Mode", ["Inside  P(a < X < b)", "Outside  P(X < a or X > b)"], label_visibility="collapsed")
    mode = "inside" if "Inside" in mode_label else "outside"

    c1, c2 = st.columns(2)
    with c1:
        st.number_input(
            "a",
            key="a",
            value=float(st.session_state["a"]),
            step=0.01,
            format="%.2f",
            on_change=on_ab_input_change
        )
    with c2:
        st.number_input(
            "b",
            key="b",
            value=float(st.session_state["b"]),
            step=0.01,
            format="%.2f",
            on_change=on_ab_input_change
        )

    a = clip4(st.session_state["a"])
    b = clip4(st.session_state["b"])
    p = area_two_side(mode, a, b)
    fig = make_figure("two", mode, a, b)
    st.plotly_chart(fig, use_container_width=True)

    st.slider(
        "Drag (a, b)",
        min_value=-4.0, max_value=4.0,
        key="ab_slider",
        value=tuple(st.session_state["ab_slider"]),
        step=0.01,
        on_change=on_ab_slider_change
    )

    lo, hi = (a, b) if a <= b else (b, a)
    st.subheader(f"Probability = {p:.4f} ")
