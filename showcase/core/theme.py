"""두 미니프로젝트 showcase의 공통 디자인 헬퍼."""

import html

import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --toss-blue:#3182f6;
          --toss-blue-dark:#1b64da;
          --toss-blue-soft:#eaf3ff;
          --ink:#191f28;
          --muted:#6b7684;
          --line:#e5e8eb;
          --surface:#ffffff;
          --canvas:#f7f8fa;
        }

        html, body, [class*="css"] {
          font-family:Pretendard,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        }
        .stApp { background:var(--canvas); color:var(--ink); }
        [data-testid="stHeader"] { background:rgba(247,248,250,.82); backdrop-filter:blur(14px); }
        [data-testid="stAppViewContainer"] > .main .block-container {
          max-width:1160px; padding-top:3rem; padding-bottom:6rem;
        }
        [data-testid="stSidebar"] { background:#fff !important; border-right:1px solid var(--line); }
        [data-testid="stSidebar"] * { color:#334155 !important; }
        [data-testid="stSidebarNav"] a { margin:.2rem .45rem; border-radius:12px; }
        [data-testid="stSidebarNav"] a:hover { background:#f1f5f9 !important; }
        [data-testid="stSidebarNav"] a[aria-current="page"] { background:#eaf3ff !important; }
        [data-testid="stSidebarNav"] a[aria-current="page"] * { color:#1b64da !important; font-weight:750; }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#17324d !important; }

        .sc-hero {
          position:relative; isolation:isolate; overflow:hidden;
          min-height:390px; padding:4.6rem 4.8rem;
          display:flex; flex-direction:column; justify-content:center;
          border:1px solid rgba(49,130,246,.08); border-radius:32px;
          background:linear-gradient(135deg,#fff 0%,#f8fbff 62%,#eef6ff 100%);
          box-shadow:0 20px 60px rgba(15,23,42,.06);
        }
        .sc-hero:before {
          content:""; position:absolute; z-index:-1; width:340px; height:340px;
          right:-70px; top:-90px; border-radius:50%;
          background:rgba(49,130,246,.12); filter:blur(4px);
        }
        .sc-hero:after {
          content:""; position:absolute; z-index:-1; width:180px; height:180px;
          right:190px; bottom:-105px; border-radius:46%; transform:rotate(28deg);
          background:rgba(139,201,255,.2);
        }
        .sc-kicker {
          width:max-content; padding:.48rem .8rem; border-radius:999px;
          background:var(--toss-blue-soft); color:var(--toss-blue-dark);
          font-weight:700; letter-spacing:.04em; font-size:.76rem;
        }
        .sc-title {
          max-width:780px; margin:1.15rem 0 1rem;
          color:var(--ink); font-size:clamp(2.65rem,5.8vw,4.8rem);
          font-weight:800; letter-spacing:-.055em; line-height:1.08;
        }
        .sc-sub {
          max-width:720px; margin:0; color:var(--muted);
          font-size:clamp(1rem,1.6vw,1.18rem); line-height:1.75;
        }
        .sc-chip-row { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.55rem; }
        .sc-chip {
          display:inline-flex; align-items:center; padding:.42rem .72rem;
          border:1px solid #e5e8eb; border-radius:999px;
          background:#fff; color:#4e5968; font-size:.82rem; font-weight:600;
        }
        .sc-author { margin:1rem 0 0; color:#8b95a1; font-size:.88rem; }
        .sc-section { margin:4.2rem 0 1.15rem; }
        .sc-eyebrow { color:var(--toss-blue); font-size:.78rem; font-weight:750; letter-spacing:.08em; }
        .sc-section-title { margin:.28rem 0 0; color:var(--ink); font-size:1.72rem; font-weight:750; letter-spacing:-.035em; }
        .sc-section-copy { margin:.45rem 0 0; color:var(--muted); font-size:.98rem; }
        .sc-note {
          padding:1.15rem 1.25rem; border:1px solid #dbeafe; border-radius:14px;
          background:#f4f8ff; color:#4e5968; line-height:1.65;
        }

        div[data-testid="stMetric"] {
          min-height:126px; padding:1.35rem 1.45rem;
          border:1px solid var(--line); border-radius:20px; background:#fff;
          box-shadow:0 8px 24px rgba(15,23,42,.035);
        }
        [data-testid="stMetricLabel"] { color:#8b95a1; font-weight:600; }
        [data-testid="stMetricValue"] { color:var(--ink); font-weight:750; letter-spacing:-.035em; }
        [data-testid="stVerticalBlockBorderWrapper"] {
          border:1px solid var(--line) !important; border-radius:24px;
          background:var(--surface); box-shadow:0 10px 30px rgba(15,23,42,.045);
          transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
          transform:translateY(-3px); border-color:#cbd5e1 !important;
          box-shadow:0 18px 42px rgba(15,23,42,.08);
        }
        [data-testid="stVerticalBlockBorderWrapper"] h3 { color:var(--ink); letter-spacing:-.03em; }
        [data-testid="stCaptionContainer"] { color:#8b95a1; }

        .stAlert { border-radius:14px; border:none; }
        a[data-testid="stPageLink-NavLink"] {
          min-height:48px; justify-content:center; margin-top:.3rem;
          border:0; border-radius:14px; background:var(--toss-blue); color:#fff !important;
          font-weight:700; transition:background .2s ease,transform .2s ease;
        }
        a[data-testid="stPageLink-NavLink"]:hover { background:var(--toss-blue-dark); transform:translateY(-1px); }
        a[data-testid="stPageLink-NavLink"] p { color:#fff !important; }
        [data-testid="stExpander"] { border:1px solid var(--line); border-radius:18px; background:#fff; }

        @media (max-width:700px) {
          [data-testid="stAppViewContainer"] > .main .block-container { padding-top:1.5rem; }
          .sc-hero { min-height:360px; padding:2.8rem 1.7rem; border-radius:24px; }
          .sc-title { font-size:2.65rem; }
          .sc-section { margin-top:3rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, name: str) -> None:
    st.markdown(
        f"""
        <section class="sc-hero">
          <div class="sc-kicker">MY DEEP LEARNING LAB</div>
          <h1 class="sc-title">{html.escape(title)}</h1>
          <p class="sc-sub">{html.escape(subtitle)}</p>
          <div class="sc-chip-row"><span class="sc-chip">Classification</span><span class="sc-chip">Regression</span><span class="sc-chip">PyTorch</span><span class="sc-chip">Streamlit</span></div>
          <p class="sc-author">Built by {html.escape(name)}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
