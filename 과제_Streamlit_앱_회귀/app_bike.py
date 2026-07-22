# 서울 자전거 수요 예측기 — 학생용 핵심 스캐폴딩
# 실행: python3.11 -m streamlit run app_bike.py
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "bike_reg.pt"
MY_NAME = "이름을 입력하세요"
REQUIRED_FEATURES = ["시간", "기온", "습도", "풍속", "가시거리", "이슬점", "일사량", "강우량", "적설량"]


class ScaffoldIncomplete(RuntimeError):
    """학생이 아직 완성하지 않은 핵심 구간을 화면에 친절하게 알립니다."""


def build_model(config: dict) -> nn.Module:
    """체크포인트 설정과 같은 회귀 신경망을 만듭니다."""
    input_dim = int(config["input_dim"])
    hidden = int(config["hidden"])
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


@st.cache_resource
def load_model(model_version: int | None = None):
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def prepare_features(values: dict[str, float], checkpoint: dict) -> torch.Tensor:
    """화면 입력을 학습 때와 같은 특성 순서·스케일로 변환합니다."""
    feature_names = list(checkpoint["feature_names"])
    mean = checkpoint["mean"].to(torch.float32)
    std = checkpoint["std"].to(torch.float32)
    raw = mean.clone()
    for name, value in values.items():
        if name in feature_names:
            raw[feature_names.index(name)] = float(value)
    return (raw - mean) / std


def predict_count(model: nn.Module, normalized: torch.Tensor) -> float:
    """표준화된 한 행으로 시간당 대여량을 예측합니다."""
    with torch.inference_mode():
        prediction = model(normalized.unsqueeze(0))
    return float(prediction.squeeze().item())


def make_toss_line_chart(
    data: pd.DataFrame,
    x_field: str,
    x_title: str,
    color: str = "#3182f6",
) -> alt.Chart:
    """밝은 배경과 충분한 축 여백을 가진 반응형 수요 차트."""
    encoding = {
        "x": alt.X(
            f"{x_field}:Q",
            title=x_title,
            axis=alt.Axis(
                grid=False,
                labelColor="#6b7684",
                titleColor="#4e5968",
                labelFontSize=12,
                titleFontSize=13,
                titlePadding=14,
                tickColor="#d1d6db",
                domainColor="#d1d6db",
            ),
        ),
        "y": alt.Y(
            "예상 대여량:Q",
            title="예상 수요 (대)",
            scale=alt.Scale(zero=False, nice=True),
            axis=alt.Axis(
                grid=True,
                gridColor="#eef1f4",
                gridOpacity=1,
                labelColor="#6b7684",
                titleColor="#4e5968",
                labelFontSize=12,
                titleFontSize=13,
                titlePadding=18,
                domain=False,
                tickSize=0,
                format=",.0f",
            ),
        ),
        "tooltip": [
            alt.Tooltip(f"{x_field}:Q", title=x_title),
            alt.Tooltip("예상 대여량:Q", title="예상 수요", format=",.0f"),
        ],
    }
    area = alt.Chart(data).mark_area(
        color=color,
        opacity=0.09,
        interpolate="monotone",
    ).encode(**encoding)
    line = alt.Chart(data).mark_line(
        color=color,
        strokeWidth=3,
        interpolate="monotone",
        point=alt.OverlayMarkDef(filled=True, size=42, color=color),
    ).encode(**encoding)
    return (
        (area + line)
        .properties(height=300, padding={"left": 12, "right": 18, "top": 12, "bottom": 10})
        .configure_view(stroke=None, fill="#ffffff")
        .configure(background="#ffffff")
    )


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#17324d; --paper:#f7f1e5; --coral:#e76f51; --mint:#2a9d8f; }
        .stApp { background: linear-gradient(135deg, #fbf8f1 0%, #eef6f3 100%); color:var(--ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background:#fff !important; border-right:1px solid #e5e8eb; }
        [data-testid="stSidebar"] * { color:#334155 !important; }
        [data-testid="stSidebarNav"] a { margin:.2rem .45rem; border-radius:12px; }
        [data-testid="stSidebarNav"] a:hover { background:#f1f5f9 !important; }
        [data-testid="stSidebarNav"] a[aria-current="page"] { background:#eaf3ff !important; }
        [data-testid="stSidebarNav"] a[aria-current="page"] * { color:#1b64da !important; font-weight:750; }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#17324d !important; }
        .mp-hero { padding:1.6rem 1.8rem; border:1px solid #d8c8ad; border-radius:18px;
          background:rgba(255,255,255,.82); box-shadow:0 12px 30px rgba(23,50,77,.08); margin-bottom:1.2rem; }
        .mp-kicker { color:var(--mint); font-weight:800; letter-spacing:.08em; font-size:.78rem; }
        .mp-title { color:var(--ink); font-size:clamp(1.8rem,4vw,3rem); line-height:1.08; margin:.35rem 0; }
        .mp-sub { color:#506579; margin:0; max-width:780px; }
        .mp-step { border-left:4px solid var(--coral); padding:.35rem .8rem; color:#40566b; }
        .mp-input-guide { margin:-.35rem 0 1.25rem; padding:.85rem 1rem; border-radius:12px;
          background:#eef6ff; color:#40566b; font-size:.92rem; line-height:1.6; }
        .mp-demand-card { padding:1.25rem 1.35rem; margin-bottom:.85rem; border-radius:16px;
          background:linear-gradient(135deg,#17324d 0%,#24577f 100%); box-shadow:0 10px 24px rgba(23,50,77,.14); }
        .mp-demand-label { color:#c7d8e6; font-size:.9rem; font-weight:750; letter-spacing:.03em; }
        .mp-demand-value { margin-top:.35rem; color:#fff; font-size:clamp(2.3rem,4vw,3.35rem);
          font-weight:850; line-height:1.05; letter-spacing:-.04em; white-space:nowrap; }
        .mp-demand-unit { margin-left:.28rem; color:#c7d8e6; font-size:1rem; font-weight:700; letter-spacing:0; }
        .mp-chart-guide { margin:.2rem 0 1rem; color:#60758a; font-size:.92rem; line-height:1.6; }
        [data-testid="stMetricLabel"] p { color:#506579 !important; font-weight:700 !important; }
        [data-testid="stMetricValue"] { color:#17324d !important; }
        [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
        [data-testid="stSlider"] label p { color:#334155 !important; font-size:.98rem !important; font-weight:750 !important; }
        [data-testid="stSlider"] [data-testid="stThumbValue"] { color:#e63f46 !important; font-weight:800 !important; }
        [data-testid="stExpander"] { border:1px solid #dce5ec !important; border-radius:14px !important;
          background:#fff !important; overflow:hidden; }
        [data-testid="stExpander"] details, [data-testid="stExpander"] summary {
          background:#fff !important; color:#17324d !important; }
        [data-testid="stExpander"] summary { padding:.25rem .35rem; }
        [data-testid="stExpander"] summary:hover { background:#f5f9fc !important; }
        [data-testid="stExpander"] summary p { color:#17324d !important; font-size:1.02rem !important; font-weight:800 !important; }
        [data-testid="stExpander"] summary svg { fill:#2a9d8f !important; color:#2a9d8f !important; }
        [data-testid="stExpander"] [data-testid="stWidgetLabel"] p { color:#334155 !important; }
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
          min-height:48px; border:1px solid #d7dee5 !important; border-radius:12px !important;
          background:#fff !important; box-shadow:none !important; }
        [data-testid="stSelectbox"] div[data-baseweb="select"] * { color:#333d4b !important; }
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover { border-color:#3182f6 !important; }
        [data-testid="stVegaLiteChart"] { overflow:visible !important; border-radius:14px; background:#fff; }
        [data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; background:rgba(255,255,255,.68); }
        .stButton>button { border-radius:12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <section class="mp-hero">
          <div class="mp-kicker">MODEL LAB · REGRESSION</div>
          <h1 class="mp-title">날씨와 시간이<br>대여량 하나가 되기까지</h1>
          <p class="mp-sub">학습 때 저장한 특성 순서와 train 통계를 그대로 복원해 미래 시점의 수요를 예측합니다.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="자전거 회귀 모델 랩", page_icon="🚲", layout="wide")
    apply_page_style()
    render_header()

    with st.expander("🧭 이 앱에서 내가 직접 완성할 핵심 4단계", expanded=False):
        st.markdown(
            """
            1. `build_model()` — 체크포인트 설정으로 회귀 구조 재구성
            2. `load_model()` — `state_dict` 복원과 추론 모드 전환
            3. `prepare_features()` — 특성 순서 복원과 train 통계 표준화
            4. `predict_count()` — 한 행을 배치로 바꿔 예측값 반환

            슬라이더와 결과 패널은 제공됩니다. 네 함수가 연결되어야 실제 예측이 시작됩니다.
            """
        )

    if not MODEL_PATH.exists():
        st.error("`bike_reg.pt`가 없습니다. 과제 노트북의 체크포인트 저장 셀을 실행하세요.")
        st.stop()

    try:
        model, checkpoint = load_model(MODEL_PATH.stat().st_mtime_ns)
    except ScaffoldIncomplete as exc:
        st.warning(str(exc))
        st.info("`app_bike.py`의 TODO 1→2 순서로 완성한 뒤 파일을 저장하면 화면이 자동으로 다시 실행됩니다.")
        st.stop()
    except (KeyError, RuntimeError, TypeError) as exc:
        st.error("학습 때의 모델 구조와 앱의 구조가 일치하지 않습니다. TODO 1·2와 checkpoint key를 확인하세요.")
        st.code(str(exc))
        st.stop()

    feature_names = list(checkpoint["feature_names"])
    missing = [name for name in REQUIRED_FEATURES if name not in feature_names]
    if missing:
        st.error(f"필수 특성이 체크포인트에 없습니다: {missing}")
        st.stop()

    metrics = checkpoint["metrics"]
    train_config = checkpoint["training_config"]
    model_config = checkpoint["model_config"]
    st.sidebar.header("MODEL PASSPORT")
    st.sidebar.metric("Validation MAE", f"{metrics['val_mae']:,.1f}대")
    st.sidebar.metric("최종 Test MAE", f"{metrics['test_mae']:,.1f}대")
    st.sidebar.caption(f"hidden {model_config['hidden']} · epochs {train_config['epochs']}")
    st.sidebar.caption(f"lr {train_config['lr']} · batch {train_config['batch']}")
    st.sidebar.caption(f"제작: {MY_NAME}")

    input_col, result_col = st.columns([1.35, 0.65], gap="large")
    with input_col:
        st.subheader("01 · 예측 조건")
        st.markdown(
            '<div class="mp-input-guide">예측하고 싶은 시간과 날씨를 설정하세요. '
            '각 항목의 <strong>?</strong> 아이콘에 마우스를 올리면 의미와 단위를 확인할 수 있습니다.</div>',
            unsafe_allow_html=True,
        )
        row1 = st.columns(3)
        hour = row1[0].slider("시간", 0, 23, 8, help="예측할 하루의 시간대입니다. 0은 자정, 12는 정오입니다.")
        temperature = row1[1].slider("기온 (°C)", -18.0, 40.0, 20.0, 0.5,
                                     help="해당 시간의 섭씨 기온입니다.")
        humidity = row1[2].slider("습도 (%)", 0, 100, 55, help="공기 중 상대습도입니다.")
        row2 = st.columns(3)
        wind = row2[0].slider("풍속 (m/s)", 0.0, 8.0, 1.5, 0.1,
                              help="초당 이동하는 바람의 속도입니다.")
        visibility = row2[1].slider("가시거리 (10m)", 0, 2000, 1500, 50,
                                    help="관측 가능한 거리입니다. 1500은 약 15km를 뜻합니다.")
        dew_point = row2[2].slider("이슬점 (°C)", -30.0, 28.0, 10.0, 0.5,
                                   help="공기 중 수증기가 물방울로 맺히기 시작하는 온도입니다.")
        with st.expander("☔ 강수·일사 조건 — 비·눈·햇빛 설정", expanded=True):
            row3 = st.columns(3)
            solar = row3[0].slider("일사량 (MJ/m²)", 0.0, 3.6, 0.5, 0.1,
                                   help="지면 1m²에 도달한 태양 에너지의 양입니다.")
            rain = row3[1].slider("강우량 (mm)", 0.0, 35.0, 0.0, 0.1,
                                  help="해당 시간 동안 내린 비의 양입니다. 0은 비가 오지 않음을 뜻합니다.")
            snow = row3[2].slider("적설량 (cm)", 0.0, 9.0, 0.0, 0.1,
                                  help="지면에 쌓인 눈의 깊이입니다. 0은 적설이 없음을 뜻합니다.")

    values = {
        "시간": hour, "기온": temperature, "습도": humidity, "풍속": wind,
        "가시거리": visibility, "이슬점": dew_point, "일사량": solar,
        "강우량": rain, "적설량": snow,
    }
    with result_col:
        st.subheader("02 · 예측 결과")
        try:
            normalized = prepare_features(values, checkpoint)
            prediction = predict_count(model, normalized)
        except ScaffoldIncomplete as exc:
            st.warning(str(exc))
            st.stop()
        with st.container(border=True):
            display_prediction = max(0, prediction)
            st.markdown(
                f"""
                <div class="mp-demand-card">
                  <div class="mp-demand-label">예상 대여량</div>
                  <div class="mp-demand-value">{display_prediction:,.0f}<span class="mp-demand-unit">대 / 시간</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Validation MAE 기준, 평균적으로 약 {metrics['val_mae']:,.0f}대 차이가 날 수 있습니다.")
        st.markdown('<p class="mp-step">슬라이더 하나만 바꿔 예측이 어떻게 움직이는지 관찰하고, 데이터에서 배운 관계인지 설명해 보세요.</p>', unsafe_allow_html=True)
        st.caption("2017-12~2018-11 운영일 자료로 학습한 교육용 모델입니다. 실제 운영에는 최신 데이터와 불확실성 검증이 필요합니다.")

    st.markdown("### 03 · 수요 변화 탐색")
    st.markdown(
        '<p class="mp-chart-guide">현재 슬라이더 조건을 기준으로 한 변수만 바꿔 예측 변화를 비교합니다. '
        '이 그래프는 모델이 학습한 관계를 보여주며 실제 인과관계를 의미하지는 않습니다.</p>',
        unsafe_allow_html=True,
    )

    curve_col, sensitivity_col = st.columns(2, gap="large")
    with curve_col:
        with st.container(border=True):
            st.markdown("#### 🕒 시간대별 예상 수요곡선")
            st.caption("날씨 조건은 현재 값으로 고정하고 시간만 0시부터 23시까지 바꿉니다.")
            hourly_rows = []
            for curve_hour in range(24):
                scenario = values.copy()
                scenario["시간"] = curve_hour
                scenario_x = prepare_features(scenario, checkpoint)
                scenario_prediction = max(0, predict_count(model, scenario_x))
                hourly_rows.append({"시간": curve_hour, "예상 대여량": scenario_prediction})
            hourly_df = pd.DataFrame(hourly_rows)
            hourly_chart = make_toss_line_chart(
                hourly_df,
                x_field="시간",
                x_title="시간대",
                color="#3182f6",
            )
            st.altair_chart(hourly_chart, use_container_width=True, theme=None)
            peak_row = hourly_df.loc[hourly_df["예상 대여량"].idxmax()]
            st.caption(
                f"현재 날씨에서 최고 수요 시간 · {int(peak_row['시간'])}시 "
                f"({peak_row['예상 대여량']:,.0f}대/시간)"
            )

    with sensitivity_col:
        with st.container(border=True):
            st.markdown("#### 🎚️ 조건 민감도")
            st.caption("선택한 조건 하나만 움직이고 나머지 값은 현재 슬라이더 값으로 고정합니다.")
            sensitivity_options = {
                "기온": ([-18 + i * 2 for i in range(30)], "°C"),
                "습도": (list(range(0, 101, 5)), "%"),
                "풍속": ([i / 2 for i in range(17)], "m/s"),
                "일사량": ([i / 10 for i in range(37)], "MJ/m²"),
                "강우량": ([float(i) for i in range(36)], "mm"),
            }
            sensitive_feature = st.selectbox(
                "변화시킬 조건",
                list(sensitivity_options),
                index=0,
                help="선택한 조건만 범위 전체에서 바꾸어 모델 예측의 민감도를 확인합니다.",
            )
            feature_values, feature_unit = sensitivity_options[sensitive_feature]
            sensitivity_rows = []
            for feature_value in feature_values:
                scenario = values.copy()
                scenario[sensitive_feature] = feature_value
                scenario_x = prepare_features(scenario, checkpoint)
                scenario_prediction = max(0, predict_count(model, scenario_x))
                sensitivity_rows.append(
                    {sensitive_feature: feature_value, "예상 대여량": scenario_prediction}
                )
            sensitivity_df = pd.DataFrame(sensitivity_rows)
            sensitivity_chart = make_toss_line_chart(
                sensitivity_df,
                x_field=sensitive_feature,
                x_title=f"{sensitive_feature} ({feature_unit})",
                color="#3182f6",
            )
            st.altair_chart(sensitivity_chart, use_container_width=True, theme=None)
            low_prediction = sensitivity_df["예상 대여량"].iloc[0]
            high_prediction = sensitivity_df["예상 대여량"].iloc[-1]
            direction = "증가" if high_prediction > low_prediction else "감소"
            st.caption(
                f"{sensitive_feature} 범위 양끝 비교 · "
                f"{feature_values[0]:g}{feature_unit} → {feature_values[-1]:g}{feature_unit}에서 "
                f"예측 수요가 {abs(high_prediction-low_prediction):,.0f}대 {direction}"
            )


if __name__ == "__main__":
    main()
