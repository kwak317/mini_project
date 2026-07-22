from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "mnist_cnn.pt"
MY_NAME = "이름을 입력하세요"


class ScaffoldIncomplete(RuntimeError):
    """학생이 아직 완성하지 않은 핵심 구간을 화면에 친절하게 알립니다."""


class CNN(nn.Module):
    def __init__(self, conv1=32, conv2=64, hidden=128, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, conv1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(conv1, conv2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv2 * 7 * 7, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@st.cache_resource
def load_model():
    """체크포인트로 학습 때와 같은 CNN을 복원합니다."""
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = CNN(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def make_mnist_canvas(image: Image.Image) -> np.ndarray:
    """다양한 사진을 MNIST와 비슷한 28×28 캔버스로 정렬합니다."""
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, "white")
    gray = ImageOps.grayscale(Image.alpha_composite(white, rgba).convert("RGB"))
    arr = np.array(gray, dtype=np.uint8)

    if float(arr.mean()) > 127:
        arr = 255 - arr
    arr[arr < 30] = 0

    mask = arr > 0
    if mask.any():
        ys, xs = np.where(mask)
        digit = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    else:
        digit = arr

    height, width = digit.shape
    side = max(height, width, 1)
    square = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - height) // 2, (side - width) // 2
    square[top:top + height, left:left + width] = digit

    resized = Image.fromarray(square).resize((20, 20), Image.Resampling.LANCZOS)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    canvas[4:24, 4:24] = np.asarray(resized, dtype=np.uint8)
    return canvas


def preprocess_image(image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """업로드 이미지 → 모델 입력 tensor와 화면용 28×28 배열."""
    canvas = make_mnist_canvas(image)
    x = torch.from_numpy(canvas).to(torch.float32).div(255.0)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, canvas


def predict_probabilities(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """모델 logit을 숫자 0~9의 확률 배열로 변환합니다."""
    with torch.inference_mode():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.squeeze(0).cpu().numpy()


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        :root { --ink:#17324d; --paper:#f7f1e5; --coral:#e76f51; --mint:#2a9d8f; }
        .stApp { background:linear-gradient(135deg,#fbf8f1 0%,#eef6f3 100%); color:var(--ink); }
        [data-testid="stHeader"] { background:transparent; }
        [data-testid="stSidebar"] { background:#fffaf0; border-right:1px solid #decfb8; }
        .mp-hero { padding:1.6rem 1.8rem; border:1px solid #d8c8ad; border-radius:18px;
          background:rgba(255,255,255,.82); box-shadow:0 12px 30px rgba(23,50,77,.08); margin-bottom:1.2rem; }
        .mp-kicker { color:var(--coral); font-weight:800; letter-spacing:.08em; font-size:.78rem; }
        .mp-title { color:var(--ink); font-size:clamp(1.8rem,4vw,3rem); line-height:1.08; margin:.35rem 0; }
        .mp-sub { color:#506579; margin:0; max-width:760px; }
        .mp-step { border-left:4px solid var(--mint); padding:.35rem .8rem; color:#40566b; }
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
          <div class="mp-kicker">MODEL LAB · CLASSIFICATION</div>
          <h1 class="mp-title">손글씨 한 장이<br>열 개의 확률이 되기까지</h1>
          <p class="mp-sub">직접 학습한 CNN의 체크포인트를 복원하고, 사진을 28×28 tensor로 바꿔 예측합니다.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="MNIST 분류 모델 랩", page_icon="✍️", layout="wide")
    apply_page_style()
    render_header()

    with st.expander("🧭 이 앱에서 완성한 핵심 4단계", expanded=False):
        st.markdown(
            """
            1. `CNN` — 학습 때와 같은 모델 구조 재구성
            2. `load_model()` — 구조와 `state_dict` 복원, 추론 모드 전환
            3. `preprocess_image()` — 사진을 `(1,1,28,28)` 입력으로 변환
            4. `predict_probabilities()` — logit을 10개 확률로 변환
            """
        )

    if not MODEL_PATH.exists():
        st.error("`mnist_cnn.pt`가 없습니다. 과제 노트북 [Step 7]을 실행해 이 폴더에 생성하세요.")
        st.stop()

    try:
        model, checkpoint = load_model()
    except (KeyError, RuntimeError, TypeError) as exc:
        st.error("학습 때의 모델 구조와 앱의 구조가 일치하지 않습니다. checkpoint key를 확인하세요.")
        st.code(str(exc))
        st.stop()

    metrics = checkpoint["metrics"]
    train_config = checkpoint["training_config"]
    model_config = checkpoint["model_config"]
    st.sidebar.header("MODEL PASSPORT")
    st.sidebar.metric("Validation 정확도", f"{metrics['val_acc']:.4f}")
    st.sidebar.metric("최종 Test 정확도", f"{metrics['test_acc']:.4f}")
    st.sidebar.metric("파라미터", f"{checkpoint['n_params']:,}")
    st.sidebar.caption(f"epochs {train_config['epochs']} · lr {train_config['lr']}")
    st.sidebar.caption(f"conv {model_config['conv1']}/{model_config['conv2']} · hidden {model_config['hidden']}")
    st.sidebar.caption(f"제작: {MY_NAME}")

    input_col, result_col = st.columns([1, 1], gap="large")
    x = None
    with input_col:
        st.subheader("01 · 입력 이미지")
        method = st.radio("입력 방식", ["이미지 업로드", "카메라 촬영"], horizontal=True)
        source = (st.file_uploader("PNG 또는 JPG", type=["png", "jpg", "jpeg"])
                  if method == "이미지 업로드" else st.camera_input("종이의 숫자를 크게 촬영하세요"))

        if source is None:
            st.info("숫자 한 개가 크게 보이는 이미지를 준비하세요. 흰 배경·검은 글씨도 자동으로 반전합니다.")
        else:
            try:
                image = Image.open(source)
                x, preview = preprocess_image(image)
            except Exception:
                st.error("이미지를 열 수 없습니다. 손상되지 않은 PNG/JPG 파일인지 확인하세요.")
                st.stop()
            before, after = st.columns(2)
            before.image(image, caption="원본", width="stretch")
            after.image(preview, caption="모델 입력 28×28", clamp=True, width="stretch")

    with result_col:
        st.subheader("02 · 모델의 판단")
        if source is None:
            with st.container(border=True):
                st.markdown("### 결과 대기 중")
                st.caption("왼쪽에서 이미지를 선택하면 예측 숫자와 클래스 확률이 여기에 표시됩니다.")
            return

        probabilities = predict_probabilities(model, x)
        prediction = int(probabilities.argmax())
        with st.container(border=True):
            st.metric("예측 숫자", prediction)
            st.caption(f"가장 높은 softmax 점수 · {probabilities[prediction] * 100:.1f}%")
            st.bar_chart({"클래스 확률": probabilities})
            top3 = probabilities.argsort()[-3:][::-1]
            st.caption("Top 3 · " + " · ".join(f"{int(i)}: {probabilities[i]:.1%}" for i in top3))


if __name__ == "__main__":
    main()
