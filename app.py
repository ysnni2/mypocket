import streamlit as st
import torch, json
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

def get_model(backbone, num_classes):
    if backbone == "resnet18":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

with open("models/class_names.json") as f:
    CLASS_NAMES = json.load(f)

st.set_page_config(page_title="Pokemon Classifier", page_icon="🎮", layout="centered")

st.markdown("""
<style>
.title { font-size: 2rem; font-weight: 700; text-align: center; margin-bottom: 0.2rem; }
.subtitle { text-align: center; color: gray; margin-bottom: 2rem; }
.result-box { background: #f8f9fa; border-radius: 12px; padding: 1.2rem; margin-bottom: 0.6rem; }
.rank1 { border-left: 5px solid #ff4b4b; }
.rank2 { border-left: 5px solid #ffa500; }
.rank3 { border-left: 5px solid #ffd700; }
.rank-other { border-left: 5px solid #cccccc; }
.pokemon-name { font-size: 1.1rem; font-weight: 600; }
.prob-text { font-size: 0.9rem; color: gray; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎮 Pokemon Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">포켓몬 이미지를 업로드하면 이름을 맞춰드립니다!</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = get_model("resnet18", len(CLASS_NAMES))
    model.load_state_dict(torch.load("models/exp2_resnet18_fullft.pth", map_location="cpu"))
    return model.eval()

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption="업로드된 이미지", use_container_width=True)

    with col2:
        with torch.no_grad():
            probs = torch.softmax(model(transform(img).unsqueeze(0)), dim=1)[0]

        top5_probs, top5_idxs = probs.topk(5)
        top1_name = CLASS_NAMES[top5_idxs[0].item()]
        top1_prob = top5_probs[0].item() * 100

        # 상단 결과 요약
        if top1_prob >= 80:
            st.success(f"**{top1_name}** ({top1_prob:.1f}%)")
        elif top1_prob >= 50:
            st.warning(f"**{top1_name}** ({top1_prob:.1f}%)")
        else:
            st.error(f"**{top1_name}** ({top1_prob:.1f}%) — 확신도 낮음")

        st.markdown("#### 🏆 Top-5 예측 결과")

        rank_colors = ["#ff4b4b", "#ffa500", "#ffd700", "#aaaaaa", "#cccccc"]
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

        for i, (prob, idx) in enumerate(zip(top5_probs, top5_idxs)):
            name = CLASS_NAMES[idx.item()]
            p = prob.item() * 100
            st.markdown(f"{medals[i]} **{name}**")
            st.progress(float(prob), text=f"{p:.1f}%")
