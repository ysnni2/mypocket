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
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

with open("models/class_names.json") as f:
    CLASS_NAMES = json.load(f)

MODEL_OPTIONS = {
    "ResNet18 Frozen":     ("exp1_resnet18_frozen",    "resnet18"),
    "ResNet18 Full FT":    ("exp2_resnet18_fullft",     "resnet18"),
    "EfficientNet-B0":     ("exp4_efficientnet_frozen", "efficientnet_b0"),
    "MobileNetV2 Full FT": ("exp5_mobilenet_fullft",    "mobilenet_v2"),
}

st.title("Pokemon Classifier")
st.write("포켓몬 이미지를 업로드하면 이름을 맞춰드립니다!")
choice = st.selectbox("모델 선택", list(MODEL_OPTIONS.keys()))

@st.cache_resource
def load(name, backbone):
    model = get_model(backbone, len(CLASS_NAMES))
    model.load_state_dict(torch.load(f"models/{name}.pth", map_location="cpu"))
    return model.eval()

name, backbone = MODEL_OPTIONS[choice]
model = load(name, backbone)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded = st.file_uploader("이미지 업로드", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_column_width=True)
    with torch.no_grad():
        probs = torch.softmax(model(transform(img).unsqueeze(0)), dim=1)[0]
    st.subheader("Top-5 Results")
    for prob, idx in zip(*probs.topk(5)):
        st.write(f"**{CLASS_NAMES[idx]}** ({prob*100:.2f}%)")
        st.progress(float(prob))
