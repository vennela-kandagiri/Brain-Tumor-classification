import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import build_model

MODEL_PATH = "outputs/best_model.pth"
NUM_CLASSES = 2
CLASS_NAMES = ["no", "yes"]
IMG_SIZE = 224

@st.cache_resource
def load_model():
    model = build_model(NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def preprocess(img):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485,0.456,0.406],
            [0.229,0.224,0.225]
        )
    ])
    return tf(img).unsqueeze(0)

def predict(img, model):
    x = preprocess(img)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return CLASS_NAMES[pred.item()], float(conf.item())

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI image to classify if tumor is present.")

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

model = load_model()

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        label, conf = predict(img, model)
        st.success(f"Prediction: **{label.upper()}**")
        st.info(f"Confidence: {conf*100:.2f}%")
