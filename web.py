import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torchvision.datasets import ImageFolder


MODEL_PATH = "student_distilled_fp32.pth"
DATASET_PATH = "dataset"

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


dataset = ImageFolder(DATASET_PATH)
class_names = dataset.classes

device = torch.device("cpu")
model = models.mobilenet_v2(weights=None, num_classes=len(class_names))
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

st.set_page_config(page_title="FoodDetector Student FP32", layout="wide")
st.title("🍽️ FoodDetector - Student FP32 (CPU)")
st.markdown(
    """
    <style>
    .pred-badge {
        display: inline-block;
        padding: 0.5em 1em;
        font-size: 1.2em;
        font-weight: bold;
        color: white;
        background: #009688;
        border-radius: 8px;
        margin-top: 0.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.write(
    "Carica una foto di cibo e il modello MobileNetV2 distillato la classificherà tra le seguenti categorie:"
)
st.markdown(", ".join([f"`{c}`" for c in class_names]))

col1, col2 = st.columns([1,2])

with col1:
    uploaded_file = st.file_uploader("Carica un'immagine", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Immagine caricata", use_container_width=True)

with col2:
    if uploaded_file is not None:
        input_tensor = test_transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = outputs.max(1)
            predicted_class = class_names[pred.item()]
        st.markdown(f"<span class='pred-badge'>✅ Predizione: {predicted_class}</span>", unsafe_allow_html=True)
        st.write("La classe predetta è evidenziata sopra. Prova con altre immagini per testare il modello!")
    else:
        st.info("Carica un'immagine per ottenere la predizione.")

st.markdown("---")
st.caption("Modello: MobileNetV2 distillato | Precisione FP32 | Dataset: {}".format(DATASET_PATH))