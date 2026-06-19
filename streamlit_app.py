import os
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from PIL import Image
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Model

st.set_page_config(
    page_title="NeuroScan AI · Princedex",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #070c18 !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1500px !important; }

/* ── Sidebar ── rich deep blue */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060f24 0%, #0a1a3e 50%, #060f24 100%) !important;
    border-right: 2px solid #1a4aaa !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.2rem 0.9rem !important; }
section[data-testid="stSidebar"] * { color: #d0e4ff !important; }
section[data-testid="stSidebar"] a { color: #5aabff !important; text-decoration: none !important; }

/* ── Topbar */
.topbar {
    background: linear-gradient(90deg, #040a18, #0a1630, #040a18);
    border-bottom: 2px solid #1a4aaa;
    padding: 15px 32px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: -1rem -2rem 2rem -2rem;
    box-shadow: 0 3px 24px rgba(26,74,170,0.35);
}
.tb-brand { font-family:'Space Grotesk',sans-serif; font-size:21px; font-weight:700; color:#fff; }
.tb-brand span { color:#5aabff; }
.tb-right { display:flex; gap:12px; align-items:center; }
.tb-dev { font-size:12px; color:#2a5a9a; }
.tb-tag {
    font-size:10px; color:#5aabff;
    background:rgba(90,171,255,0.12); border:1px solid rgba(90,171,255,0.5);
    border-radius:20px; padding:4px 14px; letter-spacing:1.5px;
    text-transform:uppercase; font-weight:600;
}

/* ── Hero */
.hero { text-align:center; padding:34px 0 22px; }
.hero-eye { font-size:10px; letter-spacing:4px; text-transform:uppercase; color:#5aabff; font-weight:600; margin-bottom:12px; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:44px; font-weight:700; color:#fff; line-height:1.15; margin-bottom:16px; }
.hero-title span { color:#5aabff; }
.hero-sub { font-size:15px; color:#5a7a9a; max-width:580px; margin:0 auto; line-height:1.8; }

/* ── Metric cards */
.mrow { display:flex; gap:12px; justify-content:center; margin:26px 0 34px; flex-wrap:wrap; }
.mc {
    background:linear-gradient(135deg,rgba(90,171,255,0.12),rgba(26,74,170,0.06));
    border:1px solid rgba(90,171,255,0.3); border-radius:14px;
    padding:16px 24px; text-align:center; min-width:110px;
    box-shadow:0 4px 18px rgba(0,80,200,0.15);
}
.mc .v { font-family:'Space Grotesk',sans-serif; font-size:26px; font-weight:700; color:#5aabff; }
.mc .l { font-size:9px; color:#2a4a7a; text-transform:uppercase; letter-spacing:1.2px; margin-top:5px; font-weight:600; }

/* ── Panels */
.panel {
    background:linear-gradient(135deg,rgba(8,18,40,0.98),rgba(6,14,30,0.95));
    border:1px solid rgba(90,171,255,0.15); border-radius:16px;
    padding:22px; margin-bottom:18px;
    box-shadow:0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(90,171,255,0.07);
}
.pt {
    font-size:9px; font-weight:700; letter-spacing:2.5px;
    text-transform:uppercase; color:#5aabff;
    margin-bottom:16px; padding-bottom:11px;
    border-bottom:1px solid rgba(90,171,255,0.12);
}

/* ── MRI placeholder */
.mri-placeholder {
    width:100%; aspect-ratio:1/1;
    background:radial-gradient(ellipse at 50% 45%, #0d2050 0%, #070e20 60%, #030710 100%);
    border-radius:12px; display:flex; align-items:center;
    justify-content:center; position:relative; overflow:hidden;
    border:1px solid rgba(90,171,255,0.2);
}
.ring {
    position:absolute; border-radius:50%;
    border:1px solid rgba(90,171,255,0.2);
    pointer-events:none;
}
.scan-text {
    position:absolute; bottom:12px; left:0; right:0;
    text-align:center; font-size:9px; color:#1a3a6a;
    letter-spacing:2px; text-transform:uppercase; font-weight:600;
}
.scan-label {
    position:absolute; top:10px; right:10px;
    background:rgba(90,171,255,0.15); border:1px solid rgba(90,171,255,0.3);
    border-radius:6px; padding:3px 8px; font-size:9px; color:#5aabff;
    letter-spacing:1px; font-weight:600;
}

/* ── Diagnosis badges */
.badge-tumor {
    background:linear-gradient(135deg,rgba(255,80,80,0.16),rgba(180,20,20,0.08));
    border:2px solid rgba(255,100,100,0.5); border-radius:12px;
    padding:16px 20px; margin-bottom:16px;
}
.badge-ok {
    background:linear-gradient(135deg,rgba(0,230,100,0.13),rgba(0,160,60,0.06));
    border:2px solid rgba(0,230,100,0.45); border-radius:12px;
    padding:16px 20px; margin-bottom:16px;
}
.badge-tumor .bn, .badge-ok .bn {
    font-family:'Space Grotesk',sans-serif; font-size:28px; font-weight:700;
}
.badge-tumor .bn { color:#ff6060; }
.badge-ok .bn { color:#00e870; }
.badge-tumor .bs { font-size:11px; color:rgba(255,100,100,0.7); margin-top:5px; }
.badge-ok .bs { font-size:11px; color:rgba(0,232,112,0.7); margin-top:5px; }

/* ── Prob bars */
.pb-row { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
.pb-lbl { font-size:11px; color:#3a5a7a; width:150px; flex-shrink:0; }
.pb-lbl.top { color:#d0e4ff; font-weight:600; }
.pb-bg { flex:1; background:rgba(255,255,255,0.04); border-radius:20px; height:8px; overflow:hidden; border:1px solid rgba(255,255,255,0.04); }
.pb-val { font-size:12px; font-weight:700; width:48px; text-align:right; flex-shrink:0; }

/* ── Info panels */
.info-card {
    background:rgba(0,0,0,0.3); border-radius:10px;
    padding:13px 15px; border-left:3px solid;
}
.info-card .ict {
    font-size:9px; color:#2a4a7a; letter-spacing:1.5px;
    text-transform:uppercase; font-weight:700; margin-bottom:7px;
}
.info-card .icb { font-size:12px; color:#6a90b8; line-height:1.75; }

.insight {
    background:rgba(100,60,220,0.09); border:1px solid rgba(130,90,255,0.3);
    border-radius:10px; padding:13px 16px; margin:13px 0;
    font-size:12px; color:#b0a0f0; line-height:1.75;
}
.warn {
    background:rgba(255,140,0,0.08); border:1px solid rgba(255,140,0,0.25);
    border-radius:10px; padding:12px 16px; margin-top:12px;
    font-size:11px; color:#ffb040; line-height:1.75;
}

/* ── Grad-CAM */
.gc-lbl {
    font-size:9px; text-align:center; color:#2a4a7a;
    letter-spacing:1.5px; text-transform:uppercase;
    margin-top:6px; font-weight:600;
}

/* ── Sidebar parts */
.sb-brand {
    text-align:center; padding-bottom:16px;
    border-bottom:1px solid rgba(90,171,255,0.15); margin-bottom:16px;
}
.sb-brand .sbn { font-family:'Space Grotesk',sans-serif; font-size:22px; font-weight:700; color:#fff !important; }
.sb-brand .sbn span { color:#5aabff !important; }
.sb-brand .sbs { font-size:9px; color:#2a4a7a !important; letter-spacing:2px; text-transform:uppercase; margin-top:4px; }

.sb-sec {
    background:rgba(0,0,0,0.2); border:1px solid rgba(90,171,255,0.12);
    border-radius:11px; padding:13px; margin-bottom:12px;
}
.sb-sec .sbt {
    font-size:9px; font-weight:700; letter-spacing:2px;
    text-transform:uppercase; color:#5aabff !important;
    margin-bottom:11px;
}
.sb-sec p, .sb-sec li { font-size:11px; color:#7090b8 !important; line-height:1.8; margin:0; }
.sb-sec b { color:#b0ccee !important; }

.tc {
    background:rgba(0,0,0,0.25); border-radius:9px;
    padding:10px 12px; margin-bottom:8px; border-left:3px solid;
}
.tc .tcn { font-size:11px; font-weight:700; margin-bottom:4px; }
.tc .tcd { font-size:10px; color:#3a5a7a !important; line-height:1.6; }
.chip {
    display:inline-block; border-radius:6px; padding:3px 10px;
    font-size:10px; font-weight:600; margin:3px 2px; border:1px solid;
}

/* ── Stbutton */
.stButton > button {
    background: linear-gradient(135deg, #0044cc, #0088ee) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; padding: 14px 20px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important; font-weight: 700 !important;
    width: 100% !important; box-shadow: 0 5px 24px rgba(0,100,220,0.4) !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

div[data-testid="stFileUploader"] {
    background: rgba(0,20,60,0.5) !important;
    border: 1px dashed rgba(90,171,255,0.35) !important;
    border-radius: 12px !important;
}

/* ── Footer */
.footer {
    text-align:center; padding:28px 0 10px;
    border-top:1px solid rgba(90,171,255,0.1); margin-top:40px;
}
.footer .fb { font-family:'Space Grotesk',sans-serif; font-size:17px; font-weight:700; color:#fff; margin-bottom:8px; }
.footer .fb span { color:#5aabff; }
.footer .fd { font-size:11px; color:#1a3a5a; line-height:2; }
.footer .fd a { color:#5aabff; text-decoration:none; }
</style>
""", unsafe_allow_html=True)

IMG_SHAPE   = (224, 224, 3)
IMG_SIZE    = (224, 224)
NUM_CLASSES = 4
CLASS_NAMES = {0:"Glioma Tumor",1:"Meningioma Tumor",2:"No Tumor",3:"Pituitary Tumor"}

CLASS_INFO = {
    "Glioma Tumor":{
        "color":"#ff5252","icon":"🔴","grade":"WHO Grade 1–4",
        "short":"Most aggressive primary brain tumor",
        "desc":"Gliomas arise from glial cells — the brain's supportive cells. They are the most common and deadly primary brain tumors, accounting for ~80% of all malignant cases. Glioblastoma (Grade 4/GBM) carries a median survival of only 15 months even with aggressive treatment.",
        "symptoms":"Headaches, seizures, memory problems, personality changes, weakness on one side",
        "treatment":"Surgical resection, radiotherapy, chemotherapy (temozolomide for GBM)",
        "sensitivity":"39.00%","specificity":"94.90%",
        "note":"⚠️ Urgent specialist referral is strongly recommended."
    },
    "Meningioma Tumor":{
        "color":"#ff9800","icon":"🟠","grade":"Usually Benign",
        "short":"Most common benign brain tumor",
        "desc":"Meningiomas arise from the meninges — three membranes surrounding the brain and spinal cord. They account for ~36% of all primary brain tumors and are most often benign. They grow slowly and are more common in women aged 40+.",
        "symptoms":"Headaches, vision changes, hearing loss, memory issues, limb weakness",
        "treatment":"Observation for small tumors, surgical resection, stereotactic radiosurgery",
        "sensitivity":"82.61%","specificity":"81.72%",
        "note":"Regular monitoring and neurological review recommended."
    },
    "No Tumor":{
        "color":"#00e870","icon":"🟢","grade":"Normal Scan",
        "short":"No tumor detected in this MRI scan",
        "desc":"No evidence of brain tumor has been detected in the submitted MRI scan. The AI model found no identifiable mass, lesion, or irregular growth. However, the model has a sensitivity of 90.48% — always confirm with a qualified neuroradiologist.",
        "symptoms":"N/A — normal scan",
        "treatment":"Routine clinical follow-up as advised by your physician",
        "sensitivity":"90.48%","specificity":"87.20%",
        "note":"✅ No tumor detected. Continue routine medical check-ups."
    },
    "Pituitary Tumor":{
        "color":"#ce93d8","icon":"🟣","grade":"Usually Benign Adenoma",
        "short":"Benign adenoma of the pituitary gland",
        "desc":"Pituitary tumors arise from the pituitary gland at the base of the brain. The vast majority are benign adenomas. Their proximity to the optic chiasm frequently causes visual disturbances. Classified as microadenomas (<10mm) or macroadenomas (≥10mm).",
        "symptoms":"Peripheral vision loss, headaches, hormonal imbalances, infertility, abnormal growth",
        "treatment":"Medication (dopamine agonists), transsphenoidal surgery, radiation therapy",
        "sensitivity":"71.62%","specificity":"97.19%",
        "note":"Endocrinological evaluation and ophthalmology review recommended."
    }
}

CLASS_COLORS = {k:v["color"] for k,v in CLASS_INFO.items()}
os.makedirs("static/uploads", exist_ok=True)

MRI_PLACEHOLDER = """
<div class="mri-placeholder">
    <div class="ring" style="width:85%;height:80%;top:10%;left:7.5%"></div>
    <div class="ring" style="width:65%;height:60%;top:20%;left:17.5%;border-color:rgba(90,171,255,0.12)"></div>
    <div class="ring" style="width:42%;height:38%;top:31%;left:29%;border-color:rgba(90,171,255,0.07)"></div>
    <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:80px;height:1px;background:rgba(90,171,255,0.15)"></div>
    <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:1px;height:80px;background:rgba(90,171,255,0.15)"></div>
    <div style="position:absolute;width:8px;height:8px;border-radius:50%;background:rgba(90,171,255,0.25);top:50%;left:50%;transform:translate(-50%,-50%)"></div>
    <div class="scan-label">T1-MRI</div>
    <div class="scan-text">Upload a brain MRI scan to analyse</div>
</div>
"""

# ── Topbar
st.markdown("""
<div class="topbar">
    <div class="tb-brand">Neuro<span>Scan</span> AI</div>
    <div class="tb-right">
        <span class="tb-dev">Stephen Ifeanyi · Princedex</span>
        <span class="tb-tag">Research Prototype</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sbn">Neuro<span>Scan</span> AI</div>
        <div class="sbs">Brain Tumor Detection System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-sec">
        <div class="sbt">🔬 About This System</div>
        <p>An AI-powered clinical decision-support tool for brain tumor
        detection and classification from MRI scans, built as a Final Year
        Project using EfficientNetV2, transfer learning, and Grad-CAM
        explainability.</p>
    </div>

    <div class="sb-sec">
        <div class="sbt">🧬 Tumor Classes</div>
        <div class="tc" style="border-color:#ff5252">
            <div class="tcn" style="color:#ff5252 !important">🔴 Glioma Tumor</div>
            <div class="tcd">Arises from glial cells. Most aggressive primary tumor.
            WHO Grade 1–4. GBM (Grade 4) has median survival of ~15 months.</div>
        </div>
        <div class="tc" style="border-color:#ff9800">
            <div class="tcn" style="color:#ff9800 !important">🟠 Meningioma Tumor</div>
            <div class="tcd">Arises from meningeal membranes. Most common primary
            brain tumor overall (~36%). Usually benign. More frequent in women 40+.</div>
        </div>
        <div class="tc" style="border-color:#ce93d8">
            <div class="tcn" style="color:#ce93d8 !important">🟣 Pituitary Tumor</div>
            <div class="tcd">Arises from the pituitary gland. Usually a benign
            adenoma. Can cause hormonal disorders and visual field defects.</div>
        </div>
        <div class="tc" style="border-color:#00e870">
            <div class="tcn" style="color:#00e870 !important">🟢 No Tumor</div>
            <div class="tcd">No tumor evidence detected. Normal brain tissue
            observed. Routine clinical follow-up still recommended.</div>
        </div>
    </div>

    <div class="sb-sec">
        <div class="sbt">⚙️ Model Architecture</div>
        <p>
        <b>Backbone:</b> EfficientNetV2-B0<br>
        <b>Pretrained:</b> ImageNet (1.28M images)<br>
        <b>Strategy:</b> Two-phase transfer learning<br>
        <b>Phase 1:</b> Train classification head only<br>
        <b>Phase 2:</b> Fine-tune top 50 backbone layers<br>
        <b>Parameters:</b> ~7.1 Million<br>
        <b>Input size:</b> 224 × 224 × 3 (RGB)<br>
        <b>Optimizer:</b> Adam  |  LR: 1e-4 → 1e-5<br>
        <b>Framework:</b> TensorFlow 2.x / Keras
        </p>
    </div>

    <div class="sb-sec">
        <div class="sbt">📊 Performance Metrics</div>
        <p>
        <b>Test Accuracy:</b> <span style="color:#5aabff !important">71.57%</span><br>
        <b>Val Accuracy:</b> <span style="color:#5aabff !important">82.20%</span><br>
        <b>AUC-ROC:</b> <span style="color:#5aabff !important">0.8928</span><br>
        <b>No Tumor Sensitivity:</b> <span style="color:#00e870 !important">90.48%</span><br>
        <b>Meningioma Sensitivity:</b> <span style="color:#ff9800 !important">82.61%</span><br>
        <b>Pituitary Sensitivity:</b> <span style="color:#ce93d8 !important">71.62%</span><br>
        <b>Glioma Sensitivity:</b> <span style="color:#ff5252 !important">39.00%</span>
        <span style="color:#1a3a5a !important;font-size:10px;"> ← active research area</span>
        </p>
    </div>

    <div class="sb-sec">
        <div class="sbt">🔍 What is Grad-CAM?</div>
        <p><b>Gradient-weighted Class Activation Mapping</b> uses
        gradients flowing into the final convolutional layer to generate
        a heatmap revealing <b>which regions of the MRI</b> the model
        focused on when making its decision.<br><br>
        <b style="color:#ff6060 !important">Red/Yellow</b> = high attention<br>
        <b style="color:#5aabff !important">Blue</b> = low attention<br><br>
        This bridges the gap between AI accuracy and clinical trust —
        showing doctors <i>why</i> the model reached a diagnosis.</p>
    </div>

    <div class="sb-sec">
        <div class="sbt">📚 Dataset</div>
        <p>
        <b>Name:</b> Kaggle Brain MRI Dataset<br>
        <b>Total images:</b> 3,264 MRI scans<br>
        <b>Modality:</b> T1-weighted MRI<br>
        <b>Train/Val/Test:</b> 70% / 15% / 15%<br>
        <b>Augmentation:</b> Rotation, flip, zoom,
        brightness, shear transformations
        </p>
    </div>

    <div class="sb-sec">
        <div class="sbt">👨‍💻 Developer</div>
        <p>
        <b>Name:</b> Stephen Ifeanyi<br>
        <b>Brand:</b> Princedex<br>
        <b>Email:</b> <a href="mailto:ifeanyistephen003@gmail.com">
        ifeanyistephen003@gmail.com</a><br>
        <b>Project:</b> Final Year Project<br>
        <b>Discipline:</b> AI · Deep Learning · Medical Imaging
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Hero
st.markdown("""
<div class="hero">
    <div class="hero-eye">Artificial Intelligence · Deep Learning · Medical Imaging · Explainability</div>
    <div class="hero-title">Brain Tumor Detection<br><span>Powered by EfficientNetV2</span></div>
    <div class="hero-sub">
        Upload a T1-weighted brain MRI scan for instant AI-powered classification
        with Grad-CAM visual explanation — revealing exactly where the model focused its attention.
    </div>
</div>
<div class="mrow">
    <div class="mc"><div class="v">71.57%</div><div class="l">Test Accuracy</div></div>
    <div class="mc"><div class="v">82.20%</div><div class="l">Val Accuracy</div></div>
    <div class="mc"><div class="v">0.8928</div><div class="l">AUC-ROC</div></div>
    <div class="mc"><div class="v">3,264</div><div class="l">MRI Images</div></div>
    <div class="mc"><div class="v">4</div><div class="l">Tumor Classes</div></div>
    <div class="mc"><div class="v">~7.1M</div><div class="l">Parameters</div></div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    inputs   = tf.keras.Input(shape=IMG_SHAPE)
    x        = tf.keras.layers.Lambda(
                   lambda img: preprocess_input(img),
                   output_shape=IMG_SHAPE,
                   name="efficientnet_preprocessing")(inputs)
    backbone = EfficientNetV2B0(include_top=False, weights=None, input_tensor=x)
    x        = backbone.output
    x        = GlobalAveragePooling2D()(x)
    x        = BatchNormalization()(x)
    x        = Dense(512, activation="relu")(x)
    x        = Dropout(0.5)(x)
    x        = Dense(256, activation="relu")(x)
    x        = Dropout(0.3)(x)
    outputs  = Dense(NUM_CLASSES, activation="softmax")(x)
    model    = Model(inputs=backbone.input, outputs=outputs)
    model.load_weights("best_model_v2_phase2.keras")
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = layer.name
            break
    return model, last_conv

def generate_gradcam(model, img_array, last_conv_layer):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        pred_index  = tf.argmax(predictions[0])
        class_score = predictions[:, pred_index]
    grads        = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out     = conv_outputs[0]
    heatmap      = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), predictions.numpy()[0]

def make_gradcam(img_pil, heatmap):
    img_np          = np.array(img_pil.resize(IMG_SIZE))
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay         = (heatmap_rgb * 0.45 + img_np * 0.55).astype(np.uint8)
    return heatmap_rgb, overlay

# ── Main layout
left, right = st.columns([1, 1.45], gap="large")

with left:
    st.markdown('<div class="panel"><div class="pt">Upload MRI Scan</div></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop your MRI image here", type=["jpg","jpeg","png"], label_visibility="collapsed")

    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        st.image(img_pil, caption="Uploaded MRI Scan", use_column_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        analyse = st.button("🔬  Analyse MRI Scan", use_container_width=True)
        st.markdown("""
        <div class="warn">
            <b>Important:</b> This is a research prototype for academic purposes only.
            It must not replace professional medical diagnosis. Always consult a
            qualified neuroradiologist or physician.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(MRI_PLACEHOLDER, unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;margin-top:14px;">
            <span class="chip" style="color:#ff5252;border-color:rgba(255,82,82,0.4);background:rgba(255,82,82,0.08)">Glioma</span>
            <span class="chip" style="color:#ff9800;border-color:rgba(255,152,0,0.4);background:rgba(255,152,0,0.08)">Meningioma</span>
            <span class="chip" style="color:#ce93d8;border-color:rgba(206,147,216,0.4);background:rgba(206,147,216,0.08)">Pituitary</span>
            <span class="chip" style="color:#00e870;border-color:rgba(0,232,112,0.4);background:rgba(0,232,112,0.08)">No Tumor</span>
        </div>""", unsafe_allow_html=True)
        analyse = False

with right:
    if uploaded and analyse:
        with st.spinner("Analysing scan..."):
            model, last_conv = load_model()
            img_r     = img_pil.resize(IMG_SIZE)
            img_arr   = np.expand_dims(np.array(img_r, dtype=np.float32), axis=0)
            heatmap, probs   = generate_gradcam(model, img_arr, last_conv)
            pred_idx         = int(np.argmax(probs))
            prediction       = CLASS_NAMES[pred_idx]
            confidence       = round(float(probs[pred_idx]) * 100, 2)
            heatmap_rgb, overlay = make_gradcam(img_pil, heatmap)
            info  = CLASS_INFO[prediction]
            color = info["color"]
            is_tumor = prediction != "No Tumor"

        badge_cls = "badge-tumor" if is_tumor else "badge-ok"
        icon      = "⚠" if is_tumor else ""
        sub_msg   = f"Tumor detected — {info['short']}" if is_tumor else info["short"]

        st.markdown(f"""
        <div class="panel">
            <div class="pt">🏥 Diagnosis Result</div>
            <div class="{badge_cls}">
                <div class="bn">{icon}  {prediction}</div>
                <div class="bs">{sub_msg}</div>
            </div>
            <div style="margin-top:18px">
                <div style="font-size:10px;color:#1a3a6a;letter-spacing:1.5px;text-transform:uppercase;font-weight:700;margin-bottom:8px">Model Confidence</div>
                <div style="display:flex;align-items:center;gap:14px">
                    <div style="flex:1;background:rgba(255,255,255,0.04);border-radius:20px;height:12px;overflow:hidden;border:1px solid rgba(255,255,255,0.05)">
                        <div style="width:{confidence}%;height:100%;border-radius:20px;background:linear-gradient(90deg,{color},{color}99)"></div>
                    </div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:28px;font-weight:700;color:{color};min-width:72px;text-align:right">{confidence}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="panel">
            <div class="pt">{info['icon']} About {prediction}</div>
            <div style="display:flex;gap:8px;margin-bottom:12px">
                <span class="chip" style="color:{color};border-color:{color}55;background:{color}11">{info['grade']}</span>
            </div>
            <p style="font-size:13px;color:#5a7a9a;line-height:1.82;margin-bottom:16px">{info['desc']}</p>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
                <div class="info-card" style="border-color:{color}">
                    <div class="ict">Common Symptoms</div>
                    <div class="icb">{info['symptoms']}</div>
                </div>
                <div class="info-card" style="border-color:{color}">
                    <div class="ict">Treatment Options</div>
                    <div class="icb">{info['treatment']}</div>
                </div>
            </div>
            <div class="insight">
                <b>AI Sensitivity for {prediction}:</b>
                <span style="color:{color};font-size:17px;font-weight:700"> {info['sensitivity']}</span>
                &nbsp;·&nbsp;
                <b>Specificity:</b>
                <span style="color:{color};font-size:17px;font-weight:700"> {info['specificity']}</span>
                <br>{info['note']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        prob_html = ""
        for i in range(NUM_CLASSES):
            cls   = CLASS_NAMES[i]
            prob  = round(float(probs[i]) * 100, 2)
            clr   = CLASS_COLORS[cls]
            top   = (i == pred_idx)
            prob_html += f"""
            <div class="pb-row">
                <div class="pb-lbl {'top' if top else ''}">{'▶ ' if top else ''}{cls}</div>
                <div class="pb-bg">
                    <div style="width:{prob}%;height:100%;border-radius:20px;background:linear-gradient(90deg,{clr},{clr}77)"></div>
                </div>
                <div class="pb-val" style="color:{clr if top else '#1a3a5a'}">{prob}%</div>
            </div>"""

        st.markdown(f'<div class="panel"><div class="pt">📊 All Class Probabilities</div>{prob_html}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="panel">
            <div class="pt">🔥 Grad-CAM Explainability</div>
            <p style="font-size:12px;color:#2a4a6a;line-height:1.75;margin-bottom:16px">
                Grad-CAM uses gradients from the final convolutional layer (<b style="color:#8ab8e8">top_conv</b>)
                to generate a heatmap revealing which MRI regions influenced the prediction.
                <b style="color:#ff6060"> Red/yellow = high attention</b> ·
                <b style="color:#5aabff"> Blue = low attention</b>
            </p>
        </div>""", unsafe_allow_html=True)

        g1, g2, g3 = st.columns(3)
        with g1:
            st.image(np.array(img_pil.resize(IMG_SIZE)), use_column_width=True)
            st.markdown('<div class="gc-lbl">Original MRI</div>', unsafe_allow_html=True)
        with g2:
            st.image(heatmap_rgb, use_column_width=True)
            st.markdown('<div class="gc-lbl">Attention Heatmap</div>', unsafe_allow_html=True)
        with g3:
            st.image(overlay, use_column_width=True)
            st.markdown('<div class="gc-lbl">Grad-CAM Overlay</div>', unsafe_allow_html=True)
    else:
        st.markdown(MRI_PLACEHOLDER, unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;margin-top:16px">
            <p style="color:#3a5a7a;font-size:13px;max-width:300px;margin:0 auto 14px;line-height:1.85;">
                Upload a brain MRI on the left and click
                <b style="color:#4f9dff">Analyse MRI Scan</b> to get
                an AI diagnosis with Grad-CAM explainability.
            </p>
            <span class="chip" style="color:#ff5252;border-color:rgba(255,82,82,0.4);background:rgba(255,82,82,0.08)">Glioma</span>
            <span class="chip" style="color:#ff9800;border-color:rgba(255,152,0,0.4);background:rgba(255,152,0,0.08)">Meningioma</span>
            <span class="chip" style="color:#ce93d8;border-color:rgba(206,147,216,0.4);background:rgba(206,147,216,0.08)">Pituitary</span>
            <span class="chip" style="color:#00dc64;border-color:rgba(0,220,100,0.4);background:rgba(0,220,100,0.08)">No Tumor</span>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <div class="fb">Neuro<span>Scan</span> AI</div>
    <div class="fd">
        Developed by <b style="color:#c0d8f8">Stephen Ifeanyi</b> · Princedex ·
        <a href="mailto:ifeanyistephen003@gmail.com">ifeanyistephen003@gmail.com</a>
        <br>
        Final Year Project · EfficientNetV2-B0 · Transfer Learning · Grad-CAM ·
        Kaggle Brain MRI Dataset (3,264 images)
        <br><br>
        Research prototype for academic purposes only — not intended for clinical use.
    </div>
</div>""", unsafe_allow_html=True)