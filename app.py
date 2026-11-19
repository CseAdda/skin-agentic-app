# import os
# import random
# import streamlit as st
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# import torchvision.models as models
# import timm
# from PIL import Image, ImageDraw
# import numpy as np
# import pandas as pd
# import cv2
# from huggingface_hub import hf_hub_download

# # =====================================================================
# # 1. PAGE SETUP & GLOBAL CONFIGURATION
# # =====================================================================
# st.set_page_config(
#     page_title="Fair & Robust Agentic Dermatology AI Hub",
#     page_icon="🔬",
#     layout="wide"
# )

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DISEASE_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# CLASS_NAMES = {
#     'akiec': "Actinic Keratoses / Bowen's Disease",
#     'bcc': "Basal Cell Carcinoma (Malignant)",
#     'bkl': "Benign Keratosis-like Lesions",
#     'df': "Dermatofibroma",
#     'mel': "Melanoma (Invasive Malignancy)",
#     'nv': "Melanocytic Nevi (Benign Mole)",
#     'vasc': "Vascular Lesion"
# }

# ANATOMICAL_SITES = [
#     'anterior torso', 'lower extremity', 'head/neck', 'upper extremity',
#     'posterior torso', 'palms/soles', 'oral/genital', 'lateral torso', 'unknown'
# ]

# FITZPATRICK_TYPES = {
#     "Type I (Very Light / Pale)": 1,
#     "Type II (Light / Fair)": 2,
#     "Type III (Intermediate / Olive)": 3,
#     "Type IV (Moderate Brown / Tan)": 4,
#     "Type V (Dark Brown)": 5,
#     "Type VI (Deeply Pigmented / Dark)": 6
# }

# # =====================================================================
# # 2. MULTI-BACKBONE ARCHITECTURES (ViT-B/16, SWIN, VISION MAMBA)
# # =====================================================================
# class AgenticMultimodalViT(nn.Module):
#     def __init__(self, num_classes=7, meta_dim=19, embed_dim=768, num_heads=8):
#         super().__init__()
#         self.vit = models.vit_b_16(weights=None)
#         self.vit.heads = nn.Identity()
#         self.meta_projector = nn.Sequential(
#             nn.Linear(meta_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim)
#         )
#         self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.classifier = nn.Sequential(
#             nn.Linear(embed_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, images, metadata):
#         v_tokens = self.vit(images).unsqueeze(1)
#         m_tokens = self.meta_projector(metadata).unsqueeze(1)
#         attn_out, attn_weights = self.cross_attention(query=v_tokens, key=m_tokens, value=m_tokens)
#         fused = self.norm1(v_tokens + attn_out).squeeze(1)
#         return self.classifier(fused), attn_weights


# class FairnessMultimodalModernBackbone(nn.Module):
#     def __init__(self, backbone_type="mamba", num_classes=7, meta_dim=19, embed_dim=512, num_heads=8):
#         super().__init__()
#         self.backbone_type = backbone_type
#         if backbone_type == "swin":
#             self.encoder = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
#             vis_dim = self.encoder.num_features
#         else:
#             self.encoder = timm.create_model("mambaout_tiny.in1k", pretrained=False, num_classes=0)
#             vis_dim = self.encoder.num_features  # 576

#         self.vis_dim = vis_dim
#         self.vis_proj = nn.Linear(vis_dim, embed_dim)
#         self.meta_projector = nn.Sequential(
#             nn.Linear(meta_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim)
#         )
#         self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.classifier = nn.Sequential(
#             nn.Linear(embed_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, images, metadata):
#         if self.backbone_type == "mamba":
#             feat = self.encoder.forward_features(images)
#             if feat.ndim == 4:
#                 feat = feat.mean(dim=(1, 2)) if feat.shape[-1] == self.vis_dim else feat.mean(dim=(2, 3))
#             elif feat.ndim == 3:
#                 feat = feat.mean(dim=1) if feat.shape[-1] == self.vis_dim else feat.mean(dim=2)
#         else:
#             feat = self.encoder(images)

#         v_feat = self.vis_proj(feat).unsqueeze(1)
#         m_feat = self.meta_projector(metadata).unsqueeze(1)
#         attn_out, attn_weights = self.cross_attention(query=v_feat, key=m_feat, value=m_feat)
#         fused = self.norm1(v_feat + attn_out).squeeze(1)
#         return self.classifier(fused), attn_weights


# @st.cache_resource
# def load_selected_model(selected_backbone):
#     file_map = {
#         "Multimodal Vision Mamba (Recommended)": ("mamba", "agentic_mamba_best.pth"),
#         "Multimodal Swin Transformer": ("swin", "agentic_swin_best.pth"),
#         "Multimodal ViT-B/16": ("vit", "agentic_vit_best.pth")
#     }
#     b_key, filename = file_map[selected_backbone]

#     if b_key == "vit":
#         model = AgenticMultimodalViT(num_classes=7, meta_dim=19).to(device)
#     else:
#         model = FairnessMultimodalModernBackbone(backbone_type=b_key, num_classes=7, meta_dim=19).to(device)

#     # Check local file first, otherwise fetch from Hugging Face
#     ckpt_path = filename if os.path.exists(filename) else None
#     if ckpt_path is None:
#         try:
#             ckpt_path = hf_hub_download(repo_id="lowanshivinay/skin-lesion", filename=filename)
#         except Exception:
#             ckpt_path = None

#     if ckpt_path is not None:
#         state_dict = torch.load(ckpt_path, map_location=device)
#         model.load_state_dict(state_dict, strict=False)
#         status_msg = f"Loaded `{filename}` successfully."
#     else:
#         status_msg = f"Checkpoint `{filename}` not found locally or on Hugging Face (running in preview mode)."

#     model.eval()
#     return model, status_msg

# # =====================================================================
# # 3. 4 INTELLIGENT AGENTS (VISION, AXIOMATIC XAI, ROBUSTNESS, CLINICAL)
# # =====================================================================
# clean_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def apply_hair_ruler_stress(pil_img):
#     """Simulates real-world clinical noise: hairs, ruler marks, blur, and illumination shifts."""
#     img = pil_img.resize((224, 224)).copy()
#     draw = ImageDraw.Draw(img)
#     rng = random.Random(42)
#     for _ in range(6):
#         x0, y0 = rng.randint(10, 210), rng.randint(10, 210)
#         x1, y1 = x0 + rng.randint(-70, 70), y0 + rng.randint(-70, 70)
#         draw.line([(x0, y0), (x1, y1)], fill=(25, 18, 15), width=2)
#     for x in range(15, 210, 15):
#         tick_h = 16 if (x % 30 == 0) else 9
#         draw.line([(x, 224), (x, 224 - tick_h)], fill=(15, 15, 15), width=2)

#     tf = transforms.Compose([
#         transforms.ColorJitter(brightness=0.35, contrast=0.35),
#         transforms.GaussianBlur(kernel_size=(5, 7), sigma=(0.8, 1.8)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return tf(img)

# def vision_agent(pil_image):
#     """Agent 1: Automated Otsu lesion isolation and binary mask extraction."""
#     np_img = np.array(pil_image)
#     gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
#     blurred = cv2.GaussianBlur(gray, (7, 7), 0)
#     _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest_c = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_c)
#         if w > 40 and h > 40:
#             cropped = Image.fromarray(np_img[y:y+h, x:x+w])
#             crop_mask = (thresh[y:y+h, x:x+w] > 0).astype(np.float32)
#             crop_mask_resized = cv2.resize(crop_mask, (224, 224))
#             return cropped, crop_mask_resized
#     mask_resized = cv2.resize((thresh > 0).astype(np.float32), (224, 224))
#     return pil_image, mask_resized

# def axiomatic_xai_agent(model, cropped_pil, mask_224, meta_tensor, target_idx):
#     """Agent 2: Gradient-based saliency attribution & Lesion Mask Overlap Audit."""
#     inp = clean_transform(cropped_pil).unsqueeze(0).to(device).requires_grad_(True)
#     logits, _ = model(inp, meta_tensor)
#     score = logits[0, target_idx]
#     model.zero_grad()
#     score.backward()

#     grad_map = inp.grad.abs().squeeze().detach().cpu().numpy().mean(axis=0)
#     grad_norm = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-8)

#     inside_energy = float(np.sum(grad_map * mask_224))
#     total_energy = float(np.sum(grad_map)) + 1e-8
#     overlap_pct = (inside_energy / total_energy) * 100.0

#     np_crop = np.array(cropped_pil.resize((224, 224)))
#     heatmap = cv2.applyColorMap(np.uint8(255 * grad_norm), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     overlay = cv2.addWeighted(np_crop, 0.55, heatmap, 0.45, 0)
#     return Image.fromarray(overlay), overlap_pct

# def robustness_agent(model, cropped_image, meta_tensor, clean_pred_idx):
#     """Agent 3: Stress-testing against hair, ruler ticks, blur, and illumination shifts."""
#     stressed_tensor = apply_hair_ruler_stress(cropped_image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         logits, _ = model(stressed_tensor, meta_tensor)
#         probs = F.softmax(logits, dim=1).cpu().squeeze().numpy()
#         stress_pred_idx = int(np.argmax(probs))
#     is_stable = (clean_pred_idx == stress_pred_idx)
#     return is_stable, probs[clean_pred_idx] * 100, probs[stress_pred_idx] * 100, DISEASE_CLASSES[stress_pred_idx]

# def clinical_agent(backbone_name, predicted_class, confidence, entropy, overlap_pct, age, sex, site, fitz_label, is_stable):
#     """Agent 4: Hallucination-free, rule-grounded structured clinical triage report."""
#     if predicted_class in ['mel', 'bcc']:
#         risk = "HIGH RISK (Malignant Suspicion)"
#         action = "Immediate excisional biopsy and urgent histopathology evaluation."
#         badge = "🚨 URGENT"
#     elif predicted_class == 'akiec':
#         risk = "MODERATE RISK (Pre-Cancerous Dysplasia)"
#         action = "Clinical review; consider cryotherapy or topical 5-FU."
#         badge = "⚠️ PRIORITY"
#     else:
#         risk = "LOW RISK (Benign Condition)"
#         action = "Routine surveillance; photograph monitoring at 6-month intervals."
#         badge = "✅ ROUTINE"

#     certainty = "CONFIDENT" if entropy < 0.60 else "BORDERLINE - CLINICAL REVIEW REQUIRED"
#     stability = "STABLE UNDER HAIR/BLUR/LIGHTING SHIFT" if is_stable else "SENSITIVE TO ACQUISITION SHIFT"
#     xai_audit = "VERIFIED (Lesion-Focused)" if overlap_pct >= 45.0 else "REVIEW (Diffuse Background Attribution)"

#     return f"""### **STRUCTURED CLINICAL TRIAGE & AUDIT REPORT**
# * **Triage Priority**: **{badge}**
# * **Active AI Backbone**: `{backbone_name}`
# * **Primary Diagnosis**: **{CLASS_NAMES[predicted_class]}** (`{predicted_class.upper()}`)
# * **Predictive Confidence**: **{confidence:.2f}%**
# * **Risk Stratification**: **{risk}**

# #### **Trustworthiness & Safety Audits**
# * **Uncertainty Calibration**: **{certainty}** (Shannon Entropy: `{entropy:.4f}`)
# * **Axiomatic XAI Mask Overlap**: **{overlap_pct:.1f}%** — `{xai_audit}`
# * **Robustness Stress Audit**: **{stability}**

# #### **Patient & Subgroup Context**
# * **Demographics**: Age `{age}` | Biological Sex: `{sex.capitalize()}`
# * **Anatomical Site**: `{site.capitalize()}`
# * **Fitzpatrick Skin Phototype**: `{fitz_label}`

# #### **Evidence-Based Clinical Action**
# > {action}
# """

# # =====================================================================
# # 4. STREAMLIT NAVIGATION & MULTI-PAGE LAYOUT
# # =====================================================================
# st.sidebar.title("🧭 Navigation & Backbone")
# app_mode = st.sidebar.radio(
#     "Select System Mode:",
#     [
#         "1. Live Multi-Agent Clinical Diagnosis",
#         "2. PhD Multi-Backbone & Fairness Results",
#         "3. Guide-Aligned System Workflow"
#     ]
# )

# selected_backbone = st.sidebar.selectbox(
#     "Select Multimodal Backbone:",
#     [
#         "Multimodal Vision Mamba (Recommended)",
#         "Multimodal Swin Transformer",
#         "Multimodal ViT-B/16"
#     ]
# )
# model, model_status = load_selected_model(selected_backbone)
# st.sidebar.caption(model_status)

# # ---------------------------------------------------------------------
# # MODE 1: LIVE MULTI-AGENT DIAGNOSIS
# # ---------------------------------------------------------------------
# if app_mode == "1. Live Multi-Agent Clinical Diagnosis":
#     st.title("🔬 Fairness-Aware & Robust Multi-Agent Dermatology System")
#     st.write("Upload a dermoscopic lesion image and configure patient metadata + Fitzpatrick skin phototype to execute the 4-agent verification pipeline.")

#     st.sidebar.header("📋 Patient & Skin-Tone Metadata")
#     input_age = st.sidebar.slider("Patient Age", min_value=1, max_value=100, value=45)
#     input_sex = st.sidebar.selectbox("Biological Sex", ["female", "male", "unknown"])
#     input_site = st.sidebar.selectbox("Anatomical Localization", ANATOMICAL_SITES, index=4)
#     input_fitz_label = st.sidebar.selectbox("Fitzpatrick Skin Type (I–VI)", list(FITZPATRICK_TYPES.keys()), index=2)
#     fitz_num = FITZPATRICK_TYPES[input_fitz_label]

#     uploaded_file = st.sidebar.file_uploader("Upload Dermoscopy Image (.jpg, .png)", type=["jpg", "png", "jpeg"])

#     if uploaded_file:
#         raw_image = Image.open(uploaded_file).convert("RGB")

#         # 19-dim Metadata Vector (Age + Sex[3] + Site[9] + Fitzpatrick[6])
#         age_scaled = (input_age - 50.0) / 15.0
#         sex_vec = [1.0 if input_sex == s else 0.0 for s in ['female', 'male', 'unknown']]
#         site_vec = [1.0 if input_site == s else 0.0 for s in ANATOMICAL_SITES]
#         fitz_vec = [1.0 if fitz_num == f else 0.0 for f in [1, 2, 3, 4, 5, 6]]
#         meta_feats = np.array([age_scaled] + sex_vec + site_vec + fitz_vec, dtype=np.float32)
#         meta_tensor = torch.tensor(meta_feats).unsqueeze(0).to(device)

#         with st.spinner("Executing 4-Agent Trust & Verification Pipeline..."):
#             # Agent 1: Vision Agent
#             cropped_image, lesion_mask = vision_agent(raw_image)

#             # Multimodal Inference
#             img_tensor = clean_transform(cropped_image).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 logits, _ = model(img_tensor, meta_tensor)
#                 probs = F.softmax(logits, dim=1).cpu().squeeze().numpy()
#                 pred_idx = int(np.argmax(probs))
#                 pred_class = DISEASE_CLASSES[pred_idx]
#                 pred_conf = probs[pred_idx] * 100
#                 entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

#             # Agent 2: Axiomatic XAI Agent
#             xai_map, overlap_pct = axiomatic_xai_agent(model, cropped_image, lesion_mask, meta_tensor, pred_idx)

#             # Agent 3: Robustness Agent (Hair + Ruler + Blur + Color Shift)
#             is_stable, retained_conf, stress_conf, stress_class = robustness_agent(
#                 model, cropped_image, meta_tensor, pred_idx
#             )

#             # Agent 4: Hallucination-Free Clinical Agent
#             report = clinical_agent(
#                 selected_backbone, pred_class, pred_conf, entropy, overlap_pct,
#                 input_age, input_sex, input_site, input_fitz_label, is_stable
#             )

#         c1, c2, c3 = st.columns([1, 1, 1.25])
#         with c1:
#             st.subheader("1. Vision & XAI Agents")
#             st.image(raw_image, caption="Original Dermoscopy Image", use_container_width=True)
#             st.image(cropped_image, caption="Vision Agent: Isolated Lesion", use_container_width=True)
#             st.image(xai_map, caption=f"Axiomatic XAI Map (Lesion Focus: {overlap_pct:.1f}%)", use_container_width=True)

#         with c2:
#             st.subheader("2. Uncertainty & Robustness")
#             st.metric("Predicted Pathology", pred_class.upper(), f"{pred_conf:.1f}% Confidence")
#             st.metric("Shannon Entropy", f"{entropy:.4f}", "Well-Calibrated" if entropy < 0.6 else "High Uncertainty", delta_color="inverse")
#             st.metric("XAI Lesion Mask Overlap", f"{overlap_pct:.1f}%", "Focused on Lesion" if overlap_pct >= 45 else "Check Attribution")

#             st.markdown("#### Hair, Ruler & Optical Stress Audit")
#             if is_stable:
#                 st.success(f"**Robust & Stable**: Diagnosis preserved under simulated hairs, ruler marks & blur ({retained_conf:.1f}% retained).")
#             else:
#                 st.error(f"**Perturbation Shift**: Shifted to `{stress_class.upper()}` under stress ({stress_conf:.1f}% confidence).")

#             st.bar_chart({CLASS_NAMES[c]: float(probs[i]) for i, c in enumerate(DISEASE_CLASSES)})

#         with c3:
#             st.subheader("3. Clinical Triage Agent")
#             st.markdown(report)
#             st.download_button("📥 Download Verified Triage Report", report, file_name="Clinical_Triage_Audit.txt")
#     else:
#         st.info("👈 Upload a dermoscopy image from the sidebar to run the multi-agent diagnostic and fairness audit pipeline.")

# # ---------------------------------------------------------------------
# # MODE 2: PhD BENCHMARK & FAIRNESS RESULTS (CLEAN 60/20/20 PROTOCOL)
# # ---------------------------------------------------------------------
# elif app_mode == "2. PhD Multi-Backbone & Fairness Results":
#     st.title("📊 PhD Experimental Benchmarks (60/20/20 Holdout Protocol)")
#     st.markdown("Evaluated on **HAM10000** using a strict stratified **60% Train / 20% Test / 20% Independent Holdout** split with **Group-DRO Fairness Regularization**.")

#     st.subheader("1. Multi-Backbone Architectural Comparison (Modern Multimodal Transformers & SSM)")
#     backbone_df = pd.DataFrame({
#         "Multimodal Backbone": [
#             "Multimodal ViT-B/16",
#             "Multimodal Swin Transformer",
#             "Multimodal Vision Mamba (Best)"
#         ],
#         "Clean Acc (%)": [86.57, 86.02, 86.77],
#         "Balanced Acc (%)": [73.70, 71.32, 78.91],
#         "Macro F1-Score": [0.7690, 0.7476, 0.7761],
#         "AUC-ROC": [0.9633, 0.9718, 0.9706],
#         "Stress Acc (%)": [79.23, 70.84, 73.24],
#         "XAI Mask Overlap (%)": ["—", "49.89%", "48.50%"]
#     })
#     st.table(backbone_df)
#     st.success("Key Finding: Multimodal Vision Mamba achieves the highest Clean Accuracy (86.77%), Balanced Accuracy (+5.21% over ViT, +7.59% over Swin), and Macro F1 (0.7761).")

#     st.subheader("2. Skin-Tone Fairness & Calibration Audit (Fitzpatrick I–VI)")
#     fairness_df = pd.DataFrame({
#         "Demographic Subgroup": [
#             "Lighter Skin Phototypes (Fitzpatrick I–III)",
#             "Darker Skin Phototypes (Fitzpatrick IV–VI)",
#             "Subgroup Disparity Gap (|Light - Dark|)"
#         ],
#         "Vision Mamba Accuracy (%)": ["86.76%", "87.10%", "0.34% (Near-Zero Bias)"],
#         "Calibration (ECE)": ["0.0718 (Overall)", "0.0718 (Overall)", "Well-Calibrated"],
#         "Mean Predictive Entropy": ["0.2916", "0.2916", "High Certainty"]
#     })
#     st.table(fairness_df)
#     st.info("Group-DRO fairness optimization eliminates diagnostic degradation on darker skin tones (87.10% vs 86.76%).")

#     st.subheader("3. Class-Wise Clinical Performance (Independent Holdout Test Set)")
#     class_report_data = {
#         "Disease Code": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
#         "Full Clinical Condition": [CLASS_NAMES[c] for c in DISEASE_CLASSES],
#         "Precision": [0.6812, 0.7157, 0.7464, 1.0000, 0.6900, 0.9315, 0.9524],
#         "Recall (Sensitivity)": [0.7231, 0.7087, 0.7091, 0.6522, 0.7085, 0.9433, 0.7143],
#         "F1-Score": [0.7015, 0.7122, 0.7273, 0.7895, 0.6991, 0.9374, 0.8163],
#         "Holdout Support": [65, 103, 220, 23, 223, 1341, 28]
#     }
#     st.dataframe(pd.DataFrame(class_report_data), use_container_width=True)

# # ---------------------------------------------------------------------
# # MODE 3: GUIDE-ALIGNED WORKFLOW
# # ---------------------------------------------------------------------
# elif app_mode == "3. Guide-Aligned System Workflow":
#     st.title("🧩 Guide-Aligned Multi-Agent Trustworthy Workflow")
#     st.markdown("""
#     This architecture implements all four revised PhD objectives:
#     1. **Vision Agent (Fairness-Aware Multimodal Backbone):** Isolates the active lesion from background skin tone via Otsu segmentation and fuses visual representations (Vision Mamba / Swin / ViT-B/16) with clinical metadata + Fitzpatrick I–VI via Cross-Attention under Group-DRO fairness loss.
#     2. **Axiomatic XAI Agent:** Computes gradient-based pixel attributions and audits the **Lesion Mask Overlap Ratio** to prove predictions rely on pathology rather than skin tone or artifacts.
#     3. **Robustness Agent:** Stress-tests model stability against simulated hairs, dermoscopic ruler ticks, optical blur, and lighting shifts under a strict 60/20/20 protocol.
#     4. **Uncertainty & Clinical Agent:** Evaluates subgroup-calibrated confidence (Shannon Entropy & ECE) and synthesizes hallucination-free triage reports.
#     """)
#     st.code("""
#     [Dermoscopic Image + Metadata (Age, Sex, Site, Fitzpatrick I-VI)]
#                                    │
#                                    ▼
#       [ 1. Vision Agent ] ──► Lesion Isolation + Vision Mamba / Swin / ViT + Group-DRO
#                                    │
#                    ┌───────────────┴───────────────┐
#                    ▼                               ▼
#       [ 2A. Axiomatic XAI Agent ]     [ 2B. Robustness Stress Agent ]
#        • Gradient Attribution Map      • Synthetic Hairs & Ruler Ticks
#        • Lesion Mask Overlap %         • Blur & Illumination Shift Audit
#                    │                               │
#                    └───────────────┬───────────────┘
#                                    ▼
#                  [ 3. Subgroup Uncertainty & Calibration ]
#                   • Shannon Entropy H(p) & ECE Check
#                                    │
#                                    ▼
#                  [ 4. Hallucination-Free Clinical Agent ]
#                   • Risk Tiering & Structured Triage Report
#     """, language="text")


# update 1

# import os
# import random
# import streamlit as st
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# import torchvision.models as models
# import timm
# from PIL import Image, ImageDraw
# import numpy as np
# import pandas as pd
# import cv2
# from huggingface_hub import hf_hub_download

# # =====================================================================
# # 1. PAGE SETUP & GLOBAL CONFIGURATION
# # =====================================================================
# st.set_page_config(
#     page_title="Fair & Robust Agentic Dermatology AI Hub",
#     page_icon="🔬",
#     layout="wide"
# )

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DISEASE_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
# CLASS_NAMES = {
#     'akiec': "Actinic Keratoses / Bowen's Disease",
#     'bcc': "Basal Cell Carcinoma (Malignant)",
#     'bkl': "Benign Keratosis-like Lesions",
#     'df': "Dermatofibroma",
#     'mel': "Melanoma (Invasive Malignancy)",
#     'nv': "Melanocytic Nevi (Benign Mole)",
#     'vasc': "Vascular Lesion"
# }

# ANATOMICAL_SITES = [
#     'anterior torso', 'lower extremity', 'head/neck', 'upper extremity',
#     'posterior torso', 'palms/soles', 'oral/genital', 'lateral torso', 'unknown'
# ]

# FITZPATRICK_TYPES = {
#     "Type I (Very Light / Pale)": 1,
#     "Type II (Light / Fair)": 2,
#     "Type III (Intermediate / Olive)": 3,
#     "Type IV (Moderate Brown / Tan)": 4,
#     "Type V (Dark Brown)": 5,
#     "Type VI (Deeply Pigmented / Dark)": 6
# }

# # =====================================================================
# # 2. MULTI-BACKBONE ARCHITECTURES (ViT-B/16, SWIN, VISION MAMBA)
# # =====================================================================
# class AgenticMultimodalViT(nn.Module):
#     def __init__(self, num_classes=7, meta_dim=19, embed_dim=768, num_heads=8):
#         super().__init__()
#         self.vit = models.vit_b_16(weights=None)
#         self.vit.heads = nn.Identity()
#         self.meta_projector = nn.Sequential(
#             nn.Linear(meta_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim)
#         )
#         self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.classifier = nn.Sequential(
#             nn.Linear(embed_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, images, metadata):
#         v_tokens = self.vit(images).unsqueeze(1)
#         m_tokens = self.meta_projector(metadata).unsqueeze(1)
#         attn_out, attn_weights = self.cross_attention(query=v_tokens, key=m_tokens, value=m_tokens)
#         fused = self.norm1(v_tokens + attn_out).squeeze(1)
#         return self.classifier(fused), attn_weights


# class FairnessMultimodalModernBackbone(nn.Module):
#     def __init__(self, backbone_type="mamba", num_classes=7, meta_dim=19, embed_dim=512, num_heads=8):
#         super().__init__()
#         self.backbone_type = backbone_type
#         if backbone_type == "swin":
#             self.encoder = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
#             vis_dim = self.encoder.num_features
#         else:
#             self.encoder = timm.create_model("mambaout_tiny.in1k", pretrained=False, num_classes=0)
#             vis_dim = self.encoder.num_features  # 576

#         self.vis_dim = vis_dim
#         self.vis_proj = nn.Linear(vis_dim, embed_dim)
#         self.meta_projector = nn.Sequential(
#             nn.Linear(meta_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
#             nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim)
#         )
#         self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.classifier = nn.Sequential(
#             nn.Linear(embed_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, images, metadata):
#         if self.backbone_type == "mamba":
#             feat = self.encoder.forward_features(images)
#             if feat.ndim == 4:
#                 feat = feat.mean(dim=(1, 2)) if feat.shape[-1] == self.vis_dim else feat.mean(dim=(2, 3))
#             elif feat.ndim == 3:
#                 feat = feat.mean(dim=1) if feat.shape[-1] == self.vis_dim else feat.mean(dim=2)
#         else:
#             feat = self.encoder(images)

#         v_feat = self.vis_proj(feat).unsqueeze(1)
#         m_feat = self.meta_projector(metadata).unsqueeze(1)
#         attn_out, attn_weights = self.cross_attention(query=v_feat, key=m_feat, value=m_feat)
#         fused = self.norm1(v_feat + attn_out).squeeze(1)
#         return self.classifier(fused), attn_weights


# @st.cache_resource
# def load_selected_model(selected_backbone):
#     file_map = {
#         "Multimodal Vision Mamba (Recommended)": ("mamba", "agentic_mamba_best.pth"),
#         "Multimodal Swin Transformer": ("swin", "agentic_swin_best.pth"),
#         "Multimodal ViT-B/16": ("vit", "agentic_vit_best.pth")
#     }
#     b_key, filename = file_map[selected_backbone]

#     if b_key == "vit":
#         model = AgenticMultimodalViT(num_classes=7, meta_dim=19).to(device)
#     else:
#         model = FairnessMultimodalModernBackbone(backbone_type=b_key, num_classes=7, meta_dim=19).to(device)

#     ckpt_path = filename if os.path.exists(filename) else None
#     if ckpt_path is None:
#         try:
#             ckpt_path = hf_hub_download(repo_id="lowanshivinay/skin-lesion", filename=filename)
#         except Exception:
#             ckpt_path = None

#     if ckpt_path is not None:
#         state_dict = torch.load(ckpt_path, map_location=device)
#         model.load_state_dict(state_dict, strict=False)
#         status_msg = f"Loaded `{filename}` successfully."
#     else:
#         status_msg = f"Checkpoint `{filename}` not found locally or on Hugging Face."

#     model.eval()
#     return model, status_msg

# # =====================================================================
# # 3. 4 AGENTS + CROSS-AGENT LOGICAL CONSISTENCY ARBITRATOR
# # =====================================================================
# clean_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# def apply_hair_ruler_stress(pil_img):
#     """Simulates real-world clinical noise: hairs, ruler marks, blur, and illumination shifts."""
#     img = pil_img.resize((224, 224)).copy()
#     draw = ImageDraw.Draw(img)
#     rng = random.Random(42)
#     for _ in range(6):
#         x0, y0 = rng.randint(10, 210), rng.randint(10, 210)
#         x1, y1 = x0 + rng.randint(-70, 70), y0 + rng.randint(-70, 70)
#         draw.line([(x0, y0), (x1, y1)], fill=(25, 18, 15), width=2)
#     for x in range(15, 210, 15):
#         tick_h = 16 if (x % 30 == 0) else 9
#         draw.line([(x, 224), (x, 224 - tick_h)], fill=(15, 15, 15), width=2)

#     tf = transforms.Compose([
#         transforms.ColorJitter(brightness=0.35, contrast=0.35),
#         transforms.GaussianBlur(kernel_size=(5, 7), sigma=(0.8, 1.8)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     return tf(img)

# def vision_agent(pil_image):
#     """Agent 1: Automated Otsu lesion isolation and binary mask extraction."""
#     np_img = np.array(pil_image)
#     gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
#     blurred = cv2.GaussianBlur(gray, (7, 7), 0)
#     _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest_c = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_c)
#         if w > 40 and h > 40:
#             cropped = Image.fromarray(np_img[y:y+h, x:x+w])
#             crop_mask = (thresh[y:y+h, x:x+w] > 0).astype(np.float32)
#             crop_mask_resized = cv2.resize(crop_mask, (224, 224))
#             return cropped, crop_mask_resized
#     mask_resized = cv2.resize((thresh > 0).astype(np.float32), (224, 224))
#     return pil_image, mask_resized

# def axiomatic_xai_agent(model, cropped_pil, mask_224, meta_tensor, target_idx):
#     """Agent 2: Gradient-based saliency attribution & Lesion Mask Overlap Audit."""
#     inp = clean_transform(cropped_pil).unsqueeze(0).to(device).requires_grad_(True)
#     logits, _ = model(inp, meta_tensor)
#     score = logits[0, target_idx]
#     model.zero_grad()
#     score.backward()

#     grad_map = inp.grad.abs().squeeze().detach().cpu().numpy().mean(axis=0)
#     grad_norm = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-8)

#     inside_energy = float(np.sum(grad_map * mask_224))
#     total_energy = float(np.sum(grad_map)) + 1e-8
#     overlap_pct = (inside_energy / total_energy) * 100.0

#     np_crop = np.array(cropped_pil.resize((224, 224)))
#     heatmap = cv2.applyColorMap(np.uint8(255 * grad_norm), cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     overlay = cv2.addWeighted(np_crop, 0.55, heatmap, 0.45, 0)
#     return Image.fromarray(overlay), overlap_pct

# def robustness_agent(model, cropped_image, meta_tensor, clean_pred_idx):
#     """Agent 3: Stress-testing against hair, ruler ticks, blur, and illumination shifts."""
#     stressed_tensor = apply_hair_ruler_stress(cropped_image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         logits, _ = model(stressed_tensor, meta_tensor)
#         probs = F.softmax(logits, dim=1).cpu().squeeze().numpy()
#         stress_pred_idx = int(np.argmax(probs))
#     is_stable = (clean_pred_idx == stress_pred_idx)
#     return is_stable, probs[clean_pred_idx] * 100, probs[stress_pred_idx] * 100, DISEASE_CLASSES[stress_pred_idx]

# def consistency_and_clinical_agent(
#     backbone_name, predicted_class, confidence, entropy,
#     overlap_pct, age, sex, site, fitz_label, is_stable, stress_class
# ):
#     """
#     Agent 4 + Cross-Agent Logical Consistency Arbitrator:
#     Enforces logical agreement across Vision, XAI, Robustness, and Clinical Priors.
#     """
#     inconsistencies = []

#     # Rule 1: XAI Grounding Check
#     if confidence > 80.0 and overlap_pct < 40.0:
#         inconsistencies.append(
#             f"XAI Grounding Mismatch: High confidence ({confidence:.1f}%) relies heavily on non-lesion background (Overlap {overlap_pct:.1f}% < 40%)."
#         )

#     # Rule 2: Robustness-Certainty Consistency
#     if not is_stable:
#         inconsistencies.append(
#             f"Acquisition Stability Conflict: Diagnosis shifts from {predicted_class.upper()} to {stress_class.upper()} under hair/blur/lighting perturbation."
#         )

#     # Rule 3: Epidemiological / Age-Pathology Consistency Check
#     if predicted_class in ["akiec", "bcc"] and age < 30:
#         inconsistencies.append(
#             f"Demographic Atypicality: {predicted_class.upper()} is clinically rare at age {age}; rule out benign inflammatory look-alikes (e.g., psoriasis, eczema, or dermatitis)."
#         )

#     # Rule 4: Unified Trust Consensus Score (0 to 100%)
#     norm_certainty = max(0.0, 1.0 - (entropy / 1.9459))
#     norm_overlap = min(1.0, overlap_pct / 60.0)
#     stability_score = 1.0 if is_stable else 0.3
#     trust_consensus = (0.35 * norm_certainty + 0.35 * norm_overlap + 0.30 * stability_score) * 100.0

#     # Base Clinical Stratification
#     if predicted_class in ['mel', 'bcc']:
#         risk = "HIGH RISK (Malignant Suspicion)"
#         action = "Immediate excisional biopsy and urgent histopathology evaluation."
#         badge = "🚨 URGENT"
#     elif predicted_class == 'akiec':
#         risk = "MODERATE RISK (Pre-Cancerous Dysplasia)"
#         action = "Clinical review; consider cryotherapy or topical 5-FU."
#         badge = "⚠️ PRIORITY"
#     else:
#         risk = "LOW RISK (Benign Condition)"
#         action = "Routine surveillance; photograph monitoring at 6-month intervals."
#         badge = "✅ ROUTINE"

#     if inconsistencies or trust_consensus < 70.0:
#         consensus_status = f"⚠️ CONFLICT DETECTED ({len(inconsistencies)} rule flag) — DERMATOLOGIST OVERRIDE ADVISED"
#         action = f"Hold autonomous triage due to cross-agent consistency alert. {action}"
#     else:
#         consensus_status = "✅ FULLY CONSISTENT ACROSS ALL 4 AGENTS"

#     conflict_block = "\n".join([f"  - {item}" for item in inconsistencies]) if inconsistencies else "  - None (All agents logically aligned)"

#     report_md = f"""### **STRUCTURED CLINICAL TRIAGE & CONSISTENCY REPORT**
# * **Cross-Agent Consensus**: **{consensus_status}**
# * **Unified Trust Consensus Score**: **{trust_consensus:.1f} / 100**
# * **Triage Priority**: **{badge}**
# * **Active AI Backbone**: `{backbone_name}`
# * **Primary Diagnosis**: **{CLASS_NAMES[predicted_class]}** (`{predicted_class.upper()}`) — `{confidence:.2f}%`
# * **Risk Stratification**: **{risk}**

# #### **Cross-Agent Logical Verification**
# * **Certainty vs. Entropy**: `{'CONFIDENT' if entropy < 0.60 else 'UNCERTAIN'}` (Shannon Entropy: `{entropy:.4f}`)
# * **Vision vs. XAI Grounding**: `{overlap_pct:.1f}%` inside lesion (`{'VERIFIED' if overlap_pct >= 40.0 else 'UNGROUNDED'}`)
# * **Robustness vs. Stability**: `{'PRESERVED' if is_stable else 'SHIFTED TO ' + stress_class.upper()}`
# * **Detected Logical Flags**:
# {conflict_block}

# #### **Patient & Subgroup Context**
# * **Demographics**: Age `{age}` | Biological Sex: `{sex.capitalize()}` | Site: `{site.capitalize()}`
# * **Fitzpatrick Phototype**: `{fitz_label}`

# #### **Evidence-Based Clinical Action**
# > {action}
# """
#     return report_md, trust_consensus, inconsistencies

# # =====================================================================
# # 4. STREAMLIT NAVIGATION & MULTI-PAGE LAYOUT
# # =====================================================================
# st.sidebar.title("🧭 Navigation & Backbone")
# app_mode = st.sidebar.radio(
#     "Select System Mode:",
#     [
#         "1. Live Multi-Agent Clinical Diagnosis",
#         "2. PhD Multi-Backbone & Fairness Results",
#         "3. Guide-Aligned System Workflow"
#     ]
# )

# selected_backbone = st.sidebar.selectbox(
#     "Select Multimodal Backbone:",
#     [
#         "Multimodal Vision Mamba (Recommended)",
#         "Multimodal Swin Transformer",
#         "Multimodal ViT-B/16"
#     ]
# )
# model, model_status = load_selected_model(selected_backbone)
# st.sidebar.caption(model_status)

# # ---------------------------------------------------------------------
# # MODE 1: LIVE MULTI-AGENT DIAGNOSIS
# # ---------------------------------------------------------------------
# if app_mode == "1. Live Multi-Agent Clinical Diagnosis":
#     st.title("🔬 Fairness-Aware & Logically Consistent Multi-Agent Dermatology System")
#     st.write("Upload a dermoscopic lesion image and configure patient metadata + Fitzpatrick skin phototype to execute the 4-agent verification and consistency pipeline.")

#     st.sidebar.header("📋 Patient & Skin-Tone Metadata")
#     input_age = st.sidebar.slider("Patient Age", min_value=1, max_value=100, value=45)
#     input_sex = st.sidebar.selectbox("Biological Sex", ["female", "male", "unknown"])
#     input_site = st.sidebar.selectbox("Anatomical Localization", ANATOMICAL_SITES, index=4)
#     input_fitz_label = st.sidebar.selectbox("Fitzpatrick Skin Type (I–VI)", list(FITZPATRICK_TYPES.keys()), index=2)
#     fitz_num = FITZPATRICK_TYPES[input_fitz_label]

#     uploaded_file = st.sidebar.file_uploader("Upload Dermoscopy Image (.jpg, .png)", type=["jpg", "png", "jpeg"])

#     if uploaded_file:
#         raw_image = Image.open(uploaded_file).convert("RGB")

#         age_scaled = (input_age - 50.0) / 15.0
#         sex_vec = [1.0 if input_sex == s else 0.0 for s in ['female', 'male', 'unknown']]
#         site_vec = [1.0 if input_site == s else 0.0 for s in ANATOMICAL_SITES]
#         fitz_vec = [1.0 if fitz_num == f else 0.0 for f in [1, 2, 3, 4, 5, 6]]
#         meta_feats = np.array([age_scaled] + sex_vec + site_vec + fitz_vec, dtype=np.float32)
#         meta_tensor = torch.tensor(meta_feats).unsqueeze(0).to(device)

#         with st.spinner("Executing 4-Agent Trust & Cross-Agent Consistency Pipeline..."):
#             cropped_image, lesion_mask = vision_agent(raw_image)

#             img_tensor = clean_transform(cropped_image).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 logits, _ = model(img_tensor, meta_tensor)
#                 probs = F.softmax(logits, dim=1).cpu().squeeze().numpy()
#                 pred_idx = int(np.argmax(probs))
#                 pred_class = DISEASE_CLASSES[pred_idx]
#                 pred_conf = probs[pred_idx] * 100
#                 entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

#             xai_map, overlap_pct = axiomatic_xai_agent(model, cropped_image, lesion_mask, meta_tensor, pred_idx)

#             is_stable, retained_conf, stress_conf, stress_class = robustness_agent(
#                 model, cropped_image, meta_tensor, pred_idx
#             )

#             report, trust_consensus, inconsistencies = consistency_and_clinical_agent(
#                 selected_backbone, pred_class, pred_conf, entropy, overlap_pct,
#                 input_age, input_sex, input_site, input_fitz_label, is_stable, stress_class
#             )

#         c1, c2, c3 = st.columns([1, 1, 1.25])
#         with c1:
#             st.subheader("1. Vision & XAI Agents")
#             st.image(raw_image, caption="Original Dermoscopy Image", use_container_width=True)
#             st.image(cropped_image, caption="Vision Agent: Isolated Lesion", use_container_width=True)
#             st.image(xai_map, caption=f"Axiomatic XAI Map (Lesion Focus: {overlap_pct:.1f}%)", use_container_width=True)

#         with c2:
#             st.subheader("2. Uncertainty & Consensus")
#             st.metric("Predicted Pathology", pred_class.upper(), f"{pred_conf:.1f}% Confidence")
#             st.metric("Unified Trust Consensus", f"{trust_consensus:.1f}%", "Consistent" if not inconsistencies else "Conflict Flagged")
#             st.metric("Shannon Entropy", f"{entropy:.4f}", "Well-Calibrated" if entropy < 0.6 else "High Uncertainty", delta_color="inverse")

#             st.markdown("#### Hair, Ruler & Optical Stress Audit")
#             if is_stable:
#                 st.success(f"**Robust & Stable**: Diagnosis preserved under simulated hairs, ruler marks & blur ({retained_conf:.1f}% retained).")
#             else:
#                 st.error(f"**Perturbation Shift**: Shifted to `{stress_class.upper()}` under stress ({stress_conf:.1f}% confidence).")

#             st.bar_chart({CLASS_NAMES[c]: float(probs[i]) for i, c in enumerate(DISEASE_CLASSES)})

#         with c3:
#             st.subheader("3. Clinical & Consistency Agent")
#             if inconsistencies:
#                 st.warning(f"Cross-Agent Consistency Arbitrator flagged {len(inconsistencies)} logical conflict(s) for clinician review.")
#             st.markdown(report)
#             st.download_button("📥 Download Verified Triage Report", report, file_name="Clinical_Triage_Audit.txt")
#     else:
#         st.info("👈 Upload a dermoscopy image from the sidebar to run the multi-agent diagnostic and consistency audit pipeline.")

# # ---------------------------------------------------------------------
# # MODE 2: PhD BENCHMARK & FAIRNESS RESULTS (CLEAN 60/20/20 PROTOCOL)
# # ---------------------------------------------------------------------
# elif app_mode == "2. PhD Multi-Backbone & Fairness Results":
#     st.title("📊 PhD Experimental Benchmarks (60/20/20 Holdout Protocol)")
#     st.markdown("Evaluated on **HAM10000** using a strict stratified **60% Train / 20% Test / 20% Independent Holdout** split with **Group-DRO Fairness Regularization**.")

#     st.subheader("1. Multi-Backbone Architectural Comparison (Modern Multimodal Transformers & SSM)")
#     backbone_df = pd.DataFrame({
#         "Multimodal Backbone": [
#             "Multimodal ViT-B/16",
#             "Multimodal Swin Transformer",
#             "Multimodal Vision Mamba (Best)"
#         ],
#         "Clean Acc (%)": [86.57, 86.02, 86.77],
#         "Balanced Acc (%)": [73.70, 71.32, 78.91],
#         "Macro F1-Score": [0.7690, 0.7476, 0.7761],
#         "AUC-ROC": [0.9633, 0.9718, 0.9706],
#         "Stress Acc (%)": [79.23, 70.84, 73.24],
#         "XAI Mask Overlap (%)": ["—", "49.89%", "48.50%"]
#     })
#     st.table(backbone_df)
#     st.success("Key Finding: Multimodal Vision Mamba achieves the highest Clean Accuracy (86.77%), Balanced Accuracy (+5.21% over ViT, +7.59% over Swin), and Macro F1 (0.7761).")

#     st.subheader("2. Skin-Tone Fairness & Calibration Audit (Fitzpatrick I–VI)")
#     fairness_df = pd.DataFrame({
#         "Demographic Subgroup": [
#             "Lighter Skin Phototypes (Fitzpatrick I–III)",
#             "Darker Skin Phototypes (Fitzpatrick IV–VI)",
#             "Subgroup Disparity Gap (|Light - Dark|)"
#         ],
#         "Vision Mamba Accuracy (%)": ["86.76%", "87.10%", "0.34% (Near-Zero Bias)"],
#         "Calibration (ECE)": ["0.0718 (Overall)", "0.0718 (Overall)", "Well-Calibrated"],
#         "Mean Predictive Entropy": ["0.2916", "0.2916", "High Certainty"]
#     })
#     st.table(fairness_df)
#     st.info("Group-DRO fairness optimization eliminates diagnostic degradation on darker skin tones (87.10% vs 86.76%).")

#     st.subheader("3. Class-Wise Clinical Performance (Independent Holdout Test Set)")
#     class_report_data = {
#         "Disease Code": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
#         "Full Clinical Condition": [CLASS_NAMES[c] for c in DISEASE_CLASSES],
#         "Precision": [0.6812, 0.7157, 0.7464, 1.0000, 0.6900, 0.9315, 0.9524],
#         "Recall (Sensitivity)": [0.7231, 0.7087, 0.7091, 0.6522, 0.7085, 0.9433, 0.7143],
#         "F1-Score": [0.7015, 0.7122, 0.7273, 0.7895, 0.6991, 0.9374, 0.8163],
#         "Holdout Support": [65, 103, 220, 23, 223, 1341, 28]
#     }
#     st.dataframe(pd.DataFrame(class_report_data), use_container_width=True)

# # ---------------------------------------------------------------------
# # MODE 3: GUIDE-ALIGNED WORKFLOW
# # ---------------------------------------------------------------------
# elif app_mode == "3. Guide-Aligned System Workflow":
#     st.title("🧩 Guide-Aligned Multi-Agent Trustworthy Workflow")
#     st.markdown("""
#     This architecture implements all four revised PhD objectives with Cross-Agent Logical Consistency:
#     1. **Vision Agent (Fairness-Aware Multimodal Backbone):** Isolates the active lesion from background skin tone via Otsu segmentation and fuses visual representations (Vision Mamba / Swin / ViT-B/16) with clinical metadata + Fitzpatrick I–VI via Cross-Attention under Group-DRO fairness loss.
#     2. **Axiomatic XAI Agent:** Computes gradient-based pixel attributions and audits the **Lesion Mask Overlap Ratio** to prove predictions rely on pathology rather than skin tone or artifacts.
#     3. **Robustness Agent:** Stress-tests model stability against simulated hairs, dermoscopic ruler ticks, optical blur, and lighting shifts under a strict 60/20/20 protocol.
#     4. **Logical Consistency Arbitrator & Clinical Agent:** Enforces cross-agent rules (XAI grounding, acquisition stability, and demographic sanity checks) to compute a **Unified Trust Consensus Score** before generating hallucination-free triage reports.
#     """)
#     st.code("""
#     [Dermoscopic Image + Metadata (Age, Sex, Site, Fitzpatrick I-VI)]
#                                    │
#                                    ▼
#       [ 1. Vision Agent ] ──► Lesion Isolation + Vision Mamba / Swin / ViT + Group-DRO
#                                    │
#                    ┌───────────────┴───────────────┐
#                    ▼                               ▼
#       [ 2A. Axiomatic XAI Agent ]     [ 2B. Robustness Stress Agent ]
#        • Gradient Attribution Map      • Synthetic Hairs & Ruler Ticks
#        • Lesion Mask Overlap %         • Blur & Illumination Shift Audit
#                    │                               │
#                    └───────────────┬───────────────┘
#                                    ▼
#             [ 3. Cross-Agent Logical Consistency Arbitrator ]
#              • XAI Grounding + Stability + Demographic Sanity Rules
#              • Unified Trust Consensus Score (0 - 100%)
#                                    │
#                                    ▼
#                  [ 4. Hallucination-Free Clinical Agent ]
#                   • Risk Tiering & Structured Triage Report
#     """, language="text")



# update 2 weighted average


import os
import random
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import timm
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import cv2
from huggingface_hub import hf_hub_download

# =====================================================================
# 1. PAGE SETUP & GLOBAL CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Fair & Robust Agentic Dermatology AI Hub",
    page_icon="🔬",
    layout="wide"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DISEASE_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_NAMES = {
    'akiec': "Actinic Keratoses / Bowen's Disease",
    'bcc': "Basal Cell Carcinoma (Malignant)",
    'bkl': "Benign Keratosis-like Lesions",
    'df': "Dermatofibroma",
    'mel': "Melanoma (Invasive Malignancy)",
    'nv': "Melanocytic Nevi (Benign Mole)",
    'vasc': "Vascular Lesion"
}

ANATOMICAL_SITES = [
    'anterior torso', 'lower extremity', 'head/neck', 'upper extremity',
    'posterior torso', 'palms/soles', 'oral/genital', 'lateral torso', 'unknown'
]

FITZPATRICK_TYPES = {
    "Type I (Very Light / Pale)": 1,
    "Type II (Light / Fair)": 2,
    "Type III (Intermediate / Olive)": 3,
    "Type IV (Moderate Brown / Tan)": 4,
    "Type V (Dark Brown)": 5,
    "Type VI (Deeply Pigmented / Dark)": 6
}

# =====================================================================
# 2. MULTI-BACKBONE ARCHITECTURES (VISION MAMBA, SWIN, ViT-B/16)
# =====================================================================
class AgenticMultimodalViT(nn.Module):
    def __init__(self, num_classes=7, meta_dim=19, embed_dim=768, num_heads=8):
        super().__init__()
        self.vit = models.vit_b_16(weights=None)
        self.vit.heads = nn.Identity()
        self.meta_projector = nn.Sequential(
            nn.Linear(meta_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, metadata):
        v_tokens = self.vit(images).unsqueeze(1)
        m_tokens = self.meta_projector(metadata).unsqueeze(1)
        attn_out, attn_weights = self.cross_attention(query=v_tokens, key=m_tokens, value=m_tokens)
        fused = self.norm1(v_tokens + attn_out).squeeze(1)
        return self.classifier(fused), attn_weights


class FairnessMultimodalModernBackbone(nn.Module):
    def __init__(self, backbone_type="mamba", num_classes=7, meta_dim=19, embed_dim=512, num_heads=8):
        super().__init__()
        self.backbone_type = backbone_type
        if backbone_type == "swin":
            self.encoder = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=0)
            vis_dim = self.encoder.num_features
        else:
            self.encoder = timm.create_model("mambaout_tiny.in1k", pretrained=False, num_classes=0)
            vis_dim = self.encoder.num_features  # 576

        self.vis_dim = vis_dim
        self.vis_proj = nn.Linear(vis_dim, embed_dim)
        self.meta_projector = nn.Sequential(
            nn.Linear(meta_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim)
        )
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, metadata):
        if self.backbone_type == "mamba":
            feat = self.encoder.forward_features(images)
            if feat.ndim == 4:
                feat = feat.mean(dim=(1, 2)) if feat.shape[-1] == self.vis_dim else feat.mean(dim=(2, 3))
            elif feat.ndim == 3:
                feat = feat.mean(dim=1) if feat.shape[-1] == self.vis_dim else feat.mean(dim=2)
        else:
            feat = self.encoder(images)

        v_feat = self.vis_proj(feat).unsqueeze(1)
        m_feat = self.meta_projector(metadata).unsqueeze(1)
        attn_out, attn_weights = self.cross_attention(query=v_feat, key=m_feat, value=m_feat)
        fused = self.norm1(v_feat + attn_out).squeeze(1)
        return self.classifier(fused), attn_weights


@st.cache_resource
def load_selected_model(selected_backbone):
    file_map = {
        "Class-Weighted Vision Mamba (Recommended)": ("mamba", "agentic_mamba_weighted_best.pth"),
        "Standard Vision Mamba (Group-DRO)": ("mamba", "agentic_mamba_best.pth"),
        "Multimodal Swin Transformer": ("swin", "agentic_swin_best.pth"),
        "Multimodal ViT-B/16": ("vit", "agentic_vit_best.pth")
    }
    b_key, filename = file_map[selected_backbone]

    if b_key == "vit":
        model = AgenticMultimodalViT(num_classes=7, meta_dim=19).to(device)
    else:
        model = FairnessMultimodalModernBackbone(backbone_type=b_key, num_classes=7, meta_dim=19).to(device)

    ckpt_path = filename if os.path.exists(filename) else None
    if ckpt_path is None:
        try:
            ckpt_path = hf_hub_download(repo_id="lowanshivinay/skin-lesion", filename=filename)
        except Exception:
            # Fallback to base mamba checkpoint if weighted file isn't uploaded yet
            if filename == "agentic_mamba_weighted_best.pth":
                try:
                    ckpt_path = hf_hub_download(repo_id="lowanshivinay/skin-lesion", filename="agentic_mamba_best.pth")
                    filename = "agentic_mamba_best.pth (Fallback)"
                except Exception:
                    ckpt_path = None
            else:
                ckpt_path = None

    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        status_msg = f"Loaded `{filename}` successfully."
    else:
        status_msg = f"Checkpoint `{filename}` not found locally or on Hugging Face."

    model.eval()
    return model, status_msg

# =====================================================================
# 3. 4 AGENTS + CROSS-AGENT LOGICAL CONSISTENCY ARBITRATOR
# =====================================================================
clean_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def apply_hair_ruler_stress(pil_img):
    """Simulates real-world clinical noise: hairs, ruler marks, blur, and illumination shifts."""
    img = pil_img.resize((224, 224)).copy()
    draw = ImageDraw.Draw(img)
    rng = random.Random(42)
    for _ in range(6):
        x0, y0 = rng.randint(10, 210), rng.randint(10, 210)
        x1, y1 = x0 + rng.randint(-70, 70), y0 + rng.randint(-70, 70)
        draw.line([(x0, y0), (x1, y1)], fill=(25, 18, 15), width=2)
    for x in range(15, 210, 15):
        tick_h = 16 if (x % 30 == 0) else 9
        draw.line([(x, 224), (x, 224 - tick_h)], fill=(15, 15, 15), width=2)

    tf = transforms.Compose([
        transforms.ColorJitter(brightness=0.35, contrast=0.35),
        transforms.GaussianBlur(kernel_size=(5, 7), sigma=(0.8, 1.8)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return tf(img)

def vision_agent(pil_image):
    """Agent 1: Automated Otsu lesion isolation and binary mask extraction."""
    np_img = np.array(pil_image)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_c)
        if w > 40 and h > 40:
            cropped = Image.fromarray(np_img[y:y+h, x:x+w])
            crop_mask = (thresh[y:y+h, x:x+w] > 0).astype(np.float32)
            crop_mask_resized = cv2.resize(crop_mask, (224, 224))
            return cropped, crop_mask_resized
    mask_resized = cv2.resize((thresh > 0).astype(np.float32), (224, 224))
    return pil_image, mask_resized

def axiomatic_xai_agent(model, cropped_pil, mask_224, meta_tensor, target_idx):
    """Agent 2: Gradient-based saliency attribution & Lesion Mask Overlap Audit."""
    inp = clean_transform(cropped_pil).unsqueeze(0).to(device).requires_grad_(True)
    logits, _ = model(inp, meta_tensor)
    score = logits[0, target_idx]
    model.zero_grad()
    score.backward()

    grad_map = inp.grad.abs().squeeze().detach().cpu().numpy().mean(axis=0)
    grad_norm = (grad_map - grad_map.min()) / (grad_map.max() - grad_map.min() + 1e-8)

    inside_energy = float(np.sum(grad_map * mask_224))
    total_energy = float(np.sum(grad_map)) + 1e-8
    overlap_pct = (inside_energy / total_energy) * 100.0

    np_crop = np.array(cropped_pil.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_norm), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(np_crop, 0.55, heatmap, 0.45, 0)
    return Image.fromarray(overlay), overlap_pct

def robustness_agent(model, cropped_image, meta_tensor, clean_pred_idx):
    """Agent 3: Stress-testing against hair, ruler ticks, blur, and illumination shifts."""
    stressed_tensor = apply_hair_ruler_stress(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(stressed_tensor, meta_tensor)
        probs = F.softmax(logits, dim=1).cpu().squeeze().numpy()
        stress_pred_idx = int(np.argmax(probs))
    is_stable = (clean_pred_idx == stress_pred_idx)
    return is_stable, probs[clean_pred_idx] * 100, probs[stress_pred_idx] * 100, DISEASE_CLASSES[stress_pred_idx]

def consistency_and_clinical_agent(
    backbone_name, predicted_class, confidence, entropy,
    overlap_pct, age, sex, site, fitz_label, is_stable, stress_class
):
    """
    Agent 4 + Cross-Agent Logical Consistency Arbitrator:
    Enforces logical agreement across Vision, XAI, Robustness, and Clinical Priors.
    """
    inconsistencies = []

    # Rule 1: XAI Grounding Check
    if confidence > 80.0 and overlap_pct < 40.0:
        inconsistencies.append(
            f"XAI Grounding Mismatch: High confidence ({confidence:.1f}%) relies heavily on non-lesion background (Overlap {overlap_pct:.1f}% < 40%)."
        )

    # Rule 2: Robustness-Certainty Consistency
    if not is_stable:
        inconsistencies.append(
            f"Acquisition Stability Conflict: Diagnosis shifts from {predicted_class.upper()} to {stress_class.upper()} under hair/blur/lighting perturbation."
        )

    # Rule 3: Epidemiological / Age-Pathology Consistency Check
    if predicted_class in ["akiec", "bcc"] and age < 30:
        inconsistencies.append(
            f"Demographic Atypicality: {predicted_class.upper()} is clinically rare at age {age}; rule out benign inflammatory look-alikes (e.g., psoriasis, eczema, or dermatitis)."
        )

    # Rule 4: Unified Trust Consensus Score (0 to 100%)
    norm_certainty = max(0.0, 1.0 - (entropy / 1.9459))
    norm_overlap = min(1.0, overlap_pct / 60.0)
    stability_score = 1.0 if is_stable else 0.3
    trust_consensus = (0.35 * norm_certainty + 0.35 * norm_overlap + 0.30 * stability_score) * 100.0

    # Base Clinical Stratification
    if predicted_class in ['mel', 'bcc']:
        risk = "HIGH RISK (Malignant Suspicion)"
        action = "Immediate excisional biopsy and urgent histopathology evaluation."
        badge = "🚨 URGENT"
    elif predicted_class == 'akiec':
        risk = "MODERATE RISK (Pre-Cancerous Dysplasia)"
        action = "Clinical review; consider cryotherapy or topical 5-FU."
        badge = "⚠️ PRIORITY"
    else:
        risk = "LOW RISK (Benign Condition)"
        action = "Routine surveillance; photograph monitoring at 6-month intervals."
        badge = "✅ ROUTINE"

    if inconsistencies or trust_consensus < 70.0:
        consensus_status = f"⚠️ CONFLICT DETECTED ({len(inconsistencies)} rule flag) — DERMATOLOGIST OVERRIDE ADVISED"
        action = f"Hold autonomous triage due to cross-agent consistency alert. {action}"
    else:
        consensus_status = "✅ FULLY CONSISTENT ACROSS ALL 4 AGENTS"

    conflict_block = "\n".join([f"  - {item}" for item in inconsistencies]) if inconsistencies else "  - None (All agents logically aligned)"

    report_md = f"""### **STRUCTURED CLINICAL TRIAGE & CONSISTENCY REPORT**
* **Cross-Agent Consensus**: **{consensus_status}**
* **Unified Trust Consensus Score**: **{trust_consensus:.1f} / 100**
* **Triage Priority**: **{badge}**
* **Active AI Backbone**: `{backbone_name}`
* **Primary Diagnosis**: **{CLASS_NAMES[predicted_class]}** (`{predicted_class.upper()}`) — `{confidence:.2f}%`
* **Risk Stratification**: **{risk}**

#### **Cross-Agent Logical Verification**
* **Certainty vs. Entropy**: `{'CONFIDENT' if entropy < 0.60 else 'UNCERTAIN'}` (Shannon Entropy: `{entropy:.4f}`)
* **Vision vs. XAI Grounding**: `{overlap_pct:.1f}%` inside lesion (`{'VERIFIED' if overlap_pct >= 40.0 else 'UNGROUNDED'}`)
* **Robustness vs. Stability**: `{'PRESERVED' if is_stable else 'SHIFTED TO ' + stress_class.upper()}`
* **Detected Logical Flags**:
{conflict_block}

#### **Patient & Subgroup Context**
* **Demographics**: Age `{age}` | Biological Sex: `{sex.capitalize()}` | Site: `{site.capitalize()}`
* **Fitzpatrick Phototype**: `{fitz_label}`

#### **Evidence-Based Clinical Action**
> {action}
"""
    return report_md, trust_consensus, inconsistencies

# =====================================================================
# 4. STREAMLIT NAVIGATION & MULTI-PAGE LAYOUT
# =====================================================================
st.sidebar.title("🧭 Navigation & Backbone")
app_mode = st.sidebar.radio(
    "Select System Mode:",
    [
        "1. Live Multi-Agent Clinical Diagnosis",
        "2. PhD Multi-Backbone & Fairness Results",
        "3. Guide-Aligned System Workflow"
    ]
)

selected_backbone = st.sidebar.selectbox(
    "Select Multimodal Backbone:",
    [
        "Class-Weighted Vision Mamba (Recommended)",
        "Standard Vision Mamba (Group-DRO)",
        "Multimodal Swin Transformer",
        "Multimodal ViT-B/16"
    ]
)
model, model_status = load_selected_model(selected_backbone)
st.sidebar.caption(model_status)

# ---------------------------------------------------------------------
# MODE 1: LIVE MULTI-AGENT DIAGNOSIS
# ---------------------------------------------------------------------
if app_mode == "1. Live Multi-Agent Clinical Diagnosis":
    st.title("🔬 Fairness-Aware & Class-Balanced Multi-Agent Dermatology System")
    st.write("Upload a dermoscopic lesion image and configure patient metadata + Fitzpatrick skin phototype to execute the 4-agent verification and consistency pipeline.")

    st.sidebar.header("📋 Patient & Skin-Tone Metadata")
    input_age = st.sidebar.slider("Patient Age", min_value=1, max_value=100, value=45)
    input_sex = st.sidebar.selectbox("Biological Sex", ["female", "male", "unknown"])
    input_site = st.sidebar.selectbox("Anatomical Localization", ANATOMICAL_SITES, index=4)
    input_fitz_label = st.sidebar.selectbox("Fitzpatrick Skin Type (I–VI)", list(FITZPATRICK_TYPES.keys()), index=2)
    fitz_num = FITZPATRICK_TYPES[input_fitz_label]

    uploaded_file = st.sidebar.file_uploader("Upload Dermoscopy Image (.jpg, .png)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        raw_image = Image.open(uploaded_file).convert("RGB")

        age_scaled = (input_age - 50.0) / 15.0
        sex_vec = [1.0 if input_sex == s else 0.0 for s in ['female', 'male', 'unknown']]
        site_vec = [1.0 if input_site == s else 0.0 for s in ANATOMICAL_SITES]
        fitz_vec = [1.0 if fitz_num == f else 0.0 for f in [1, 2, 3, 4, 5, 6]]
        meta_feats = np.array([age_scaled] + sex_vec + site_vec + fitz_vec, dtype=np.float32)
        meta_tensor = torch.tensor(meta_feats).unsqueeze(0).to(device)

        with st.spinner("Executing 4-Agent Trust & Cross-Agent Consistency Pipeline..."):
            cropped_image, lesion_mask = vision_agent(raw_image)

            img_tensor = clean_transform(cropped_image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(img_tensor, meta_tensor)
                probs = F.softmax(logits, dim=1).cpu().squeeze().numpy()
                pred_idx = int(np.argmax(probs))
                pred_class = DISEASE_CLASSES[pred_idx]
                pred_conf = probs[pred_idx] * 100
                entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

            xai_map, overlap_pct = axiomatic_xai_agent(model, cropped_image, lesion_mask, meta_tensor, pred_idx)

            is_stable, retained_conf, stress_conf, stress_class = robustness_agent(
                model, cropped_image, meta_tensor, pred_idx
            )

            report, trust_consensus, inconsistencies = consistency_and_clinical_agent(
                selected_backbone, pred_class, pred_conf, entropy, overlap_pct,
                input_age, input_sex, input_site, input_fitz_label, is_stable, stress_class
            )

        c1, c2, c3 = st.columns([1, 1, 1.25])
        with c1:
            st.subheader("1. Vision & XAI Agents")
            st.image(raw_image, caption="Original Dermoscopy Image", use_container_width=True)
            st.image(cropped_image, caption="Vision Agent: Isolated Lesion", use_container_width=True)
            st.image(xai_map, caption=f"Axiomatic XAI Map (Lesion Focus: {overlap_pct:.1f}%)", use_container_width=True)

        with c2:
            st.subheader("2. Uncertainty & Consensus")
            st.metric("Predicted Pathology", pred_class.upper(), f"{pred_conf:.1f}% Confidence")
            st.metric("Unified Trust Consensus", f"{trust_consensus:.1f}%", "Consistent" if not inconsistencies else "Conflict Flagged")
            st.metric("Shannon Entropy", f"{entropy:.4f}", "Well-Calibrated" if entropy < 0.6 else "High Uncertainty", delta_color="inverse")

            st.markdown("#### Hair, Ruler & Optical Stress Audit")
            if is_stable:
                st.success(f"**Robust & Stable**: Diagnosis preserved under simulated hairs, ruler marks & blur ({retained_conf:.1f}% retained).")
            else:
                st.error(f"**Perturbation Shift**: Shifted to `{stress_class.upper()}` under stress ({stress_conf:.1f}% confidence).")

            st.bar_chart({CLASS_NAMES[c]: float(probs[i]) for i, c in enumerate(DISEASE_CLASSES)})

        with c3:
            st.subheader("3. Clinical & Consistency Agent")
            if inconsistencies:
                st.warning(f"Cross-Agent Consistency Arbitrator flagged {len(inconsistencies)} logical conflict(s) for clinician review.")
            st.markdown(report)
            st.download_button("📥 Download Verified Triage Report", report, file_name="Clinical_Triage_Audit.txt")
    else:
        st.info("👈 Upload a dermoscopy image from the sidebar to run the multi-agent diagnostic and consistency audit pipeline.")

# ---------------------------------------------------------------------
# MODE 2: PhD BENCHMARK & FAIRNESS RESULTS (CLASS-WEIGHTED + 60/20/20)
# ---------------------------------------------------------------------
elif app_mode == "2. PhD Multi-Backbone & Fairness Results":
    st.title("📊 PhD Experimental Benchmarks (60/20/20 Holdout Protocol)")
    st.markdown("Evaluated on **HAM10000** ($N=10,015$) using a strict stratified **60% Train ($6,009$) / 20% Val ($2,003$) / 20% Holdout Test ($2,003$)** split with **Inverse Class-Frequency Weighting** ($w_i = \\frac{N}{K \\times n_i}$) and **Group-DRO Fairness**.")

    st.subheader("1. Architectural & Class-Imbalance Ablation Comparison")
    backbone_df = pd.DataFrame({
        "Multimodal Architecture & Training Loss": [
            "Multimodal Swin Transformer (Unweighted)",
            "Multimodal ViT-B/16 (Unweighted)",
            "Multimodal Vision Mamba (Standard Group-DRO)",
            "Class-Weighted Vision Mamba (Proposed Best)"
        ],
        "Clean Acc (%)": [86.02, 86.57, 86.77, 84.02],
        "Balanced Acc (%)": [71.32, 73.70, 78.91, 84.88],
        "Macro F1-Score": [0.7476, 0.7690, 0.7761, 0.7873],
        "AUC-ROC": [0.9718, 0.9633, 0.9706, 0.9646],
        "Melanoma (MEL) Recall": ["—", "70.85%", "70.85%", "83.41% (+12.56%)"]
    })
    st.table(backbone_df)
    st.success("Breakthrough Finding: Applying Inverse Class-Frequency Weighting (w_i = N / (K * n_i)) boosted Balanced Accuracy from 78.91% to 84.88% (+5.97%), Macro F1 to 0.7873, and Melanoma sensitivity to 83.41%.")

    st.subheader("2. Skin-Tone Fairness & Calibration Audit (Fitzpatrick I–VI)")
    fairness_df = pd.DataFrame({
        "Demographic Subgroup": [
            "Lighter Skin Phototypes (Fitzpatrick I–III)",
            "Darker Skin Phototypes (Fitzpatrick IV–VI)",
            "Subgroup Disparity Gap (|Light - Dark|)"
        ],
        "Standard Vision Mamba": ["86.76%", "87.10%", "0.34% (Near-Zero Bias)"],
        "Class-Weighted Vision Mamba": ["84.10%", "82.95%", "1.15% (Equitable Parity)"],
        "Calibration Status": ["ECE: 0.0718 (Well-Calibrated)", "ECE: 0.0718 (Well-Calibrated)", "Fairness Preserved"]
    })
    st.table(fairness_df)

    st.subheader("3. Class-Wise Sensitivity & F1 Breakdown (Class-Weighted Vision Mamba Holdout)")
    class_report_data = {
        "Disease Code": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "Full Clinical Condition": [CLASS_NAMES[c] for c in DISEASE_CLASSES],
        "Class Weight (w_i)": [4.3575, 2.7871, 1.3026, 12.4410, 1.2851, 0.2134, 10.0992],
        "Precision": [0.6000, 0.7909, 0.7004, 0.6897, 0.5552, 0.9776, 0.8966],
        "Recall (Sensitivity)": [0.8308, 0.8447, 0.7864, 0.8696, 0.8341, 0.8479, 0.9286],
        "F1-Score": [0.6968, 0.8169, 0.7409, 0.7692, 0.6667, 0.9081, 0.9123],
        "Holdout Support": [65, 103, 220, 23, 223, 1341, 28]
    }
    st.dataframe(pd.DataFrame(class_report_data), use_container_width=True)

# ---------------------------------------------------------------------
# MODE 3: GUIDE-ALIGNED WORKFLOW
# ---------------------------------------------------------------------
elif app_mode == "3. Guide-Aligned System Workflow":
    st.title("🧩 Guide-Aligned Multi-Agent Trustworthy Workflow")
    st.markdown("""
    This architecture implements all four revised PhD objectives with Inverse Class-Frequency Balancing and Cross-Agent Logical Consistency:
    1. **Vision Agent (Class-Balanced & Fairness-Aware Multimodal Backbone):** Isolates the active lesion via Otsu segmentation and fuses visual representations (Vision Mamba / Swin / ViT-B/16) with clinical metadata + Fitzpatrick I–VI via Cross-Attention under Inverse Class-Weighted ($w_i = \\frac{N}{K \\times n_i}$) Group-DRO loss.
    2. **Axiomatic XAI Agent:** Computes gradient-based pixel attributions and audits the **Lesion Mask Overlap Ratio** to prove predictions rely on pathology rather than skin tone or artifacts.
    3. **Robustness Agent:** Stress-tests model stability against simulated hairs, dermoscopic ruler ticks, optical blur, and lighting shifts under a strict 60/20/20 protocol.
    4. **Logical Consistency Arbitrator & Clinical Agent:** Enforces cross-agent rules (XAI grounding, acquisition stability, and demographic sanity checks) to compute a **Unified Trust Consensus Score** before generating hallucination-free triage reports.
    """)
    st.code("""
    [Dermoscopic Image + Metadata (Age, Sex, Site, Fitzpatrick I-VI)]
                                   │
                                   ▼
      [ 1. Vision Agent ] ──► Lesion Isolation + Class-Weighted Vision Mamba + Group-DRO
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
      [ 2A. Axiomatic XAI Agent ]     [ 2B. Robustness Stress Agent ]
       • Gradient Attribution Map      • Synthetic Hairs & Ruler Ticks
       • Lesion Mask Overlap %         • Blur & Illumination Shift Audit
                   │                               │
                   └───────────────┬───────────────┘
                                   ▼
            [ 3. Cross-Agent Logical Consistency Arbitrator ]
             • XAI Grounding + Stability + Demographic Sanity Rules
             • Unified Trust Consensus Score (0 - 100%)
                                   │
                                   ▼
                 [ 4. Hallucination-Free Clinical Agent ]
                  • Risk Tiering & Structured Triage Report
    """, language="text")
