# Explainable Pneumonia Diagnosis via Counterfactual Image Generation

**Project Deliverable 1.2: From Pitch to Prototype**

This project provides a robust, explainable AI system for pneumonia diagnosis in chest X-rays. It combines ResNet-18 classification with CycleGAN-based counterfactual generation to provide clinicians with transparent and actionable diagnostic evidence.

---

## 🚀 Quick Start

### Installation
1. Clone the repository and navigate to the root.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run the Dashboard
To launch the interactive clinical cockpit:
```bash
streamlit run ui/dashboard.py
```

### Reproduce Analysis (Jupyter)
To verify the environment and explore the data:
```bash
jupyter notebook notebooks/setup.ipynb
```

---

## 🖼️ Visual Showcase
Our "Diagnostic Cockpit" provides four distinct layers of proof for every diagnosis:
1. **Grad-CAM Attention**: See where the AI is looking.
2. **Counterfactual Simulation**: See a "Healthy" version of the patient's scan.
3. **Difference Mapping**: See exactly which pathological pixels were removed.
4. **Stability Metrics**: See how much you should trust the result.

A high-fidelity layout of the system can be seen in [**`docs/ui_mockup.png`**](file:///Users/hima/Desktop/Counterfactual-Medical-Image-Generation-for-Explainable-Disease-Diagnosis/docs/ui_mockup.png).

---

## 🧠 Key Features
- **Multi-layered Explainability**: Grad-CAM (attention), Counterfactuals (generative "what-if"), and Difference Maps (pixel changes).
- **Statistical Rigor**: ROC Curves, Precision-Recall Curves, and Confidence Metrics included in the dashboard.
- **Clinical Workflow**: Real-time inference for new uploads, clinician feedback loop, and automated diagnostic reporting.
- **Global Analysis**: Latent space visualization via PCA to track disease trajectories.

---

## 📊 Dataset Information
- **Source**: Chest X-Ray Images (Pneumonia) dataset.
- **Type**: 5,856 grayscale X-ray images (converted to RGB for analysis).
- **Structure**: Split into `train`, `val`, and `test` across `NORMAL` and `PNEUMONIA` classes.
- **Preprocessing**: 256x256 resizing and normalization.

---

## 🛠 Repository Structure
- `src/` → Core logic (models, training, inference, evaluation, utils).
- `ui/` → Streamlit dashboard interface.
- `results/` → Pre-generated explainability artifacts and statistical summaries.
- `notebooks/` → Setup and EDA notebooks.
- `docs/` → Technical blueprint, UI mockups, and architecture diagrams.
- `data/` → Raw and sample X-ray data.

---

## 👨‍💻 Author
- **Name**: Hima Yalavarthi
- **Contact**: hima.yalavarthi@example.com

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🛡️ Reproducibility Guarantee
This project was developed with a focus on strict reproducibility. All dependencies are pinned in `requirements.txt`, and the `setup.ipynb` notebook provides a step-by-step verification process to ensure the code executes cleanly and produces visible output on your first run.
