import streamlit as st
import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms
import sys
import json
from src.utils.sensitivity import calculate_stability
from src.utils.report_gen import generate_markdown_report
from src.utils.feedback_manager import save_feedback, get_feedback_summary
from src.utils.inference_engine import run_live_inference
from src.models.classifier import get_resnet18_classifier

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(page_title="Counterfactual Medical AI", layout="wide")

st.title("🫁 Clinical Diagnostic Interface (XAI)")
st.markdown("""
Welcome to the **Explainable Medical AI Diagnostics** system. 
Analyze pulmonary cases using deep learning, counterfactuals, and uncertainty estimation.
""")

# 0. Device Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. File Uploader for Live Analysis
st.sidebar.header("🧪 Live Analysis")
uploaded_file = st.sidebar.file_uploader("Upload a new Chest X-ray", type=["jpg", "jpeg", "png"])

# 1. Load Data
@st.cache_data
def load_predictions():
    csv_path = 'results/predictions.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_metrics():
    metrics_path = 'results/evaluation_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

df = load_predictions()
metrics_data = load_metrics()

if df is not None:
    st.sidebar.header("Filter Images")
    # ... existing filters ...
    
    if metrics_data:
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Global Project Stats")
        if global_summary:
            st.sidebar.metric("Classifier Accuracy", f"{global_summary['classifier']['overall_accuracy']:.2%}")
            st.sidebar.metric("P2N Flip Rate", f"{global_summary['generator'].get('flip_rate', 0):.2%}")
            st.sidebar.metric("Mean SSIM", f"{global_summary['generator'].get('mean_ssim', 0):.4f}")
            st.sidebar.caption(f"Last updated: {global_summary['metadata']['generated_at']}")
        else:
            st.sidebar.warning("Run 'evaluation/global_summary.py' for full stats.")

    # Load feedback summary
    feedback_summary = get_feedback_summary()
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Lab Validation Status")
    st.sidebar.metric("Cases Validated", feedback_summary['total'])
    if feedback_summary['discrepancies'] > 0:
        st.sidebar.warning(f"Discrepancies found in {feedback_summary['discrepancies']} cases.")

    # Load classifier for sensitivity analysis
    @st.cache_resource
    def load_classifier():
        model = get_resnet18_classifier(pretrained=False)
        model_path = 'src/models/best_classifier.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model

    classifier = load_classifier()

    # Load latent coordinates
    @st.cache_data
    def load_latent_coords():
        path = 'results/latent_coordinates.csv'
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    latent_df = load_latent_coords()

    # Load global summary
    @st.cache_data
    def load_global_summary():
        path = 'results/global_dataset_summary.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    global_summary = load_global_summary()

    # Load statistical metrics
    @st.cache_data
    def load_stats():
        path = 'results/statistical_metrics.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    stats_data = load_stats()

    # 1. LIVE SESSION VIEW
    if uploaded_file is not None:
        st.markdown("---")
        st.header("🚨 Live Session: New Case Analysis")
        
        live_img = Image.open(uploaded_file).convert('RGB')
        
        with st.spinner("Performing real-time AI diagnostic..."):
            live_res = run_live_inference(live_img, classifier, device=device)
            
        l_col1, l_col2, l_col3 = st.columns([1, 1, 1])
        
        with l_col1:
            st.subheader("1. Uploaded Image")
            st.image(live_img, use_container_width=True)
            st.write(f"**Predicted:** {live_res['label']}")
            st.write(f"**Confidence:** {live_res['confidence']:.2%}")

        with l_col2:
            st.subheader("2. AI Focus (Grad-CAM)")
            st.image(live_res['heatmap_img'], use_container_width=True)
            st.info("Heatmap generated in real-time.")

        with l_col3:
            st.subheader("3. Reliability Metrics")
            st.metric("Stability Score", f"{live_res['stability_score']:.2f}/1.0")
            # Probabilities
            prob_df = pd.DataFrame({
                'Class': ['Normal', 'Pneumonia'],
                'Probability': [live_res['probs']['NORMAL'], live_res['probs']['PNEUMONIA']]
            })
            st.bar_chart(prob_df.set_index('Class'))

        # Live Report
        st.markdown("---")
        report_notes = st.text_area("Live Clinician Notes:", placeholder="Enter observations for this new case.")
        
        fake_item = {
            'filename': uploaded_file.name,
            'true_label': 'Unknown (Live)',
            'predicted_label': live_res['label'],
            'confidence': live_res['confidence'],
            'prob_normal': live_res['probs']['NORMAL'],
            'prob_pneumonia': live_res['probs']['PNEUMONIA'],
            'ssim': 'N/A (Live)',
            'lpips': 'N/A (Live)'
        }
        live_report = generate_markdown_report(fake_item, live_res['stability_score'], {}, report_notes)
        
        st.download_button(
            label="📥 Export Live Diagnostic Report",
            data=live_report,
            file_name=f"live_report_{uploaded_file.name.replace('.jpeg', '.md')}",
            mime="text/markdown"
        )
        st.markdown("---")
        st.info("Switch back to browsing the dataset by clearing the file upload above.")
        return # Skip the rest of the dashboard for now
    split = st.sidebar.selectbox("Select Split", df['split'].unique())
    label = st.sidebar.selectbox("Select Class", df['true_label'].unique())
    correct = st.sidebar.radio("Correctly Classified?", ["All", "Yes", "No"])

    filtered_df = df[(df['split'] == split) & (df['true_label'] == label)]
    if correct == "Yes":
        filtered_df = filtered_df[filtered_df['correct'] == True]
    elif correct == "No":
        filtered_df = filtered_df[filtered_df['correct'] == False]

    st.sidebar.write(f"Showing {len(filtered_df)} images")

    # 2. Display Result Grid
    st.header(f"Results for {split} - {label}")
    
    # 1. Selection
    st.sidebar.markdown("---")
    st.sidebar.header("Select Image")
    selected_idx = st.sidebar.selectbox("Choose an image to analyze", filtered_df.index, 
                                format_func=lambda x: filtered_df.loc[x, 'filename'])
    
    item = filtered_df.loc[selected_idx]

    # 2. Tabs for Different Views
    tab1, tab2, tab3 = st.tabs(["🧐 Local Explanations", "🌐 Global Feature Space", "📊 Project Metrics"])

    with tab1:
        # 2. Main Visualization Row
        st.markdown("### 🔍 Model Explanations")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("1. Original Image")
            img_path = os.path.join('data/chest_xray', item['filename'])
            img_orig = Image.open(img_path).convert('RGB')
            st.image(img_orig, use_container_width=True)
            st.write(f"**True Label:** {item['true_label']}")
            st.write(f"**Predicted:** {item['predicted_label']} ({item['confidence']:.2%})")

        with col2:
            st.subheader("2. AI Focus (Grad-CAM)")
            cam_path = os.path.join('results/gradcam', item['filename'])
            if os.path.exists(cam_path):
                cam_img = Image.open(cam_path)
                st.image(cam_img, use_container_width=True)
                st.info("Heatmap showing regions the classifier focuses on.")
            else:
                st.warning("Grad-CAM not available.")

        with col3:
            target_label = 'NORMAL' if item['predicted_label'] == 'PNEUMONIA' else 'PNEUMONIA'
            st.subheader(f"3. {item['predicted_label']} ➔ {target_label}")
            cf_path = os.path.join('results/counterfactuals', item['filename'])
            if os.path.exists(cf_path):
                cf_img = Image.open(cf_path)
                st.image(cf_img, use_container_width=True)
                st.success(f"Visualizing the '{target_label}' version of this scan.")
            else:
                st.warning("No counterfactual available.")

        # 3. Analysis Row
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.subheader("4. Difference Map (Changes)")
            diff_path = os.path.join('results/diff_maps', item['filename'])
            if os.path.exists(diff_path):
                diff_img = Image.open(diff_path)
                st.image(diff_img, use_container_width=True)
                st.caption("Highlights specific pixels modified by the GAN to achieve target classification.")
            else:
                st.warning("Difference map not available.")

        with res_col2:
            st.subheader("5. Prediction Stability & Metrics")
            
            # Real-time Sensitivity Analysis
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img_orig).unsqueeze(0)
            
            with st.spinner("Analyzing prediction stability..."):
                stability_res = calculate_stability(classifier, img_tensor)
                
            st.metric("Stability Score", f"{stability_res['stability_score']:.2f}/1.0", 
                    help="Measures how robust the prediction is to small input perturbations.")
            
            st.write(f"**Confidence Score:** {item['confidence']:.4f}")
            st.write(f"**Prob (Normal):** {item['prob_normal']:.4f}")
            st.write(f"**Prob (Pneumonia):** {item['prob_pneumonia']:.4f}")
            
            # Plot probabilities
            prob_df = pd.DataFrame({
                'Class': ['Normal', 'Pneumonia'],
                'Probability': [item['prob_normal'], item['prob_pneumonia']]
            })
            st.bar_chart(prob_df.set_index('Class'))

            st.markdown("---")
            st.subheader("👨‍⚕️ Clinician Feedback")
            
            # Check if already validated
            patient_id = item['filename']
            existing = [f for f in feedback_summary.get('feedback_data', []) if f['patient_id'] == patient_id]
            
            if existing:
                status = existing[0]['clinician_label']
                st.info(f"Previously Validated as: **{status}**")
            
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                if st.button("✅ Confirm (Correct)", use_container_width=True):
                    save_feedback(patient_id, item['predicted_label'], item['predicted_label'], notes)
                    st.success("AI prediction confirmed!")
                    st.rerun()
            with f_col2:
                other_label = "NORMAL" if item['predicted_label'] == "PNEUMONIA" else "PNEUMONIA"
                if st.button(f"❌ Correct to {other_label}", use_container_width=True):
                    save_feedback(patient_id, item['predicted_label'], other_label, notes)
                    st.warning(f"Prediction corrected to {other_label}")
                    st.rerun()

            st.markdown("---")
            st.subheader("📝 Clinician Analysis")
            notes = st.text_area("Add diagnostic notes for the report:", 
                               placeholder="e.g., Opacities observed in right lower lobe, consistent with focal pneumonia.")
            
            # Generate Report
            report_md = generate_markdown_report(item, stability_res['stability_score'], metrics_data, notes)
            
            st.download_button(
                label="📥 Export Diagnostic Report",
                data=report_md,
                file_name=f"report_{item['filename'].split('/')[-1].replace('.jpeg', '.md')}",
                mime="text/markdown",
                help="Download a structured summary of the AI analysis and your notes."
            )

    with tab2:
        st.header("🌐 Global Latent Space (PCA)")
        st.markdown("""
        How "Normal" do our counterfactuals look to the AI? This map shows the distribution 
        of real and generated images in the classifier's feature space.
        """)
        
        if latent_df is not None:
            # Highlight current patient
            patient_fname = item['filename']
            cf_fname = f"cf_{patient_fname}"
            
            # Filter latent_df for visualization (subset)
            chart_df = latent_df.copy()
            
            # Create a Plotly chart or use Streamlit native
            import plotly.express as px
            
            fig = px.scatter(chart_df, x='x', y='y', color='label', symbol='type', 
                            hover_data=['filename'], opacity=0.4,
                            title="Global Distribution of Real and Counterfactual Images")
            
            # Add specific point for current patient if in latent_df
            p_data = latent_df[latent_df['filename'] == patient_fname]
            c_data = latent_df[latent_df['filename'] == cf_fname]
            
            if not p_data.empty and not c_data.empty:
                # Add highlighting trace for the trajectory
                fig.add_scatter(x=[p_data.iloc[0]['x'], c_data.iloc[0]['x']], 
                               y=[p_data.iloc[0]['y'], c_data.iloc[0]['y']],
                               mode='lines+markers', name='Current Patient Trajectory',
                               line=dict(color='black', width=3),
                               marker=dict(size=12, symbol=['circle', 'star'], color='yellow'))
                st.info(f"Trajectory shown for {patient_fname}: Original (Circle) ➔ Counterfactual (Star)")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Latent coordinates not found. Run 'evaluation/visualize_latent_space.py' map.")

    with tab3:
        st.header("📊 Global Performance Analysis")
        st.markdown("""
        Rigorous statistical evaluation of the classifier's performance across the entire dataset.
        """)
        
        if stats_data:
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("AUC-ROC", f"{stats_data['auc_roc']:.4f}")
            with m_col2:
                st.metric("AUC-PR", f"{stats_data['auc_pr']:.4f}")
            with m_col3:
                f1 = stats_data['classification_report']['1']['f1-score']
                st.metric("F1-Score (Pneumonia)", f"{f1:.4f}")
            
            st.markdown("---")
            st.subheader("Performance Curves & Confusion Matrix")
            
            # Show the generated plot
            plot_path = 'results/performance_plots.png'
            if os.path.exists(plot_path):
                st.image(Image.open(plot_path), use_container_width=True)
            
            st.markdown("---")
            st.subheader("Classification Report Breakdown")
            report_df = pd.DataFrame(stats_data['classification_report']).transpose()
            st.dataframe(report_df.style.format(precision=4))
        else:
            st.warning("Statistical metrics not found. Run 'evaluation/plot_performance.py'.")

else:
    st.error("Predictions CSV not found. Please run batch inference first.")
