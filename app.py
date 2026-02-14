"""
ML Assignment 2 - Stramlit Web Application
Mushroom Classification
"""

import streamlit as st
import pandas as pd
import json
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / 'model'

#Set page info
st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_results():
    """Load precomputed results from trailing"""
    results_path = MODEL_DIR / 'results.json'
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Results file not found at {results_path}. Please run train_models.py first.")
        return None


@st.cache_resource
def load_model(model_name):
    """Load trained model"""
    model_filename = MODEL_DIR / f"{model_name.lower().replace(' ', '_')}_model.pkl"
    try:
        with open(model_filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file {model_filename} not found.")
        return None
    

@st.cache_data
def load_encoders():
    """Load label encoders"""
    try:
        with open(MODEL_DIR / 'label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open(MODEL_DIR / 'target_encoder.pkl', 'rb') as f:
            target_encoder = pickle.load(f)
        return label_encoders, target_encoder
    except FileNotFoundError:
        st.error("Encoder files not found. Please run train_models.py first.")
        return None, None
    

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Edible', 'Poisonous'],
                yticklabels=['Edible', 'Poisonous'],
                ax=ax
                )
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    return fig


def plot_metrics_comparison(results):
    """Plot comparaion of all models"""
    models = list(results.keys())
    metrics_names = ['accuracy', 'auc', 'precision', 'recall', 'f1', 'mcc']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics_names):
        values = [results[model]['metrics'][metric] for model in models]
        axes[idx].bar(range(len(models)), values, color='skyblue')
        axes[idx].set_xticks(range(len(models)))
        axes[idx].set_xticklabels(models, rotation=45, ha='right')
        axes[idx].set_ylabel(metric.upper())
        axes[idx].set_title(f'{metric.upper()} Comparision')
        axes[idx].set_ylim([0, 1.1])
        axes[idx].grid(axis='y', alpha=0.3)

        #Add value lables on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<p style="font-size: 50px; font-weight: bold; margin-bottom: 0px;">🍄 Mushroom Classification System</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 30px; color: gray;">Binary Classification: Edible vs Poisonous Mushrooms</p>', unsafe_allow_html=True)

    # Load results
    results = load_results()
    if results is None:
        st.stop()

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page",
                            ["Home", "Model Comparison", "Model Testing", "Dataset Info"])

    # Model selection
    model_names = list(results.keys())

    if page == "Home":
        st.header(" Project Overview")

        st.markdown("""
        ### About This Project
        This application demonstrates a comprehensive machine learning solution for mushroom classification.
        The system predicts whether a mushroom is **edible** or **poisonous** based on its physical characteristics.

        ### Features
        - **6 Classification Models** implemented and compared
        - **6 Evaluation Metrics** for comprehensive performance analysis
        - **Interactive Interface** for testing models
        - **Visual Analytics** for better understanding

        ### 📚 Dataset
        - **Source**: Kaggle - Mushroom Classification
        - **Instances**: 8,124 mushrooms
        - **Features**: 22 categorical attributes
        - **Classes**: Edible (e) and Poisonous (p)
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Models", "6")
        with col2:
            st.metric("Evaluation Metrics", "6")
        with col3:
            st.metric("Features", "22")

        st.markdown("---")
        st.subheader(" Best Performing Models")

        # Find best models by accuracy
        best_models = sorted(results.items(),
                           key=lambda x: x[1]['metrics']['accuracy'],
                           reverse=True)[:3]

        for rank, (model_name, result) in enumerate(best_models, 1):
            with st.expander(f"#{rank} {model_name} - Accuracy: {result['metrics']['accuracy']:.4f}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Precision", f"{result['metrics']['precision']:.4f}")
                col2.metric("Recall", f"{result['metrics']['recall']:.4f}")
                col3.metric("F1 Score", f"{result['metrics']['f1']:.4f}")
        
        # --- ADD CONCLUSION HERE ---
        st.markdown("---")
        st.subheader(" Key Findings & Conclusion")
        st.success("""
        **Performance Summary:**
        The ensemble and tree-based models (**Random Forest, XGBoost, and Decision Tree**) achieved perfect 
        **100% accuracy**, demonstrating that the mushroom features provide clear, non-linear decision boundaries. 
        
        **Insights:**
        * Tree-based models are exceptionally well-suited for this categorical dataset.
        * While **Logistic Regression** and **Naive Bayes** performed well (>92%), their lower scores suggest 
        that the relationships between mushroom attributes are complex and not purely linear.
        * The high **MCC (Matthews Correlation Coefficient)** across all models confirms reliable 
        performance on both edible and poisonous classes.
        """)

    elif page == "Model Comparison":
        st.header("📈 Model Performance Comparison")

        # Comparison table
        st.subheader("Performance Metrics Table")

        comparison_data = []
        for model_name, result in results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'AUC': f"{metrics['auc']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1']:.4f}",
                'MCC': f"{metrics['mcc']:.4f}"
            })

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, width='stretch', hide_index=True)

        # Visual comparison
        st.subheader("Visual Comparison")
        fig = plot_metrics_comparison(results)
        st.pyplot(fig)

        # Download results
        st.subheader("📥 Download Results")
        csv = df_comparison.to_csv(index=False)
        st.download_button(
            label="Download Comparison Table (CSV)",
            data=csv,
            file_name="model_comparison.csv",
            mime="text/csv"
        )

    elif page == "Model Testing":
        st.header(" Model Testing & Evaluation")

        # Model selection
        selected_model = st.sidebar.selectbox("Select Model", model_names)

        if selected_model:
            st.subheader(f"Results for {selected_model}")

            model_results = results[selected_model]
            metrics = model_results['metrics']

            # Display metrics
            st.markdown("### 📊 Performance Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("AUC Score", f"{metrics['auc']:.4f}")
            col3.metric("Precision", f"{metrics['precision']:.4f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Recall", f"{metrics['recall']:.4f}")
            col5.metric("F1 Score", f"{metrics['f1']:.4f}")
            col6.metric("MCC Score", f"{metrics['mcc']:.4f}")

            # Confusion Matrix
            st.markdown("### Confusion Matrix")
            cm = np.array(model_results['confusion_matrix'])
            fig = plot_confusion_matrix(cm, f"{selected_model} - Confusion Matrix")
            st.pyplot(fig)

            # Classification Report
            st.markdown("### 📋 Classification Report")
            cr = model_results['classification_report']

            # Create a cleaner classification report
            report_data = []
            for label, values in cr.items():
                if label not in ['accuracy', 'macro avg', 'weighted avg']:
                    report_data.append({
                        'Class': 'Edible' if label == '0' else 'Poisonous',
                        'Precision': f"{values['precision']:.4f}",
                        'Recall': f"{values['recall']:.4f}",
                        'F1-Score': f"{values['f1-score']:.4f}",
                        'Support': int(values['support'])
                    })

            df_report = pd.DataFrame(report_data)
            st.dataframe(df_report, width='stretch', hide_index=True)

        # CSV Upload Section
        st.markdown("---")
        st.subheader("📤 Upload Test Data")
        st.info("Upload a CSV file with mushroom features to make predictions")

        # Provide download option for test_data.csv
        test_csv_path = MODEL_DIR / 'test_data.csv'
        if test_csv_path.exists():
            test_df_download = pd.read_csv(test_csv_path)
            st.download_button(
                label=f"📥 Download Full Test Data ({len(test_df_download)} samples)",
                data=test_df_download.to_csv(index=False),
                file_name="test_data.csv",
                mime="text/csv",
                help=f"Download the complete test dataset with {len(test_df_download)} mushroom samples to test the model"
            )
            st.caption(f"✨ Test data contains {len(test_df_download)} samples with all 22 features + class label")

        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                # Load data
                test_df = pd.read_csv(uploaded_file)
                st.write(f"Uploaded data shape: {test_df.shape}")
                st.dataframe(test_df.head(), width='stretch')

                # Load model and encoders
                model = load_model(selected_model)
                label_encoders, target_encoder = load_encoders()

                if model is not None and label_encoders is not None:
                    # Check if 'class' column exists
                    has_target = 'class' in test_df.columns

                    if has_target:
                        y_true = test_df['class'].values
                        X_test = test_df.drop('class', axis=1)
                    else:
                        X_test = test_df

                    # Make predictions
                    if st.button("Make Predictions"):
                        with st.spinner("Making predictions..."):
                            y_pred = model.predict(X_test)
                            y_pred_labels = target_encoder.inverse_transform(y_pred)

                            # Show predictions
                            result_df = test_df.copy()
                            result_df['Prediction'] = y_pred_labels
                            result_df['Prediction_Label'] = ['Edible' if x == 'e' else 'Poisonous'
                                                             for x in y_pred_labels]

                            st.success("Predictions completed!")
                            st.dataframe(result_df.head(20), width='stretch')

                            # If true labels exist, show metrics
                            if has_target:
                                from sklearn.metrics import accuracy_score, classification_report
                                acc = accuracy_score(y_true, y_pred)
                                st.metric("Prediction Accuracy", f"{acc:.4f}")

                                # Show confusion matrix
                                cm_upload = confusion_matrix(y_true, y_pred)
                                fig = plot_confusion_matrix(cm_upload, "Uploaded Data - Confusion Matrix")
                                st.pyplot(fig)

                            # Download predictions
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions (CSV)",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    elif page == "Dataset Info":
        st.header("📚 Dataset Information")

        st.markdown("""
        ### Mushroom Dataset

        **Source**: Kaggle - Mushroom Classification Dataset
        **Platform**: Kaggle Datasets

        ### Description
        This dataset includes descriptions of hypothetical samples corresponding to 23 species of
        gilled mushrooms in the Agaricus and Lepiota Family. Each species is identified as
        definitely edible, definitely poisonous, or of unknown edibility and not recommended.

        ### Dataset Statistics
        - **Total Instances**: 8,124
        - **Number of Features**: 22 (all categorical)
        - **Target Classes**: 2 (Edible, Poisonous)
        - **Missing Values**: Yes (denoted by '?')

        ### Features
        1. cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
        2. cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s
        3. cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
        4. bruises: bruises=t, no=f
        5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
        6. gill-attachment: attached=a, descending=d, free=f, notched=n
        7. gill-spacing: close=c, crowded=w, distant=d
        8. gill-size: broad=b, narrow=n
        9. gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y
        10. stalk-shape: enlarging=e, tapering=t
        11. stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
        12. And 11 more features...

        ### Class Distribution
        - **Edible (e)**: 4,208 instances (51.8%)
        - **Poisonous (p)**: 3,916 instances (48.2%)

        The dataset is well-balanced, making it ideal for binary classification tasks.
        """)

        # Show sample data if available
        test_data_path = MODEL_DIR / 'test_data.csv'
        if test_data_path.exists():
            st.subheader("Sample Data")
            sample_df = pd.read_csv(test_data_path)
            st.dataframe(sample_df.head(10), width='stretch')

            st.download_button(
                label="Download Sample Test Data",
                data=sample_df.to_csv(index=False),
                file_name="mushroom_test_data.csv",
                mime="text/csv"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>2025AA05958 - Chaitra Shree - ML Assignment 2 - Mushroom Classification System</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()