import streamlit as st
import os
from script_evaluator import ScriptEvaluator
import json
import pandas as pd
import plotly.graph_objects as go
from typing import List
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for required environment variables
required_env_vars = ["LANGSMITH_API_KEY", "OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

def create_radar_chart(scores: dict) -> go.Figure:
    """Create a radar chart for evaluation scores."""
    categories = list(scores.keys())
    values = [scores[cat] for cat in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Evaluation Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Script Evaluation Scores"
    )
    
    return fig

def display_evaluation_results(results: dict):
    """Display evaluation results in a structured way."""
    # Overall Score
    st.header("Overall Score")
    st.metric("Score", f"{results['overall_score']:.2f}/1.00")
    
    # Category Scores
    st.subheader("Category Scores")
    scores_df = pd.DataFrame({
        'Category': list(results['category_scores'].keys()),
        'Score': list(results['category_scores'].values())
    })
    st.dataframe(scores_df)
    
    # Radar Chart
    st.plotly_chart(create_radar_chart(results['category_scores']))
    
    # Detailed Metrics
    st.subheader("Detailed Metrics")
    for category, metrics in results['detailed_metrics'].items():
        with st.expander(f"{category.replace('_', ' ').title()}"):
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
                    else:
                        st.text(f"{key.replace('_', ' ').title()}: {value}")
    
    # Feedback
    st.subheader("Feedback")
    for feedback in results['feedback']:
        st.info(feedback)
    
    # Recommendations
    st.subheader("Recommendations")
    for rec in results['recommendations']:
        st.warning(rec)

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary directory and return path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def main():
    st.set_page_config(
        page_title="Script Evaluator",
        page_icon="📝",
        layout="wide"
    )
    
    st.title("📝 Script Evaluation System")
    st.markdown("""
    This tool evaluates scripts across multiple dimensions including content quality, structure, 
    technical accuracy, clarity, completeness, and originality.
    """)
    
    # Check for missing environment variables
    if missing_vars:
        st.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        st.info("Please add these variables to your .env file and restart the app.")
        
        # Show environment variable setup form
        with st.expander("Environment Variable Setup"):
            st.markdown("""
            1. Create a `.env` file in the project root with the following variables:
            ```
            LANGSMITH_TRACING=true
            LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
            LANGSMITH_API_KEY="<your-langsmith-api-key>"
            OPENAI_API_KEY="<your-openai-api-key>"
            ```
            
            2. Replace `<your-langsmith-api-key>` with your LangSmith API key
            3. Replace `<your-openai-api-key>` with your OpenAI API key
            4. Restart the Streamlit app
            """)
        return
    
    # Initialize evaluator
    evaluator = ScriptEvaluator()
    
    # File upload section
    st.header("Upload Script")
    script_file = st.file_uploader(
        "Upload your script (PDF, DOCX, or CSV)",
        type=['pdf', 'docx', 'csv']
    )
    
    # Reference documents section
    st.header("Reference Documents (Optional)")
    reference_files = st.file_uploader(
        "Upload reference documents for comparison",
        type=['pdf', 'docx', 'csv'],
        accept_multiple_files=True
    )
    
    if script_file:
        try:
            # Save uploaded files
            script_path = save_uploaded_file(script_file)
            reference_paths = [save_uploaded_file(ref) for ref in reference_files]
            
            # Run evaluation
            with st.spinner("Evaluating script..."):
                results = evaluator.evaluate_script(script_path, reference_paths)
                report = evaluator.generate_report(results)
                
                # Display results
                display_evaluation_results(report)
                
                # Download report
                st.download_button(
                    label="Download Evaluation Report",
                    data=json.dumps(report, indent=4),
                    file_name="evaluation_report.json",
                    mime="application/json"
                )
            
            # Cleanup temporary files
            os.unlink(script_path)
            for ref_path in reference_paths:
                os.unlink(ref_path)
                
        except Exception as e:
            st.error(f"An error occurred during evaluation: {str(e)}")
            st.exception(e)  # Show detailed error information
    
    # Add information about the evaluation criteria
    with st.sidebar:
        st.header("Evaluation Criteria")
        st.markdown("""
        The script is evaluated across the following dimensions:
        
        1. **Content Quality**
           - Vocabulary and grammar
           - Writing style
           - Language proficiency
        
        2. **Structure & Organization**
           - Document structure
           - Section organization
           - Flow and coherence
        
        3. **Technical Accuracy**
           - Factual correctness
           - Reference alignment
           - Technical precision
        
        4. **Clarity & Readability**
           - Clear expression
           - Readability
           - Audience appropriateness
        
        5. **Completeness**
           - Required sections
           - Content coverage
           - Missing elements
        
        6. **Originality**
           - Unique content
           - Creative elements
           - Reference differentiation
        """)

if __name__ == "__main__":
    main() 