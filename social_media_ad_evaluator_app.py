import streamlit as st
import os
import json
import pandas as pd
import plotly.graph_objects as go
import tempfile
from typing import List, Dict, Any
from dotenv import load_dotenv
from social_media_ad_evaluator import SocialMediaAdEvaluator
import re
import datetime

# Load environment variables
load_dotenv()

# Check for required environment variables
required_env_vars = ["LANGSMITH_API_KEY", "OPENAI_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary directory and return path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

def save_text_to_file(text: str) -> str:
    """Save text to a temporary file and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
        tmp_file.write(text.encode('utf-8'))
        return tmp_file.name

def create_radar_chart(scores: Dict[str, float]) -> go.Figure:
    """Create a radar chart for evaluation scores."""
    categories = [cat.replace('_', ' ').title() for cat in scores.keys()]
    values = list(scores.values())
    
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
                range=[0, 10]  # Scale is 0-10
            )),
        showlegend=False,
        title="Ad Script Evaluation Scores"
    )
    
    return fig

def display_evaluation_results(report: Dict[str, Any]):
    """Display evaluation results in a structured format."""
    st.header("Evaluation Results")

    if "evaluations" in report:
        # Multiple briefs case
        st.write(f"Total Briefs Evaluated: {report['total_briefs']}")
        
        for brief_eval in report["evaluations"]:
            st.subheader(f"Brief #{brief_eval['brief_number']}: {brief_eval['brief_title']}")
            
            # Display scores
            st.metric("Total Score", 
                     f"{brief_eval['total_score']:.1f}/{brief_eval['max_possible_score']} ({brief_eval['percentage_score']:.1f}%)")
            
            # Display category scores in columns
            st.write("Category Scores:")
            cols = st.columns(4)  # Adjust number of columns as needed
            for idx, (category, score) in enumerate(brief_eval['category_scores'].items()):
                with cols[idx % 4]:
                    st.metric(category.replace('_', ' ').title(), f"{score}/10")
            
            # Display verification result
            st.write("Fact Verification:", 
                    "✅ Passed" if brief_eval['verification_result'] else "❌ Failed")
            
            # Display detailed feedback in an expander
            with st.expander("See Detailed Feedback"):
                for category, feedback in brief_eval['detailed_feedback'].items():
                    st.write(f"**{category.replace('_', ' ').title()}:**")
                    st.write(feedback)
            
            # Display recommendations
            st.write("Recommendations:")
            for rec in brief_eval['recommendations']:
                st.write(f"- {rec}")
            
            st.divider()  # Add visual separator between briefs
    elif "total_score" in report:
        # Single brief case (backwards compatibility)
        # Display scores
        st.metric("Total Score", 
                 f"{report['total_score']:.1f}/{report['max_possible_score']} ({report['percentage_score']:.1f}%)")
        
        # Display category scores in columns
        st.write("Category Scores:")
        cols = st.columns(4)  # Adjust number of columns as needed
        for idx, (category, score) in enumerate(report['category_scores'].items()):
            with cols[idx % 4]:
                st.metric(category.replace('_', ' ').title(), f"{score}/10")
        
        # Display verification result
        st.write("Fact Verification:", 
                "✅ Passed" if report['verification_result'] else "❌ Failed")
        
        # Display detailed feedback in an expander
        with st.expander("See Detailed Feedback"):
            for category, feedback in report['detailed_feedback'].items():
                st.write(f"**{category.replace('_', ' ').title()}:**")
                st.write(feedback)
        
        # Display recommendations
        st.write("Recommendations:")
        for rec in report['recommendations']:
            st.write(f"- {rec}")
    else:
        st.error("Invalid report format. Could not display evaluation results.")
        st.write("Report keys:", list(report.keys()))
    
    # Add download buttons for reports
    if ("evaluations" in report) or ("total_score" in report):
        st.subheader("Download Reports")
        col1, col2 = st.columns(2)
        
        # Initialize evaluator to access report generation methods
        evaluator = SocialMediaAdEvaluator()
        
        # Get current timestamp for unique filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate CSV report
        csv_report = evaluator.generate_csv_report(report)
        col1.download_button(
            label="Download CSV Report",
            data=csv_report,
            file_name=f"ad_evaluation_report_{timestamp}.csv",
            mime="text/csv",
            help="Download a CSV spreadsheet with evaluation scores and feedback"
        )
        
        # Generate HTML report
        html_report = evaluator.generate_html_report(report)
        col2.download_button(
            label="Download Detailed HTML Report",
            data=html_report,
            file_name=f"ad_evaluation_report_{timestamp}.html",
            mime="text/html",
            help="Download a formatted HTML report with complete evaluation details"
        )

def main():
    st.set_page_config(
        page_title="Social Media Ad Evaluator",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Social Media Ad Script Evaluator")
    st.markdown("""
    This tool evaluates LLM-generated social media ad scripts using a comprehensive analysis across 
    multiple dimensions and compares it with winning examples.
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
    
    # Initialize session state for evaluation results if not exists
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None
    
    # Initialize input method state if not exists
    if 'input_method' not in st.session_state:
        st.session_state.input_method = 'File Upload'
    
    # Add a sidebar with information about processing time
    with st.sidebar:
        st.subheader("⏱️ Processing Information")
        st.write("""
        **Please note:** Evaluation takes approximately 1-2 minutes per brief.
        
        For a document with multiple briefs, processing may take 5-10 minutes or more.
        
        During evaluation, you'll see real-time progress updates.
        """)
        
        # Add explanation of scoring
        st.subheader("📊 Scoring System")
        st.write("""
        Each brief is evaluated across 8 criteria, with a maximum of 10 points per criteria:
        
        - Creativity
        - Ad Brief Alignment
        - Debrief Analysis
        - Hallucination Check
        - Relevance & Clarity
        - Emotional Appeal
        - Natural Language
        - Winning Ads Comparison
        
        Total maximum score: 80 points (100%)
        
        Categories that cannot be evaluated (due to missing sections) will receive 0 points.
        """)
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ['File Upload', 'Text Input'],
        key='input_method'
    )
    
    # Script input
    script_file = None
    script_text = ""
    reference_files = []
    reference_text = ""
    winning_ads_files = []
    winning_ads_text = ""
    
    if input_method == 'File Upload':
        # Main layout with three columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1. Upload Script")
            script_file = st.file_uploader(
                "Upload script file",
                type=["txt", "pdf", "docx"],
                help="Upload the file containing brief(s), script(s), and debrief(s)",
                key="script_file"
            )
        
        with col2:
            st.subheader("2. References (Optional)")
            reference_files = st.file_uploader(
                "Upload reference files",
                type=["txt", "pdf", "docx", "csv"],
                help="Upload files containing reference information for fact verification",
                accept_multiple_files=True,
                key="reference_files"
            )
        
        with col3:
            st.subheader("3. Winning Ads (Optional)")
            winning_ads_files = st.file_uploader(
                "Upload winning ad examples",
                type=["txt", "pdf", "docx"],
                help="Upload examples of successful ads to compare against",
                accept_multiple_files=True,
                key="winning_ads"
            )
    else:
        # Direct text entry with three columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1. Enter Script")
            script_text = st.text_area(
                "Paste script here",
                height=300,
                help="Include brief, script, and debrief sections",
                key="script_text"
            )
        
        with col2:
            st.subheader("2. References (Optional)")
            reference_text = st.text_area(
                "Paste reference information",
                height=300,
                help="Paste reference information for fact verification",
                key="reference_text"
            )
        
        with col3:
            st.subheader("3. Winning Ads (Optional)")
            winning_ads_text = st.text_area(
                "Paste winning ad examples",
                height=300,
                help="Paste examples of successful ads to compare against",
                key="winning_ads_text"
            )
    
    # Evaluation button
    if st.button("Evaluate Ad Script", type="primary"):
        try:
            # Input validation
            if input_method == 'File Upload':
                if script_file is None:
                    st.error("Please upload a script file.")
                    return
                script_path = save_uploaded_file(script_file)
            else:
                if not script_text.strip():
                    st.error("Please enter script text.")
                    return
                script_path = save_text_to_file(script_text)
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process reference documents
            status_text.text("Processing reference documents...")
            ref_paths = []
            if input_method == 'File Upload' and reference_files:
                for ref_file in reference_files:
                    ref_paths.append(save_uploaded_file(ref_file))
            elif input_method == 'Text Input' and reference_text:
                ref_path = save_text_to_file(reference_text)
                ref_paths.append(ref_path)
            
            # Process winning ads
            status_text.text("Processing winning ads examples...")
            winning_paths = []
            if input_method == 'File Upload' and winning_ads_files:
                for win_file in winning_ads_files:
                    winning_paths.append(save_uploaded_file(win_file))
            elif input_method == 'Text Input' and winning_ads_text:
                win_path = save_text_to_file(winning_ads_text)
                winning_paths.append(win_path)
            
            progress_bar.progress(10)
            
            # Parse main document
            status_text.text("Parsing input documents and extracting briefs...")
            evaluator = SocialMediaAdEvaluator()
            
            # Initial document parsing (estimate 20% of work)
            progress_bar.progress(20)
            
            # Extract briefs to calculate total evaluation steps
            with st.spinner("Extracting briefs..."):
                # Use a simpler method to just count briefs without full evaluation
                full_text = evaluator.script_evaluator.parse_document(script_path)
                brief_splits = re.split(r'(?i)creative\s+brief\s+#\d+:|brief\s+#\d+:', full_text)
                num_briefs = len(brief_splits) - 1 if len(brief_splits) > 1 else 1
                
                # Show brief count
                status_text.text(f"Found {num_briefs} {'briefs' if num_briefs > 1 else 'brief'} to evaluate")
                progress_bar.progress(30)
            
            # Run evaluation with progress updates
            with st.spinner(f"Evaluating {num_briefs} {'briefs' if num_briefs > 1 else 'brief'}..."):
                # Calculate progress steps
                # 30% already done, 70% left for evaluation and report generation
                # Each brief gets equal portion of the remaining 70%
                brief_progress_portion = 60 / num_briefs
                
                # Create a status container for detailed progress
                detailed_status = st.empty()
                
                def progress_callback(brief_num, total_briefs, step, total_steps, criteria=None):
                    """Update progress bar and status during evaluation"""
                    # Calculate progress percentage
                    brief_progress = (brief_num - 1) * brief_progress_portion
                    step_progress = (step / total_steps) * brief_progress_portion
                    total_progress = 30 + brief_progress + step_progress
                    
                    # Update progress bar
                    progress_bar.progress(min(int(total_progress), 90))
                    
                    # Update status text
                    if criteria:
                        status_text.text(f"Evaluating Brief #{brief_num}/{total_briefs}: {criteria}")
                        detailed_status.text(f"Step {step}/{total_steps}: Analyzing {criteria}")
                    else:
                        status_text.text(f"Evaluating Brief #{brief_num}/{total_briefs}")
                
                # Run actual evaluation
                results = evaluator.evaluate_ad_script(
                    script_path=script_path,
                    reference_docs=ref_paths if ref_paths else None,
                    winning_ads_paths=winning_paths if winning_paths else None,
                    progress_callback=progress_callback
                )
                
                # Final processing
                status_text.text("Generating final report...")
                progress_bar.progress(95)
                
                # Generate report
                report = evaluator.generate_report(results)
                
                # Complete progress
                progress_bar.progress(100)
                status_text.text("Evaluation complete!")
                detailed_status.empty()
                
                # Store results in session state
                st.session_state.evaluation_results = report
                
                # Display results
                display_evaluation_results(report)
            
            # Clean up
            try:
                os.unlink(script_path)
                for path in ref_paths + winning_paths:
                    if os.path.exists(path):
                        os.unlink(path)
            except Exception as e:
                st.warning(f"Could not clean up temporary files: {e}")
        
        except Exception as e:
            st.error(f"An error occurred during evaluation: {str(e)}")
            st.exception(e)
    
    # Display previous results if they exist
    elif st.session_state.evaluation_results is not None:
        display_evaluation_results(st.session_state.evaluation_results)
    
    # Add information about the evaluation criteria
    with st.sidebar:
        st.header("Evaluation Criteria")
        
        criteria_descriptions = {
            "Creativity (0-10)": """
            - Does the ad stand out and capture attention effectively?
            - Is the idea original and creatively executed?
            - Does the script evoke emotional engagement, curiosity, or intrigue?
            """,
            
            "Ad Brief Alignment (0-10)": """
            - Does the script accurately align with the provided brief?
            - Are all critical elements mentioned in the brief clearly reflected?
            - Is the tone and messaging consistent with the brief's specified audience and objective?
            """,
            
            "Debrief Analysis (0-10)": """
            - Has the LLM addressed key feedback and insights from previous successful ad scripts?
            - Is there an improvement in identified areas from past evaluations?
            - Does the script avoid previously noted weaknesses?
            """,
            
            "Hallucination Check (0-10)": """
            - Has every claim been verified against provided documentation?
            - Are there any instances of hallucination or inaccuracies?
            - Is all factual information accurately represented and verifiable?
            """,
            
            "Relevance & Clarity (0-10)": """
            - Does the script clearly convey the intended message and benefits?
            - Is the language simple, concise, and easily understandable?
            - Does the script directly address audience pain points or needs?
            """,
            
            "Emotional Appeal (0-10)": """
            - Does the ad evoke genuine emotions or resonate with the target audience?
            - Is it persuasive enough to drive action?
            - Does the ad script use language that naturally appeals to human experiences?
            """,
            
            "Natural Language (0-10)": """
            - Does the language flow naturally, conversationally, and authentically?
            - Is the ad script free from robotic or mechanical wording?
            - Does the language feel human-like and relatable?
            """,
            
            "Winning Ads Comparison (0-10)": """
            - How well does the script match the quality, tone, and style of the winning examples?
            - Does it incorporate similar successful elements or approaches?
            - Does it follow patterns found in successful ads?
            - Are there elements from winning ads that should be incorporated?
            """,
            
            "Verification (Pass/Fail)": """
            - Has every claim been verified against provided documentation?
            - Any discrepancies or unverifiable statements are identified.
            """
        }
        
        for criteria, description in criteria_descriptions.items():
            with st.expander(criteria):
                st.markdown(description)
        
        st.header("About")
        st.markdown("""
        This evaluation system uses LangSmith to provide comprehensive 
        analysis of social media ad scripts. It combines multiple evaluation methods including:
        
        - LLM-based content analysis
        - Semantic similarity comparison
        - Structured evaluation criteria
        - Automated metrics calculation
        
        The evaluation is performed using the GPT-4 model for accurate assessment.
        """)

if __name__ == "__main__":
    main() 