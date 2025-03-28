import streamlit as st
import os
import json
import pandas as pd
import plotly.graph_objects as go
import tempfile
from typing import List, Dict, Any
from dotenv import load_dotenv
from social_media_ad_evaluator import SocialMediaAdEvaluator
from evaluation_criteria import EVALUATION_CRITERIA
import re
import datetime
from evaluation_questions import EVALUATION_QUESTIONS, CATEGORY_DISPLAY_NAMES

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

def create_radar_chart(scores: Dict[str, float], max_scores: Dict[str, float]) -> go.Figure:
    """Create a radar chart for evaluation scores."""
    categories = [cat.replace('_', ' ').title() for cat in scores.keys()]
    values = list(scores.values())
    
    # Find the highest max_score for radar chart scale
    max_value = max(max(max_scores.values()), 10)  # At least 10 for visibility
    
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
                range=[0, max_value]  # Scale based on max scores
            )),
        showlegend=False,
        title="Ad Script Evaluation Scores"
    )
    
    return fig

def add_download_buttons(report, location_prefix="main"):
    """Add buttons to download the report in various formats."""
    st.subheader("📥 Download Report")
    
    col1, col2, col3 = st.columns(3)
    
    # Create an instance of SocialMediaAdEvaluator to access the report generation methods
    evaluator = SocialMediaAdEvaluator()
    
    # Generate a timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON download
    with col1:
        json_data = json.dumps(report, indent=4)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"ad_evaluation_{timestamp}.json",
            mime="application/json",
            help="Download the full evaluation results in JSON format",
            key=f"{location_prefix}_download_json"
        )
    
    # CSV download
    with col2:
        csv_data = evaluator.generate_csv_report(report)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"ad_evaluation_{timestamp}.csv",
            mime="text/csv",
            help="Download the evaluation results in CSV format for easy spreadsheet analysis",
            key=f"{location_prefix}_download_csv"
        )
    
    # HTML download
    with col3:
        html_data = evaluator.generate_html_report(report)
        st.download_button(
            label="Download HTML Report",
            data=html_data,
            file_name=f"ad_evaluation_{timestamp}.html",
            mime="text/html",
            help="Download a formatted HTML report with visualizations",
            key=f"{location_prefix}_download_html"
        )

def display_evaluation_results(report):
    """Display evaluation results in a structured format."""
    st.header("Evaluation Results")
    
    # Add download buttons at the top
    add_download_buttons(report, location_prefix="results")
    
    # Handle both single evaluation and multiple evaluations
    if "evaluations" in report:
        for brief in report["evaluations"]:
            st.subheader(f"Brief #{brief['brief_number']}: {brief['brief_title']}")
            
            # Display overall score
            st.metric("Overall Score", f"{brief['total_score']}/{brief['max_possible_score']} ({brief['percentage_score']:.1f}%)")
            
            # Display category scores
            st.subheader("Category Scores")
            for category, score in brief['category_scores'].items():
                max_score = brief['category_max_scores'].get(category, 0)
                st.metric(category.replace("_", " ").title(), f"{score}/{max_score}")
            
            # Display verification status
            st.subheader("Verification Status")
            verification_status = "PASSED" if brief['verification_result'] else "FAILED"
            st.metric("Status", verification_status, delta=None)
            
            # Display recommendations
            st.subheader("Recommendations")
            for rec in brief['recommendations']:
                st.write(f"• {rec}")
            
            # Display detailed feedback
            st.subheader("Detailed Feedback")
            for category, feedback in brief['detailed_feedback'].items():
                with st.expander(category.replace("_", " ").title()):
                    if isinstance(feedback, str):
                        # Try to parse feedback into question-answer-reasoning format
                        # Look for patterns like "Question: X, Answer: Y, Reasoning: Z"
                        items = []
                        # Split by question format
                        question_pattern = r'- ([^:]+): (Yes|No) - (.+?)(?=- [^:]+: |$)'
                        matches = re.findall(question_pattern, feedback, re.DOTALL)
                        
                        if matches:
                            for question, answer, reasoning in matches:
                                # Display with proper formatting
                                st.markdown(f"**Question:** {question.strip()}")
                                
                                # Color-code the answer
                                if answer.lower() == 'yes':
                                    st.success(f"**Answer:** {answer}")
                                else:
                                    st.error(f"**Answer:** {answer}")
                                
                                st.markdown(f"**Reasoning:** {reasoning.strip()}")
                                st.divider()
                        else:
                            # Try another pattern: splitting by line breaks and looking for Question, Answer, Reasoning prefixes
                            lines = feedback.split('\n')
                            i = 0
                            while i < len(lines):
                                line = lines[i].strip()
                                if line.startswith("Question:") or line.startswith("- "):
                                    question = line.replace("Question:", "").replace("- ", "").strip()
                                    
                                    # Look for Answer
                                    answer = ""
                                    reasoning = ""
                                    
                                    # Try to find answer in the same line or next lines
                                    if "Answer:" in line or ": Yes " in line or ": No " in line:
                                        # Answer in same line
                                        if ": Yes " in line:
                                            answer = "Yes"
                                            parts = line.split(": Yes ")
                                            reasoning = parts[1] if len(parts) > 1 else ""
                                        elif ": No " in line:
                                            answer = "No"
                                            parts = line.split(": No ")
                                            reasoning = parts[1] if len(parts) > 1 else ""
                                        else:
                                            parts = line.split("Answer:")
                                            answer = parts[1].strip() if len(parts) > 1 else ""
                                    elif i + 1 < len(lines) and ("Answer:" in lines[i+1] or "Yes -" in lines[i+1] or "No -" in lines[i+1]):
                                        # Answer in next line
                                        i += 1
                                        line = lines[i].strip()
                                        if "Yes -" in line:
                                            answer = "Yes"
                                            reasoning = line.split("Yes -")[1].strip()
                                        elif "No -" in line:
                                            answer = "No"
                                            reasoning = line.split("No -")[1].strip()
                                        else:
                                            answer = line.replace("Answer:", "").strip()
                                            
                                            # Look for Reasoning in next line
                                            if i + 1 < len(lines) and "Reasoning:" in lines[i+1]:
                                                i += 1
                                                reasoning = lines[i].replace("Reasoning:", "").strip()
                                    
                                    # Display with proper formatting
                                    st.markdown(f"**Question:** {question}")
                                    
                                    # Color-code the answer
                                    if answer.lower() == 'yes':
                                        st.success(f"**Answer:** {answer}")
                                    elif answer.lower() == 'no':
                                        st.error(f"**Answer:** {answer}")
                                    else:
                                        st.write(f"**Answer:** {answer}")
                                    
                                    st.markdown(f"**Reasoning:** {reasoning}")
                                    st.divider()
                                i += 1
                            
                            # If we couldn't parse the format, just display as is
                            if i == 0:
                                st.write(feedback)
                    elif isinstance(feedback, list):
                        for item in feedback:
                            if isinstance(item, dict):
                                question = item.get('Question', '')
                                answer = item.get('Answer', '')
                                reasoning = item.get('Reasoning', '')
                                
                                st.markdown(f"**Question:** {question}")
                                
                                # Color-code the answer
                                if answer.lower() == 'yes':
                                    st.success(f"**Answer:** {answer}")
                                elif answer.lower() == 'no':
                                    st.error(f"**Answer:** {answer}")
                                else:
                                    st.write(f"**Answer:** {answer}")
                                
                                st.markdown(f"**Reasoning:** {reasoning}")
                                st.divider()
                            else:
                                st.write(item)
                    else:
                        st.write(str(feedback))
    else:
        # Single brief case
        st.subheader(f"Brief: {report['brief_title']}")
        
        # Display overall score
        st.metric("Overall Score", f"{report['total_score']}/{report['max_possible_score']} ({report['percentage_score']:.1f}%)")
        
        # Display category scores
        st.subheader("Category Scores")
        for category, score in report['category_scores'].items():
            max_score = report['category_max_scores'].get(category, 0)
            st.metric(category.replace("_", " ").title(), f"{score}/{max_score}")
        
        # Display verification status
        st.subheader("Verification Status")
        verification_status = "PASSED" if report['verification_result'] else "FAILED"
        st.metric("Status", verification_status, delta=None)
        
        # Display recommendations
        st.subheader("Recommendations")
        for rec in report['recommendations']:
            st.write(f"• {rec}")
        
        # Display detailed feedback
        st.subheader("Detailed Feedback")
        for category, feedback in report['detailed_feedback'].items():
            with st.expander(category.replace("_", " ").title()):
                if isinstance(feedback, str):
                    # Try to parse feedback into question-answer-reasoning format
                    # Look for patterns like "Question: X, Answer: Y, Reasoning: Z"
                    items = []
                    # Split by question format
                    question_pattern = r'- ([^:]+): (Yes|No) - (.+?)(?=- [^:]+: |$)'
                    matches = re.findall(question_pattern, feedback, re.DOTALL)
                    
                    if matches:
                        for question, answer, reasoning in matches:
                            # Display with proper formatting
                            st.markdown(f"**Question:** {question.strip()}")
                            
                            # Color-code the answer
                            if answer.lower() == 'yes':
                                st.success(f"**Answer:** {answer}")
                            else:
                                st.error(f"**Answer:** {answer}")
                            
                            st.markdown(f"**Reasoning:** {reasoning.strip()}")
                            st.divider()
                    else:
                        # Try another pattern: splitting by line breaks and looking for Question, Answer, Reasoning prefixes
                        lines = feedback.split('\n')
                        i = 0
                        while i < len(lines):
                            line = lines[i].strip()
                            if line.startswith("Question:") or line.startswith("- "):
                                question = line.replace("Question:", "").replace("- ", "").strip()
                                
                                # Look for Answer
                                answer = ""
                                reasoning = ""
                                
                                # Try to find answer in the same line or next lines
                                if "Answer:" in line or ": Yes " in line or ": No " in line:
                                    # Answer in same line
                                    if ": Yes " in line:
                                        answer = "Yes"
                                        parts = line.split(": Yes ")
                                        reasoning = parts[1] if len(parts) > 1 else ""
                                    elif ": No " in line:
                                        answer = "No"
                                        parts = line.split(": No ")
                                        reasoning = parts[1] if len(parts) > 1 else ""
                                    else:
                                        parts = line.split("Answer:")
                                        answer = parts[1].strip() if len(parts) > 1 else ""
                                elif i + 1 < len(lines) and ("Answer:" in lines[i+1] or "Yes -" in lines[i+1] or "No -" in lines[i+1]):
                                    # Answer in next line
                                    i += 1
                                    line = lines[i].strip()
                                    if "Yes -" in line:
                                        answer = "Yes"
                                        reasoning = line.split("Yes -")[1].strip()
                                    elif "No -" in line:
                                        answer = "No"
                                        reasoning = line.split("No -")[1].strip()
                                    else:
                                        answer = line.replace("Answer:", "").strip()
                                        
                                        # Look for Reasoning in next line
                                        if i + 1 < len(lines) and "Reasoning:" in lines[i+1]:
                                            i += 1
                                            reasoning = lines[i].replace("Reasoning:", "").strip()
                                    
                                    # Display with proper formatting
                                    st.markdown(f"**Question:** {question}")
                                    
                                    # Color-code the answer
                                    if answer.lower() == 'yes':
                                        st.success(f"**Answer:** {answer}")
                                    elif answer.lower() == 'no':
                                        st.error(f"**Answer:** {answer}")
                                    else:
                                        st.write(f"**Answer:** {answer}")
                                    
                                    st.markdown(f"**Reasoning:** {reasoning}")
                                    st.divider()
                                i += 1
                            
                            # If we couldn't parse the format, just display as is
                            if i == 0:
                                st.write(feedback)
                elif isinstance(feedback, list):
                    for item in feedback:
                        if isinstance(item, dict):
                            question = item.get('Question', '')
                            answer = item.get('Answer', '')
                            reasoning = item.get('Reasoning', '')
                            
                            st.markdown(f"**Question:** {question}")
                            
                            # Color-code the answer
                            if answer.lower() == 'yes':
                                st.success(f"**Answer:** {answer}")
                            elif answer.lower() == 'no':
                                st.error(f"**Answer:** {answer}")
                            else:
                                st.write(f"**Answer:** {answer}")
                            
                            st.markdown(f"**Reasoning:** {reasoning}")
                            st.divider()
                        else:
                            st.write(item)
                else:
                    st.write(str(feedback))

    # Display radar chart of scores
    if report["evaluations"] and len(report["evaluations"]) > 0:
        st.subheader("Score Visualization")
        fig = create_radar_chart(report["evaluations"][0]["category_scores"], report["evaluations"][0]["category_max_scores"])
        st.plotly_chart(fig)

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
        Each brief is evaluated across 9 criteria:
        
        - Creativity
        - Emotional Appeal 
        - Relevance & Clarity
        - Natural Language
        - System 1 Assessment
        - System 2 Validation
        - Cognitive Harmony
        - Red Flags
        - Hallucination Check
        
        For each section, the score equals the number of 'Yes' answers to evaluation questions.
        The total score is the sum of all 'Yes' answers across all categories.
        
        The maximum score depends on the total number of questions evaluated across all criteria.
        
        Categories that cannot be evaluated (due to missing sections) will receive 0 points.
        """)
    
    # Use tabs to show different sections
    tab1, tab2 = st.tabs(["Upload & Evaluate", "About"])
    
    with tab1:
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize evaluator with the API key
        evaluator = SocialMediaAdEvaluator(api_key=api_key)
        
        # Add file upload widgets
        st.header("Upload Files")
        
        script_file = st.file_uploader("Upload Ad Script", type=["txt", "docx", "pdf"], key="script")
        
        ref_expander = st.expander("Reference Materials (Optional)")
        with ref_expander:
            # Multiple reference document upload
            reference_files = st.file_uploader("Upload Reference Documents", 
                                             type=["txt", "docx", "pdf"], 
                                             accept_multiple_files=True,
                                             key="references")
            
            # Reference text input
            reference_text = st.text_area("Or enter reference text directly", 
                                        height=200,
                                        placeholder="Enter product details, facts, or market research...",
                                        key="reference_text")
        
        winning_expander = st.expander("Winning Examples (Optional)")
        with winning_expander:
            # Multiple winning ad upload
            winning_files = st.file_uploader("Upload Winning Ad Examples", 
                                           type=["txt", "docx", "pdf"], 
                                           accept_multiple_files=True,
                                           key="winning")
            
            # Winning examples text input
            winning_text = st.text_area("Or enter winning ad examples directly", 
                                      height=200,
                                      placeholder="Enter examples of successful ads in this format...",
                                      key="winning_text")
    
    # Evaluation button
    if st.button("Evaluate Ad Script", type="primary"):
        try:
            # Input validation
            if script_file is None:
                st.error("Please upload a script file.")
                return
            script_path = save_uploaded_file(script_file)
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process reference documents
            status_text.text("Processing reference documents...")
            ref_paths = []
            if reference_files:
                for ref_file in reference_files:
                    ref_paths.append(save_uploaded_file(ref_file))
            elif reference_text:
                ref_path = save_text_to_file(reference_text)
                ref_paths.append(ref_path)
            
            # Process winning ads
            status_text.text("Processing winning ads examples...")
            winning_paths = []
            if winning_files:
                for win_file in winning_files:
                    winning_paths.append(save_uploaded_file(win_file))
            elif winning_text:
                win_path = save_text_to_file(winning_text)
                winning_paths.append(win_path)
            
            progress_bar.progress(10)
            
            # Parse main document
            status_text.text("Parsing input documents and extracting briefs...")
            
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
        
        # Add a separate "Download Results" section if desired
        st.sidebar.markdown("---")
        if st.sidebar.expander("Download Options", expanded=False):
            add_download_buttons(st.session_state.evaluation_results, location_prefix="sidebar")
    
    # Add information about the evaluation criteria
    with st.sidebar:
        st.header("Evaluation Criteria")
        
        for category, questions in EVALUATION_QUESTIONS.items():
            with st.expander(CATEGORY_DISPLAY_NAMES[category]):
                for question in questions:
                    st.markdown(f"- {question}")

if __name__ == "__main__":
    main() 