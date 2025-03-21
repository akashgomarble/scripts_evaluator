# Script and Ad Evaluation System

A comprehensive tool for evaluating scripts and social media ad content using LangSmith and OpenAI.

## Features

### General Script Evaluation
- Evaluates scripts across six dimensions: content quality, structure, technical accuracy, clarity, completeness, and originality
- Supports multiple file formats: PDF, DOCX, and CSV
- Provides detailed feedback and recommendations
- Visualizes results with interactive charts

### Social Media Ad Evaluation
- Specialized evaluation for social media ad scripts with 7 tailored criteria:
  - Creativity (0-10)
  - Ad Brief Alignment (0-10)
  - Debrief Analysis (0-10)
  - Hallucination Check (0-10)
  - Relevance & Clarity (0-10)
  - Emotional Appeal (0-10)
  - Natural Language (0-10)
  - Verification (Pass/Fail)
- Comprehensive feedback and actionable recommendations
- Support for ad briefs and reference materials for fact-checking

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/script-evaluation-system.git
cd script-evaluation-system
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```bash
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="<your-langsmith-api-key>"
OPENAI_API_KEY="<your-openai-api-key>"
```

## Usage

### General Script Evaluation

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Upload your script file (PDF, DOCX, or CSV)

3. Optionally, upload reference documents for comparison

4. View the evaluation results and download the report

### Social Media Ad Evaluation

1. Run the specialized ad evaluation app:
```bash
streamlit run social_media_ad_evaluator_app.py
```

2. Choose to either:
   - Upload your ad script, brief, and reference files
   - Enter text directly in the provided fields

3. Click "Evaluate Ad Script" to analyze the content

4. Review the detailed evaluation across all criteria

5. Download the comprehensive evaluation report

## Command Line Usage

You can also use the evaluation system from the command line:

```bash
python main.py --script path/to/script.pdf --reference-docs path/to/ref1.pdf path/to/ref2.pdf
```

## Evaluation Criteria

### General Script Evaluation

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

### Social Media Ad Evaluation

1. **Creativity (0-10)**
   - Does the ad stand out and capture attention effectively?
   - Is the idea original and creatively executed?
   - Does the script evoke emotional engagement, curiosity, or intrigue?

2. **Ad Brief Alignment (0-10)**
   - Does the script accurately align with the provided brief?
   - Are all critical elements mentioned in the brief clearly reflected?
   - Is the tone and messaging consistent with the brief's audience and objective?

3. **Debrief Analysis (0-10)**
   - Has the LLM addressed key feedback from previous successful ad scripts?
   - Is there an improvement in identified areas from past evaluations?
   - Does the script avoid previously noted weaknesses?

4. **Hallucination Check (0-10)**
   - Has every claim been verified against provided documentation?
   - Are there any instances of hallucination or inaccuracies?
   - Is all factual information accurately represented and verifiable?

5. **Relevance & Clarity (0-10)**
   - Does the script clearly convey the intended message and benefits?
   - Is the language simple, concise, and easily understandable?
   - Does the script directly address audience pain points or needs?

6. **Emotional Appeal (0-10)**
   - Does the ad evoke genuine emotions or resonate with the target audience?
   - Is it persuasive enough to drive action?
   - Does the ad script use language that naturally appeals to human experiences?

7. **Natural Language (0-10)**
   - Does the language flow naturally, conversationally, and authentically?
   - Is the ad script free from robotic or mechanical wording?
   - Does the language feel human-like and relatable?

8. **Verification (Pass/Fail)**
   - Has every claim been verified against provided documentation?
   - Any discrepancies or unverifiable statements are identified.

## API Keys

To use this application, you'll need:

1. **OpenAI API Key**: Get one from [OpenAI's website](https://platform.openai.com/api-keys)
2. **LangSmith API Key**: Get one from [LangSmith's website](https://smith.langchain.com/)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 