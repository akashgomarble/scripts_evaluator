"""
This module contains all evaluation criteria and questions for the social media ad evaluator.
Each criterion has 10 boolean questions that are used to evaluate different aspects of the ad script.
"""

EVALUATION_CRITERIA = {
    "creativity": {
        "name": "Creativity",
        "description": "Evaluate the creativity of the social media ad script",
        "questions": [
            "Does the ad have a unique and memorable hook? Please answer Yes if it has otherwise reply No",
            "Does the script use unexpected or surprising elements? Please answer Yes if it has otherwise reply No",
            "Does the ad tell a compelling story or narrative? Please answer Yes if it has otherwise reply No",
            "Does the script use creative metaphors or analogies? Please answer Yes if it has otherwise reply No",
            "Does the ad have a distinctive voice or personality? Please answer Yes if it has otherwise reply No",
            "Does the script use creative wordplay or clever phrasing? Please answer Yes if it has otherwise reply No",
            "Does the ad create emotional resonance? Please answer Yes if it has otherwise reply No",
            "Does the script maintain originality while staying on-brand? Please answer Yes if it has otherwise reply No",
            "Does the ad have a memorable and impactful ending? Please answer Yes if it has otherwise reply No",
            "Does the script avoid clichés and common advertising tropes? Please answer Yes if it has otherwise reply No"
        ]
    },
    "natural_language": {
        "name": "Natural Language",
        "description": "Evaluate the natural language and tone",
        "questions": [
            "Does the script use conversational, everyday language? Please answer Yes if it has otherwise reply No",
            "Is the flow of words natural and easy to follow? Please answer Yes if it has otherwise reply No",
            "Does the script avoid overly formal or technical language? Please answer Yes if it has otherwise reply No",
            "Are the sentences structured naturally? Please answer Yes if it has otherwise reply No",
            "Does the script use contractions appropriately? Please answer Yes if it has otherwise reply No",
            "Is the tone consistent throughout the script? Please answer Yes if it has otherwise reply No",
            "Does the script avoid repetitive or redundant phrases? Please answer Yes if it has otherwise reply No",
            "Are transitions between ideas smooth and natural? Please answer Yes if it has otherwise reply No",
            "Does the script use appropriate punctuation? Please answer Yes if it has otherwise reply No",
            "Does the language feel authentic to the target audience? Please answer Yes if it has otherwise reply No"
        ]
    },
    "ad_brief_alignment": {
        "name": "Ad Brief Alignment",
        "description": "Evaluate how well the ad script aligns with the provided brief",
        "questions": [
            "Does the script address all key objectives from the brief? Please answer Yes if it has otherwise reply No",
            "Is the target audience clearly reflected in the messaging? Please answer Yes if it has otherwise reply No",
            "Are all mandatory brand elements included? Please answer Yes if it has otherwise reply No",
            "Does the script follow the specified tone and style? Please answer Yes if it has otherwise reply No",
            "Are all key product features mentioned? Please answer Yes if it has otherwise reply No",
            "Does the script address the specified pain points? Please answer Yes if it has otherwise reply No",
            "Is the call-to-action aligned with brief objectives? Please answer Yes if it has otherwise reply No",
            "Does the script stay within the specified time/length constraints? Please answer Yes if it has otherwise reply No",
            "Are all mandatory legal/compliance elements included? Please answer Yes if it has otherwise reply No",
            "Does the script maintain brand voice consistency? Please answer Yes if it has otherwise reply No"
        ]
    },
    "relevance_clarity": {
        "name": "Relevance & Clarity",
        "description": "Evaluate the relevance and clarity of the ad script",
        "questions": [
            "Is the main message immediately clear? Please answer Yes if it has otherwise reply No",
            "Does the script address the target audience's needs? Please answer Yes if it has otherwise reply No",
            "Are all key points from the brief clearly communicated? Please answer Yes if it has otherwise reply No",
            "Is the language appropriate for the target audience? Please answer Yes if it has otherwise reply No",
            "Does the script avoid unnecessary complexity? Please answer Yes if it has otherwise reply No",
            "Is the call-to-action clear and compelling? Please answer Yes if it has otherwise reply No",
            "Does the script maintain focus on key objectives? Please answer Yes if it has otherwise reply No",
            "Are transitions between ideas logical and clear? Please answer Yes if it has otherwise reply No",
            "Does the script avoid confusing or ambiguous language? Please answer Yes if it has otherwise reply No",
            "Is the overall message memorable and impactful? Please answer Yes if it has otherwise reply No"
        ]
    },
    "debrief_analysis": {
        "name": "Debrief Analysis",
        "description": "Evaluate how well the ad script incorporates feedback from the debrief",
        "questions": [
            "Have all critical feedback points been addressed? Please answer Yes if it has otherwise reply No",
            "Are previous successful elements maintained? Please answer Yes if it has otherwise reply No",
            "Have identified weaknesses been improved? Please answer Yes if it has otherwise reply No",
            "Does the script show meaningful progress? Please answer Yes if it has otherwise reply No",
            "Are recommended changes implemented effectively? Please answer Yes if it has otherwise reply No",
            "Does the script avoid previously identified issues? Please answer Yes if it has otherwise reply No",
            "Are new improvements aligned with feedback? Please answer Yes if it has otherwise reply No",
            "Does the script build on previous successes? Please answer Yes if it has otherwise reply No",
            "Are all feedback points properly integrated? Please answer Yes if it has otherwise reply No",
            "Does the script show overall improvement? Please answer Yes if it has otherwise reply No"
        ]
    },
    "hallucination_check": {
        "name": "Hallucination Check",
        "description": "Evaluate the hallucination check",
        "questions": [
            "Are all product claims verified in references? Please answer Yes if it has otherwise reply No",
            "Are all features mentioned supported by documentation? Please answer Yes if it has otherwise reply No",
            "Are all benefits stated factual and verifiable? Please answer Yes if it has otherwise reply No",
            "Are all statistics or data points accurate? Please answer Yes if it has otherwise reply No",
            "Are all comparisons supported by references? Please answer Yes if it has otherwise reply No",
            "Are all technical specifications correct? Please answer Yes if it has otherwise reply No",
            "Are all customer testimonials authentic? Please answer Yes if it has otherwise reply No",
            "Are all pricing claims accurate? Please answer Yes if it has otherwise reply No",
            "Are all performance claims verified? Please answer Yes if it has otherwise reply No",
            "Are all brand statements accurate? Please answer Yes if it has otherwise reply No"
        ]
    },
    "emotional_appeal": {
        "name": "Emotional Appeal",
        "description": "Evaluate the emotional appeal and persuasiveness",
        "questions": [
            "Does the script evoke genuine emotions? Please answer Yes if it has otherwise reply No",
            "Is the emotional tone appropriate for the audience? Please answer Yes if it has otherwise reply No",
            "Does the script create personal resonance? Please answer Yes if it has otherwise reply No",
            "Are emotional triggers effectively used? Please answer Yes if it has otherwise reply No",
            "Does the script avoid manipulative tactics? Please answer Yes if it has otherwise reply No",
            "Is the emotional journey logical? Please answer Yes if it has otherwise reply No",
            "Does the script maintain emotional authenticity? Please answer Yes if it has otherwise reply No",
            "Are emotional elements well-integrated? Please answer Yes if it has otherwise reply No",
            "Does the script balance emotion and logic? Please answer Yes if it has otherwise reply No",
            "Is the emotional payoff satisfying? Please answer Yes if it has otherwise reply No"
        ]
    },
    "verification": {
        "name": "Verification",
        "description": "Evaluate the verification and fact-checking",
        "questions": [
            "Are all product claims verified? Please answer Yes if it has otherwise reply No",
            "Are all features documented? Please answer Yes if it has otherwise reply No",
            "Are all benefits supported? Please answer Yes if it has otherwise reply No",
            "Are all statistics accurate? Please answer Yes if it has otherwise reply No",
            "Are all comparisons valid? Please answer Yes if it has otherwise reply No",
            "Are all technical specs correct? Please answer Yes if it has otherwise reply No",
            "Are all testimonials authentic? Please answer Yes if it has otherwise reply No",
            "Are all pricing claims accurate? Please answer Yes if it has otherwise reply No",
            "Are all performance claims verified? Please answer Yes if it has otherwise reply No",
            "Are all brand statements accurate? Please answer Yes if it has otherwise reply No"
        ]
    },
    "winning_ads_comparison": {
        "name": "Winning Ads Comparison",
        "description": "Compare the ad script to winning ad examples",
        "questions": [
            "Does the script match winning ads' quality level? Please answer Yes if it has otherwise reply No",
            "Is the tone consistent with winning examples? Please answer Yes if it has otherwise reply No",
            "Does it use similar successful elements? Please answer Yes if it has otherwise reply No",
            "Does it avoid elements missing from winners? Please answer Yes if it has otherwise reply No",
            "Does it have similar persuasive techniques? Please answer Yes if it has otherwise reply No",
            "Does it match winning ads' emotional appeal? Please answer Yes if it has otherwise reply No",
            "Does it have similar clarity and brevity? Please answer Yes if it has otherwise reply No",
            "Does it follow winning patterns? Please answer Yes if it has otherwise reply No",
            "Does it avoid common losing patterns? Please answer Yes if it has otherwise reply No",
            "Does it have similar impact potential? Please answer Yes if it has otherwise reply No"
        ]
    }
}

def get_evaluation_prompt(criteria_type, script, context=None):
    """Generate the evaluation prompt for a specific criteria."""
    if criteria_type not in EVALUATION_CRITERIA:
        raise ValueError(f"Unknown criteria type: {criteria_type}")
    
    questions = EVALUATION_CRITERIA[criteria_type]["questions"]
    
    # Build the prompt
    prompt = f"""Please evaluate the following ad script based on {criteria_type.replace('_', ' ')} criteria.

Ad Script:
{script}

"""
    
    # Add context if provided
    if context:
        prompt += f"""
Additional Context:
{context}

"""
    
    # Add the questions
    prompt += "Please answer each of these questions with a 'Yes' or 'No' and provide brief reasoning:\n\n"
    for i, question in enumerate(questions, 1):
        prompt += f"{i}. {question}\n"
    
    # Add the format instructions for JSON output
    prompt += """
IMPORTANT: Return your evaluation as a JSON array directly (not wrapped in any object) in this exact format:

[
  {
    "Question": "Question 1 text here",
    "Answer": "Yes", 
    "Reasoning": "Reason for yes/no answer here"
  },
  {
    "Question": "Question 2 text here",
    "Answer": "No",
    "Reasoning": "Reason for yes/no answer here"
  },
  ...and so on for all 10 questions
]

CRITICAL NOTES:
1. Return a JSON array directly - do NOT wrap it in another object with a key like "evaluation" or "results"
2. Ensure each "Answer" is exactly "Yes" or "No" (case sensitive)
3. Include the full question text in each "Question" field
4. The output must be a valid JSON array at the top level
"""
    
    return prompt 