import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
from dotenv import load_dotenv
from openai import OpenAI
import re
import tempfile
import csv
import io
import html

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client with API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Get model name from environment
model_name = os.getenv("MODEL_NAME", "gpt-4")  # Default to gpt-4 if not set
openai_client = OpenAI(api_key=api_key)

# Import script evaluator
from script_evaluator import ScriptEvaluator

class AdEvaluationCriteria(BaseModel):
    score: float = Field(description="Score between 0 and 10")
    reasoning: str = Field(description="Detailed reasoning for the score")

class VerificationResult(BaseModel):
    passed: bool = Field(description="Whether the verification passed (True) or failed (False)")
    details: str = Field(description="Details about the verification result")

class ComparisonResult(BaseModel):
    score: float = Field(description="Score between 0 and 10 for how well the ad compares to winning examples")
    analysis: str = Field(description="Detailed analysis comparing the ad to winning examples")

class SocialMediaAdEvaluator:
    def __init__(self):
        # Initialize the script evaluator
        self.script_evaluator = ScriptEvaluator()
        self.openai_client = openai_client
        self.model_name = model_name
        
        # Define evaluation instructions for each criteria
        self.criteria_instructions = {
            "creativity": """Evaluate the creativity of the social media ad script.
            
            Provide your response in the following format:
            Score: [number between 0-10]
            
            Reasoning: [Your detailed analysis]
            
            Consider these aspects:
            - Does the ad stand out and capture attention effectively?
            - Is the idea original and creatively executed?
            - Does the script evoke emotional engagement, curiosity, or intrigue?""",
            
            "ad_brief_alignment": """Evaluate how well the ad script aligns with the provided brief.
            
            Provide your response in the following format:
            Score: [number between 0-10]
            
            Reasoning: [Your detailed analysis]
            
            Consider these aspects:
            - Does the script accurately align with the provided brief?
            - Are all critical elements mentioned in the brief clearly reflected?
            - Is the tone and messaging consistent with the brief's specified audience and objective?""",
            
            "debrief_analysis": """Evaluate how well the ad script incorporates feedback from the debrief/previous feedback.

            Provide your response in the following format:
            Score: [number between 0-10]

            Reasoning: [Your detailed analysis covering each aspect below]

            Evaluate these specific aspects:
            1. Feedback Implementation:
               - List each key point from the debrief/previous feedback
               - For each point, analyze how well it has been addressed in the new script
               - Identify any feedback points that were missed or poorly addressed

            2. Improvement Areas:
               - Compare the current script to the specific weaknesses mentioned in the debrief
               - Evaluate if previous issues have been resolved
               - Assess if any new issues have been introduced

            3. Best Practices Adoption:
               - Check if successful elements mentioned in the debrief are maintained
               - Evaluate if recommended improvements have been implemented effectively
               - Assess if the script builds upon previous successes

            4. Overall Progress:
               - Compare the overall quality to the expectations set in the debrief
               - Evaluate if the script shows meaningful improvement in key areas
               - Assess if the script maintains previously successful elements while addressing weaknesses

            Provide specific examples from both the debrief and the current script to support your evaluation.""",
            
            "hallucination_check": """Evaluate the hallucination check (score 0-10):
            - Has every claim, feature, or benefit stated by the LLM been verified against provided documentation?
            - Are there any instances of hallucination or inaccuracies?
            - Is all factual information accurately represented and verifiable against provided documents?
            Provide a score between 0 and 10 with detailed reasoning.""",
            
            "relevance_clarity": """Evaluate the relevance and clarity of the ad script.

            Provide your response in the following format:
            Score: [number between 0-10]

            Reasoning: [Your detailed analysis covering each aspect below]

            Evaluate these specific aspects:
            1. Message Relevance:
               - How well does the script align with the brief's core message?
               - Are all key points from the brief addressed?
               - Is the messaging consistent with the brand voice described in the brief?

            2. Target Audience Alignment:
               - Does the content specifically address the target audience defined in the brief?
               - Are audience pain points and needs clearly addressed?
               - Is the language and tone appropriate for the target demographic?

            3. Call-to-Action Relevance:
               - Is the CTA clear and aligned with the brief's objectives?
               - Does it naturally flow from the message?
               - Is it compelling for the target audience?

            4. Clarity and Comprehension:
               - Is the message immediately clear and understandable?
               - Are complex concepts simplified appropriately?
               - Is the language concise and impactful?

            For each aspect, provide specific examples from the script to support your evaluation.""",
            
            "emotional_appeal": """Evaluate the emotional appeal and persuasiveness (score 0-10):
            - Does the ad evoke genuine emotions or resonate personally with the target audience?
            - Is it persuasive enough to drive action or influence purchasing decisions?
            - Does the ad script use language that naturally appeals to human experiences and feelings, avoiding a mechanical or overly formal tone?
            Provide a score between 0 and 10 with detailed reasoning.""",
            
            "natural_language": """Evaluate the natural language and tone (score 0-10):
            - Does the language flow naturally, conversationally, and authentically?
            - Is the ad script free from robotic, unnatural, or overly mechanical wording?
            - Does the language feel human-like and relatable rather than artificially generated?
            Provide a score between 0 and 10 with detailed reasoning."""
        }
        
        # Define verification instructions
        self.verification_instructions = """Evaluate the verification and fact-checking (Pass/Fail):
        - Has every claim, feature, or benefit stated by the LLM been verified against provided documentation?
        - Clearly identify any discrepancies or unverifiable statements.
        Return True if all claims are verified, False otherwise, with detailed reasoning."""
        
        # Define winning ads comparison instructions
        self.winning_ads_comparison_instructions = """Compare the ad script to the winning ad examples (score 0-10):
        - How well does the script match the quality, tone, and style of the winning examples?
        - Does it incorporate similar successful elements or approaches?
        - Does it avoid elements that are missing from the winning examples?
        - Does it have similar persuasive techniques and emotional appeal?
        - Does it have similar clarity, brevity, and impact?
        
        Provide a score between 0 and 10 along with detailed analysis that includes:
        1. Specific similarities to winning ads
        2. Key differences from winning ads
        3. Whether the ad script follows patterns found in successful ads
        4. Areas where the script could be modified to better match winning examples
        """
    
    def _extract_brief_sections(self, full_text: str) -> List[Dict[str, str]]:
        """
        Extract multiple briefs and their corresponding sections from the input text.
        
        Args:
            full_text: The complete input text containing multiple briefs
            
        Returns:
            List of dictionaries containing sections for each brief
        """
        briefs = []
        
        # Split by "Creative Brief #" or similar patterns
        brief_splits = re.split(r'(?i)creative\s+brief\s+#\d+:|brief\s+#\d+:', full_text)
        
        if len(brief_splits) <= 1:
            # If no clear brief splits found, treat the whole text as one brief
            return self._extract_single_brief_sections(full_text)
        
        # Process each brief section
        for brief_text in brief_splits[1:]:  # Skip the first split as it's before the first brief
            brief_sections = {
                'brief': '',
                'script': '',
                'debrief': '',
                'references': '',
                'reasoning': ''
            }
            
            # Extract main script section
            script_match = re.search(r'(?i)main\s+script:?(.*?)(?=(?:hook\s+variations:|document\s+references:|debrief\s+analysis:|$))', brief_text, re.DOTALL)
            if script_match:
                brief_sections['script'] = script_match.group(1).strip()
            
            # Extract brief content (everything before Main Script)
            brief_content = brief_text[:brief_text.lower().find('main script')] if 'main script' in brief_text.lower() else ''
            brief_sections['brief'] = brief_content.strip()
            
            # Extract debrief analysis
            debrief_match = re.search(r'(?i)debrief\s+analysis:?(.*?)(?=(?:technical\s+requirements:|$))', brief_text, re.DOTALL)
            if debrief_match:
                brief_sections['debrief'] = debrief_match.group(1).strip()
            
            # Extract document references
            references_match = re.search(r'(?i)document\s+references\s+and\s+reasoning:?(.*?)(?=(?:debrief\s+analysis:|$))', brief_text, re.DOTALL)
            if references_match:
                brief_sections['references'] = references_match.group(1).strip()
            
            # Extract reasoning if available
            reasoning_match = re.search(r'(?i)reasoning:?(.*?)(?=(?:technical\s+requirements:|$))', brief_text, re.DOTALL)
            if reasoning_match:
                brief_sections['reasoning'] = reasoning_match.group(1).strip()
            
            briefs.append(brief_sections)
        
        return briefs

    def evaluate_ad_script(self, 
                          script_path: str, 
                          brief_path: Optional[str] = None,
                          reference_docs: List[str] = None,
                          debrief_path: Optional[str] = None,
                          winning_ads_paths: List[str] = None,
                          progress_callback = None) -> Dict[str, Any]:
        """
        Evaluate a social media ad script based on specified criteria
        
        Args:
            script_path: Path to the file containing multiple briefs and their sections
            brief_path: Not used when briefs are in the main file
            reference_docs: Additional reference documents (optional)
            debrief_path: Not used when debriefs are in the main file
            winning_ads_paths: List of paths to winning ad examples
            progress_callback: Optional callback function to report progress
            
        Returns:
            Dictionary with evaluation results for each brief
        """
        # Parse the main document
        full_text = self.script_evaluator.parse_document(script_path)
        print("\n=== DOCUMENT PARSING COMPLETE ===")
        print(f"Document length: {len(full_text)} characters")
        
        # Extract all briefs and their sections
        briefs = self._extract_brief_sections(full_text)
        print(f"\n=== EXTRACTED {len(briefs)} BRIEFS ===")
        
        # Process reference documents if provided
        reference_texts = []
        winning_ads_texts = []
        
        if reference_docs:
            for doc in reference_docs:
                reference_texts.append(self.script_evaluator.parse_document(doc))
            print(f"\n=== PROCESSED {len(reference_texts)} REFERENCE DOCUMENTS ===")
        
        if winning_ads_paths:
            for doc in winning_ads_paths:
                winning_ads_texts.append(self.script_evaluator.parse_document(doc))
            print(f"\n=== PROCESSED {len(winning_ads_texts)} WINNING ADS DOCUMENTS ===")
        
        # Combine additional reference texts
        additional_reference_text = "\n\n".join(reference_texts)
        winning_ads_text = "\n\n".join(winning_ads_texts)
        
        # Evaluate each brief
        all_results = []
        total_briefs = len(briefs)
        
        # Calculate total evaluation steps per brief
        # Base steps: creativity, natural_language (always evaluated)
        total_steps_per_brief = 2
        
        # For each brief, determine additional steps
        for idx, brief_sections in enumerate(briefs, 1):
            # Add brief-dependent steps (ad_brief_alignment, relevance_clarity)
            if brief_sections['brief']:
                total_steps_per_brief += 2
                
            # Add debrief-dependent steps
            if brief_sections['debrief']:
                total_steps_per_brief += 1
                
            # Add reference-dependent steps (hallucination_check, emotional_appeal, verification)
            if brief_sections['references'] or reference_texts:
                total_steps_per_brief += 3
                
            # Add winning ads steps
            if winning_ads_text:
                total_steps_per_brief += 1
                
            # Only need to calculate once
            if idx == 1:
                print(f"Total evaluation steps per brief: {total_steps_per_brief}")
                break
                
        # Process each brief
        for idx, brief_sections in enumerate(briefs, 1):
            print(f"\n\n========== EVALUATING BRIEF #{idx} ==========")
            print(f"Brief title: {self._extract_brief_title(brief_sections['brief'])}")
            print(f"Sections found: {', '.join([k for k, v in brief_sections.items() if v])}")
            print(f"Script length: {len(brief_sections['script'])} characters")
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(idx, total_briefs, 0, total_steps_per_brief)
            
            # Combine references from brief with additional references
            combined_references = brief_sections['references']
            if additional_reference_text:
                combined_references += "\n\n" + additional_reference_text
            
            evaluation_results = {}
            current_step = 0
            
            # Evaluate criteria that don't need reference documents
            for criteria in ["creativity", "natural_language"]:
                current_step += 1
                if progress_callback:
                    progress_callback(idx, total_briefs, current_step, total_steps_per_brief, criteria)
                    
                print(f"\n--- Evaluating {criteria} ---")
                result = self._evaluate_criteria(
                    script_text=brief_sections['script'],
                    criteria=criteria,
                    instructions=self.criteria_instructions[criteria]
                )
                print(f"SCORE: {result.get('score', 0)}/10")
                evaluation_results[criteria] = result
            
            # Evaluate criteria that need brief
            if brief_sections['brief']:
                for criteria in ["ad_brief_alignment", "relevance_clarity"]:
                    current_step += 1
                    if progress_callback:
                        progress_callback(idx, total_briefs, current_step, total_steps_per_brief, criteria)
                        
                    print(f"\n--- Evaluating {criteria} ---")
                    result = self._evaluate_criteria_with_brief(
                        script_text=brief_sections['script'],
                        brief_text=brief_sections['brief'],
                        criteria=criteria,
                        instructions=self.criteria_instructions[criteria]
                    )
                    print(f"SCORE: {result.get('score', 0)}/10")
                    evaluation_results[criteria] = result
            
            # Evaluate criteria that need debrief
            if brief_sections['debrief']:
                current_step += 1
                if progress_callback:
                    progress_callback(idx, total_briefs, current_step, total_steps_per_brief, "debrief_analysis")
                    
                print(f"\n--- Evaluating debrief_analysis ---")
                result = self._evaluate_criteria_with_debrief(
                    script_text=brief_sections['script'],
                    debrief_text=brief_sections['debrief'],
                    criteria="debrief_analysis",
                    instructions=self.criteria_instructions["debrief_analysis"]
                )
                print(f"SCORE: {result.get('score', 0)}/10")
                evaluation_results["debrief_analysis"] = result
            
            # Evaluate criteria that need reference documents
            if combined_references:
                for criteria in ["hallucination_check", "emotional_appeal"]:
                    current_step += 1
                    if progress_callback:
                        progress_callback(idx, total_briefs, current_step, total_steps_per_brief, criteria)
                        
                    print(f"\n--- Evaluating {criteria} ---")
                    result = self._evaluate_criteria_with_references(
                        script_text=brief_sections['script'],
                        reference_text=combined_references,
                        criteria=criteria,
                        instructions=self.criteria_instructions[criteria]
                    )
                    print(f"SCORE: {result.get('score', 0)}/10")
                    evaluation_results[criteria] = result
                
                # Verification (pass/fail)
                current_step += 1
                if progress_callback:
                    progress_callback(idx, total_briefs, current_step, total_steps_per_brief, "verification")
                    
                print(f"\n--- Evaluating verification ---")
                verification = self._verify_facts(
                    script_text=brief_sections['script'],
                    reference_text=combined_references,
                    instructions=self.verification_instructions
                )
                print(f"PASSED: {verification.get('passed', False)}")
                evaluation_results["verification"] = verification
            
            # Evaluate comparison with winning ads if provided
            if winning_ads_text:
                current_step += 1
                if progress_callback:
                    progress_callback(idx, total_briefs, current_step, total_steps_per_brief, "winning_ads_comparison")
                    
                print(f"\n--- Evaluating winning_ads_comparison ---")
                winning_ads_comparison = self._compare_with_winning_ads(
                    script_text=brief_sections['script'],
                    winning_ads_text=winning_ads_text,
                    instructions=self.winning_ads_comparison_instructions
                )
                print(f"SCORE: {winning_ads_comparison.get('score', 0)}/10")
                evaluation_results["winning_ads_comparison"] = winning_ads_comparison
            
            # Generate report for this brief
            brief_report = self._generate_single_brief_report(evaluation_results)
            brief_report["brief_number"] = idx
            brief_report["brief_title"] = self._extract_brief_title(brief_sections['brief'])
            
            # Print brief summary
            print(f"\n=== BRIEF #{idx} EVALUATION SUMMARY ===")
            print(f"Title: {brief_report['brief_title']}")
            print(f"Total Score: {brief_report['total_score']}/{brief_report['max_possible_score']} ({brief_report['percentage_score']:.1f}%)")
            print("Category Scores:")
            for category, score in brief_report['category_scores'].items():
                print(f"  - {category}: {score}/10")
            print(f"Verification: {'PASSED' if brief_report['verification_result'] else 'FAILED'}")
            print("Recommendations:")
            for rec in brief_report['recommendations']:
                print(f"  - {rec}")
            
            all_results.append(brief_report)
            
            # Final callback for this brief
            if progress_callback:
                progress_callback(idx, total_briefs, total_steps_per_brief, total_steps_per_brief)
        
        print("\n\n========== EVALUATION COMPLETE ==========")
        print(f"Total Briefs Evaluated: {len(briefs)}")
        
        return {
            "total_briefs": len(briefs),
            "evaluations": all_results
        }

    def _extract_brief_title(self, brief_text: str) -> str:
        """Extract the title of the brief from the brief text."""
        # Look for patterns like 'Creative Brief #2: "The Hair Loss Solution"'
        title_match = re.search(r'(?i)"([^"]+)"', brief_text)
        if title_match:
            return title_match.group(1)
        
        # If no quoted title, try to get the first line
        first_line = brief_text.split('\n')[0].strip()
        if first_line:
            return first_line
        
        return "Untitled Brief"
    
    def _evaluate_criteria(self, script_text: str, criteria: str, instructions: str) -> Dict[str, Any]:
        """Evaluate a single criteria using LLM."""
        try:
            print(f"Starting evaluation for {criteria}...")
            
            # Create prompt with detailed instructions
            prompt = f"""
            You are evaluating a social media ad script based on specific criteria.

            CRITERIA TO EVALUATE: {criteria}

            EVALUATION INSTRUCTIONS:
            {instructions}

            SCRIPT TO EVALUATE:
            {script_text}

            YOUR TASK:
            1. Evaluate the script on a scale of 0-10 for the given criteria
            2. Provide detailed reasoning for your score
            
            SCORING GUIDELINES:
            - 0-2: Poor (fails on most aspects)
            - 3-4: Below Average (meets few aspects)
            - 5-6: Average (meets some aspects)
            - 7-8: Good (meets most aspects)
            - 9-10: Excellent (exceeds expectations)
            
            FORMAT YOUR RESPONSE AS JSON:
            {{
                "score": [0-10 integer score],
                "reasoning": [detailed reasoning for the score]
            }}
            """
            
            # Call the OpenAI API
            print(f"Calling OpenAI API for {criteria} evaluation...")
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # Extract the response content
            response_content = response.choices[0].message.content
            print(f"Received response from OpenAI for {criteria}")
            
            # Parse JSON response
            try:
                result = json.loads(response_content)
                
                # Validate the result has the expected structure
                if not isinstance(result, dict):
                    print(f"ERROR: Invalid result format for {criteria}. Expected dict, got {type(result)}")
                    return {"score": 0, "reasoning": f"Error: Invalid evaluation format for {criteria}"}
                
                # Ensure score is an integer and in the correct range
                if "score" not in result:
                    print(f"ERROR: Missing score in evaluation result for {criteria}")
                    result["score"] = 0
                else:
                    try:
                        result["score"] = int(result["score"])
                        result["score"] = max(0, min(10, result["score"]))  # Ensure score is between 0-10
                        print(f"Raw score for {criteria}: {result['score']}")
                    except (ValueError, TypeError):
                        print(f"ERROR: Invalid score format in {criteria}: {result['score']}")
                        result["score"] = 0
                
                # Ensure reasoning is present
                if "reasoning" not in result:
                    print(f"ERROR: Missing reasoning in evaluation result for {criteria}")
                    result["reasoning"] = f"Error: Missing reasoning for {criteria} evaluation"
                else:
                    reasoning_preview = result["reasoning"][:100] + "..." if len(result["reasoning"]) > 100 else result["reasoning"]
                    print(f"Reasoning preview: {reasoning_preview}")
                
                print(f"Final score for {criteria}: {result['score']}/10")
                return result
                
            except json.JSONDecodeError as e:
                print(f"ERROR: JSON parsing error for {criteria}: {e}")
                print(f"Raw response: {response_content[:200]}...")
                return {"score": 0, "reasoning": f"Error: Could not parse evaluation for {criteria}"}
                
        except Exception as e:
            print(f"ERROR: Exception during {criteria} evaluation: {str(e)}")
            return {"score": 0, "reasoning": f"Error during evaluation: {str(e)}"}
    
    def _evaluate_criteria_with_brief(self, script_text: str, brief_text: str, criteria: str, instructions: str) -> Dict[str, Any]:
        """Evaluate criteria that requires brief document."""
        try:
            print(f"Starting evaluation for {criteria} with brief...")
            
            # Create prompt with detailed instructions
            prompt = f"""
            You are evaluating a social media ad script based on specific criteria.

            CRITERIA TO EVALUATE: {criteria}

            EVALUATION INSTRUCTIONS:
            {instructions}

            AD BRIEF:
            {brief_text}

            SCRIPT TO EVALUATE:
            {script_text}

            YOUR TASK:
            1. Evaluate the script on a scale of 0-10 for how well it aligns with the brief
            2. Provide detailed reasoning for your score
            
            SCORING GUIDELINES:
            - 0-2: Poor (fails on most aspects)
            - 3-4: Below Average (meets few aspects)
            - 5-6: Average (meets some aspects)
            - 7-8: Good (meets most aspects)
            - 9-10: Excellent (exceeds expectations)
            
            FORMAT YOUR RESPONSE AS JSON:
            {{
                "score": [0-10 integer score],
                "reasoning": [detailed reasoning for the score]
            }}
            """
            
            # Call the OpenAI API
            print(f"Calling OpenAI API for {criteria} evaluation with brief...")
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # Extract the response content
            response_content = response.choices[0].message.content
            print(f"Received response from OpenAI for {criteria} with brief")
            
            # Parse JSON response
            try:
                result = json.loads(response_content)
                
                # Validate the result has the expected structure
                if not isinstance(result, dict):
                    print(f"ERROR: Invalid result format for {criteria}. Expected dict, got {type(result)}")
                    return {"score": 0, "reasoning": f"Error: Invalid evaluation format for {criteria}"}
                
                # Ensure score is an integer and in the correct range
                if "score" not in result:
                    print(f"ERROR: Missing score in evaluation result for {criteria}")
                    result["score"] = 0
                else:
                    try:
                        result["score"] = int(result["score"])
                        result["score"] = max(0, min(10, result["score"]))  # Ensure score is between 0-10
                        print(f"Raw score for {criteria}: {result['score']}")
                    except (ValueError, TypeError):
                        print(f"ERROR: Invalid score format in {criteria}: {result['score']}")
                        result["score"] = 0
                
                # Ensure reasoning is present
                if "reasoning" not in result:
                    print(f"ERROR: Missing reasoning in evaluation result for {criteria}")
                    result["reasoning"] = f"Error: Missing reasoning for {criteria} evaluation"
                else:
                    reasoning_preview = result["reasoning"][:100] + "..." if len(result["reasoning"]) > 100 else result["reasoning"]
                    print(f"Reasoning preview: {reasoning_preview}")
                
                print(f"Final score for {criteria}: {result['score']}/10")
                return result
                
            except json.JSONDecodeError as e:
                print(f"ERROR: JSON parsing error for {criteria}: {e}")
                print(f"Raw response: {response_content[:200]}...")
                return {"score": 0, "reasoning": f"Error: Could not parse evaluation for {criteria}"}
                
        except Exception as e:
            print(f"ERROR: Exception during {criteria} evaluation with brief: {str(e)}")
            return {"score": 0, "reasoning": f"Error during evaluation: {str(e)}"}
    
    def _evaluate_criteria_with_debrief(self, script_text: str, debrief_text: str, criteria: str, instructions: str) -> Dict[str, Any]:
        """Evaluate a criteria that requires the debrief document."""
        try:
            print(f"Starting evaluation for {criteria} with debrief using model {self.model_name}")
            
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Script: {script_text}\n\nDebrief (Previous Feedback): {debrief_text}\n\nPlease provide a detailed evaluation following the format in the instructions. For each point in the debrief, explicitly state whether and how it was addressed in the new script."}
                ],
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            print(f"Raw response for {criteria}: {content}")
            
            # Parse response
            try:
                if "Score:" in content:
                    parts = content.split("Score:", 1)
                    score_text = parts[1].split("\n")[0].strip()
                    score = float(score_text)
                    reasoning = parts[1].split("\n", 1)[1].strip() if len(parts[1].split("\n")) > 1 else parts[0].strip()
                    
                    # Extract feedback implementation details
                    feedback_analysis = {
                        "addressed_points": [],
                        "missed_points": [],
                        "improvements": [],
                        "remaining_issues": []
                    }
                    
                    # Try to extract structured feedback
                    sections = ["Feedback Implementation", "Improvement Areas", "Best Practices", "Overall Progress"]
                    current_section = None
                    for line in reasoning.split("\n"):
                        line = line.strip()
                        if any(section in line for section in sections):
                            current_section = line
                        elif line.startswith("- ") and current_section:
                            if "Feedback Implementation" in current_section:
                                if "addressed" in line.lower() or "implemented" in line.lower():
                                    feedback_analysis["addressed_points"].append(line[2:])
                                elif "missed" in line.lower() or "not" in line.lower():
                                    feedback_analysis["missed_points"].append(line[2:])
                            elif "Improvement Areas" in current_section:
                                if "improved" in line.lower() or "resolved" in line.lower():
                                    feedback_analysis["improvements"].append(line[2:])
                                elif "still" in line.lower() or "remains" in line.lower():
                                    feedback_analysis["remaining_issues"].append(line[2:])
                    
                    return {
                        "score": min(max(score, 0), 10),
                        "reasoning": reasoning,
                        "feedback_analysis": feedback_analysis
                    }
                else:
                    first_line = content.split("\n")[0]
                    import re
                    score_match = re.search(r'\b(\d+(\.\d+)?)\b', first_line)
                    score = float(score_match.group(1)) if score_match else 7.0
                    reasoning = content.strip()
                
                return {
                    "score": min(max(score, 0), 10),
                    "reasoning": reasoning
                }
            except Exception as e:
                print(f"WARNING: Error parsing response for {criteria}: {str(e)}")
                return {
                    "score": 7.0,
                    "reasoning": content
                }
                
        except Exception as e:
            print(f"ERROR: OpenAI API error for {criteria}: {str(e)}")
            return {
                "score": 0,
                "reasoning": f"Error evaluating {criteria}: {str(e)}"
            }
    
    def _evaluate_criteria_with_references(self, script_text: str, reference_text: str, criteria: str, instructions: str) -> Dict[str, Any]:
        """Evaluate a criteria that requires reference documents."""
        try:
            print(f"Starting evaluation for {criteria} with reference documents...")
            
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Script: {script_text}\n\nReference Documents: {reference_text}"}
                ],
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Parse response similar to _evaluate_criteria
            try:
                if "Score:" in content:
                    parts = content.split("Score:", 1)
                    score_text = parts[1].split("\n")[0].strip()
                    score = float(score_text)
                    reasoning = parts[1].split("\n", 1)[1].strip() if len(parts[1].split("\n")) > 1 else parts[0].strip()
                else:
                    first_line = content.split("\n")[0]
                    import re
                    score_match = re.search(r'\b(\d+(\.\d+)?)\b', first_line)
                    score = float(score_match.group(1)) if score_match else 7.0
                    reasoning = content.strip()
                
                return {
                    "score": min(max(score, 0), 10),
                    "reasoning": reasoning
                }
            except Exception as e:
                print(f"WARNING: Error parsing response for {criteria}: {str(e)}")
                return {
                    "score": 7.0,
                    "reasoning": content
                }
                
        except Exception as e:
            print(f"ERROR: OpenAI API error for {criteria}: {str(e)}")
            return {
                "score": 0,
                "reasoning": f"Error evaluating {criteria}: {str(e)}"
            }
    
    def _verify_facts(self, script_text: str, reference_text: str, instructions: str) -> Dict[str, Any]:
        """Verify facts in the script against reference documents."""
        try:
            print(f"Starting verification for script against reference documents...")
            
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Script: {script_text}\n\nReference Documents: {reference_text}"}
                ],
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Parse response to determine pass/fail and details
            try:
                # Look for explicit pass/fail indication
                passed = any(word in content.lower() for word in ["pass", "passed", "verified", "true"])
                not_passed = any(word in content.lower() for word in ["fail", "failed", "false", "not verified"])
                
                if passed and not not_passed:
                    passed = True
                elif not_passed:
                    passed = False
                else:
                    # Default to True if no clear indication
                    passed = True
                
                return {
                    "passed": passed,
                    "details": content.strip()
                }
            except Exception as e:
                print(f"WARNING: Error parsing verification response: {str(e)}")
                return {
                    "passed": True,
                    "details": content.strip()
                }
                
        except Exception as e:
            print(f"ERROR: OpenAI API error for verification: {str(e)}")
            return {
                "passed": False,
                "details": f"Error during verification: {str(e)}"
            }
            
    def _compare_with_winning_ads(self, script_text: str, winning_ads_text: str, instructions: str) -> Dict[str, Any]:
        """Compare the ad script with winning ad examples."""
        try:
            print(f"Starting comparison with winning ads...")
            
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Script to evaluate: {script_text}\n\nWinning ad examples: {winning_ads_text}"}
                ],
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Parse response similar to _evaluate_criteria
            try:
                if "Score:" in content:
                    parts = content.split("Score:", 1)
                    score_text = parts[1].split("\n")[0].strip()
                    score = float(score_text)
                    analysis = parts[1].split("\n", 1)[1].strip() if len(parts[1].split("\n")) > 1 else parts[0].strip()
                else:
                    first_line = content.split("\n")[0]
                    import re
                    score_match = re.search(r'\b(\d+(\.\d+)?)\b', first_line)
                    score = float(score_match.group(1)) if score_match else 7.0
                    analysis = content.strip()
                
                return {
                    "score": min(max(score, 0), 10),
                    "reasoning": analysis
                }
            except Exception as e:
                print(f"WARNING: Error parsing comparison response: {str(e)}")
                return {
                    "score": 7.0,
                    "reasoning": content
                }
                
        except Exception as e:
            print(f"ERROR: OpenAI API error for comparison: {str(e)}")
            return {
                "score": 0,
                "reasoning": f"Error comparing with winning ads: {str(e)}"
            }
    
    def generate_report(self, evaluation_results: Dict[str, Any], output_path: str = None) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        # Handle both single evaluation and multiple evaluations
        if "evaluations" in evaluation_results:
            # Multiple briefs case - just return as is, we already processed each report
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(evaluation_results, f, indent=4)
            
            print(f"Returning final report with {len(evaluation_results['evaluations'])} briefs")
            # Debug the structure of the first brief to confirm it has the proper format
            if evaluation_results['evaluations']:
                first_brief = evaluation_results['evaluations'][0]
                print(f"First brief keys: {list(first_brief.keys())}")
                print(f"First brief score: {first_brief.get('total_score', 'N/A')}/{first_brief.get('max_possible_score', 'N/A')}")
                
            return evaluation_results
        else:
            # Single brief case (backwards compatibility)
            single_report = self._generate_single_brief_report(evaluation_results)
            
            # Save report if output path provided
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(single_report, f, indent=4)
            
            return single_report
    
    def _extract_single_brief_sections(self, full_text: str) -> List[Dict[str, str]]:
        """
        Extract sections from a text assuming it's a single brief.
        
        Args:
            full_text: The complete text to parse
            
        Returns:
            List with a single dictionary containing brief sections
        """
        brief_sections = {
            'brief': '',
            'script': '',
            'debrief': '',
            'references': '',
            'reasoning': ''
        }
        
        # Try standard regex extraction first
        brief_sections = self._extract_sections_with_regex(full_text)
        
        # If debrief is missing but we have other sections, try to use LLM to find the debrief
        if not brief_sections['debrief'] and (brief_sections['brief'] or brief_sections['script']):
            print("Debrief section not found with regex. Trying LLM extraction...")
            brief_sections['debrief'] = self._extract_debrief_with_llm(full_text, brief_sections['brief'], brief_sections['script'])
            if brief_sections['debrief']:
                print("Successfully extracted debrief section using LLM")
        
        # Final fallback if still missing key sections
        if not brief_sections['brief'] or not brief_sections['script'] or len([k for k, v in brief_sections.items() if v]) < 3:
            print("WARNING: Insufficient sections found, using fallback approach")
            brief_sections = self._extract_sections_with_fallback(full_text, brief_sections)
        
        return [brief_sections]
        
    def _extract_sections_with_regex(self, full_text: str) -> Dict[str, str]:
        """Extract document sections using regex patterns."""
        brief_sections = {
            'brief': '',
            'script': '',
            'debrief': '',
            'references': '',
            'reasoning': ''
        }
        
        # Look for "Main Script" or similar headers to extract script content
        script_patterns = [
            r'(?i)main\s+script:?(.*?)(?=(?:hook\s+variations:|document\s+references:|debrief\s+analysis:|$))',
            r'(?i)content:?(.*?)(?=(?:hook\s+variations:|document\s+references:|debrief\s+analysis:|$))',
            r'(?i)script\s+content:?(.*?)(?=(?:hook\s+variations:|document\s+references:|debrief\s+analysis:|$))',
            r'(?i)ad\s+script:?(.*?)(?=(?:hook\s+variations:|document\s+references:|debrief\s+analysis:|$))'
        ]
        
        # Extract script section
        for pattern in script_patterns:
            script_match = re.search(pattern, full_text, re.DOTALL)
            if script_match:
                brief_sections['script'] = script_match.group(1).strip()
                break
        
        # If no explicit script section, use the entire text as the script
        if not brief_sections['script']:
            brief_sections['script'] = full_text
        
        # Find debrief analysis - can be anywhere in the document
        debrief_patterns = [
            r'(?i)debrief\s+analysis:?(.*?)(?=(?:technical\s+requirements:|$))',
            r'(?i)feedback:?(.*?)(?=(?:technical\s+requirements:|$))',
            r'(?i)analysis:?(.*?)(?=(?:technical\s+requirements:|$))'
        ]
        
        for pattern in debrief_patterns:
            debrief_match = re.search(pattern, full_text, re.DOTALL)
            if debrief_match:
                brief_sections['debrief'] = debrief_match.group(1).strip()
                break
        
        # Extract document references - can be anywhere in the document
        references_patterns = [
            r'(?i)document\s+references\s+and\s+reasoning:?(.*?)(?=(?:debrief\s+analysis:|analysis:|feedback:|$))',
            r'(?i)document\s+references:?(.*?)(?=(?:debrief\s+analysis:|analysis:|feedback:|$))',
            r'(?i)references:?(.*?)(?=(?:debrief\s+analysis:|analysis:|feedback:|$))'
        ]
        
        for pattern in references_patterns:
            references_match = re.search(pattern, full_text, re.DOTALL)
            if references_match:
                brief_sections['references'] = references_match.group(1).strip()
                break
        
        # Find brief content - usually at the beginning
        # We do this last so that we can exclude other identified sections
        # Create a modified text by removing the sections we've already found
        modified_text = full_text
        for section_type, content in brief_sections.items():
            if content and section_type != 'brief':
                # Replace the content with a placeholder to maintain structure
                modified_text = modified_text.replace(content, f"[{section_type.upper()}_CONTENT]")
        
        # Find text before the Main Script or similar headers
        brief_before_script = ""
        script_header_match = re.search(r'(?i)(main\s+script|content|script\s+content|ad\s+script):', modified_text)
        if script_header_match:
            brief_before_script = modified_text[:script_header_match.start()].strip()
        
        # Look for explicit brief headers
        brief_patterns = [
            r'(?i)creative\s+brief:?(.*?)(?=(?:\[|main\s+script:|content:|script\s+content:|ad\s+script:|$))',
            r'(?i)brief:?(.*?)(?=(?:\[|main\s+script:|content:|script\s+content:|ad\s+script:|$))'
        ]
        
        brief_content = ""
        for pattern in brief_patterns:
            brief_match = re.search(pattern, modified_text, re.DOTALL)
            if brief_match:
                brief_content = brief_match.group(1).strip()
                break
        
        # Use either the explicit brief content or the text before script
        if brief_content:
            brief_sections['brief'] = brief_content
        elif brief_before_script:
            brief_sections['brief'] = brief_before_script
        
        # Log what sections were found
        print(f"Brief sections extracted with regex: {', '.join([k for k, v in brief_sections.items() if v])}")
        
        # Final check - if debrief is empty but the brief contains debrief-like content,
        # try to extract it from the brief
        if not brief_sections['debrief'] and brief_sections['brief']:
            for pattern in debrief_patterns:
                debrief_match = re.search(pattern, brief_sections['brief'], re.DOTALL)
                if debrief_match:
                    debrief_content = debrief_match.group(1).strip()
                    brief_sections['debrief'] = debrief_content
                    # Remove the debrief content from the brief to avoid duplication
                    brief_text = brief_sections['brief']
                    brief_sections['brief'] = brief_text.replace(debrief_content, "").strip()
                    print("Found debrief section inside brief content")
                    break
        
        return brief_sections
    
    def _extract_debrief_with_llm(self, full_text: str, brief_text: str, script_text: str) -> str:
        """
        Use LLM to identify and extract the debrief/feedback section from the text.
        
        Args:
            full_text: The full document text
            brief_text: The already extracted brief section
            script_text: The already extracted script section
            
        Returns:
            Extracted debrief section as a string
        """
        try:
            print("Calling LLM to extract debrief section...")
            
            # Remove already extracted sections from the text to focus on remaining content
            modified_text = full_text
            if brief_text:
                modified_text = modified_text.replace(brief_text, "[BRIEF_CONTENT]")
            if script_text:
                modified_text = modified_text.replace(script_text, "[SCRIPT_CONTENT]")
            
            # Create prompt for LLM
            prompt = f"""
            You are tasked with identifying and extracting the debrief/feedback section from a social media ad document.
            
            DOCUMENT TEXT:
            {modified_text}
            
            TASK:
            1. Identify any sections that contain feedback, analysis, or debrief information about a previous version of the ad.
            2. Extract only that specific section.
            3. Look for content that discusses previous issues, things to improve, or analytical comments.
            
            FORMAT YOUR RESPONSE AS JSON:
            {{
                "debrief_found": true/false,
                "debrief_text": "the extracted debrief/feedback section"
            }}
            
            If no debrief or feedback section is found, set "debrief_found" to false and "debrief_text" to an empty string.
            """
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # Parse response
            response_content = response.choices[0].message.content
            result = json.loads(response_content)
            
            if result.get("debrief_found", False):
                debrief_text = result.get("debrief_text", "").strip()
                if debrief_text:
                    print(f"LLM successfully identified debrief section ({len(debrief_text)} characters)")
                    return debrief_text
            
            # If still not found, try one more approach with the full text
            if not brief_text and not script_text:
                print("Trying one more LLM approach with full text...")
                
                prompt = f"""
                You are tasked with analyzing a social media ad document and extracting specific sections.
                
                DOCUMENT TEXT:
                {full_text}
                
                TASK:
                Identify and extract these three distinct sections:
                1. The creative brief (instructions/requirements for the ad)
                2. The actual ad script/content
                3. Any feedback, analysis, or debrief sections
                
                FORMAT YOUR RESPONSE AS JSON:
                {{
                    "brief": "extracted brief section",
                    "script": "extracted script section",
                    "debrief": "extracted debrief/feedback section"
                }}
                
                If any section is not found, provide an empty string for that section.
                """
                
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                
                response_content = response.choices[0].message.content
                result = json.loads(response_content)
                
                return result.get("debrief", "").strip()
            
            return ""
        except Exception as e:
            print(f"Error during LLM debrief extraction: {str(e)}")
            return ""
    
    def _extract_sections_with_fallback(self, full_text: str, existing_sections: Dict[str, str]) -> Dict[str, str]:
        """Use fallback approach to extract sections when regular methods fail."""
        brief_sections = existing_sections.copy()
        
        # Split the text into chunks and make educated guesses
        lines = full_text.split('\n')
        line_count = len(lines)
        
        # Assign first 1/4 to brief if not already assigned
        if not brief_sections['brief'] and line_count > 4:
            brief_end = line_count // 4
            brief_sections['brief'] = '\n'.join(lines[:brief_end])
        
        # If we have a script, use that, otherwise use the middle half
        if not brief_sections['script']:
            script_start = line_count // 4
            script_end = line_count * 3 // 4
            brief_sections['script'] = '\n'.join(lines[script_start:script_end])
        
        # Assign last 1/4 to debrief if not already assigned
        if not brief_sections['debrief'] and line_count > 4:
            debrief_start = line_count * 3 // 4
            brief_sections['debrief'] = '\n'.join(lines[debrief_start:])
        
        print("Applied fallback section extraction")
        print(f"Final sections: {', '.join([k for k, v in brief_sections.items() if v])}")
        
        return brief_sections
    
    def _generate_single_brief_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report for a single brief evaluation."""
        # Log what's in the evaluation results
        print(f"Generating report for evaluation results with keys: {list(evaluation_results.keys())}")
        
        # Define all possible score categories
        score_categories = [
            "creativity", "ad_brief_alignment", "debrief_analysis", 
            "hallucination_check", "relevance_clarity", "emotional_appeal", 
            "natural_language", "winning_ads_comparison"
        ]
        
        # Fixed maximum possible score - 8 categories, 10 points each
        max_possible_score = 80
        
        # Calculate total score (excluding verification which is pass/fail)
        total_score = 0
        
        # Initialize category scores and feedback for all categories
        category_scores = {}
        detailed_feedback = {}
        
        # Set default values for all categories
        for category in score_categories:
            category_scores[category] = 0
            detailed_feedback[category] = "Not evaluated"
        
        # Update with actual scores and feedback where available
        for category in score_categories:
            if category in evaluation_results:
                category_result = evaluation_results[category]
                if isinstance(category_result, dict) and "score" in category_result:
                    score = category_result["score"]
                    print(f"Found score for {category}: {score}")
                    total_score += score
                    category_scores[category] = score
                    if "reasoning" in category_result:
                        detailed_feedback[category] = category_result["reasoning"]
                else:
                    print(f"WARNING: Category {category} has invalid structure: {category_result}")
            else:
                print(f"WARNING: Category {category} not found in evaluation results - using default score of 0")
        
        print(f"Total score: {total_score}/{max_possible_score} ({(total_score/max_possible_score)*100:.1f}%)")
        
        # Generate recommendations based on scores
        recommendations = self._generate_recommendations(evaluation_results)
        
        # Create final report
        report = {
            "brief_number": evaluation_results.get("brief_number", 1),
            "brief_title": evaluation_results.get("brief_title", "Untitled Brief"),
            "total_score": total_score,
            "max_possible_score": max_possible_score,
            "percentage_score": (total_score / max_possible_score) * 100,
            "category_scores": category_scores,
            "verification_result": False,
            "detailed_feedback": detailed_feedback,
            "recommendations": recommendations
        }
        
        # Add verification result if available
        if "verification" in evaluation_results:
            verification_result = evaluation_results["verification"]
            if isinstance(verification_result, dict):
                report["verification_result"] = verification_result.get("passed", False)
                report["detailed_feedback"]["verification"] = verification_result.get("details", "Not evaluated")
        
        return report
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Add strengths
        strengths = []
        for category, result in results.items():
            if category in ["verification", "brief_number", "brief_title"]:
                continue
                
            if isinstance(result, dict) and "score" in result:
                score = result["score"]
                if score >= 8:
                    strengths.append(category.replace("_", " ").title())
        
        if strengths:
            recommendations.append(f"Key Strengths: {', '.join(strengths)}")
        
        # Add areas for improvement
        improvements = []
        for category, result in results.items():
            if category in ["verification", "brief_number", "brief_title"]:
                continue
                
            if isinstance(result, dict) and "score" in result:
                score = result["score"]
                if score < 6:
                    reasoning = result.get("reasoning", "").split(".")[0] if result.get("reasoning") else ""
                    improvements.append(f"{category.replace('_', ' ').title()}: {reasoning}.")
        
        if improvements:
            recommendations.append("Areas for Improvement:")
            recommendations.extend(improvements)
        
        # Add verification recommendation if failed
        verification = results.get("verification", {})
        if isinstance(verification, dict):
            if not verification.get("passed", False):
                details = verification.get("details", "").split(".")[0] if verification.get("details") else ""
                recommendations.append(f"Fact Verification Failed: {details}.")
        
        # Add winning ads comparison highlights
        winning_comparison = results.get("winning_ads_comparison", {})
        if isinstance(winning_comparison, dict) and "score" in winning_comparison:
            winning_score = winning_comparison["score"]
            if winning_score >= 7:
                recommendations.append("Strong Alignment with Winning Ads: This ad matches successful examples well.")
            else:
                recommendations.append("Improve Alignment with Winning Ads: Consider incorporating more elements from successful examples.")
        
        # Add general recommendation if needed
        if not recommendations:
            recommendations.append("No specific recommendations. The ad script performs well across all criteria.")
        
        return recommendations

    def save_text_to_file(self, text: str) -> str:
        """Save text to a temporary file and return the path."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(text.encode('utf-8'))
            return tmp_file.name

    def cleanup_temp_files(self):
        """Clean up any temporary files created during processing."""
        # This function can be expanded to clean up specific directories or file patterns
        pass

    def generate_csv_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a CSV report for download.
        
        Args:
            evaluation_results: The evaluation results dictionary
            
        Returns:
            CSV string of the evaluation results
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Check if we have multiple evaluations
        if "evaluations" in evaluation_results:
            # Header row for multiple briefs
            writer.writerow(["Brief #", "Brief Title", "Total Score", "Max Score", "Percentage", 
                             "Verification", "Creativity", "Natural Language", "Ad Brief Alignment", 
                             "Relevance & Clarity", "Debrief Analysis", "Hallucination Check", 
                             "Emotional Appeal", "Winning Ads Comparison"])
            
            # Data rows for each brief
            for brief in evaluation_results["evaluations"]:
                writer.writerow([
                    brief.get("brief_number", ""),
                    brief.get("brief_title", ""),
                    f"{brief.get('total_score', 0):.1f}",
                    brief.get("max_possible_score", 0),
                    f"{brief.get('percentage_score', 0):.1f}%",
                    "PASS" if brief.get("verification_result", False) else "FAIL",
                    f"{brief.get('category_scores', {}).get('creativity', 0)}/10",
                    f"{brief.get('category_scores', {}).get('natural_language', 0)}/10",
                    f"{brief.get('category_scores', {}).get('ad_brief_alignment', 0)}/10",
                    f"{brief.get('category_scores', {}).get('relevance_clarity', 0)}/10",
                    f"{brief.get('category_scores', {}).get('debrief_analysis', 0)}/10",
                    f"{brief.get('category_scores', {}).get('hallucination_check', 0)}/10",
                    f"{brief.get('category_scores', {}).get('emotional_appeal', 0)}/10",
                    f"{brief.get('category_scores', {}).get('winning_ads_comparison', 0)}/10"
                ])
        else:
            # Single brief case
            brief = evaluation_results
            
            # Header row
            writer.writerow(["Category", "Score"])
            
            # Summary data
            writer.writerow(["Brief Title", brief.get("brief_title", "Untitled")])
            writer.writerow(["Total Score", f"{brief.get('total_score', 0)}/{brief.get('max_possible_score', 0)}"]) 
            writer.writerow(["Percentage", f"{brief.get('percentage_score', 0):.1f}%"])
            writer.writerow(["Verification", "PASS" if brief.get("verification_result", False) else "FAIL"])
            
            # Category scores
            writer.writerow([])
            writer.writerow(["CATEGORY SCORES", ""])
            for category, score in brief.get("category_scores", {}).items():
                writer.writerow([category.replace("_", " ").title(), f"{score}/10"])
            
            # Recommendations
            writer.writerow([])
            writer.writerow(["RECOMMENDATIONS", ""])
            for rec in brief.get("recommendations", []):
                writer.writerow([rec, ""])
        
        return output.getvalue()
    
    def generate_html_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate an HTML report for download.
        
        Args:
            evaluation_results: The evaluation results dictionary
            
        Returns:
            HTML string of the evaluation results
        """
        # Start with basic HTML structure and styles
        html_output = '''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Social Media Ad Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; color: #333; }
                h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                h2 { color: #2980b9; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .score { font-weight: bold; }
                .high-score { color: green; }
                .medium-score { color: orange; }
                .low-score { color: red; }
                .pass { color: green; }
                .fail { color: red; }
                .recommendations { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; }
                .detailed-feedback { margin-top: 10px; font-size: 0.9em; color: #555; }
            </style>
        </head>
        <body>
            <h1>Social Media Ad Evaluation Report</h1>
        '''
        
        # Multiple briefs case
        if "evaluations" in evaluation_results:
            html_output += f'''
            <p><strong>Total Briefs Evaluated:</strong> {evaluation_results.get("total_briefs", 0)}</p>
            <table>
                <tr>
                    <th>Brief #</th>
                    <th>Brief Title</th>
                    <th>Total Score</th>
                    <th>Percentage</th>
                    <th>Verification</th>
                </tr>
            '''
            
            # Add summary rows for each brief
            for brief in evaluation_results["evaluations"]:
                percentage = brief.get("percentage_score", 0)
                score_class = "high-score" if percentage >= 70 else "medium-score" if percentage >= 50 else "low-score"
                verification_class = "pass" if brief.get("verification_result", False) else "fail"
                verification_text = "PASS" if brief.get("verification_result", False) else "FAIL"
                
                html_output += f'''
                <tr>
                    <td>{brief.get("brief_number", "")}</td>
                    <td>{html.escape(brief.get("brief_title", ""))}</td>
                    <td class="score {score_class}">{brief.get("total_score", 0)}/{brief.get("max_possible_score", 0)}</td>
                    <td class="score {score_class}">{percentage:.1f}%</td>
                    <td class="{verification_class}">{verification_text}</td>
                </tr>
                '''
            
            html_output += '</table>'
            
            # Add detailed sections for each brief
            for brief in evaluation_results["evaluations"]:
                brief_title = html.escape(brief.get("brief_title", "Untitled Brief"))
                html_output += f'''
                <h2>Brief #{brief.get("brief_number", "")}: {brief_title}</h2>
                
                <h3>Category Scores</h3>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Score</th>
                    </tr>
                '''
                
                # Add rows for each category
                categories = brief.get("category_scores", {})
                for category, score in categories.items():
                    category_display = category.replace("_", " ").title()
                    score_class = "high-score" if score >= 7 else "medium-score" if score >= 5 else "low-score"
                    
                    html_output += f'''
                    <tr>
                        <td>{category_display}</td>
                        <td class="score {score_class}">{score}/10</td>
                    </tr>
                    '''
                
                html_output += '</table>'
                
                # Add recommendations
                html_output += '''
                <h3>Recommendations</h3>
                <div class="recommendations">
                    <ul>
                '''
                
                for rec in brief.get("recommendations", []):
                    html_output += f'<li>{html.escape(rec)}</li>'
                
                html_output += '''
                    </ul>
                </div>
                
                <h3>Detailed Feedback</h3>
                '''
                
                # Add detailed feedback for each category
                detailed_feedback = brief.get("detailed_feedback", {})
                for category, feedback in detailed_feedback.items():
                    category_display = category.replace("_", " ").title()
                    html_output += f'''
                    <div>
                        <h4>{category_display}</h4>
                        <div class="detailed-feedback">{html.escape(feedback).replace("\n", "<br>")}</div>
                    </div>
                    '''
        else:
            # Single brief case
            brief = evaluation_results
            brief_title = html.escape(brief.get("brief_title", "Untitled Brief"))
            percentage = brief.get("percentage_score", 0)
            score_class = "high-score" if percentage >= 70 else "medium-score" if percentage >= 50 else "low-score"
            
            html_output += f'''
            <h2>Brief: {brief_title}</h2>
            
            <table>
                <tr>
                    <th>Total Score</th>
                    <td class="score {score_class}">{brief.get("total_score", 0)}/{brief.get("max_possible_score", 0)} ({percentage:.1f}%)</td>
                </tr>
                <tr>
                    <th>Verification</th>
                    <td class="{("pass" if brief.get("verification_result", False) else "fail")}">
                        {("PASS" if brief.get("verification_result", False) else "FAIL")}
                    </td>
                </tr>
            </table>
            
            <h3>Category Scores</h3>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Score</th>
                </tr>
            '''
            
            # Add rows for each category
            categories = brief.get("category_scores", {})
            for category, score in categories.items():
                category_display = category.replace("_", " ").title()
                score_class = "high-score" if score >= 7 else "medium-score" if score >= 5 else "low-score"
                
                html_output += f'''
                <tr>
                    <td>{category_display}</td>
                    <td class="score {score_class}">{score}/10</td>
                </tr>
                '''
            
            html_output += '</table>'
            
            # Add recommendations
            html_output += '''
            <h3>Recommendations</h3>
            <div class="recommendations">
                <ul>
            '''
            
            for rec in brief.get("recommendations", []):
                html_output += f'<li>{html.escape(rec)}</li>'
            
            html_output += '''
                </ul>
            </div>
            
            <h3>Detailed Feedback</h3>
            '''
            
            # Add detailed feedback for each category
            detailed_feedback = brief.get("detailed_feedback", {})
            for category, feedback in detailed_feedback.items():
                category_display = category.replace("_", " ").title()
                html_output += f'''
                <div>
                    <h4>{category_display}</h4>
                    <div class="detailed-feedback">{html.escape(feedback).replace("\n", "<br>")}</div>
                </div>
                '''
        
        # Close HTML tags
        html_output += '''
        </body>
        </html>
        '''
        
        return html_output 