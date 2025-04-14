import os
import json
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
from dotenv import load_dotenv
from openai import OpenAI
import re
import tempfile
import csv
import io
import html
from evaluation_criteria import get_evaluation_prompt, EVALUATION_CRITERIA
from evaluation_questions import EVALUATION_QUESTIONS, CATEGORY_DISPLAY_NAMES

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
model_name = os.getenv("MODEL_NAME", "gpt-4o")  # Default to gpt-4-turbo-preview if not set
openai_client = OpenAI(api_key = api_key)

# Import script evaluator
from script_evaluator import ScriptEvaluator

class SocialMediaAdEvaluator:
    def __init__(self, api_key=None, model_name="gpt-4o"):
        """Initialize the evaluator with OpenAI API key and model name."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided and OPENAI_API_KEY environment variable is not set")
        self.openai_client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.category_display_names = CATEGORY_DISPLAY_NAMES
        self.evaluation_questions = EVALUATION_QUESTIONS
        
        # Initialize the script evaluator
        self.script_evaluator = ScriptEvaluator()
        
        # Load evaluation instructions
        self.evaluation_instructions = {
            "creativity": "Evaluate the creative aspects of the ad script. Consider originality, innovation, and ability to capture attention.",
            "emotional_appeal": "Assess how well the ad connects emotionally with the audience and creates desire.",
            "relevance_clarity": "Evaluate how clear, relevant, and understandable the ad is for the target audience.",
            "natural_language": "Assess how natural and human-like the language feels.",
            "system1_assessment": "Evaluate the immediate, intuitive impact of the ad.",
            "system2_validation": "Assess the logical and analytical aspects of the ad.",
            "cognitive_harmony": "Evaluate how well the ad balances emotional and rational elements.",
            "red_flags": "Check for potential issues or concerns in the ad content.",
            "hallucination_check": "Verify that all claims, features, benefits, and specifications mentioned in the ad are factually accurate and supported by the reference materials."
        }
        
        # Define verification instructions
        self.verification_instructions = f"""Evaluate the verification and fact-checking by answering these 10 boolean questions.
For each question, answer Yes or No and provide a brief explanation.

{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(EVALUATION_CRITERIA['verification']['questions']))}

FORMAT YOUR RESPONSE AS JSON ARRAY:
[
    {{"Question": "Question 1 text", "Answer": "Yes/No", "Reasoning": "Your reasoning"}},
    {{"Question": "Question 2 text", "Answer": "Yes/No", "Reasoning": "Your reasoning"}},
    ...
]"""
        
        # Define winning ads comparison instructions
        self.winning_ads_comparison_instructions = f"""Compare the ad script to winning ad examples by answering these 10 boolean questions.
For each question, answer Yes or No and provide a brief explanation.

{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(EVALUATION_CRITERIA['winning_ads_comparison']['questions']))}

FORMAT YOUR RESPONSE AS JSON ARRAY:
[
    {{"Question": "Question 1 text", "Answer": "Yes/No", "Reasoning": "Your reasoning"}},
    {{"Question": "Question 2 text", "Answer": "Yes/No", "Reasoning": "Your reasoning"}},
    ...
]"""
    
    def _extract_brief_sections(self, full_text: str) -> List[Dict[str, str]]:
        """
        Extract brief sections from the full text, handling multiple briefs if present.
        Uses LLM to identify and separate briefs in a document.
        
        Args:
            full_text: The complete document text
            
        Returns:
            List of dictionaries, each containing brief, script, debrief, and references sections
        """
        print("Using LLM to extract briefs from document...")
        
        # Create prompt for LLM to identify multiple briefs
        prompt = f"""
        This document contains one or more social media ad briefs (possibly 4 or more briefs).
        
        Your task is to carefully analyze this document and separate each distinct brief. 
        Briefs may be labeled as "Creative Brief #1", "Brief #2", etc., or may use other heading formats.
        
        Each brief typically contains:
        - A brief number or identifier
        - Product/company information
        - Target audience details
        - Brand voice/tone guidelines
        - Key messages or benefits
        - A script section (which may contain dialogue, visuals, timing information)
        
        For each distinct brief you identify, extract:
        1. Brief Number (e.g. Brief #1, Brief #2, etc.)
        2. The full content of that brief including all sections
        
        FORMAT YOUR RESPONSE AS JSON:
        [
          {{
            "brief_number": 1,
            "content": "Full text of Brief #1 including script..."
          }},
          {{
            "brief_number": 2,
            "content": "Full text of Brief #2 including script..."
          }},
          ...
        ]
        
        IMPORTANT INSTRUCTIONS:
        - Be thorough in identifying ALL briefs in the document.
        - Do not merge multiple briefs together.
        - Include brief numbers in your response.
        - Ensure each "content" field contains the complete text of that brief.
        - Look for section headings or patterns that indicate separate briefs.
        
        Document to analyze:
        {full_text}
        """
        
        try:
            # Call the OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specialized in extracting structured content from documents containing ad briefs."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Process the LLM response
            briefs = []
            
            # Check if we have a list of briefs directly
            if isinstance(result, list) and len(result) > 0:
                print(f"LLM extraction found {len(result)} briefs")
                for brief_item in result:
                    brief_number = brief_item.get('brief_number', 0)
                    brief_content = brief_item.get('content', '')
                    
                    if brief_content:
                        brief_sections, missing_sections = self._extract_single_brief_sections(brief_content)
                        brief_sections['brief_number'] = brief_number
                        if missing_sections:
                            brief_sections['missing_sections'] = missing_sections
                        briefs.append(brief_sections)
                
                if briefs:
                    return briefs
            
            # Check if the result has a 'briefs' key (alternative format)
            elif isinstance(result, dict) and 'briefs' in result and isinstance(result['briefs'], list):
                print(f"LLM extraction found {len(result['briefs'])} briefs")
                for brief_item in result['briefs']:
                    brief_number = brief_item.get('brief_number', 0)
                    brief_content = brief_item.get('content', '')
                    
                    if brief_content:
                        brief_sections, missing_sections = self._extract_single_brief_sections(brief_content)
                        brief_sections['brief_number'] = brief_number
                        if missing_sections:
                            brief_sections['missing_sections'] = missing_sections
                        briefs.append(brief_sections)
                
                if briefs:
                    return briefs
            
            print("LLM extraction didn't yield usable results, falling back to regex approach")
            
        except Exception as e:
            print(f"Error during LLM brief extraction: {str(e)}")
            print("Falling back to regex-based brief extraction")
        
        # Fallback: Use regex patterns to extract briefs
        briefs = []
        
        # Try to split on Creative Brief headers with various formats
        # This improved pattern should catch more varieties of brief headers
        brief_pattern = r'(?i)(?:creative\s+)?brief(?:\s+#?\d+)?(?:\s*:|\s*-|\n)'
        brief_splits = re.split(brief_pattern, full_text)
        
        if len(brief_splits) > 1:
            # Found multiple briefs
            print(f"Regex approach found {len(brief_splits)-1} briefs in the document")
            
            # Get all matches to preserve the brief headers
            header_matches = re.finditer(brief_pattern, full_text)
            headers = [m.group(0) for m in header_matches]
            
            # Process each brief, skipping the text before the first brief header
            for i, brief_text in enumerate(brief_splits[1:], 1):
                # Add the header back if available, otherwise use a generic one
                header = headers[i-1] if i-1 < len(headers) else f"Brief #{i}:"
                brief_with_header = f"{header} {brief_text}"
                
                print(f"\nProcessing Brief #{i}")
                
                # Extract sections for this brief
                brief_sections, missing_sections = self._extract_single_brief_sections(brief_with_header)
                brief_sections['brief_number'] = i
                if missing_sections:
                    brief_sections['missing_sections'] = missing_sections
                briefs.append(brief_sections)
        else:
            # Single brief
            print("Document appears to contain a single brief")
            brief_sections, missing_sections = self._extract_single_brief_sections(full_text)
            brief_sections['brief_number'] = 1
            if missing_sections:
                brief_sections['missing_sections'] = missing_sections
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
        
        if reference_docs:
            for doc in reference_docs:
                reference_texts.append(self.script_evaluator.parse_document(doc))
            print(f"\n=== PROCESSED {len(reference_texts)} REFERENCE DOCUMENTS ===")
        
        # Combine additional reference texts
        additional_reference_text = "\n\n".join(reference_texts)
        
        # Evaluate each brief
        all_results = []
        total_briefs = len(briefs)
        
        # Calculate total evaluation steps per brief based on new categories
        total_steps_per_brief = len(self.category_display_names)
        
        # Process each brief
        for idx, brief_sections in enumerate(briefs, 1):
            print(f"\nProcessing Brief #{idx}")
            print(f"Total evaluation steps: {total_steps_per_brief}")
            
            # Initialize current step
            current_step = 1
            
            # Process each brief
            brief_results = {}
            
            # Evaluate all categories
            for criteria in self.category_display_names.keys():
                if progress_callback:
                    progress_callback(idx, total_briefs, current_step, total_steps_per_brief, criteria)
                
                print(f"\n--- Evaluating {self.category_display_names[criteria]} ---")
                
                # Choose evaluation method based on criteria
                # if criteria in ["relevance_clarity"] and brief_sections['brief']:
                #     result = self._evaluate_criteria_with_brief(
                #         script_text=brief_sections['script'],
                #         brief_text=brief_sections['brief'],
                #         criteria=criteria,
                #         instructions=self.evaluation_instructions[criteria]
                #     )
                if criteria in ["system2_validation"] and brief_sections['debrief']:
                    result = self._evaluate_criteria_with_debrief(
                        script_text=brief_sections['script'],
                        debrief_text=brief_sections['debrief'],
                        criteria=criteria,
                        instructions=self.evaluation_instructions[criteria]
                    )
                elif criteria in ["emotional_appeal", "system1_assessment", "hallucination_check",'relevance_clarity'] and (brief_sections['references'] or reference_texts):
                    references_text = f"References in the brief from the documents: {brief_sections['references']} \n\n Script: {brief_sections['script']} \n\n Debrief: {brief_sections['debrief']}"
                    if reference_texts:
                        references_text = references_text + "\n\n" + additional_reference_text if references_text else additional_reference_text
                    
                    result = self._evaluate_criteria_with_references(
                        script_text=brief_sections['script'],
                        references_text=references_text,
                        criteria=criteria,
                        instructions=self.evaluation_instructions[criteria]
                    )
                else:
                    result = self._evaluate_criteria(
                        script_text=brief_sections['script'],
                        criteria=criteria,
                        instructions=self.evaluation_instructions[criteria]
                    )
                
                brief_results[criteria] = result
                current_step += 1
            
            # Generate report for this brief
            brief_report = self._generate_single_brief_report(brief_results)
            brief_report["brief_number"] = idx
            brief_report["brief_title"] = self._extract_brief_title(brief_sections['brief'])
            all_results.append(brief_report)
            
            # Print brief summary
            print(f"\n=== BRIEF #{idx} EVALUATION SUMMARY ===")
            print(f"Title: {brief_report['brief_title']}")
            print(f"Total Score: {brief_report['total_score']}/{brief_report['max_possible_score']} ({brief_report['percentage_score']:.1f}%)")
            print("Category Scores:")
            for category, score in brief_report['category_scores'].items():
                max_score = brief_report['category_max_scores'].get(category, 0)
                print(f"  - {self.category_display_names[category]}: {score}/{max_score}")
            print("Recommendations:")
            for rec in brief_report['recommendations']:
                print(f"  - {rec}")
        
        # Return final results
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
    
    def _evaluate_with_context(self, script_text: str, context_text: str, context_type: str, 
                          criteria: str, instructions: str) -> Dict[str, Any]:
        """Generic evaluation helper that handles various context types.
        
        Args:
            script_text: The ad script text to evaluate
            context_text: Additional context to include (brief, debrief, references)
            context_type: Type of context ("brief", "debrief", "references", or "none")
            criteria: The evaluation criteria to use
            instructions: The evaluation instructions
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Get questions for this category
            questions = self.get_questions_for_category(criteria)
            total_questions = len(questions)
            
            context_description = f" with {context_type} context" if context_type != "none" else ""
            print(f"\nEvaluating {criteria}{context_description}")
            print(f"Script length: {len(script_text)} characters")
            if context_type != "none":
                print(f"{context_type.capitalize()} length: {len(context_text)} characters")
            print(f"Number of questions: {total_questions}")
            
            if total_questions == 0:
                print(f"Warning: No questions found for category {criteria}")
                return {
                    "score": 0.0,
                    "total_questions": total_questions,
                    "yes_count": 0,
                    "evaluation": [],
                    "feedback": f"No questions found for category {criteria}"
                }
            
            # Create the questions JSON array for the prompt
            questions_json = []
            for i, question in enumerate(questions, 1):
                questions_json.append({"id": i, "text": question})
            
            # Build the prompt based on context type
            prompt = f"""Please evaluate this ad script based on the specified criteria."""
            
            prompt += f"""

AD SCRIPT TO EVALUATE:
{script_text}

"""
            
            # Add appropriate context section if available
            if context_type != "none" and context_text:
                prompt += f"""{context_type.upper()} CONTEXT:
{context_text}

"""
            
            # Add criteria and questions
            prompt += f"""EVALUATION CRITERIA:
{instructions}

QUESTIONS TO ANSWER:
{json.dumps(questions_json, indent=2)}

INSTRUCTIONS:
1. Answer each question with 'Yes' or 'No'
2. Provide brief reasoning for each answer
3. Return your evaluation ONLY as JSON with this EXACT structure:

{{
  "evaluation": [
    {{
      "Question": "question text here",
      "Answer": "Yes",
      "Reasoning": "brief explanation here"
    }},
    {{
      "Question": "question text here",
      "Answer": "No",
      "Reasoning": "brief explanation here"
    }}
  ]
}}

DO NOT include any text outside of this JSON structure.
"""
            
            print("\nSending request to OpenAI API...")
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a specialized evaluator that returns well-formatted JSON responses according to the exact schema requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            print("Received response from OpenAI API")
            
            # Parse the response
            try:
                response_content = response.choices[0].message.content.strip()
                print("Raw response:", response_content[:200] + "..." if len(response_content) > 200 else response_content)
                
                # Parse the JSON response
                parsed_json = json.loads(response_content)
                
                # Extract the evaluation array
                if "evaluation" in parsed_json and isinstance(parsed_json["evaluation"], list):
                    evaluation = parsed_json["evaluation"]
                else:
                    # Try to find any array in the response
                    found_array = False
                    for key, value in parsed_json.items():
                        if isinstance(value, list) and len(value) > 0:
                            evaluation = value
                            found_array = True
                            break
                    
                    if not found_array:
                        print("No array found in response")
                        evaluation = []
                
                if not evaluation:
                    print("Failed to extract valid evaluation from response")
                    return {
                        "score": 0.0,
                        "total_questions": total_questions,
                        "yes_count": 0,
                        "evaluation": [],
                        "feedback": "Failed to extract evaluation from response"
                    }
                
                print(f"Parsed {len(evaluation)} evaluation items")
                
                # Calculate raw score (count of "Yes" answers)
                score = sum(1 for item in evaluation if isinstance(item, dict) and item.get("Answer", "").lower() == "yes")
                
                print(f"\nEvaluation Results:")
                print(f"Total Questions: {total_questions}")
                print(f"Yes Answers: {score}")
                print(f"Score: {score}/{total_questions}")
                
                print("\nDetailed Feedback:")
                feedback_items = []
                for item in evaluation:
                    if isinstance(item, dict):
                        question = item.get("Question", "Unknown question")
                        answer = item.get("Answer", "No answer")
                        reasoning = item.get("Reasoning", "No reasoning provided")
                        print(f"Q: {question}")
                        print(f"A: {answer}")
                        print(f"R: {reasoning}\n")
                        feedback_items.append(f"- {question}: {answer} - {reasoning}")
                
                return {
                    "score": score,
                    "total_questions": total_questions,
                    "yes_count": score,
                    "evaluation": evaluation,
                    "feedback": "\n".join(feedback_items)
                }
                
            except json.JSONDecodeError as e:
                print(f"Error parsing evaluation response: {str(e)}")
                print("Raw response:", response.choices[0].message.content)
                return {
                    "score": 0.0,
                    "total_questions": total_questions,
                    "yes_count": 0,
                    "evaluation": [],
                    "feedback": f"Error parsing evaluation response: {str(e)}"
                }
                
        except Exception as e:
            print(f"Error in criteria evaluation: {str(e)}")
            return {
                "score": 0.0,
                "total_questions": total_questions,
                "yes_count": 0,
                "evaluation": [],
                "feedback": f"Error in criteria evaluation: {str(e)}"
            }

    def _evaluate_criteria(self, script_text: str, criteria: str, instructions: str) -> Dict[str, Any]:
        """Evaluate the script based on specified criteria without additional context."""
        return self._evaluate_with_context(
            script_text=script_text,
            context_text="",
            context_type="none",
            criteria=criteria,
            instructions=instructions
        )
    
    def _evaluate_criteria_with_brief(self, script_text: str, brief_text: str, criteria: str, instructions: str) -> Dict[str, Any]:
        """Evaluate a single criteria using LLM with brief context."""
        return self._evaluate_with_context(
            script_text=script_text,
            context_text=brief_text,
            context_type="brief",
            criteria=criteria,
            instructions=instructions
        )
    
    def _evaluate_criteria_with_debrief(self, script_text: str, debrief_text: str, criteria: str, instructions: str) -> Dict[str, Any]:
        """Evaluate a single criteria using LLM with debrief context."""
        return self._evaluate_with_context(
            script_text=script_text,
            context_text=debrief_text,
            context_type="debrief",
            criteria=criteria,
            instructions=instructions
        )
    
    def _evaluate_criteria_with_references(self, script_text: str, references_text: str, criteria: str, instructions: str) -> Dict[str, Any]:
        """Evaluate a single criteria using LLM with references context."""
        return self._evaluate_with_context(
            script_text=script_text,
            context_text=references_text,
            context_type="references",
            criteria=criteria,
            instructions=instructions
        )
    
    def _compare_with_winning_ads(self, script_text: str, winning_ads_text: str, instructions: str) -> Dict[str, Any]:
        """Compare script with winning ad examples and evaluate it."""
        try:
            print(f"\n=== Comparing with Winning Ads ===")
            print(f"Script length: {len(script_text)} characters")
            print(f"Winning ads length: {len(winning_ads_text)} characters")
            
            # Create prompt with detailed instructions
            prompt = f"""
            You are comparing a social media ad script with examples of winning ads.
            
            EVALUATION INSTRUCTIONS:
            {instructions}
            
            SCRIPT TO EVALUATE:
            {script_text}
            
            WINNING ADS TO COMPARE AGAINST:
            {winning_ads_text}
            
            IMPORTANT: Format your response as a JSON array with the following structure:
            [
                {{
                    "Question": "Full question text",
                    "Answer": "Yes/No",
                    "Reasoning": "Detailed explanation with specific references"
                }},
                ...
            ]
            """
            
            print("\nSending request to OpenAI API...")
            # Call the OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert ad evaluator. Compare the script with winning examples and provide responses in the exact JSON format specified."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )
            
            print("Received response from OpenAI API")
            # Parse the response
            evaluation = json.loads(response.choices[0].message.content)
            print(f"Parsed {len(evaluation)} comparison items")
            
            # Calculate score based on Yes/No answers
            total_questions = len(evaluation)
            yes_answers = sum(1 for item in evaluation if item["Answer"].lower() == "yes")
            score = (yes_answers / total_questions) * 10 if total_questions > 0 else 0
            
            print(f"\nComparison Results:")
            print(f"Total Questions: {total_questions}")
            print(f"Yes Answers: {yes_answers}")
            print(f"Score: {score:.1f}/10")
            
            # Format feedback as a string
            feedback_lines = []
            for item in evaluation:
                feedback_lines.append(f"{item['Question']}\nAnswer: {item['Answer']}\nReasoning: {item['Reasoning']}\n")
            formatted_feedback = "\n".join(feedback_lines)
            
            print("\nDetailed Feedback:")
            print(formatted_feedback)
            
            return {
                "score": round(score, 2),
                "reasoning": formatted_feedback,
                "details": evaluation
            }
            
        except Exception as e:
            logger.error(f"Error during winning ads comparison: {str(e)}")
            print(f"\nError during comparison: {str(e)}")
            return {
                "score": 0,
                "reasoning": f"Error during comparison: {str(e)}",
                "details": []
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
    
    def _extract_single_brief_sections(self, text: str) -> tuple[Dict[str, str], List[str]]:
        """
        Extract different sections from a brief document using primarily LLM.
        
        Args:
            text: The brief document text
            
        Returns:
            Tuple containing:
                - Dictionary with extracted brief, script, debrief, and references sections
                - List of missing section names
        """
        # Initialize sections
        sections = {
            "brief": "",
            "script": "",
            "debrief": "",
            "references": ""
        }
        
        # Track missing sections
        missing_sections = []
        
        # Try LLM extraction first
        print("Using LLM to extract sections from brief...")
        extracted_sections = self._extract_all_sections_with_llm(text)
        
        # Update sections with LLM results
        if extracted_sections:
            for section_name, content in extracted_sections.items():
                if content and len(content.strip()) > 0:
                    sections[section_name] = content
                    
        # If script section is missing or empty, try to find it with pattern matching
        if not sections["script"] or len(sections["script"].strip()) == 0:
            print("Script section missing, trying pattern matching")
            script_lines = []
            in_script_section = False
            for line in text.split("\n"):
                line = line.strip()
                if line.lower() == "script:" or line.lower().startswith("script:"):
                    in_script_section = True
                    # Extract any content after "Script:"
                    if ":" in line:
                        script_content = line.split(":", 1)[1].strip()
                        if script_content:
                            script_lines.append(script_content)
                elif in_script_section and line and not any(line.lower().startswith(s) for s in ["brief", "debrief", "references", "thumbnail hook:", "video hook:"]):
                    script_lines.append(line)
                elif line.lower().startswith(("brief", "debrief", "references", "thumbnail hook:", "video hook:")):
                    in_script_section = False
            
            if script_lines:
                sections["script"] = "\n".join(script_lines)
                print(f"Found script section using pattern matching: {len(sections['script'])} characters")
        
        # Check for any missing sections
        for section, content in sections.items():
            if not content or len(content.strip()) == 0:
                missing_sections.append(section)
        
        # Log what we found
        present_sections = [s for s, c in sections.items() if c and len(c.strip()) > 0]
        print(f"Sections extracted: {', '.join(present_sections) if present_sections else 'none'}")
        if missing_sections:
            print(f"Missing sections after extraction: {', '.join(missing_sections)}")
            
        return sections, missing_sections
    
    def _extract_all_sections_with_llm(self, text: str) -> Dict[str, str]:
        """Use LLM to extract all sections from the document in one call."""
        prompt = f"""
        Please analyze this document and extract these specific sections from a social media ad brief:
        
        1. Brief: Extract the creative brief section that includes information about the product, target audience, 
           and marketing goals. This typically includes elements like elevator pitch, length, production style, 
           talent info, brand talking points, etc.
        
        2. Script: Extract the actual ad script section that contains dialogue, visuals, and timing information.
           This is typically labeled as "Main Script" or similar, and often contains time-coded segments
           (e.g., "0-5 sec: [action] VO: dialogue")
        
        3. Debrief: Extract any analytical section that discusses the ad's strategy, principles, or approach.
           This might be labeled as "Debrief", "Analysis", or appear at the end without a clear header.
        
        4. References: Extract any references to external documents, studies, or winning ads mentioned.
        
        Format your response as JSON with these exact keys:
        {{
            "brief": "extracted brief content",
            "script": "extracted script content",
            "debrief": "extracted debrief content",
            "references": "extracted references content"
        }}
        
        If a section is not found, provide an empty string for that key.
        
        Document to analyze:
        {text}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that accurately extracts sections from documents."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate and clean up the result
            for key in ["brief", "script", "debrief", "references"]:
                if key not in result:
                    print(f"Warning: LLM extraction missing '{key}' section")
                    result[key] = ""
                elif result[key] is None:
                    result[key] = ""
            
            return result
        except Exception as e:
            print(f"Error during LLM section extraction: {str(e)}")
            return {}
    
    def _generate_single_brief_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a report for a single brief evaluation."""
        # Log what's in the evaluation results
        print(f"Generating report for evaluation results with keys: {list(evaluation_results.keys())}")
        
        # Define all possible score categories
        score_categories = [
            "creativity", "emotional_appeal", "relevance_clarity", "natural_language",
            "system1_assessment", "system2_validation", "cognitive_harmony",
            "red_flags", "hallucination_check"
        ]
        
        # Initialize total questions count for calculating maximum possible score
        max_possible_score = 0
        
        # Calculate total score
        total_score = 0
        
        # Initialize category scores and feedback for all categories
        category_scores = {}
        category_max_scores = {}
        detailed_feedback = {}
        
        # Set default values for all categories
        for category in score_categories:
            category_scores[category] = 0
            category_max_scores[category] = 0
            detailed_feedback[category] = "Not evaluated"
        
        # Update with actual scores and feedback where available
        for category in score_categories:
            if category in evaluation_results:
                category_result = evaluation_results[category]
                if isinstance(category_result, dict) and "score" in category_result:
                    score = category_result["score"]
                    total_questions = category_result.get("total_questions", 0)
                    print(f"Found score for {category}: {score}/{total_questions}")
                    
                    total_score += score
                    max_possible_score += total_questions
                    
                    category_scores[category] = score
                    category_max_scores[category] = total_questions
                    
                    # Handle feedback based on different formats
                    if "feedback" in category_result:
                        detailed_feedback[category] = category_result["feedback"]
                    elif "reasoning" in category_result:
                        detailed_feedback[category] = category_result["reasoning"]
                    elif "evaluation" in category_result:
                        # Format evaluation items
                        detailed_feedback[category] = category_result["evaluation"]
                else:
                    print(f"WARNING: Category {category} has invalid structure: {category_result}")
            else:
                print(f"WARNING: Category {category} not found in evaluation results - using default score of 0")
        
        print(f"Total score: {total_score}/{max_possible_score} ({(total_score/max_possible_score)*100:.1f}% if max_possible_score > 0 else 0)")
        
        # Generate recommendations based on scores
        recommendations = self._generate_recommendations(evaluation_results, category_max_scores)
        
        # Create final report
        report = {
            "brief_number": evaluation_results.get("brief_number", 1),
            "brief_title": evaluation_results.get("brief_title", "Untitled Brief"),
            "total_score": total_score,
            "max_possible_score": max_possible_score,
            "percentage_score": (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0,
            "category_scores": category_scores,
            "category_max_scores": category_max_scores,
            "verification_result": True,  # Default to true since we're not using separate verification
            "detailed_feedback": detailed_feedback,
            "recommendations": recommendations
        }
        
        return report
    
    def _generate_recommendations(self, results: Dict[str, Any], category_max_scores: Dict[str, int]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Add strengths
        strengths = []
        for category, result in results.items():
            if category in ["verification", "brief_number", "brief_title"]:
                continue
                
            if isinstance(result, dict) and "score" in result:
                score = result["score"]
                max_score = category_max_scores.get(category, 0)
                if max_score > 0 and score / max_score >= 0.8:  # 80% or better is a strength
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
                max_score = category_max_scores.get(category, 0)
                if max_score > 0 and score / max_score < 0.6:  # Less than 60% needs improvement
                    reasoning = result.get("reasoning", "")
                    # Handle both string and dictionary reasoning
                    if isinstance(reasoning, str):
                        reasoning_text = reasoning.split(".")[0] if reasoning else ""
                    elif isinstance(reasoning, dict):
                        reasoning_text = reasoning.get("details", "") or reasoning.get("summary", "") or ""
                    else:
                        reasoning_text = ""
                    improvements.append(f"{category.replace('_', ' ').title()}: {reasoning_text}.")
        
        if improvements:
            recommendations.append("Areas for Improvement:")
            recommendations.extend(improvements)
        
        # Add verification recommendation if failed
        verification = results.get("verification", {})
        if isinstance(verification, dict):
            if not verification.get("passed", False):
                details = verification.get("details", "")
                if isinstance(details, str):
                    details_text = details.split(".")[0] if details else ""
                elif isinstance(details, dict):
                    details_text = details.get("summary", "") or ""
                else:
                    details_text = ""
                recommendations.append(f"Fact Verification Failed: {details_text}.")
        
        # Add winning ads comparison highlights
        winning_comparison = results.get("winning_ads_comparison", {})
        if isinstance(winning_comparison, dict) and "score" in winning_comparison:
            winning_score = winning_comparison["score"]
            winning_max = winning_comparison.get("total_questions", 10)
            if winning_max > 0 and winning_score / winning_max >= 0.7:  # 70% or better
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
                           "Creativity", "Emotional Appeal", "Relevance & Clarity", "Natural Language",
                           "System 1 Assessment", "System 2 Validation", "Cognitive Harmony",
                           "Red Flags", "Hallucination Check"])
            
            # Data rows for each brief
            for brief in evaluation_results["evaluations"]:
                category_scores = brief.get("category_scores", {})
                category_max_scores = brief.get("category_max_scores", {})
                
                writer.writerow([
                    brief.get("brief_number", ""),
                    brief.get("brief_title", ""),
                    f"{brief.get('total_score', 0)}",
                    brief.get("max_possible_score", 0),
                    f"{brief.get('percentage_score', 0):.1f}%",
                    f"{category_scores.get('creativity', 0)}/{category_max_scores.get('creativity', 0)}",
                    f"{category_scores.get('emotional_appeal', 0)}/{category_max_scores.get('emotional_appeal', 0)}",
                    f"{category_scores.get('relevance_clarity', 0)}/{category_max_scores.get('relevance_clarity', 0)}",
                    f"{category_scores.get('natural_language', 0)}/{category_max_scores.get('natural_language', 0)}",
                    f"{category_scores.get('system1_assessment', 0)}/{category_max_scores.get('system1_assessment', 0)}",
                    f"{category_scores.get('system2_validation', 0)}/{category_max_scores.get('system2_validation', 0)}",
                    f"{category_scores.get('cognitive_harmony', 0)}/{category_max_scores.get('cognitive_harmony', 0)}",
                    f"{category_scores.get('red_flags', 0)}/{category_max_scores.get('red_flags', 0)}",
                    f"{category_scores.get('hallucination_check', 0)}/{category_max_scores.get('hallucination_check', 0)}"
                ])
        else:
            # Single brief case
            brief = evaluation_results
            category_scores = brief.get("category_scores", {})
            category_max_scores = brief.get("category_max_scores", {})
            
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
            for category, score in category_scores.items():
                max_score = category_max_scores.get(category, 0)
                writer.writerow([category.replace("_", " ").title(), f"{score}/{max_score}"])
            
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
                category_max_scores = brief.get("category_max_scores", {})
                for category, score in categories.items():
                    category_display = category.replace("_", " ").title()
                    max_score = category_max_scores.get(category, 0)
                    score_percentage = (score / max_score * 100) if max_score > 0 else 0
                    score_class = "high-score" if score_percentage >= 70 else "medium-score" if score_percentage >= 50 else "low-score"
                    
                    html_output += f'''
                    <tr>
                        <td>{category_display}</td>
                        <td class="score {score_class}">{score}/{max_score}</td>
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
            category_max_scores = brief.get("category_max_scores", {})
            for category, score in categories.items():
                category_display = category.replace("_", " ").title()
                max_score = category_max_scores.get(category, 0)
                score_percentage = (score / max_score * 100) if max_score > 0 else 0
                score_class = "high-score" if score_percentage >= 70 else "medium-score" if score_percentage >= 50 else "low-score"
                
                html_output += f'''
                <tr>
                    <td>{category_display}</td>
                    <td class="score {score_class}">{score}/{max_score}</td>
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

    def _evaluate_creativity(self, script: str) -> Tuple[float, str]:
        """Evaluate the creativity of the ad script."""
        try:
            prompt = get_evaluation_prompt("creativity", script)
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            content = response.choices[0].message.content
            
            # Parse the JSON response
            parsed_response = json.loads(content)
            
            # Count the number of "Yes" answers
            yes_count = 0
            evaluation_details = []
            
            # Ensure we have a list of questions with answers
            if isinstance(parsed_response, list):
                for item in parsed_response:
                    if isinstance(item, dict) and "Answer" in item:
                        answer = item.get("Answer", "").strip().lower()
                        if answer == "yes":
                            yes_count += 1
                        
                        # Format the item for display
                        question = item.get("Question", "Unknown question")
                        reasoning = item.get("Reasoning", "No reasoning provided")
                        evaluation_details.append(f"{question}: {answer.upper()} - {reasoning}")
            
            # Cap the score at 10
            score = min(yes_count, 10)
            
            # Format the reasoning as a detailed list
            formatted_reasoning = "\n\n".join(evaluation_details)
            
            return score, formatted_reasoning
        except Exception as e:
            print(f"Error in creativity evaluation: {str(e)}")
            return 0.0, f"Error in creativity evaluation: {str(e)}"

    def _evaluate_natural_language(self, script: str) -> Tuple[float, str]:
        """Evaluate the natural language and tone."""
        try:
            prompt = get_evaluation_prompt("natural_language", script)
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            content = response.choices[0].message.content
            
            # Parse the JSON response
            parsed_response = json.loads(content)
            
            # Count the number of "Yes" answers
            yes_count = 0
            evaluation_details = []
            
            # Ensure we have a list of questions with answers
            if isinstance(parsed_response, list):
                for item in parsed_response:
                    if isinstance(item, dict) and "Answer" in item:
                        answer = item.get("Answer", "").strip().lower()
                        if answer == "yes":
                            yes_count += 1
                        
                        # Format the item for display
                        question = item.get("Question", "Unknown question")
                        reasoning = item.get("Reasoning", "No reasoning provided")
                        evaluation_details.append(f"{question}: {answer.upper()} - {reasoning}")
            
            # Cap the score at 10
            score = min(yes_count, 10)
            
            # Format the reasoning as a detailed list
            formatted_reasoning = "\n\n".join(evaluation_details)
            
            return score, formatted_reasoning
        except Exception as e:
            print(f"Error in natural language evaluation: {str(e)}")
            return 0.0, f"Error in natural language evaluation: {str(e)}"

    def _evaluate_ad_brief_alignment(self, script: str, brief: str) -> Tuple[float, str]:
        """Evaluate how well the ad script aligns with the provided brief."""
        try:
            prompt = get_evaluation_prompt("ad_brief_alignment", script, f"Brief:\n{brief}")
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            content = response.choices[0].message.content
            
            # Parse the JSON response
            parsed_response = json.loads(content)
            
            # Count the number of "Yes" answers
            yes_count = 0
            evaluation_details = []
            
            # Ensure we have a list of questions with answers
            if isinstance(parsed_response, list):
                for item in parsed_response:
                    if isinstance(item, dict) and "Answer" in item:
                        answer = item.get("Answer", "").strip().lower()
                        if answer == "yes":
                            yes_count += 1
                        
                        # Format the item for display
                        question = item.get("Question", "Unknown question")
                        reasoning = item.get("Reasoning", "No reasoning provided")
                        evaluation_details.append(f"{question}: {answer.upper()} - {reasoning}")
            
            # Cap the score at 10
            score = min(yes_count, 10)
            
            # Format the reasoning as a detailed list
            formatted_reasoning = "\n\n".join(evaluation_details)
            
            return score, formatted_reasoning
        except Exception as e:
            print(f"Error in ad brief alignment evaluation: {str(e)}")
            return 0.0, f"Error in ad brief alignment evaluation: {str(e)}"

    def _evaluate_relevance_clarity(self, script: str, brief: str) -> Tuple[float, str]:
        """Evaluate the relevance and clarity of the ad script."""
        try:
            prompt = get_evaluation_prompt("relevance_clarity", script, f"Brief:\n{brief}")
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            content = response.choices[0].message.content
            
            # Parse the JSON response
            parsed_response = json.loads(content)
            
            # Count the number of "Yes" answers
            yes_count = 0
            evaluation_details = []
            
            # Ensure we have a list of questions with answers
            if isinstance(parsed_response, list):
                for item in parsed_response:
                    if isinstance(item, dict) and "Answer" in item:
                        answer = item.get("Answer", "").strip().lower()
                        if answer == "yes":
                            yes_count += 1
                        
                        # Format the item for display
                        question = item.get("Question", "Unknown question")
                        reasoning = item.get("Reasoning", "No reasoning provided")
                        evaluation_details.append(f"{question}: {answer.upper()} - {reasoning}")
            
            # Cap the score at 10
            score = min(yes_count, 10)
            
            # Format the reasoning as a detailed list
            formatted_reasoning = "\n\n".join(evaluation_details)
            
            return score, formatted_reasoning
        except Exception as e:
            print(f"Error in relevance and clarity evaluation: {str(e)}")
            return 0.0, f"Error in relevance and clarity evaluation: {str(e)}"

    def _evaluate_debrief_analysis(self, script: str, debrief: str) -> Tuple[float, str]:
        """Evaluate how well the ad script incorporates feedback from the debrief."""
        try:
            prompt = get_evaluation_prompt("debrief_analysis", script, f"Debrief:\n{debrief}")
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            content = response.choices[0].message.content
            
            # Parse the JSON response
            parsed_response = json.loads(content)
            
            # Count the number of "Yes" answers
            yes_count = 0
            evaluation_details = []
            
            # Ensure we have a list of questions with answers
            if isinstance(parsed_response, list):
                for item in parsed_response:
                    if isinstance(item, dict) and "Answer" in item:
                        answer = item.get("Answer", "").strip().lower()
                        if answer == "yes":
                            yes_count += 1
                        
                        # Format the item for display
                        question = item.get("Question", "Unknown question")
                        reasoning = item.get("Reasoning", "No reasoning provided")
                        evaluation_details.append(f"{question}: {answer.upper()} - {reasoning}")
            
            # Cap the score at 10
            score = min(yes_count, 10)
            
            # Format the reasoning as a detailed list
            formatted_reasoning = "\n\n".join(evaluation_details)
            
            return score, formatted_reasoning
        except Exception as e:
            print(f"Error in debrief analysis evaluation: {str(e)}")
            return 0.0, f"Error in debrief analysis evaluation: {str(e)}"

    def _evaluate_emotional_appeal(self, script: str, references: List[str]) -> Tuple[float, str]:
        """Evaluate the emotional appeal and persuasiveness."""
        try:
            prompt = get_evaluation_prompt("emotional_appeal", script, f"References:\n{chr(10).join(references)}")
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            content = response.choices[0].message.content
            
            # Parse the JSON response
            parsed_response = json.loads(content)
            
            # Count the number of "Yes" answers
            yes_count = 0
            evaluation_details = []
            
            # Ensure we have a list of questions with answers
            if isinstance(parsed_response, list):
                for item in parsed_response:
                    if isinstance(item, dict) and "Answer" in item:
                        answer = item.get("Answer", "").strip().lower()
                        if answer == "yes":
                            yes_count += 1
                        
                        # Format the item for display
                        question = item.get("Question", "Unknown question")
                        reasoning = item.get("Reasoning", "No reasoning provided")
                        evaluation_details.append(f"{question}: {answer.upper()} - {reasoning}")
            
            # Cap the score at 10
            score = min(yes_count, 10)
            
            # Format the reasoning as a detailed list
            formatted_reasoning = "\n\n".join(evaluation_details)
            
            return score, formatted_reasoning
        except Exception as e:
            print(f"Error in emotional appeal evaluation: {str(e)}")
            return 0.0, f"Error in emotional appeal evaluation: {str(e)}"

    def _verify_facts(self, script_text: str, reference_text: str, instructions: str) -> Dict[str, Any]:
        """Verify facts in the script against reference documents."""
        try:
            print(f"\n=== Verifying Facts ===")
            print(f"Script length: {len(script_text)} characters")
            print(f"Reference text length: {len(reference_text)} characters")
            
            # Create prompt with detailed instructions
            prompt = f"""
            You are verifying facts in a social media ad script against reference documents.
            
            EVALUATION INSTRUCTIONS:
            {instructions}
            
            SCRIPT TO VERIFY:
            {script_text}
            
            REFERENCE DOCUMENTS:
            {reference_text}
            
            IMPORTANT: Format your response as a JSON array with the following structure:
            [
                {{
                    "Question": "Full question text",
                    "Answer": "Yes/No",
                    "Reasoning": "Detailed explanation with specific references"
                }},
                ...
            ]
            """
            
            print("\nSending request to OpenAI API...")
            # Call the OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert fact-checker. Verify all claims against reference documents and provide responses in the exact JSON format specified."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )
            
            print("Received response from OpenAI API")
            # Parse the response
            evaluation = json.loads(response.choices[0].message.content)
            print(f"Parsed {len(evaluation)} verification items")
            
            # Calculate score based on Yes/No answers
            total_questions = len(evaluation)
            yes_answers = sum(1 for item in evaluation if item["Answer"].lower() == "yes")
            score = (yes_answers / total_questions) * 10 if total_questions > 0 else 0
            
            # Verification passes only if all answers are "Yes"
            passed = yes_answers == total_questions
            
            print(f"\nVerification Results:")
            print(f"Total Questions: {total_questions}")
            print(f"Yes Answers: {yes_answers}")
            print(f"Score: {score:.1f}/10")
            print(f"Verification Status: {'PASSED' if passed else 'FAILED'}")
            
            # Format feedback as a string
            feedback_lines = []
            for item in evaluation:
                feedback_lines.append(f"{item['Question']}\nAnswer: {item['Answer']}\nReasoning: {item['Reasoning']}\n")
            formatted_feedback = "\n".join(feedback_lines)
            
            print("\nDetailed Feedback:")
            print(formatted_feedback)
            
            return {
                "passed": passed,
                "score": round(score, 2),
                "reasoning": formatted_feedback,
                "details": evaluation
            }
            
        except Exception as e:
            logger.error(f"Error during fact verification: {str(e)}")
            print(f"\nError during verification: {str(e)}")
            return {
                "passed": False,
                "score": 0,
                "reasoning": f"Error during verification: {str(e)}",
                "details": []
            }

    def _evaluate_winning_ads_comparison(self, script: str, winning_ads: List[str]) -> Tuple[float, str]:
        """Compare the ad script to winning ad examples."""
        try:
            prompt = get_evaluation_prompt("winning_ads_comparison", script, f"Winning Ads:\n{chr(10).join(winning_ads)}")
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            content = response.choices[0].message.content
            
            # Parse the JSON response
            parsed_response = json.loads(content)
            
            # Count the number of "Yes" answers
            yes_count = 0
            evaluation_details = []
            
            # Ensure we have a list of questions with answers
            if isinstance(parsed_response, list):
                for item in parsed_response:
                    if isinstance(item, dict) and "Answer" in item:
                        answer = item.get("Answer", "").strip().lower()
                        if answer == "yes":
                            yes_count += 1
                        
                        # Format the item for display
                        question = item.get("Question", "Unknown question")
                        reasoning = item.get("Reasoning", "No reasoning provided")
                        evaluation_details.append(f"{question}: {answer.upper()} - {reasoning}")
            
            # Cap the score at 10
            score = min(yes_count, 10)
            
            # Format the reasoning as a detailed list
            formatted_reasoning = "\n\n".join(evaluation_details)
            
            return score, formatted_reasoning
        except Exception as e:
            print(f"Error in winning ads comparison evaluation: {str(e)}")
            return 0.0, f"Error in winning ads comparison evaluation: {str(e)}"

    def get_questions_for_category(self, category: str) -> list:
        """Get the list of questions for a specific category."""
        return self.evaluation_questions.get(category, [])

def main():
    """
    Main function to test the SocialMediaAdEvaluator with sample inputs.
    """
    # Sample data
    sample_brief = """
    Creative Brief #1: "The Hair Loss Solution"
    
    Product: HairMax Pro - Advanced hair growth treatment
    Target Audience: Men and women aged 25-45 experiencing hair loss
    Key Message: Revolutionary hair growth solution with proven results
    
    Brand Voice: Professional yet approachable
    Tone: Informative and solution-focused
    Key Benefits:
    - Clinically proven results
    - Natural ingredients
    - 90-day money-back guarantee
    - Fast-acting formula
    
    Call to Action: Visit HairMaxPro.com for a free consultation
    """

    sample_script = """
    [0-5 sec]
    VISUAL: Close-up of healthy, thick hair
    VO: "Tired of watching your hair thin day by day?"

    [6-15 sec]
    VISUAL: Split screen showing before/after results
    VO: "Introducing HairMax Pro, the revolutionary hair growth treatment that's changing lives."

    [16-25 sec]
    VISUAL: Scientific laboratory footage
    VO: "With clinically proven results and natural ingredients, HairMax Pro works from the inside out."

    [26-35 sec]
    VISUAL: Satisfied customer testimonials
    VO: "Join thousands who have restored their confidence with HairMax Pro."

    [36-45 sec]
    VISUAL: Product demonstration
    VO: "Try HairMax Pro risk-free with our 90-day money-back guarantee."

    [46-60 sec]
    VISUAL: Call-to-action screen
    VO: "Visit HairMaxPro.com today for your free consultation. Your journey to thicker, healthier hair starts now."
    """

    sample_debrief = """
    Previous Ad Analysis:
    - Strengths: Clear value proposition, strong emotional appeal
    - Areas for Improvement: 
      * Add more specific product benefits
      * Include social proof elements
      * Strengthen call to action
    - Target Audience Response: Positive to before/after visuals
    """

    sample_references = """
    Product Research:
    - Clinical trials show 85% improvement in hair growth
    - Natural ingredients include biotin, keratin, and vitamin E
    - 90% customer satisfaction rate
    
    Market Research:
    - Target demographic spends $500+ annually on hair care
    - 60% of adults experience hair loss by age 35
    - Natural hair growth solutions growing 25% annually
    """

    sample_winning_ads = """
    Winning Ad Example 1:
    [0-5 sec] Emotional hook showing hair loss impact
    [6-15 sec] Scientific credibility with lab footage
    [16-25 sec] Before/after transformations
    [26-35 sec] Customer testimonials
    [36-45 sec] Product benefits and features
    [46-60 sec] Strong call to action

    Winning Ad Example 2:
    [0-5 sec] Problem statement with relatable scenario
    [6-15 sec] Solution introduction with product demo
    [16-25 sec] Key benefits with visual support
    [26-35 sec] Social proof elements
    [36-45 sec] Risk reversal (money-back guarantee)
    [46-60 sec] Clear call to action
    """

    # Create temporary files for testing
    import tempfile
    import os

    def create_temp_file(content, prefix):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, prefix=prefix, suffix='.txt') as f:
            f.write(content)
            return f.name

    # Create temporary files
    script_file = create_temp_file(sample_script, 'script_')
    brief_file = create_temp_file(sample_brief, 'brief_')
    debrief_file = create_temp_file(sample_debrief, 'debrief_')
    references_file = create_temp_file(sample_references, 'references_')
    winning_ads_file = create_temp_file(sample_winning_ads, 'winning_ads_')

    try:
        # Initialize evaluator
        evaluator = SocialMediaAdEvaluator(api_key=api_key, model_name=model_name)

        # Define progress callback
        def progress_callback(brief_num, total_briefs, current_step, total_steps, current_criteria=None):
            print(f"\nProgress: Brief {brief_num}/{total_briefs}")
            print(f"Step {current_step}/{total_steps}")
            if current_criteria:
                print(f"Evaluating: {current_criteria}")

        # Run evaluation
        print("\nStarting evaluation...")
        results = evaluator.evaluate_ad_script(
            script_path=script_file,
            brief_path=brief_file,
            reference_docs=[references_file],
            debrief_path=debrief_file,
            winning_ads_paths=[winning_ads_file],
            progress_callback=progress_callback
        )

        # Print results
        print("\nEvaluation Results:")
        print("=" * 50)
        
        for brief in results['evaluations']:
            print(f"\nBrief #{brief['brief_number']}: {brief['brief_title']}")
            print(f"Total Score: {brief['total_score']}/{brief['max_possible_score']} ({brief['percentage_score']:.1f}%)")
            print("\nCategory Scores:")
            for category, score in brief['category_scores'].items():
                max_score = brief['category_max_scores'].get(category, 0)
                print(f"  - {self.category_display_names[category]}: {score}/{max_score}")
            print(f"Verification: {'PASSED' if brief['verification_result'] else 'FAILED'}")
            print("\nRecommendations:")
            for rec in brief['recommendations']:
                print(f"  - {rec}")

        # Generate reports
        print("\nGenerating reports...")
        
        # JSON Report
        json_report = evaluator.generate_report(results)
        print("\nJSON Report generated successfully")
        
        # CSV Report
        csv_report = evaluator.generate_csv_report(results)
        print("CSV Report generated successfully")
        
        # HTML Report
        html_report = evaluator.generate_html_report(results)
        print("HTML Report generated successfully")

    finally:
        # Clean up temporary files
        for file in [script_file, brief_file, debrief_file, references_file, winning_ads_file]:
            try:
                os.unlink(file)
            except Exception as e:
                print(f"Error deleting temporary file {file}: {e}")

if __name__ == "__main__":
    main() 