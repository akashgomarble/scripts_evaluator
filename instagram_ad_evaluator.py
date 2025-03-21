import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langsmith import wrappers, Client
from openai import OpenAI
from script_evaluator import ScriptEvaluator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class InstagramAdEvaluationCriteria(BaseModel):
    score: float = Field(description="Score between 0 and 10")
    reasoning: str = Field(description="Detailed reasoning for the score")

class VerificationResult(BaseModel):
    passed: bool = Field(description="Whether the verification passed (True) or failed (False)")
    details: str = Field(description="Details about the verification result")

class InstagramAdEvaluator:
    def __init__(self):
        # Initialize LangSmith client and OpenAI wrapper
        self.client = Client()
        self.openai_client = wrappers.wrap_openai(OpenAI())
        self.script_evaluator = ScriptEvaluator()
        
        # Define evaluation instructions for each criteria
        self.criteria_instructions = {
            "creativity": """Evaluate the creativity of the Instagram ad script (score 0-10):
            - Does the ad stand out and capture attention effectively?
            - Is the idea original and creatively executed?
            - Does the script evoke emotional engagement, curiosity, or intrigue?
            Provide a score between 0 and 10 with detailed reasoning.""",
            
            "ad_brief_alignment": """Evaluate the ad brief alignment (score 0-10):
            - Does the script accurately align with the provided brief?
            - Are all critical elements mentioned in the brief clearly reflected?
            - Is the tone and messaging consistent with the brief's specified audience and objective?
            Provide a score between 0 and 10 with detailed reasoning.""",
            
            "debrief_analysis": """Evaluate the debrief analysis (score 0-10):
            - Has the LLM addressed key feedback and insights from previous successful ad scripts?
            - Is there an improvement in identified areas from past evaluations?
            - Does the script avoid previously noted weaknesses?
            Provide a score between 0 and 10 with detailed reasoning.""",
            
            "hallucination_check": """Evaluate the hallucination check (score 0-10):
            - Has every claim, feature, or benefit stated by the LLM been verified against provided documentation?
            - Are there any instances of hallucination or inaccuracies?
            - Is all factual information accurately represented and verifiable against provided documents?
            Provide a score between 0 and 10 with detailed reasoning.""",
            
            "relevance_clarity": """Evaluate the relevance and clarity (score 0-10):
            - Does the script clearly convey the intended message and product/service benefits?
            - Is the language simple, concise, and easily understandable for the target audience?
            - Does the script directly address audience pain points, desires, or needs as mentioned in the brief?
            Provide a score between 0 and 10 with detailed reasoning.""",
            
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
    
    def evaluate_ad_script(self, 
                          script_path: str, 
                          brief_path: Optional[str] = None,
                          reference_docs: List[str] = None,
                          debrief_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate an Instagram ad script based on specified criteria
        
        Args:
            script_path: Path to the ad script file
            brief_path: Path to the ad brief file
            reference_docs: List of paths to reference documents
            debrief_path: Path to the debrief document (previous feedback)
            
        Returns:
            Dictionary with evaluation results
        """
        # Parse documents
        script_text = self.script_evaluator.parse_document(script_path)
        brief_text = self.script_evaluator.parse_document(brief_path) if brief_path else ""
        debrief_text = self.script_evaluator.parse_document(debrief_path) if debrief_path else ""
        reference_texts = []
        
        if reference_docs:
            for doc in reference_docs:
                reference_texts.append(self.script_evaluator.parse_document(doc))
        
        reference_text = "\n\n".join(reference_texts)
        
        # Create dataset for tracking
        dataset = self.client.create_dataset(
            dataset_name="Instagram Ad Evaluation",
            description="Evaluation of Instagram ad script"
        )
        
        # Prepare inputs for evaluation
        eval_inputs = {
            "script": script_text,
            "brief": brief_text,
            "reference_docs": reference_text,
            "debrief": debrief_text
        }
        
        # Add example to dataset
        self.client.create_examples(
            inputs=[eval_inputs],
            outputs=[{}],
            dataset_id=dataset.id
        )
        
        # Evaluate each criteria
        evaluation_results = {}
        
        # Criteria that don't need reference documents
        for criteria in ["creativity", "natural_language"]:
            result = self._evaluate_criteria(
                script_text=script_text,
                criteria=criteria,
                instructions=self.criteria_instructions[criteria]
            )
            evaluation_results[criteria] = result
        
        # Criteria that need brief
        if brief_text:
            for criteria in ["ad_brief_alignment", "relevance_clarity"]:
                result = self._evaluate_criteria_with_brief(
                    script_text=script_text,
                    brief_text=brief_text,
                    criteria=criteria,
                    instructions=self.criteria_instructions[criteria]
                )
                evaluation_results[criteria] = result
        else:
            for criteria in ["ad_brief_alignment", "relevance_clarity"]:
                evaluation_results[criteria] = {
                    "score": 0,
                    "reasoning": "Brief document not provided. Cannot evaluate this criteria."
                }
        
        # Criteria that need debrief
        if debrief_text:
            result = self._evaluate_criteria_with_debrief(
                script_text=script_text,
                debrief_text=debrief_text,
                criteria="debrief_analysis",
                instructions=self.criteria_instructions["debrief_analysis"]
            )
            evaluation_results["debrief_analysis"] = result
        else:
            evaluation_results["debrief_analysis"] = {
                "score": 0,
                "reasoning": "Debrief document not provided. Cannot evaluate this criteria."
            }
        
        # Criteria that need reference documents
        if reference_text:
            for criteria in ["hallucination_check", "emotional_appeal"]:
                result = self._evaluate_criteria_with_references(
                    script_text=script_text,
                    reference_text=reference_text,
                    criteria=criteria,
                    instructions=self.criteria_instructions[criteria]
                )
                evaluation_results[criteria] = result
                
            # Verification (pass/fail)
            verification = self._verify_facts(
                script_text=script_text,
                reference_text=reference_text,
                instructions=self.verification_instructions
            )
            evaluation_results["verification"] = verification
        else:
            for criteria in ["hallucination_check", "emotional_appeal"]:
                evaluation_results[criteria] = {
                    "score": 0,
                    "reasoning": "Reference documents not provided. Cannot evaluate this criteria."
                }
            
            evaluation_results["verification"] = {
                "passed": False,
                "details": "Reference documents not provided. Cannot verify facts."
            }
        
        return evaluation_results
    
    def _evaluate_criteria(self, script_text: str, criteria: str, instructions: str) -> Dict[str, Any]:
        """Evaluate a single criteria using LLM."""
        try:
            response = self.openai_client.beta.chat.completions.parse(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Script: {script_text}"}
                ],
                response_format=InstagramAdEvaluationCriteria
            )
            
            result = response.choices[0].message.parsed
            
            return {
                "score": result.score,
                "reasoning": result.reasoning
            }
        except Exception as e:
            return {
                "score": 0,
                "reasoning": f"Error evaluating {criteria}: {str(e)}"
            }
    
    def _evaluate_criteria_with_brief(self, script_text: str, brief_text: str, criteria: str, instructions: str) -> Dict[str, Any]:
        """Evaluate a criteria that requires the brief document."""
        try:
            response = self.openai_client.beta.chat.completions.parse(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Script: {script_text}\n\nBrief: {brief_text}"}
                ],
                response_format=InstagramAdEvaluationCriteria
            )
            
            result = response.choices[0].message.parsed
            
            return {
                "score": result.score,
                "reasoning": result.reasoning
            }
        except Exception as e:
            return {
                "score": 0,
                "reasoning": f"Error evaluating {criteria}: {str(e)}"
            }
    
    def _evaluate_criteria_with_debrief(self, script_text: str, debrief_text: str, criteria: str, instructions: str) -> Dict[str, Any]:
        """Evaluate a criteria that requires the debrief document."""
        try:
            response = self.openai_client.beta.chat.completions.parse(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Script: {script_text}\n\nDebrief (Previous Feedback): {debrief_text}"}
                ],
                response_format=InstagramAdEvaluationCriteria
            )
            
            result = response.choices[0].message.parsed
            
            return {
                "score": result.score,
                "reasoning": result.reasoning
            }
        except Exception as e:
            return {
                "score": 0,
                "reasoning": f"Error evaluating {criteria}: {str(e)}"
            }
    
    def _evaluate_criteria_with_references(self, script_text: str, reference_text: str, criteria: str, instructions: str) -> Dict[str, Any]:
        """Evaluate a criteria that requires reference documents."""
        try:
            response = self.openai_client.beta.chat.completions.parse(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Script: {script_text}\n\nReference Documents: {reference_text}"}
                ],
                response_format=InstagramAdEvaluationCriteria
            )
            
            result = response.choices[0].message.parsed
            
            return {
                "score": result.score,
                "reasoning": result.reasoning
            }
        except Exception as e:
            return {
                "score": 0,
                "reasoning": f"Error evaluating {criteria}: {str(e)}"
            }
    
    def _verify_facts(self, script_text: str, reference_text: str, instructions: str) -> Dict[str, Any]:
        """Verify facts in the script against reference documents."""
        try:
            response = self.openai_client.beta.chat.completions.parse(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Script: {script_text}\n\nReference Documents: {reference_text}"}
                ],
                response_format=VerificationResult
            )
            
            result = response.choices[0].message.parsed
            
            return {
                "passed": result.passed,
                "details": result.details
            }
        except Exception as e:
            return {
                "passed": False,
                "details": f"Error during verification: {str(e)}"
            }
    
    def generate_report(self, evaluation_results: Dict[str, Any], output_path: str = None) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        # Calculate total score (excluding verification which is pass/fail)
        score_categories = [
            "creativity", "ad_brief_alignment", "debrief_analysis", 
            "hallucination_check", "relevance_clarity", "emotional_appeal", "natural_language"
        ]
        
        total_score = sum(
            evaluation_results.get(category, {}).get("score", 0) 
            for category in score_categories
        )
        
        max_possible_score = 10 * len(score_categories)  # 7 categories, 10 points each
        
        # Generate recommendations based on scores
        recommendations = self._generate_recommendations(evaluation_results)
        
        # Create final report
        report = {
            "total_score": total_score,
            "max_possible_score": max_possible_score,
            "percentage_score": (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0,
            "category_scores": {
                category: evaluation_results.get(category, {}).get("score", 0)
                for category in score_categories
            },
            "verification_result": evaluation_results.get("verification", {}).get("passed", False),
            "detailed_feedback": {
                category: evaluation_results.get(category, {}).get("reasoning", "Not evaluated")
                for category in score_categories + ["verification"]
            },
            "recommendations": recommendations
        }
        
        # Save report if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)
        
        return report
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Add strengths
        strengths = []
        for category, result in results.items():
            if category == "verification":
                continue
                
            score = result.get("score", 0)
            if score >= 8:
                strengths.append(category.replace("_", " ").title())
        
        if strengths:
            recommendations.append(f"Key Strengths: {', '.join(strengths)}")
        
        # Add areas for improvement
        improvements = []
        for category, result in results.items():
            if category == "verification":
                continue
                
            score = result.get("score", 0)
            if score < 6:
                improvements.append(f"{category.replace('_', ' ').title()}: {result.get('reasoning', '').split('.')[0]}.")
        
        if improvements:
            recommendations.append("Areas for Improvement:")
            recommendations.extend(improvements)
        
        # Add verification recommendation if failed
        verification = results.get("verification", {})
        if not verification.get("passed", False):
            recommendations.append(f"Fact Verification Failed: {verification.get('details', '').split('.')[0]}.")
        
        # Add general recommendation if needed
        if not recommendations:
            recommendations.append("No specific recommendations. The ad script performs well across all criteria.")
        
        return recommendations 