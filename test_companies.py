import os
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

# Import from the social_media_ad_evaluator_app.py
from social_media_ad_evaluator import SocialMediaAdEvaluator
from social_media_ad_evaluator_app import save_uploaded_file, save_text_to_file, add_download_buttons, create_radar_chart

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompanyFolderEvaluator:
    """Evaluates social media ad scripts organized in company folders."""
    
    def __init__(self, main_folder: str, output_folder: str, api_key: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            main_folder: Root folder containing company folders
            output_folder: Where to save results
            api_key: Optional OpenAI API key
        """
        self.main_folder = os.path.abspath(main_folder)
        self.output_folder = os.path.abspath(output_folder)
        self.api_key = api_key
        
        # Initialize the evaluator class from the original module
        self.evaluator = SocialMediaAdEvaluator(api_key=self.api_key)
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Store all results
        self.results = {}
        
        # Add to __init__ method
        self.patch_evaluator_methods()
    
    def patch_evaluator_methods(self):
        """
        Patch methods in the SocialMediaAdEvaluator class to prioritize LLM extraction
        for handling multiple briefs and script sections.
        """
        # Store original methods for potential fallback
        original_extract_briefs = self.evaluator._extract_brief_sections
        
        def patched_extract_briefs(full_text):
            """
            Enhanced version of _extract_brief_sections that prioritizes LLM extraction
            for identifying multiple briefs in a document.
            """
            logger.info("Attempting to extract briefs using LLM")
            
            # First try: Use LLM to identify and separate briefs
            prompt = f"""
            This document contains multiple social media ad briefs.
            
            Please analyze this document and separate each distinct brief.
            
            For each brief, extract:
            1. Brief Number (e.g. Brief #1, Brief #2, etc.)
            2. The full content of that brief including any script sections
            
            Format your response as a JSON array of briefs:
            [
              {{
                "brief_number": 1,
                "content": "Full text of Brief #1 including script..."
              }},
              {{
                "brief_number": 2,
                "content": "Full text of Brief #2 including script..."
              }}
            ]
            
            Document to analyze:
            {full_text}
            """
            
            try:
                response = self.evaluator.openai_client.chat.completions.create(
                    model=self.evaluator.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specialized in extracting structured content from documents containing multiple ad briefs."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Check if we received an array of briefs
                if isinstance(result, list) and len(result) > 0:
                    # Process each brief returned by the LLM
                    briefs = []
                    for brief_item in result:
                        brief_number = brief_item.get('brief_number', 0)
                        brief_content = brief_item.get('content', '')
                        
                        if brief_content:
                            brief_sections, missing_sections = self.evaluator._extract_single_brief_sections(brief_content)
                            brief_sections['brief_number'] = brief_number
                            if missing_sections:
                                brief_sections['missing_sections'] = missing_sections
                            briefs.append(brief_sections)
                
                    if briefs:
                        logger.info(f"LLM extraction found {len(briefs)} briefs")
                        return briefs
                else:
                    # Check if the result has a 'briefs' key (alternative format)
                    if isinstance(result, dict) and 'briefs' in result and isinstance(result['briefs'], list):
                        briefs = []
                        for brief_item in result['briefs']:
                            brief_number = brief_item.get('brief_number', 0)
                            brief_content = brief_item.get('content', '')
                            
                            if brief_content:
                                brief_sections, missing_sections = self.evaluator._extract_single_brief_sections(brief_content)
                                brief_sections['brief_number'] = brief_number
                                if missing_sections:
                                    brief_sections['missing_sections'] = missing_sections
                                briefs.append(brief_sections)
                    
                        if briefs:
                            logger.info(f"LLM extraction found {len(briefs)} briefs")
                            return briefs
                
                logger.warning("LLM brief extraction didn't yield usable results, falling back to regex")
            except Exception as e:
                logger.error(f"Error during LLM brief extraction: {str(e)}")
                logger.warning("Falling back to regex-based brief extraction")
            
            # Fallback: Use modified regex pattern to handle briefs with or without colons
            brief_pattern = r'(?i)(?:creative\s+)?brief\s+#\d+:?'
            brief_splits = re.split(brief_pattern, full_text)
            
            briefs = []
            if len(brief_splits) > 1:
                # Found multiple briefs
                logger.info(f"Regex found {len(brief_splits)-1} briefs in the document")
                
                # Process each brief, skipping the text before the first brief header
                for i, brief_text in enumerate(brief_splits[1:], 1):
                    # Add the header back
                    brief_with_header = f"Brief #{i}: {brief_text}"
                    
                    logger.info(f"Processing Brief #{i}")
                    
                    # Extract sections for this brief
                    brief_sections, missing_sections = self.evaluator._extract_single_brief_sections(brief_with_header)
                    brief_sections['brief_number'] = i
                    if missing_sections:
                        brief_sections['missing_sections'] = missing_sections
                    briefs.append(brief_sections)
            else:
                # Single brief or no briefs found
                logger.info("Document appears to contain a single brief or no briefs")
                brief_sections, missing_sections = self.evaluator._extract_single_brief_sections(full_text)
                brief_sections['brief_number'] = 1
                if missing_sections:
                    brief_sections['missing_sections'] = missing_sections
                briefs.append(brief_sections)
            
            return briefs
        
        # Enhance the extraction of sections within each brief
        original_extract_single_brief = self.evaluator._extract_single_brief_sections
        
        def patched_extract_single_brief(text):
            """Enhanced version of _extract_single_brief_sections that prioritizes LLM extraction."""
            logger.info("Extracting sections from brief using LLM")
            
            prompt = f"""
            Please analyze this brief and extract these specific sections:
            
            1. Brief: The creative brief section that includes information about the product, target audience, etc.
                This is typically the information that comes before the script.
            
            2. Script: The actual ad script section. This may be labeled as "Script:", or may simply be the dialogue.
                Look for any content that appears to be ad copy, dialogue, or script content.
            
            3. Debrief: Any analytical section that comes after the script.
            
            4. References: Any references to external documents or materials.
            
            Format your response as JSON with these exact keys:
            {{
                "brief": "extracted brief content",
                "script": "extracted script content",
                "debrief": "extracted debrief content",
                "references": "extracted references content"
            }}
            
            If a section is not found, provide an empty string for that key.
            
            Brief to analyze:
            {text}
            """
            
            try:
                response = self.evaluator.openai_client.chat.completions.create(
                    model=self.evaluator.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specialized in extracting structured content from ad briefs."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                
                extracted_sections = json.loads(response.choices[0].message.content)
                
                # Validate and clean up the sections
                sections = {
                    "brief": "",
                    "script": "",
                    "debrief": "",
                    "references": ""
                }
                
                # Update with extracted content
                for key in sections.keys():
                    if key in extracted_sections and extracted_sections[key]:
                        sections[key] = extracted_sections[key]
                
                # Check for any missing sections
                missing_sections = [key for key, value in sections.items() if not value]
                
                # Log what we found
                present_sections = [key for key, value in sections.items() if value]
                logger.info(f"LLM extraction found sections: {', '.join(present_sections) if present_sections else 'none'}")
                if missing_sections:
                    logger.info(f"Missing sections after LLM extraction: {', '.join(missing_sections)}")
                
                # Special check for script section - if missing, try to find it with pattern matching
                if "script" in missing_sections or not sections["script"]:
                    logger.info("Script section missing, trying pattern matching")
                    # Look for lines that start with "Script:" or lines after a line containing just "Script:"
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
                        elif in_script_section and line and not any(line.lower().startswith(s) for s in ["brief #", "thumbnail hook:", "video hook:"]):
                            script_lines.append(line)
                        elif line.lower().startswith(("brief #", "thumbnail hook:", "video hook:")):
                            in_script_section = False
                    
                    if script_lines:
                        sections["script"] = "\n".join(script_lines)
                        if "script" in missing_sections:
                            missing_sections.remove("script")
                        logger.info(f"Found script section using pattern matching: {len(sections['script'])} characters")
                
                return sections, missing_sections
                
            except Exception as e:
                logger.error(f"Error during LLM section extraction: {str(e)}")
                logger.warning("Falling back to original extraction method")
                return original_extract_single_brief(text)
        
        # Replace the original methods with our patched versions
        self.evaluator._extract_brief_sections = patched_extract_briefs
        self.evaluator._extract_single_brief_sections = patched_extract_single_brief
        logger.info("Patched extraction methods to prioritize LLM for handling multiple briefs and script sections")
    
    def process_companies(self):
        """Process all company folders in the main folder."""
        logger.info(f"Looking for company folders in {self.main_folder}")
        
        # Find all company folders
        company_folders = []
        for item in os.listdir(self.main_folder):
            folder_path = os.path.join(self.main_folder, item)
            if os.path.isdir(folder_path):
                company_folders.append(folder_path)
        
        logger.info(f"Found {len(company_folders)} company folders")
        
        # Process each company folder
        for company_folder in company_folders:
            company_name = os.path.basename(company_folder)
            logger.info(f"Processing company: {company_name}")
            
            try:
                # Create company output folder
                company_output_folder = os.path.join(self.output_folder, company_name)
                os.makedirs(company_output_folder, exist_ok=True)
                
                # Process this company
                company_results = self.process_company(company_folder, company_output_folder)
                
                # Store results
                self.results[company_name] = company_results
                
            except Exception as e:
                logger.error(f"Error processing company {company_name}: {e}")
    
    def process_company(self, company_folder: str, output_folder: str) -> Dict[str, Any]:
        """
        Process a single company folder.
        
        Args:
            company_folder: Path to company folder
            output_folder: Where to save results for this company
            
        Returns:
            Dictionary with evaluation results
        """
        company_name = os.path.basename(company_folder)
        logger.info(f"Evaluating company: {company_name}")
        
        # Find the specific subfolders
        scripts_folder = os.path.join(company_folder, "Scripts")
        ref_folder = os.path.join(company_folder, "Reference_docs")
        winning_folder = os.path.join(company_folder, "Winning_Scripts")
        
        # Check if these folders exist
        if not os.path.exists(scripts_folder):
            logger.warning(f"Scripts folder not found for {company_name}")
            return {"error": "Scripts folder not found"}
        
        # Get all script files
        script_files = []
        if os.path.exists(scripts_folder):
            for file in os.listdir(scripts_folder):
                file_path = os.path.join(scripts_folder, file)
                if os.path.isfile(file_path):
                    script_files.append(file_path)
        
        # Get all reference files
        reference_files = []
        if os.path.exists(ref_folder):
            for file in os.listdir(ref_folder):
                file_path = os.path.join(ref_folder, file)
                if os.path.isfile(file_path):
                    reference_files.append(file_path)
        
        # Get all winning script files
        winning_files = []
        if os.path.exists(winning_folder):
            for file in os.listdir(winning_folder):
                file_path = os.path.join(winning_folder, file)
                if os.path.isfile(file_path):
                    winning_files.append(file_path)
        
        logger.info(f"Found {len(script_files)} script files, {len(reference_files)} reference files, and {len(winning_files)} winning script files")
        
        # Results for all scripts
        company_results = {
            "company_name": company_name,
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluations": []
        }
        
        # Define a progress callback
        def progress_callback(brief_num, total_briefs, current_step, total_steps, current_criteria=None):
            """Update progress during evaluation"""
            if current_criteria:
                logger.info(f"Progress: Brief {brief_num}/{total_briefs}, Step {current_step}/{total_steps}, Criteria: {current_criteria}")
            else:
                logger.info(f"Progress: Brief {brief_num}/{total_briefs}, Step {current_step}/{total_steps}")
        
        # Process each script file
        for script_path in script_files:
            script_name = os.path.basename(script_path)
            logger.info(f"Evaluating script: {script_name}")
            
            try:
                # Use the evaluator from social_media_ad_evaluator.py - same as in the app
                results = self.evaluator.evaluate_ad_script(
                    script_path=script_path,
                    reference_docs=reference_files if reference_files else None,
                    winning_ads_paths=winning_files if winning_files else None,
                    progress_callback=progress_callback
                )
                
                # Generate the report - same as in the app
                report = self.evaluator.generate_report(results)
                
                # Add script name to report
                report["script_name"] = script_name
                
                # Add to company results
                company_results["evaluations"].append(report)
                
                # Generate file basename
                base_name = os.path.splitext(script_name)[0]
                
                # Save JSON report
                json_path = os.path.join(output_folder, f"{base_name}_report.json")
                with open(json_path, 'w') as f:
                    json.dump(report, f, indent=4)
                
                # Save HTML report
                html_path = os.path.join(output_folder, f"{base_name}_report.html")
                html_content = self.evaluator.generate_html_report(report)
                with open(html_path, 'w') as f:
                    f.write(html_content)
                
                # Save CSV report
                csv_path = os.path.join(output_folder, f"{base_name}_report.csv")
                csv_content = self.evaluator.generate_csv_report(report)
                with open(csv_path, 'w') as f:
                    f.write(csv_content)
                
                logger.info(f"Saved reports for {script_name}")
                
            except Exception as e:
                logger.error(f"Error evaluating {script_name}: {str(e)}")
                company_results["evaluations"].append({
                    "script_name": script_name,
                    "error": str(e)
                })
        
        # Save company summary report
        summary_path = os.path.join(output_folder, f"{company_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(company_results, f, indent=4)
        
        return company_results
    
    def generate_summary_report(self):
        """Generate a summary report of all evaluations."""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "companies_evaluated": len(self.results),
            "company_summaries": []
        }
        
        # Create company summaries
        for company_name, company_results in self.results.items():
            if "error" in company_results:
                company_summary = {
                    "company_name": company_name,
                    "status": "Error",
                    "error": company_results["error"]
                }
            else:
                # Count evaluations
                total_scripts = len(company_results.get("evaluations", []))
                error_scripts = sum(1 for script in company_results.get("evaluations", []) if "error" in script)
                successful_scripts = total_scripts - error_scripts
                
                # Calculate average scores
                all_briefs = []
                total_score = 0
                max_score = 0
                
                for script in company_results.get("evaluations", []):
                    if "error" in script:
                        continue
                        
                    if "evaluations" in script:
                        for brief in script["evaluations"]:
                            all_briefs.append(brief)
                            total_score += brief.get("total_score", 0)
                            max_score += brief.get("max_possible_score", 0)
                
                avg_percentage = (total_score / max_score * 100) if max_score > 0 else 0
                
                company_summary = {
                    "company_name": company_name,
                    "status": "Success",
                    "scripts_evaluated": total_scripts,
                    "successful_scripts": successful_scripts,
                    "error_scripts": error_scripts,
                    "briefs_evaluated": len(all_briefs),
                    "average_score": f"{total_score}/{max_score}",
                    "average_percentage": round(avg_percentage, 2)
                }
            
            summary["company_summaries"].append(company_summary)
        
        # Save summary report
        summary_path = os.path.join(self.output_folder, "evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Save CSV summary
        csv_path = os.path.join(self.output_folder, "evaluation_summary.csv")
        
        # Create CSV content
        csv_lines = ["Company,Scripts,Success,Errors,Briefs,Avg Score %"]
        
        for company_summary in summary["company_summaries"]:
            if company_summary["status"] == "Error":
                csv_lines.append(f"{company_summary['company_name']},Error,0,0,0,0.00")
            else:
                csv_lines.append(
                    f"{company_summary['company_name']},"
                    f"{company_summary['scripts_evaluated']},"
                    f"{company_summary['successful_scripts']},"
                    f"{company_summary['error_scripts']},"
                    f"{company_summary['briefs_evaluated']},"
                    f"{company_summary['average_percentage']:.2f}"
                )
        
        with open(csv_path, 'w') as f:
            f.write("\n".join(csv_lines))
        
        logger.info(f"Summary reports saved to {self.output_folder}")

def main():
    """Main entry point."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate social media ad scripts for multiple companies")
    parser.add_argument("--main_folder", default="/Users/marble-dev-01/script_eval/Original_Scripts",
                        help="Path to the main folder containing company subfolders")
    parser.add_argument("--output_folder", default="./evaluation_results",
                        help="Path to save evaluation results")
    parser.add_argument("--api_key", help="OpenAI API key (optional)")
    
    args = parser.parse_args()
    
    # Create the evaluator
    evaluator = CompanyFolderEvaluator(
        main_folder=args.main_folder,
        output_folder=args.output_folder,
        api_key=args.api_key
    )
    
    # Process all companies
    evaluator.process_companies()
    
    # Generate summary report
    evaluator.generate_summary_report()
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()