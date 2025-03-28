import os
import sys
from social_media_ad_evaluator import SocialMediaAdEvaluator
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)
    
    # Get the file path from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_extraction.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    print(f"Testing extraction on file: {file_path}")
    
    # Create the evaluator with API key
    evaluator = SocialMediaAdEvaluator(api_key=api_key)
    
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the sections
    sections, missing_sections = evaluator._extract_single_brief_sections(content)
    
    # Display results
    print("\n=== EXTRACTION RESULTS ===")
    print(f"Sections found: {', '.join([k for k, v in sections.items() if v])}")
    if missing_sections:
        print(f"Missing sections: {', '.join(missing_sections)}")
    
    # Print section summaries
    for section, content in sections.items():
        if content:
            lines = content.strip().split('\n')
            line_count = len(lines)
            first_line = lines[0][:50] + ('...' if len(lines[0]) > 50 else '')
            print(f"\n--- {section.upper()} ({line_count} lines) ---")
            print(f"First line: {first_line}")
            print(f"Length: {len(content)} characters")
            # Print first few lines of content for verification
            preview = "\n".join(lines[:3]) + ("..." if line_count > 3 else "")
            print(f"Preview:\n{preview}")

if __name__ == "__main__":
    main() 