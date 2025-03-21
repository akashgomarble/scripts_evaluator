from script_evaluator import ScriptEvaluator
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM-generated scripts')
    parser.add_argument('--script', required=True, help='Path to the script to evaluate')
    parser.add_argument('--reference-docs', nargs='+', help='List of reference documents for verification')
    parser.add_argument('--output', help='Output path for the evaluation report')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ScriptEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_script(args.script, args.reference_docs)
    
    # Generate report
    report = evaluator.generate_report(results, args.output)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Overall Score: {report['overall_score']:.2f}")
    print("\nKey Metrics:")
    print(f"- Writing Quality Score: {1 - report['detailed_metrics']['writing_quality']['stop_word_ratio']:.2f}")
    print(f"- Structure Score: {'Complete' if report['detailed_metrics']['structure']['section_presence']['brief'] else 'Incomplete'}")
    print(f"- Comprehensiveness Score: {min(1, report['detailed_metrics']['comprehensiveness']['unique_entities'] / 20):.2f}")
    
    if args.reference_docs:
        print(f"- Reference Verification Score: {report['detailed_metrics']['reference_verification']['avg_similarity']:.2f}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")

if __name__ == "__main__":
    main() 