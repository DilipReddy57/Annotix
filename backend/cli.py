import argparse
import sys
import os
import asyncio
from backend.pipeline.orchestrator import AnnotationPipeline
from backend.core.logger import get_logger

logger = get_logger("cli")

def main():
    parser = argparse.ArgumentParser(description="Autonomous Annotation Agent CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process Command
    process_parser = subparsers.add_parser("process", help="Process an image or directory")
    process_parser.add_argument("input", help="Path to image or directory")
    process_parser.add_argument("--prompt", help="Text prompt for PCS", default=None)

    # Analytics Command
    subparsers.add_parser("analytics", help="Show dataset analytics")

    args = parser.parse_args()

    if args.command == "process":
        run_process(args.input, args.prompt)
    elif args.command == "analytics":
        run_analytics()
    else:
        parser.print_help()

def run_process(input_path: str, prompt: str):
    pipeline = AnnotationPipeline()
    
    if os.path.isfile(input_path):
        logger.info(f"Processing single file: {input_path}")
        result = pipeline.process_image(input_path, prompt)
        print(f"Result: {result['status']}, Annotations: {len(result['annotations'])}")
    elif os.path.isdir(input_path):
        logger.info(f"Processing directory: {input_path}")
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    try:
                        result = pipeline.process_image(full_path, prompt)
                        print(f"Processed {file}: {len(result['annotations'])} objects")
                    except Exception as e:
                        logger.error(f"Failed {file}: {e}")

def run_analytics():
    pipeline = AnnotationPipeline()
    data = pipeline.aggregator.get_analytics()
    print("\n=== Dataset Analytics ===")
    print(f"Total Images:      {data['total_images']}")
    print(f"Total Annotations: {data['total_annotations']}")
    print(f"Categories:        {data['categories']}")
    print("\nTop Classes:")
    for cls, count in data['class_distribution'].items():
        print(f"  - {cls}: {count}")
    print("=========================\n")

if __name__ == "__main__":
    main()
