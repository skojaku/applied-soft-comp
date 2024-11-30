"""
A script to generate Marp presentation slides from quiz questions stored in JSON format.

This script takes a JSON file containing quiz questions and generates a Markdown file formatted
for Marp presentations. Each question is placed on its own slide. The output follows Marp's
markdown syntax with appropriate slide separators and formatting.

Functions:
    create_marp_header() -> str: Creates the YAML front matter for Marp slides
    format_question_slide(question_data: dict) -> str: Formats a single question into a slide
    generate_marp_slides(json_file: str, output_file: str): Generates the complete slide deck
    main(): Entry point that processes default input/output files

The JSON file should be structured as a list of question objects. Each question object
should have the following format:

{
    "question": "What is the capital of France?",
    "options": [
        "London",
        "Paris",
        "Berlin",
        "Madrid"
    ],
    "correct_answer": 1  # Index of correct option (0-based)
}

Example JSON file:
[
    {
        "question": "What is the capital of France?",
        "options": ["London", "Paris", "Berlin", "Madrid"],
        "correct_answer": 1
    },
    {
        "question": "Which programming language is this script written in?",
        "options": ["JavaScript", "Python", "Java", "C++"],
        "correct_answer": 1
    }
]

The script processes this JSON structure to create presentation slides, focusing on
displaying the questions clearly for classroom or review purposes.
"""

import json

def create_marp_header():
    return """---
marp: true
theme: default
paginate: true
size: 4:3
---"""

def format_question_slide(question_data):
    # Get just the question text, without the options
    question_text = question_data['question'].strip()

    # Create the slide with just the question
    slide = f"""
{question_text}
"""
    return slide

def generate_marp_slides(json_file, output_file):
    # Read the JSON file
    with open(json_file, 'r') as f:
        quiz_data = json.load(f)

    # Start with the header
    marp_content = [create_marp_header()]

    # Create a slide for each question
    for question in quiz_data:
        marp_content.append(format_question_slide(question))

    # Join all slides with slide separators
    full_content = "\n---\n".join(marp_content)

    # Write to output file
    with open(output_file, 'w') as f:
        f.write(full_content)

    print(f"Successfully created Marp slides in {output_file}")

def main():
    input_file = 'sample-quiz.json'
    output_file = 'quiz_slides.md'
    generate_marp_slides(input_file, output_file)

if __name__ == "__main__":
    main()
