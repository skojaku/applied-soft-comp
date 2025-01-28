"""
A script to convert quiz questions from JSON format to Brightspace-compatible CSV format.

This script takes a JSON file containing quiz questions and generates a CSV file that can be
imported into Brightspace's quiz system. Each question is formatted according to Brightspace's
specific requirements, including question text, options, correct answers, and metadata.

Classes:
    QuizQuestion: Represents a single multiple choice question with its properties and conversion logic

Functions:
    convert_json_to_brightspace_csv(json_file: str, output_file: str, course_code: str):
        Converts a JSON quiz file to Brightspace CSV format

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

The script generates a CSV file with the following Brightspace-specific columns:
- NewQuestion: Indicates the start of a new question
- ID: Unique identifier for the question
- Title: Question title
- QuestionText: The actual question text
- Points: Question point value
- Difficulty: Question difficulty level
- Image: Reference to associated image
- Option: Answer choices with points and feedback
"""
import sys
import json
import csv
from typing import List, Dict

class QuizQuestion:
    def __init__(self, question_id: int, question_text: str, options: Dict[str, str],
                 correct_answer: str, course_code: str = "COURSE101", explanation: str = ""):
        self.question_id = question_id
        self.question_text = question_text
        self.options = options
        self.correct_answer = correct_answer
        self.course_code = course_code
        self.explanation = explanation

    def to_rows(self) -> List[List[str]]:
        """Convert question to CSV rows format"""
        rows = []

        # Start new question
        rows.append(["NewQuestion", "MC", "", "", ""])
        rows.append(["ID", f"{self.course_code}-{self.question_id}", "", "", ""])
        rows.append(["Title", f"Question {self.question_id}", "", "", ""])
        rows.append(["QuestionText", self.question_text, "", "", ""])
        rows.append(["Points", "1", "", "", ""])
        rows.append(["Difficulty", "1", "", "", ""])

        # Add image reference
        rows.append(["Image", f"sample-quiz/output_{self.question_id}.jpg", "", "", ""])

        # Add options
        for letter, text in self.options.items():
            points = 100 if letter == self.correct_answer else 0
            feedback = self.explanation 
            rows.append(["Option", str(points), text, "", feedback])

        # Add empty row for separation
        rows.append(["", "", "", "", ""])
        return rows

def convert_json_to_brightspace_csv(json_file: str, output_file: str, course_code: str = "COURSE101"):
    """Convert JSON quiz file to Brightspace CSV format"""
    # Read JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert each question
    questions = []
    for i, item in enumerate(data):
        question = QuizQuestion(
            question_id=i,
            question_text=item['question'],
            options=item['options'],
            correct_answer=item['correct_answer'],
            course_code=course_code,
            explanation = item["explanation"] if "explanation" in item else "" 
        )
        questions.append(question)

    # Write CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header comments
        writer.writerow(['// Brightspace Quiz - Multiple Choice Questions'])
        writer.writerow(['// Images are located in /content/enforced/353075-32628.202490/sample-quiz/'])
        writer.writerow([''])

        # Write each question
        for question in questions:
            rows = question.to_rows()
            writer.writerows(rows)

# Example usage
# You can change these parameters as needed
json_file = sys.argv[1]  # Your input JSON file
output_file = sys.argv[2]  # The output CSV file
course_code = sys.argv[3]  # Your course code

convert_json_to_brightspace_csv(json_file, output_file, course_code)
