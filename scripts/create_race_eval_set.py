"""
Create MCQ evaluation set from the RACE dataset.

RACE (ReAding Comprehension from Examinations) is a reading comprehension
dataset collected from English exams for middle/high school students in China.
Each question has 4 options and is grounded in a passage.

Usage:
    python scripts/create_race_eval_set.py --split train --samples 7000
    python scripts/create_race_eval_set.py --split validation
"""

import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset


def create_race_eval_set(
    num_samples: int,
    output_dir: str = "data/eval_sets",
    seed: int = 42,
    split: str = "train",
    subset: str = "all",
    tag: str = None,
):
    random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading RACE ({subset}, {split}) from HuggingFace...")
    dataset = load_dataset("ehovy/race", subset, split=split)
    print(f"Loaded {len(dataset)} samples")

    letters = ['A', 'B', 'C', 'D']
    valid_samples = list(dataset)

    if num_samples is None or num_samples > len(valid_samples):
        num_samples = len(valid_samples)

    random.shuffle(valid_samples)
    selected = valid_samples[:num_samples]

    mcq_samples = []
    for i, sample in enumerate(selected):
        passage = sample['article']
        question = sample['question']
        options_list = sample['options']
        answer_letter = sample['answer']
        answer_idx = letters.index(answer_letter)

        options = {letters[j]: text for j, text in enumerate(options_list)}

        mcq_prompt = f"Passage: {passage}\n\nQuestion: {question}\n"
        for j, text in enumerate(options_list):
            mcq_prompt += f"{letters[j]}. {text}\n"
        mcq_prompt += "Answer:"

        correct_answer = options_list[answer_idx]
        distractors = [text for j, text in enumerate(options_list) if j != answer_idx]

        mcq_samples.append({
            'id': sample.get('example_id', f'race_{i}'),
            'question': question,
            'mcq_prompt': mcq_prompt,
            'options': options,
            'correct_letter': answer_letter,
            'correct_answer': correct_answer,
            'distractors': distractors,
        })

    position_counts = {l: 0 for l in letters}
    for s in mcq_samples:
        position_counts[s['correct_letter']] += 1

    print(f"\nCorrect answer position distribution:")
    for l, count in position_counts.items():
        print(f"  {l}: {count} ({100*count/len(mcq_samples):.1f}%)")

    split_tag = tag if tag else f"{split}_{num_samples}_discovery"
    output_file = output_dir / f"eval_set_mcq_race_{subset}_{split_tag}.json"
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "dataset": f"race_{subset}",
                "format": "mcq",
                "split": f"{split}_discovery",
                "num_samples": num_samples,
                "num_options": 4,
                "seed": seed,
            },
            "samples": mcq_samples
        }, f, indent=2)

    print(f"\nSaved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create RACE MCQ evaluation set")
    parser.add_argument("--samples", type=int, default=7000)
    parser.add_argument("--output-dir", type=str, default="data/eval_sets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "test"])
    parser.add_argument("--subset", type=str, default="all",
                        choices=["all", "high", "middle"])
    parser.add_argument("--tag", type=str, default=None,
                        help="Custom filename tag (overrides default)")
    args = parser.parse_args()
    create_race_eval_set(
        num_samples=args.samples,
        output_dir=args.output_dir,
        seed=args.seed,
        split=args.split,
        subset=args.subset,
        tag=args.tag,
    )
