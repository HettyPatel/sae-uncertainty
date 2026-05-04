"""
Self-Abstention Experiment

Add "E. I don't know" as a 5th option to MCQ questions and measure:
  1. How often does the model self-abstain?
  2. Does it abstain more on questions it would have gotten wrong?
  3. How does accuracy change on answered (non-abstained) questions?

Usage:
    python experiments/self_abstain_experiment.py \
        --model meta-llama/Llama-3.1-8B \
        --eval-set data/eval_sets/eval_set_mcq_mmlu_test_14042_validation.json \
        --output-dir results/sae_self_abstain_experiment/
"""

import csv
import json
import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from src.utils import seed_everything, load_eval_set
from src.model import load_model, get_letter_token_ids


def get_predicted_letter_5(logits, letter_token_ids_5):
    letter_logits = {l: logits[tid].item() for l, tid in letter_token_ids_5.items()}
    return max(letter_logits, key=letter_logits.get)


def compute_entropy_5(logits, letter_token_ids_5):
    abcde_logits = torch.tensor([logits[tid].item() for tid in letter_token_ids_5.values()])
    probs = torch.softmax(abcde_logits, dim=0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    return entropy


def add_abstain_option(prompt):
    """Add 'E. I don't know' before 'Answer:'"""
    return prompt.replace("\nAnswer:", "\nE. I don't know\nAnswer:")


def main():
    parser = argparse.ArgumentParser(description="Self-Abstention Experiment")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--eval-set', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results/sae_self_abstain_experiment/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of samples (for quick testing)')
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer, model_config = load_model(args.model, device=args.device)

    # Get token IDs for A-E
    letter_token_ids_4 = {}
    letter_token_ids_5 = {}
    for letter in ['A', 'B', 'C', 'D', 'E']:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if ids:
            tid = ids[-1]
            if letter in ['A', 'B', 'C', 'D']:
                letter_token_ids_4[letter] = tid
            letter_token_ids_5[letter] = tid
    print(f"Letter token IDs (5): {letter_token_ids_5}")

    # Load eval set
    samples = load_eval_set(args.eval_set)
    if args.max_samples:
        samples = samples[:args.max_samples]

    # Run both conditions: original (4 options) and with abstain option (5 options)
    results = []

    for sample in tqdm(samples, desc="Evaluating"):
        prompt_4 = sample.get('mcq_prompt', sample.get('prompt', ''))
        prompt_5 = add_abstain_option(prompt_4)
        correct_letter = sample.get('correct_letter', None)

        # Original 4-option
        inputs_4 = tokenizer(prompt_4, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs_4 = model(**inputs_4)
        logits_4 = outputs_4.logits[0, -1, :].detach()
        pred_4 = get_predicted_letter_5(logits_4, letter_token_ids_4)
        correct_4 = (pred_4 == correct_letter) if correct_letter else None
        entropy_4 = compute_entropy_5(logits_4, letter_token_ids_4)

        # 5-option with abstain option
        inputs_5 = tokenizer(prompt_5, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs_5 = model(**inputs_5)
        logits_5 = outputs_5.logits[0, -1, :].detach()
        pred_5 = get_predicted_letter_5(logits_5, letter_token_ids_5)
        correct_5 = (pred_5 == correct_letter) if correct_letter else None
        chose_abstain = (pred_5 == 'E')
        entropy_5 = compute_entropy_5(logits_5, letter_token_ids_5)

        # Get E probability in both conditions
        e_logit_4 = logits_4[letter_token_ids_5['E']].item()
        abcd_logits_4 = torch.tensor([logits_4[letter_token_ids_4[l]].item() for l in ['A', 'B', 'C', 'D']])
        e_prob_in_4 = torch.softmax(torch.tensor([*abcd_logits_4.tolist(), e_logit_4]), dim=0)[4].item()

        abcde_logits_5 = torch.tensor([logits_5[letter_token_ids_5[l]].item() for l in ['A', 'B', 'C', 'D', 'E']])
        probs_5 = torch.softmax(abcde_logits_5, dim=0)
        e_prob_5 = probs_5[4].item()

        results.append({
            'id': sample.get('id', ''),
            'correct_letter': correct_letter,
            'pred_4opt': pred_4,
            'correct_4opt': correct_4,
            'entropy_4opt': entropy_4,
            'pred_5opt': pred_5,
            'correct_5opt': correct_5,
            'chose_abstain': chose_abstain,
            'entropy_5opt': entropy_5,
            'e_prob_without_abstain_prompt': e_prob_in_4,
            'e_prob_with_abstain_prompt': e_prob_5,
        })

    # Analysis
    total = len(results)
    n_abstain = sum(r['chose_abstain'] for r in results)
    n_correct_4 = sum(r['correct_4opt'] for r in results)
    n_correct_5 = sum(r['correct_5opt'] for r in results)

    # Of questions where model chose to abstain, how many would it have gotten right/wrong?
    abstain_questions = [r for r in results if r['chose_abstain']]
    abstain_would_correct = sum(r['correct_4opt'] for r in abstain_questions)
    abstain_would_incorrect = len(abstain_questions) - abstain_would_correct

    # Of questions where model didn't choose to abstain
    non_abstain = [r for r in results if not r['chose_abstain']]
    non_abstain_correct = sum(r['correct_5opt'] for r in non_abstain)

    # Breakdown by original correctness
    originally_correct = [r for r in results if r['correct_4opt']]
    originally_incorrect = [r for r in results if not r['correct_4opt']]
    abstain_rate_on_correct = sum(r['chose_abstain'] for r in originally_correct) / len(originally_correct) * 100
    abstain_rate_on_incorrect = sum(r['chose_abstain'] for r in originally_incorrect) / len(originally_incorrect) * 100

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total questions: {total}")
    print(f"Original accuracy (4 options): {n_correct_4}/{total} ({n_correct_4/total*100:.2f}%)")
    print(f"With abstain option accuracy (5 options): {n_correct_5}/{total} ({n_correct_5/total*100:.2f}%)")
    print(f"")
    print(f"Chose to abstain: {n_abstain}/{total} ({n_abstain/total*100:.2f}%)")
    print(f"  Of those, would have been correct: {abstain_would_correct} ({abstain_would_correct/max(n_abstain,1)*100:.1f}%)")
    print(f"  Of those, would have been incorrect: {abstain_would_incorrect} ({abstain_would_incorrect/max(n_abstain,1)*100:.1f}%)")
    print(f"")
    print(f"Abstain rate on originally correct questions: {abstain_rate_on_correct:.2f}%")
    print(f"Abstain rate on originally incorrect questions: {abstain_rate_on_incorrect:.2f}%")
    print(f"")
    if non_abstain:
        print(f"Accuracy on non-abstain questions: {non_abstain_correct}/{len(non_abstain)} ({non_abstain_correct/len(non_abstain)*100:.2f}%)")
    print(f"")
    print(f"Mean E probability (without abstain option in prompt): {np.mean([r['e_prob_without_abstain_prompt'] for r in results]):.4f}")
    print(f"Mean E probability (with abstain option in prompt):    {np.mean([r['e_prob_with_abstain_prompt'] for r in results]):.4f}")
    print(f"  On originally correct:   {np.mean([r['e_prob_with_abstain_prompt'] for r in originally_correct]):.4f}")
    print(f"  On originally incorrect: {np.mean([r['e_prob_with_abstain_prompt'] for r in originally_incorrect]):.4f}")

    # Save CSV
    csv_file = output_dir / "self_abstain_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_file}")

    # Save summary
    summary_file = output_dir / "self_abstain_summary.csv"
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['total_questions', total])
        writer.writerow(['accuracy_4opt', n_correct_4/total])
        writer.writerow(['accuracy_5opt', n_correct_5/total])
        writer.writerow(['n_chose_abstain', n_abstain])
        writer.writerow(['abstain_rate', n_abstain/total])
        writer.writerow(['abstain_would_correct', abstain_would_correct])
        writer.writerow(['abstain_would_incorrect', abstain_would_incorrect])
        writer.writerow(['abstain_rate_on_correct', abstain_rate_on_correct/100])
        writer.writerow(['abstain_rate_on_incorrect', abstain_rate_on_incorrect/100])
        writer.writerow(['accuracy_non_abstain', non_abstain_correct/len(non_abstain) if non_abstain else 0])
    print(f"Saved: {summary_file}")
    print("DONE")


if __name__ == '__main__':
    main()
