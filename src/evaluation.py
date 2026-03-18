import torch
import numpy as np
from tqdm import tqdm


def get_predicted_letter(logits, letter_token_ids):
    letter_logits = {l: logits[tid].item() for l, tid in letter_token_ids.items()}
    return max(letter_logits, key=letter_logits.get)


def compute_entropy(logits, letter_token_ids):
    abcd_logits = torch.tensor([logits[tid].item() for tid in letter_token_ids.values()])
    probs = torch.softmax(abcd_logits, dim=0)
    return -torch.sum(probs * torch.log(probs + 1e-10)).item()


def run_eval(model, tokenizer, samples, letter_token_ids, device, desc="Eval"):
    """Run MCQ evaluation, return list of per-question result dicts."""
    results = []
    for sample in tqdm(samples, desc=desc):
        prompt = sample.get('mcq_prompt', sample.get('prompt', ''))
        correct_letter = sample.get('correct_letter', None)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :].detach()
        predicted = get_predicted_letter(logits, letter_token_ids)
        correct = (predicted == correct_letter) if correct_letter else None
        entropy = compute_entropy(logits, letter_token_ids)
        results.append({
            'predicted': predicted,
            'correct_letter': correct_letter,
            'correct': correct,
            'entropy': entropy,
        })
    return results


def summarise(results):
    acc = sum(r['correct'] for r in results) / len(results)
    ent = np.mean([r['entropy'] for r in results])
    return acc, ent


def compute_flips(baseline_results, suppressed_results):
    flipped_correct = sum(
        1 for bl, sp in zip(baseline_results, suppressed_results)
        if not bl['correct'] and sp['correct']
    )
    flipped_incorrect = sum(
        1 for bl, sp in zip(baseline_results, suppressed_results)
        if bl['correct'] and not sp['correct']
    )
    return flipped_correct, flipped_incorrect
