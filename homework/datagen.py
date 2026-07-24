def _clean_generation(generation: str) -> str | None:
    """
    Keep only the reasoning up to and including the closing </answer> tag.

    Generations run to `max_new_tokens`, so anything after </answer> is trailing filler that
    would teach the fine-tuned model to keep rambling instead of stopping at the answer.
    """
    if "<answer>" not in generation or "</answer>" not in generation:
        return None
    head, _, tail = generation.partition("</answer>")
    if "<answer>" not in head:
        return None
    return (head + "</answer>").strip()


def generate_dataset(
    output_json: str = "rft.json",
    oversample: int = 10,
    temperature: float = 0.6,
    batch_size: int = 16,
    max_chars: int = 400,
):
    import json
    from pathlib import Path

    from .cot import CoTModel
    from .data import DATA_DIR, Dataset, is_answer_valid

    dataset = Dataset("train")
    model = CoTModel()
    model.model.eval()

    output_path = Path(output_json)
    if not output_path.is_absolute():
        output_path = DATA_DIR / output_path

    accepted: list[list] = []
    rejected: list[dict] = []

    items = list(dataset.data)
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        prompts = [model.format_prompt(question) for question, _ in chunk]
        batch_generations = model.batched_generate(
            prompts,
            num_return_sequences=oversample,
            temperature=temperature,
        )

        for (question, answer), generations in zip(chunk, batch_generations):
            candidates = []
            failures = []
            for generation in generations:
                cleaned = _clean_generation(generation)
                if cleaned is None:
                    failures.append(("missing or unclosed <answer> tag", generation, None))
                    continue
                parsed = model.parse_answer(cleaned)
                if parsed != parsed:  # NaN
                    failures.append(("answer tag did not parse as a float", cleaned, parsed))
                elif not is_answer_valid(parsed, answer):
                    failures.append(("answer outside 5% tolerance", cleaned, parsed))
                elif len(cleaned) > max_chars:
                    failures.append(("reasoning too long to fit the generation budget", cleaned, parsed))
                else:
                    candidates.append(cleaned)

            if candidates:
                # Prefer the shortest correct chain: it trains faster and is far less likely to
                # get truncated by max_new_tokens at inference time.
                accepted.append([question, answer, min(candidates, key=len)])
            else:
                reason, generation, parsed = failures[0] if failures else ("no generations", "", None)
                rejected.append(
                    {
                        "question": question,
                        "expected_answer": answer,
                        "generation": generation,
                        "parsed_answer": parsed,
                        "reason": reason,
                    }
                )

        kept = len(accepted)
        print(f"[datagen] {start + len(chunk)}/{len(items)} questions | kept {kept} ({kept / (start + len(chunk)):.1%})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(accepted, f, indent=2)

    with output_path.with_name(f"{output_path.stem}_rejected.json").open("w") as f:
        json.dump(rejected, f, indent=2)

    print(f"[datagen] wrote {len(accepted)} examples to {output_path} ({len(rejected)} questions rejected)")
    return accepted


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
