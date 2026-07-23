def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
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
    for question, answer in dataset.data:
        prompt = model.format_prompt(question)
        generations = model.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature,
        )[0]

        found_valid = False
        for generation in generations:
            parsed_answer = model.parse_answer(generation)
            if "<answer>" in generation and "</answer>" in generation and is_answer_valid(parsed_answer, answer):
                accepted.append([question, answer, generation])
                found_valid = True
                break

        if not found_valid:
            continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(accepted, f, indent=2)

    return accepted


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
