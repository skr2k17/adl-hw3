from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You convert units. Be concise: state the conversion factor, show one "
                    "multiplication, and end with <answer>number</answer>. "
                    "Use 1 year = 365.2422 days, 1 month = 30.4368 days, and decimal byte "
                    "prefixes (1 KB = 1000 B, 1 B = 8 bit)."
                ),
            },
            {"role": "user", "content": "How many gram are there per 6 kg?"},
            {"role": "assistant", "content": "1 kg = 1000 g. 6 * 1000 = <answer>6000</answer>"},
            {"role": "user", "content": "Express 4 centuries as a quantity of week."},
            {
                "role": "assistant",
                "content": "1 century = 5217.7457 weeks. 4 * 5217.7457 = <answer>20870.983</answer>",
            },
            {"role": "user", "content": "Could you provide the value of 2 pound in ounce?"},
            {"role": "assistant", "content": "1 pound = 16 ounce. 2 * 16 = <answer>32</answer>"},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
