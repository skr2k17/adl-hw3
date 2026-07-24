from .base_llm import BaseLLM
from .data import Dataset, benchmark

# Chain-of-thought answers need more room than the plain <answer> completions SFT produces.
RFT_MAX_NEW_TOKENS = 96


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = str(Path(__file__).parent / model_name)

    llm = BaseLLM()
    llm.max_new_tokens = RFT_MAX_NEW_TOKENS
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str = "homework/rft_model",
    **kwargs,
):
    from pathlib import Path

    from .datagen import generate_dataset
    from .sft import train_model as sft_train_model

    data_dir = Path(__file__).resolve().parent.parent / "data"
    rft_path = data_dir / "rft.json"
    if not rft_path.exists():
        generate_dataset(
            output_json="rft.json",
            oversample=kwargs.pop("oversample", 10),
            temperature=kwargs.pop("temperature", 0.6),
        )

    kwargs.setdefault("num_train_epochs", 8)
    kwargs.setdefault("learning_rate", 2e-4)
    return sft_train_model(output_dir=output_dir, dataset_name="rft", **kwargs)


def test_model(ckpt_path: str):
    from peft import PeftModel

    testset = Dataset("valid")
    llm = BaseLLM()
    llm.max_new_tokens = RFT_MAX_NEW_TOKENS
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
