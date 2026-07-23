from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = str(Path(__file__).parent / model_name)

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str = "homework/rft_model",
    **kwargs,
):
    from pathlib import Path

    from .sft import train_model as sft_train_model

    data_dir = Path(__file__).resolve().parent.parent / "data"
    dataset_name = "rft" if (data_dir / "rft.json").exists() else "train"

    return sft_train_model(output_dir=output_dir, dataset_name=dataset_name, **kwargs)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
