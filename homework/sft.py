from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = str(Path(__file__).parent / model_name)

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str, max_length: int = 128):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=max_length)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_number(value: float) -> str:
    """
    Render a float with as few tokens as possible while staying well inside the 5% grading
    tolerance. Three decimals is lossless for every answer in this dataset (smallest is 0.125),
    and dropping the trailing ".0" saves two tokens on the majority of examples.
    """
    text = f"{round(float(value), 3):f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


def format_example(prompt: str, answer: str, reasoning: str | None = None) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    if reasoning is not None:
        answer_text = reasoning
    else:
        answer_text = f"<answer>{format_number(answer)}</answer>"

    return {"question": prompt, "answer": answer_text}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn, max_length: int = 128):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formatted_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, max_length=self.max_length, **formatted_data)


def train_model(
    output_dir: str = "homework/sft_model",
    dataset_name: str = "train",
    **kwargs,
):
    from pathlib import Path

    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    llm = BaseLLM()
    llm.model.train()
    llm.model.enable_input_require_grads()

    r = kwargs.pop("r", 8)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=kwargs.pop("lora_alpha", 4 * r),
        lora_dropout=kwargs.pop("lora_dropout", 0.05),
        target_modules=kwargs.pop("target_modules", "all-linear"),
        bias=kwargs.pop("bias", "none"),
        task_type=kwargs.pop("task_type", "CAUSAL_LM"),
    )
    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.config.use_cache = False
    llm.model.print_trainable_parameters()

    dataset = Dataset(dataset_name)
    # Plain <answer>...</answer> targets are ~25 tokens; RFT reasoning targets need the full budget.
    max_length = kwargs.pop("max_length", 128 if dataset_name != "train" else 64)
    train_dataset = TokenizedDataset(llm.tokenizer, dataset, format_example, max_length=max_length)

    num_train_epochs = kwargs.pop("num_train_epochs", 5)
    per_device_train_batch_size = kwargs.pop("per_device_train_batch_size", 32)
    learning_rate = kwargs.pop("learning_rate", 2e-4)
    gradient_accumulation_steps = kwargs.pop("gradient_accumulation_steps", 1)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        logging_dir=str(output_path),
        report_to="tensorboard",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="no",
        logging_strategy="steps",
        remove_unused_columns=False,
        save_total_limit=1,
        bf16=llm.device == "cuda",
        **kwargs,
    )

    trainer = Trainer(model=llm.model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    trainer.save_model(str(output_path))

    test_model(str(output_path), max_new_tokens=50 if dataset_name == "train" else 96)


def test_model(ckpt_path: str, max_new_tokens: int = 50):
    testset = Dataset("valid")
    llm = BaseLLM()
    llm.max_new_tokens = max_new_tokens

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
