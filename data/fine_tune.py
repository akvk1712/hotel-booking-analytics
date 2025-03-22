from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import os

def fine_tune_model():
    # ✅ Model & Tokenizer
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # ✅ Load dataset
    if not os.path.exists("data/processed_data.csv"):
        raise FileNotFoundError("❌ processed_data.csv not found! Make sure the data file exists.")
    
    dataset = load_dataset("csv", data_files="data/processed_data.csv")

    # ✅ Training arguments
    training_args = TrainingArguments(
        output_dir="./gpt2-hotel-model",
        per_device_train_batch_size=4,
        logging_dir="./logs",
        num_train_epochs=2,
        save_strategy="epoch"
    )

    # ✅ Initialize trainer and start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"]
    )
    
    # ✅ Start training
    trainer.train()

    # ✅ Save the trained model
    trainer.save_model("./gpt2-hotel-model")
    tokenizer.save_pretrained("./gpt2-hotel-model")

    print("🎉✅ Fine-tuning complete! Model saved to './gpt2-hotel-model'")

if __name__ == "__main__":
    fine_tune_model() 