import os
import torch
import gc
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Configuration
MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct"
DATASET_FILE = "data/finetuning_dataset.jsonl"
OUTPUT_DIR = "models/llama3.1-bim-rag-lora-v2"
MAX_SEQ_LENGTH = 8192

def prepare_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, # Auto detection
        load_in_4bit = True, # Use QLoRA
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Dropout = 0 is optimized
        bias = "none",    
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
    )
    return model, tokenizer

def run_finetuning():
    if not os.path.exists(DATASET_FILE) or os.path.getsize(DATASET_FILE) == 0:
        print("No new data to finetune on. Exiting.")
        return

    print("Loading Dataset...")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    def format_chat_template(examples):
        # We assume the `db_logger.py` exported in ChatML or standard message list.
        # Ensure our FastLanguageModel formats it properly.
        texts = []
        for msgs in examples["messages"]:
            # Simple fallback stringification if not applying real chat_template
            concat = ""
            for m in msgs:
                concat += f"<{m['role']}>\n{m['content']}\n</{m['role']}>\n"
            texts.append(concat)
        return {"text": texts}

    dataset = dataset.map(format_chat_template, batched=True)

    print("Preparing Model...")
    model, tokenizer = prepare_model()

    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, # Can make true for speedups on short sequences
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1, # Fast iteration
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    print("Training...")
    trainer_stats = trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
    # NOTE: In a true continuous loop, you might version this (v1, v2, v3) 
    # so you can upload the specific version to vLLM.
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Empty dataset file after successful training
    open(DATASET_FILE, 'w').close()
    
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    run_finetuning()
