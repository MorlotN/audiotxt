from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import notebook_login
#Use a sharded model to fine-tune in the free version of Google Colab.
base_model = "mistralai/Mistral-7B-v0.1" #bn22/Mistral-7B-Instruct-v0.1-sharded
# dataset_name, new_model = "gathnex/Gath_baize", "gathnex/Gath_mistral_7b"
new_model = "gathnex/Gath_mistral_7b"
dataset_name = "gathnex/Gath_baize"

# Loading a Gath_baize dataset
dataset = load_dataset(dataset_name, split="train")
dataset["chat_sample"][0]

# Load base model(Mistral 7B)
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map={"": 0}
)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token
