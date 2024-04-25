import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
# from trl.sft import SFTTrainer
from trl import SFTTrainer
import wandb

# Configuration de Weights & Biases
wandb.login()  # Assurez-vous que votre clé API est définie ou qu'elle est définie dans vos variables d'environnement

# Charger les données
dataset = load_dataset('gathnex/Gath_baize', split='train')

# Initialiser le tokenizer et le modèle avec des configurations spécifiques pour l'efficacité de la mémoire
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1', use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.1',
    revision='main',  # ou le hash de commit spécifique pour assurer la reproductibilité
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    low_cpu_mem_usage=True
)

# Assurez-vous que tous les poids du modèle sont sur le même appareil, réduisez l'utilisation de la mémoire
if torch.cuda.is_available():
    model.to('cuda')

# Configurer les arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Réglez cela plus bas si vous rencontrez des problèmes de mémoire CUDA
    logging_steps=10,
    save_steps=500,
    report_to='wandb',
    fp16=torch.cuda.is_available(),  # Utilisez la précision mixte pour économiser de la mémoire sur GPU
)

# Configurer SFTTrainer avec une longueur de séquence minimale pour économiser de la mémoire
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    compute_metrics=None  # Définissez votre fonction pour calculer les métriques si nécessaire
)

# Exécuter l'entraînement
trainer.train()

# Sauvegarder votre modèle (optionnel)
model_path = "./your_model_dir"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Nettoyage pour économiser de la mémoire
del model
torch.cuda.empty_cache()  # Videz le cache de la mémoire si CUDA est utilisé
