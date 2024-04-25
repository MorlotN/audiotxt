from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# Définir le chemin local où le modèle et le tokenizer sont sauvegardés
local_model_path = '/home/morlot/code/audiotxt/your_model_dir'

# Charger le modèle
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Exemple d'utilisation du modèle
with open('/home/morlot/code/audiotxt/input/entretien michel_test.txt', 'r') as file:
    data = file.read().replace('\n', '')
# df = pd.read_csv('/home/morlot/code/audiotxt/input/entretien michel.txt')
inputs = tokenizer(data, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)  # ou utilisez max_length selon vos besoins

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

summ = tokenizer.decode(outputs[0], skip_special_tokens=True)
with open('/home/morlot/code/audiotxt/output/entretien michel_test.txt', "w") as text_file:
    text_file.write(summ)