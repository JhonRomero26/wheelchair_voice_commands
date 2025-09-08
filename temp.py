import os

output_dataset = 'commands_augmented'
output_dir = f'{output_dataset}/negative'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    link_root = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
    filenames = ['dinner_party.zip', 'dinner_party_eval.zip', 'no_speech.zip', 'speech.zip']
    for fname in filenames:
        link = link_root + fname

        zip_path = f"{output_dir}/{fname}"
        os.system(f"wget -O {zip_path} {link}")
        os.system(f"unzip -q {zip_path} -d {output_dir}")