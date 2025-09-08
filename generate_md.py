import os

input_dir = os.path.join("microwakeword")
output_file = "speech-commands.md"

with open(output_file, "w") as f:
    body = ""
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.startswith("_") and file.endswith(".py"):
                current_path = os.path.join(root, file)
                with open(current_path, "r") as fpy:
                    body += f"```{file}\n"
                    body += fpy.read()
                    body += "```\n\n"
    
    yaml_params = 'training_parameters.yaml'
    with open(yaml_params, 'r') as ft:
        body += f"```{yaml_params}\n"
        body += ft.read()
        body += "```\n\n"

    main_file = 'model_train_eval.py'
    with open(main_file, 'r') as ft:
        body += f"```{main_file}\n"
        body += ft.read()
        body += "```\n\n"

    f.write(body)


