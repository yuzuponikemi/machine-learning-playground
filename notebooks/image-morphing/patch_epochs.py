import glob
import json

notebooks = glob.glob("3*.ipynb")

for nb_path in notebooks:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    modified = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            new_source = []
            for line in cell["source"]:
                # Reduce training epochs
                if "range(15):" in line:
                    line = line.replace("range(15):", "range(2):")
                    modified = True
                elif "range(20):" in line:
                    line = line.replace("range(20):", "range(2):")
                    modified = True
                elif "range(10):" in line:
                    if "for epoch in" in line:
                        line = line.replace("range(10):", "range(2):")
                        modified = True
                
                # Reduce diffusion steps for speed
                if "n_inference_steps=50" in line:
                    line = line.replace("n_inference_steps=50", "n_inference_steps=5")
                    modified = True
                elif "n_steps=50" in line:
                    line = line.replace("n_steps=50", "n_steps=5")
                    modified = True
                elif "n_steps=100" in line:
                    # Don't change schedule setup, only inference
                    pass 
                
                # GAN optimization steps
                if "n_steps = 200" in line:
                    line = line.replace("n_steps = 200", "n_steps = 10")
                    modified = True
                if "n_steps=200" in line:
                    line = line.replace("n_steps=200", "n_steps=10")
                    modified = True
                
                new_source.append(line)
            cell["source"] = new_source
            
    if modified:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Patched {nb_path} for faster execution.")
print("Done patching.")
