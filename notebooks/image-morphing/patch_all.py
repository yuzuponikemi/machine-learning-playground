import glob
import json
import re

notebooks = glob.glob("3*.ipynb")

for nb_path in notebooks:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    modified = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            new_source = []
            for line in cell["source"]:
                # Patch epochs assignments and parameters
                new_line = re.sub(r"n_epochs\s*=\s*\d+", "n_epochs=2", line)
                
                # Patch for loops with ranges, but be careful not to patch range(10) which is for digits (0-9)
                # Usually training loops are something like "for epoch in range(20):"
                if "for epoch in" in new_line:
                    new_line = re.sub(r"range\(\d+\)", "range(2)", new_line)
                
                # GAN training steps
                if "n_steps" in new_line and "range(n_steps)" not in new_line and "n_steps_list" not in new_line:
                    # Don't touch schedule generation: get_schedule(n_steps=100)
                    if "get_schedule(" not in new_line and "def get_schedule" not in new_line and "schedule(" not in new_line:
                        new_line = re.sub(r"n_steps\s*=\s*200", "n_steps=10", new_line)
                        new_line = re.sub(r"n_inference_steps\s*=\s*50", "n_inference_steps=5", new_line)
                        new_line = re.sub(r"n_steps\s*=\s*50", "n_steps=5", new_line)
                        new_line = re.sub(r"n_steps\s*=\s*100\)", "n_steps=5)", new_line) # some other step
                
                if new_line != line:
                    modified = True
                new_source.append(new_line)
            cell["source"] = new_source
            
    if modified:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Patched {nb_path} with regex.")
print("Done patching all.")
