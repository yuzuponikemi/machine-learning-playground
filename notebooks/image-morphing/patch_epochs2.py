import glob
import json

notebooks = glob.glob("301*.ipynb") + glob.glob("3*.ipynb")

for nb_path in set(notebooks):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    modified = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            new_source = []
            for line in cell["source"]:
                if "n_epochs = 15" in line:
                    line = line.replace("n_epochs = 15", "n_epochs = 2")
                    modified = True
                if "n_epochs = 20" in line:
                    line = line.replace("n_epochs = 20", "n_epochs = 2")
                    modified = True
                if "n_epochs=15" in line:
                    line = line.replace("n_epochs=15", "n_epochs=2")
                    modified = True
                if "n_epochs = 10" in line:
                    line = line.replace("n_epochs = 10", "n_epochs = 2")
                    modified = True
                
                new_source.append(line)
            cell["source"] = new_source
            
    if modified:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Patched variables in {nb_path}.")
print("Done patching variables.")
