import glob
import json

for nb_path in glob.glob("3*.ipynb"):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    modified = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            new_source = []
            for line in cell["source"]:
                if "n_iter=1000" in line:
                    line = line.replace(", n_iter=1000", "").replace("n_iter=1000", "")
                    modified = True
                new_source.append(line)
            cell["source"] = new_source
    if modified:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Patched", nb_path)
