import glob
import json

for nb_path in glob.glob("31*.ipynb"):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    modified = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            new_source = []
            for line in cell["source"]:
                if "% (n_inference_steps // 10)" in line:
                    line = line.replace("(n_inference_steps // 10)", "max(1, n_inference_steps // 10)")
                    modified = True
                if "% (steps // 10)" in line:
                    line = line.replace("(steps // 10)", "max(1, steps // 10)")
                    modified = True
                new_source.append(line)
            cell["source"] = new_source
    if modified:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Patched", nb_path)
