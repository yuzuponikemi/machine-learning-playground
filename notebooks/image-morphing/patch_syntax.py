import glob
import json

for nb_path in glob.glob("3*.ipynb"):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    modified = False
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            new_source = []
            for i, line in enumerate(cell["source"]):
                # find strings that are unterminated (e.g., have a single quote open but not closed before newline)
                if "fig.suptitle('" in line and "'," not in line and "')" not in line and "\\n" not in line:
                    if line.endswith("\n"):
                        # If the next line continues the string... we should join them or escape newline
                        # A quick hack for Jupyter notebook format is to just escape the newline in python code
                        pass
                
                # Let's just do a specific replace for the exact strings
                if "DDIM Inversion + ノイズ空間Lerp補間\n" in line:
                    line = line.replace("DDIM Inversion + ノイズ空間Lerp補間\n", "DDIM Inversion + ノイズ空間Lerp補間\\n")
                    modified = True
                
                if "DDIM Inversion + ノイズ空間Slerp補間\n" in line:
                    line = line.replace("DDIM Inversion + ノイズ空間Slerp補間\n", "DDIM Inversion + ノイズ空間Slerp補間\\n")
                    modified = True

                new_source.append(line)
            cell["source"] = new_source
    if modified:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Patched", nb_path)
