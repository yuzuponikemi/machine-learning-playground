import glob
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import sys

notebooks = sorted([f for f in glob.glob("31*.ipynb") if "310" not in f])

for nb_path in notebooks:
    print(f"Executing {nb_path}...", flush=True)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    client = NotebookClient(nb, timeout=1200, kernel_name='python3')
    
    try:
        client.execute()
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Success: {nb_path}", flush=True)
    except Exception as e:
        print(f"Error in {nb_path}:", flush=True)
        print(e, flush=True)
        sys.exit(1)
