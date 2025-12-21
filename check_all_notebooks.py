#!/usr/bin/env python3
"""Check execution status of all notebooks"""

import json
import sys
from pathlib import Path

def check_notebook_execution(notebook_path):
    """Check if all cells in a notebook have been executed."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        cells = nb.get('cells', [])
        code_cells = [cell for cell in cells if cell.get('cell_type') == 'code']

        if not code_cells:
            return True, 0, 0, "No code cells"

        executed = 0
        for cell in code_cells:
            # Check if cell has execution_count (None or a number means it's been processed)
            # If execution_count key doesn't exist, cell hasn't been executed
            if 'execution_count' in cell:
                executed += 1

        total = len(code_cells)
        all_executed = executed == total

        return all_executed, executed, total, "OK" if all_executed else "Partial"

    except Exception as e:
        return False, 0, 0, f"Error: {str(e)}"

def main():
    notebooks_dir = Path('notebooks')
    notebooks = sorted(notebooks_dir.glob('*_improved_v2.ipynb'))

    print("=" * 80)
    print("NOTEBOOK EXECUTION STATUS CHECK")
    print("=" * 80)
    print(f"{'#':<4} {'Notebook':<50} {'Status':<12} {'Executed':<10}")
    print("-" * 80)

    summary = {
        'total': 0,
        'fully_executed': 0,
        'partially_executed': 0,
        'not_executed': 0,
        'error': 0
    }

    for i, nb_path in enumerate(notebooks, 1):
        nb_name = nb_path.name
        nb_num = nb_name.split('_')[0]

        all_exec, executed, total, status = check_notebook_execution(nb_path)

        summary['total'] += 1

        if status.startswith("Error"):
            summary['error'] += 1
            status_symbol = "❌ ERROR"
        elif all_exec:
            summary['fully_executed'] += 1
            status_symbol = "✅ FULL"
        elif executed > 0:
            summary['partially_executed'] += 1
            status_symbol = "⚠️  PARTIAL"
        else:
            summary['not_executed'] += 1
            status_symbol = "❌ NONE"

        exec_info = f"{executed}/{total}" if total > 0 else "N/A"

        print(f"{nb_num:<4} {nb_name[:48]:<50} {status_symbol:<12} {exec_info:<10}")

    print("=" * 80)
    print("\nSUMMARY:")
    print(f"  Total notebooks: {summary['total']}")
    print(f"  ✅ Fully executed: {summary['fully_executed']}")
    print(f"  ⚠️  Partially executed: {summary['partially_executed']}")
    print(f"  ❌ Not executed: {summary['not_executed']}")
    print(f"  ❌ Errors: {summary['error']}")
    print("=" * 80)

    # Return exit code based on results
    if summary['fully_executed'] == summary['total']:
        print("\n🎉 All notebooks are fully executed!")
        return 0
    else:
        print(f"\n⚠️  {summary['total'] - summary['fully_executed']} notebook(s) need attention")
        return 1

if __name__ == '__main__':
    sys.exit(main())
