#!/usr/bin/env python3
"""Wrap eval_v11.py into a single-cell Kaggle notebook."""
import json
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE / "eval_v11.py").read_text(encoding="utf-8")

nb = {
    "cells": [
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": src.splitlines(keepends=True),
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = HERE / "notebook.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size} bytes)")
