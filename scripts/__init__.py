import importlib.util
import sys
from pathlib import Path

def _load_digit_module(filename: str, alias: str):
    """Load a module whose filename starts with a digit and register it under alias."""
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"scripts.{alias}"] = module
    spec.loader.exec_module(module)
    return module

s01_finetune = _load_digit_module("01_finetune.py", "s01_finetune")
s02_build_rag_index = _load_digit_module("02_build_rag_index.py", "s02_build_rag_index")
s03_perturbation_agent = _load_digit_module("03_perturbation_agent.py", "s03_perturbation_agent")
