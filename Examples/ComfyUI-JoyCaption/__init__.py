import os
import importlib
import sys
from pathlib import Path

current_dir = Path(__file__).parent

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

EXCLUDE_FILES = ['__init__.py', '__pycache__']

for file in current_dir.glob('*.py'):
    if file.name not in EXCLUDE_FILES:
        module_name = file.stem
        spec = importlib.util.spec_from_file_location(module_name, str(file))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        if hasattr(module, 'NODE_CLASS_MAPPINGS'):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        
        if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']