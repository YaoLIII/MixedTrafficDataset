# -*- coding: utf-8 -*-
"""
list the structure of current folder
"""

import os

def generate_markdown_tree(current_path, prefix=""):
    items = os.listdir(current_path)
    tree_structure = ""
    for idx, item in enumerate(sorted(items)):
        item_path = os.path.join(current_path, item)
        is_last = (idx == len(items) - 1)
        tree_structure += f"{prefix}├── {item}\n" if not is_last else f"{prefix}└── {item}\n"
        if os.path.isdir(item_path):
            sub_prefix = prefix + ("│   " if not is_last else "    ")
            tree_structure += generate_markdown_tree(item_path, sub_prefix)
    return tree_structure

# Specify the directory path you want to represent
root_directory = "Folder contents"
markdown_structure = generate_markdown_tree(os.path.dirname(os.path.abspath(__file__)))
print(markdown_structure)
