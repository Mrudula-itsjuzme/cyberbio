import os
import ast

def get_classes_and_functions(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read(), filename=file_path)
    
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    return classes, functions

base_dir = "/home/mrudula/Downloads/DL_cyberbio/materials-adversarial/src"
for root, dirs, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".py"):
            path = os.path.join(root, f)
            c, fn = get_classes_and_functions(path)
            rel_path = os.path.relpath(path, base_dir)
            if c or fn:
                print(f"File: {rel_path}")
                if c: print(f"  Classes: {c}")
                if fn: print(f"  Functions: {fn}")
