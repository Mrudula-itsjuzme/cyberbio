import os
import glob

def replace_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. 0.404 -> 0.614
    content = content.replace("0.404", "0.614")

    # 3. 3.19 eV description
    content = content.replace(
        "substitutions/deletions produced a 3.19 eV maximum",
        "The repaired bounded adaptive multi-substitution search produced a maximum observed GraphMPNN prediction drift of approximately 3.19 eV."
    )
    content = content.replace(
        "substitutions and deletions). The repaired bounded adaptive multi-substitution search produced a maximum observed GraphMPNN prediction drift of approximately 3.19 eV",
        "substitutions and deletions). The repaired bounded adaptive multi-substitution search produced a maximum observed GraphMPNN prediction drift of approximately 3.19 eV"
    )

    with open(filepath, 'w') as f:
        f.write(content)

for md_file in glob.glob("docs/*.md"):
    replace_in_file(md_file)
if os.path.exists("README.md"):
    replace_in_file("README.md")
