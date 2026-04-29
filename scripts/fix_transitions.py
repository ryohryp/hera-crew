import os
import re
from pathlib import Path

SCENARIO_DIR = Path(r"i:\04_develop\---The-Bottom-of-Thirst\data\scenarios\raw")

def fix_scenarios():
    scenes = {} # {id: filename}
    scene_regex = re.compile(r'\[scene:\s*([\w_]+)\]')
    jump_regex = re.compile(r'(\((?:next|choice):.*?->\s*|jump:\s*|nextSceneId:\s*|next:\s*)([\w_]+)')
    
    files = list(SCENARIO_DIR.glob("*.txt"))
    
    # First pass: Collect all valid scene IDs
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                sc_match = scene_regex.search(line)
                if sc_match:
                    scenes[sc_match.group(1)] = file.name

    print(f"Collected {len(scenes)} valid scenes.")

    # Second pass: Fix broken links
    fix_count = 0
    suffixes = ["_narration", "_souma", "_saya", "_kagami", "_leo", "_hikawa", "_mikoshiba", "_ren"]
    
    for file in files:
        lines = []
        changed = False
        with open(file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                def replace_func(match):
                    nonlocal changed, fix_count
                    prefix = match.group(1)
                    target = match.group(2)
                    
                    if target == 'ZAPPING' or target in scenes:
                        return match.group(0)
                    
                    # Try suffixes
                    for s in suffixes:
                        alt = target + s
                        if alt in scenes:
                            print(f"FIX: {file.name}:{i} '{target}' -> '{alt}'")
                            changed = True
                            fix_count += 1
                            return f"{prefix}{alt}"
                    
                    # Also try target + "_1" etc. if needed, but the pattern seems to be _name
                    return match.group(0)

                new_line = jump_regex.sub(replace_func, line)
                lines.append(new_line)
        
        if changed:
            with open(file, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"Updated {file.name}")

    print(f"\nFinished. Total fixes: {fix_count}")

if __name__ == "__main__":
    fix_scenarios()
