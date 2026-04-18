import os
import re
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

SCENARIO_DIR = Path(r"i:\04_develop\---The-Bottom-of-Thirst\data\scenarios\raw")

def audit_scenarios():
    scenes = {} # {id: filename}
    jumps = []  # [(filename, line, target_id)]
    
    scene_regex = re.compile(r'\[scene:\s*([\w_]+)\]')
    jump_regex = re.compile(r'\(next:\s*([\w_]+)\)|jump:\s*([\w_]+)|nextSceneId:\s*([\w_]+)')
    
    files = list(SCENARIO_DIR.glob("*.txt"))
    print(f"Auditing {len(files)} files in {SCENARIO_DIR}")
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                # Find scene definitions
                sc_match = scene_regex.search(line)
                if sc_match:
                    scene_id = sc_match.group(1)
                    if scene_id in scenes:
                        print(f"ERROR: Duplicate scene ID '{scene_id}' in {file.name} (already in {scenes[scene_id]})")
                    scenes[scene_id] = file.name
                
                # Find jumps
                jp_matches = jump_regex.finditer(line)
                for match in jp_matches:
                    target = next(g for g in match.groups() if g)
                    jumps.append((file.name, i, target))

    print(f"\nCollected {len(scenes)} scenes and {len(jumps)} transitions.\n")
    
    error_count = 0
    for file_name, line, target in jumps:
        if target == 'ZAPPING':
            # ZAPPING is handled specially by the engine
            continue
        if target not in scenes:
            print(f"ERROR: Broken link in {file_name}:{line} -> '{target}' (Not found)")
            error_count += 1
            
    # Check for the ZAPPING loop in zapping.txt
    if 'zapping_connect' in scenes:
        # Check if it jumps to ZAPPING (which it does, but we need to see how it's handled)
        pass

    if error_count == 0:
        print("✅ No broken links found!")
    else:
        print(f"❌ Found {error_count} broken links.")

if __name__ == "__main__":
    audit_scenarios()
