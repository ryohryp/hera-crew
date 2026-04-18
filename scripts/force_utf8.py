import os
from pathlib import Path

SCENARIO_DIR = Path(r"i:\04_develop\---The-Bottom-of-Thirst\data\scenarios\raw")

def force_utf8():
    for file in SCENARIO_DIR.glob("*.txt"):
        try:
            # Try reading as binary and decoding
            data = open(file, 'rb').read()
            
            # Try common encodings
            content = None
            for enc in ['utf-8', 'cp932', 'euc-jp', 'utf-16']:
                try:
                    content = data.decode(enc)
                    print(f"Read {file.name} as {enc}")
                    break
                except:
                    continue
            
            if content:
                # Always save as UTF-8 (no BOM)
                with open(file, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                print(f"FAILED to decode {file.name}")
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

if __name__ == "__main__":
    force_utf8()
