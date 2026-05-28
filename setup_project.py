import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

denominations = ["R10", "R20", "R50", "R100", "R200"]
designs       = ["old", "new"]
sides         = ["front", "back"]

output_roots = [
    os.path.join(BASE_DIR, "dataset", "processed", "train"),
    os.path.join(BASE_DIR, "dataset", "processed", "test", "clean"),
    os.path.join(BASE_DIR, "dataset", "processed", "test", "augmented"),
]

for root in output_roots:
    for denomination in denominations:
        for design in designs:
            for side in sides:
                folder = os.path.join(root, denomination, design, side)
                os.makedirs(folder, exist_ok=True)

                # Place .gitkeep so GitHub tracks the empty folder
                gitkeep_path = os.path.join(folder, ".gitkeep")
                if not os.path.exists(gitkeep_path):
                    open(gitkeep_path, 'w').close()

                print(f"Created: {folder}")

print("\nFolder structure ready.")