import os
import json
import re

# --- CONFIG ---
BASE_DIR = "./youtube-vos/JPEGImages"
JSON_PATH = "./youtube-vos/train.json"
OUTPUT_REPORT = "./frame_check_report.json"

FRAME_PATTERN = re.compile(r"^\d{5}\.jpg$")


def get_frame_indices(folder_path):
    indices = set()

    for fname in os.listdir(folder_path):
        if FRAME_PATTERN.match(fname):
            idx = int(fname.split(".")[0])
            indices.add(idx)

    return indices


def main():
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    all_ok = True
    report = {}

    for folder_name, expected_count in data.items():
        folder_path = os.path.join(BASE_DIR, folder_name)

        print(f"\nChecking: {folder_name}")
        folder_issues = {}

        # 1. Check folder exists
        if not os.path.isdir(folder_path):
            print(f"  ❌ Folder missing")
            folder_issues["error"] = "folder_missing"
            report[folder_name] = folder_issues
            all_ok = False
            continue

        # 2. Get actual frames
        indices = get_frame_indices(folder_path)
        actual_count = len(indices)

        # 3. Compare counts
        if actual_count != expected_count:
            print(f"  ❌ Count mismatch: expected {expected_count}, found {actual_count}")
            folder_issues["expected_count"] = expected_count
            folder_issues["actual_count"] = actual_count
            all_ok = False
        else:
            print(f"  ✅ Count matches: {actual_count}")

        # 4. Check missing/extra frames
        expected_indices = set(range(expected_count))
        missing = sorted(expected_indices - indices)
        extra = sorted(indices - expected_indices)

        if missing:
            print(f"  ❌ Missing frames: {missing[:10]}{'...' if len(missing) > 10 else ''}")
            folder_issues["missing_frames"] = missing
            all_ok = False
        else:
            print("  ✅ No missing frames")

        if extra:
            print(f"  ❌ Extra frames: {extra[:10]}{'...' if len(extra) > 10 else ''}")
            folder_issues["extra_frames"] = extra
            all_ok = False
        else:
            print("  ✅ No extra frames")

        # Save only if issues exist
        if folder_issues:
            report[folder_name] = folder_issues

    # --- Write report file ---
    with open(OUTPUT_REPORT, "w") as f:
        json.dump(report, f, indent=4)

    print("\n====================")
    if all_ok:
        print("✅ All folders are consistent with JSON")
    else:
        print(f"❌ Some folders have issues (saved to {OUTPUT_REPORT})")


if __name__ == "__main__":
    main()
