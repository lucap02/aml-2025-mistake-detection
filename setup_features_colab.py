##############################
# COPIARE E INCOLLARE QUESTO #
# SCRIPT IN UNA CELLA        #
# DI COLAB PER SETUP         #
# DELLE FEATURES             #   
##############################

import os
import shutil
import zipfile
from pathlib import Path
from typing import List

from tqdm import tqdm

def mount_google_drive() -> bool:
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        print("✓ Google Drive mounted at /content/drive")
        return True
    except Exception:
        print("⚠ Not running on Colab; skipping drive mount")
        return False


def extract_zip(zip_path: str, destination: str) -> Path:
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"📦 Extracting {zip_path} -> {destination} ({len(members)} entries)")
        for member in tqdm(members, desc="Extract", unit="file"):
            zf.extract(member, destination)
    return destination_path


def extract_inner_zips(root: Path) -> List[Path]:
    extracted_dirs = []
    for zip_file in root.rglob("*.zip"):
        # Skip the outer archive itself if present in the scan
        if zip_file.samefile(root):
            continue
        target_dir = zip_file.with_suffix("")
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_file, "r") as zf:
            members = zf.namelist()
            print(f"📦 Extracting inner {zip_file} -> {target_dir} ({len(members)} entries)")
            for member in tqdm(members, desc=f"Extract {zip_file.name}", unit="file"):
                zf.extract(member, target_dir)
        extracted_dirs.append(target_dir)
    return extracted_dirs


def ensure_target_dirs(base: Path) -> None:
    (base / "video" / "omnivore").mkdir(parents=True, exist_ok=True)
    (base / "video" / "slowfast").mkdir(parents=True, exist_ok=True)
    (base / "audio").mkdir(parents=True, exist_ok=True)


def decide_target(file_path: Path, target_root: Path) -> Path:
    lower_parts = [p.lower() for p in file_path.parts]
    name_lower = file_path.name.lower()

    if "omnivore" in lower_parts or "omnivore" in name_lower:
        return target_root / "video" / "omnivore" / file_path.name
    if "slowfast" in lower_parts or "slowfast" in name_lower:
        return target_root / "video" / "slowfast" / file_path.name
    if "audio" in lower_parts or name_lower.endswith("_audio.npz"):
        return target_root / "audio" / file_path.name
    # Fallback: treat as omnivore video
    return target_root / "video" / "omnivore" / file_path.name


def move_npz_files(source_root: Path, target_root: Path) -> int:
    ensure_target_dirs(target_root)
    moved = 0
    for npz in source_root.rglob("*.npz"):
        target = decide_target(npz, target_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.move(str(npz), str(target))
            moved += 1
    return moved


def verify(target_root: Path) -> None:
    print("\n✅ Verification:")
    for label, rel in [
        ("Video Omnivore", target_root / "video" / "omnivore"),
        ("Video SlowFast", target_root / "video" / "slowfast"),
        ("Audio", target_root / "audio"),
    ]:
        count = len(list(rel.glob("*.npz"))) if rel.exists() else 0
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {label}: {count} files in {rel}")


def cleanup(temp_dir: Path) -> None:
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"🧹 Removed {temp_dir}")


def main():
    on_colab = mount_google_drive()

    outer_zip = Path("/content/drive/MyDrive/CaptainCook4D/features.zip" if on_colab else "./features.zip")
    if not outer_zip.exists():
        raise FileNotFoundError(f"Zip not found at {outer_zip}")

    temp_root = Path("./features_temp")
    target_root = Path("/content/code/data") # Direct extraction to /content/code/data

    # Step 1: Extract outer zip
    extract_zip(str(outer_zip), str(temp_root))

    # Step 2: Extract inner zips (omnivore.zip, slowfast.zip, etc.)
    extract_inner_zips(temp_root)

    # Step 3: Move npz files into expected layout
    moved = move_npz_files(temp_root, target_root)
    print(f"\n📂 Moved {moved} .npz files into {target_root}")

    # Step 4: Verify
    verify(target_root)

    # Step 5: Cleanup temp
    cleanup(temp_root)

    print("\nDone. You can now run training/eval with data under ./data")


if __name__ == "__main__":
    main()
