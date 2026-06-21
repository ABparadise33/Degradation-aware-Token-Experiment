import argparse
import os
import shutil
import subprocess
import tempfile


UIEB_REPOSITORY = "https://huggingface.co/datasets/Edddddd8787/UIEB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and arrange the UIEB dataset.")
    parser.add_argument("--output-root", default="./datasets", help="Dataset output root.")
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_root)
    full_dir = os.path.join(output_root, "full")
    raw_dir = os.path.join(full_dir, "raw")
    gt_dir = os.path.join(full_dir, "GT")

    if os.path.isdir(raw_dir) and os.path.isdir(gt_dir):
        print(f"UIEB is already prepared at {full_dir}")
        return
    if os.path.exists(full_dir):
        raise FileExistsError(
            f"{full_dir} already exists but is incomplete. Remove or rename it, then run this script again."
        )

    os.makedirs(output_root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="uieb_download_", dir=output_root) as temp_dir:
        clone_dir = os.path.join(temp_dir, "UIEB")
        subprocess.run(["git", "clone", "--depth", "1", UIEB_REPOSITORY, clone_dir], check=True)

        source_raw = os.path.join(clone_dir, "raw-890")
        source_gt = os.path.join(clone_dir, "reference-890")
        if not os.path.isdir(source_raw) or not os.path.isdir(source_gt):
            raise FileNotFoundError("Downloaded UIEB repository does not contain raw-890/reference-890.")

        os.makedirs(full_dir)
        shutil.move(source_raw, raw_dir)
        shutil.move(source_gt, gt_dir)

    print(f"UIEB prepared at {full_dir}")
    print(f"Raw images: {raw_dir}")
    print(f"References: {gt_dir}")


if __name__ == "__main__":
    main()
