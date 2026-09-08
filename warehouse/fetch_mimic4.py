"""Check the landing zone for a full MIMIC-IV release, and print how to fetch it.

This script deliberately **does not download anything and never handles a
credential.** Full MIMIC-IV is credentialed access: it requires a PhysioNet
account, completed CITI training, and a signed data use agreement, and the
files are served over HTTP basic auth. Those credentials belong to the person
who signed the DUA, so the download is a command *you* run, with `wget`
prompting for the password interactively -- nothing here reads, stores, echoes,
or forwards it.

What this does do is the part that is easy to get wrong: tell you whether the
directory is complete before you spend an hour building a warehouse from a
half-finished download, and whether this machine has the disk to hold it.

Usage:
    python warehouse/fetch_mimic4.py                       # check the default path
    python warehouse/fetch_mimic4.py --data-dir data/raw/mimic-iv-3.1
    python warehouse/fetch_mimic4.py --verify-checksums    # slow; hashes every file
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "mimic-iv-3.1"
PHYSIONET_URL = "https://physionet.org/files/mimiciv/3.1/"

# The tables this project actually reads: warehouse/build_duckdb.py loads every
# file it finds, but these are the ones whose absence breaks a downstream step,
# so they are what "complete" is judged against.
REQUIRED = {
    "hosp": [
        "admissions.csv.gz",
        "patients.csv.gz",
        "labevents.csv.gz",
        "d_labitems.csv.gz",
        "transfers.csv.gz",
        "services.csv.gz",
        "diagnoses_icd.csv.gz",
        "d_icd_diagnoses.csv.gz",
        "prescriptions.csv.gz",
        "pharmacy.csv.gz",
        "poe.csv.gz",
        "microbiologyevents.csv.gz",
        "omr.csv.gz",
    ],
    "icu": [
        "icustays.csv.gz",
        "chartevents.csv.gz",
        "d_items.csv.gz",
        "inputevents.csv.gz",
        "outputevents.csv.gz",
        "procedureevents.csv.gz",
        "datetimeevents.csv.gz",
        "ingredientevents.csv.gz",
    ],
}

# Rough guidance only, and labelled as such wherever it is printed. The point is
# the order of magnitude -- that a full build is a tens-of-gigabytes commitment,
# not a few hundred megabytes -- not a precise figure to plan a disk against.
APPROX_DOWNLOAD_GB = 8
APPROX_FULL_WAREHOUSE_GB = 40


def human_gb(n_bytes: float) -> str:
    return f"{n_bytes / 1024**3:.1f} GB"


def survey(data_dir: Path) -> tuple[list[str], list[str], int]:
    """Returns (present, missing, total_bytes) for the required files."""
    present: list[str] = []
    missing: list[str] = []
    total = 0
    for module, files in REQUIRED.items():
        for name in files:
            path = data_dir / module / name
            rel = f"{module}/{name}"
            if path.exists() and path.stat().st_size > 0:
                present.append(rel)
                total += path.stat().st_size
            else:
                missing.append(rel)
    return present, missing, total


def verify_checksums(data_dir: Path) -> tuple[int, list[str]]:
    """Hash every file listed in the release's own SHA256SUMS.txt.

    A truncated download is the failure this catches: it leaves a valid gzip
    prefix that reads fine for thousands of rows and then fails partway through
    a multi-hour load, which is a miserable way to discover it.
    """
    sums = data_dir / "SHA256SUMS.txt"
    if not sums.exists():
        return 0, [f"no SHA256SUMS.txt in {data_dir} -- nothing to verify against"]

    checked, problems = 0, []
    for line in sums.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        expected, rel = parts[0], parts[1].lstrip("*./")
        path = data_dir / rel
        if not path.exists():
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        checked += 1
        if digest.hexdigest() != expected:
            problems.append(f"checksum mismatch: {rel}")
    return checked, problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument(
        "--verify-checksums",
        action="store_true",
        help="hash every present file against the release's SHA256SUMS.txt (slow)",
    )
    args = ap.parse_args()

    print(f"== landing zone: {args.data_dir}")
    if not args.data_dir.exists():
        print("   directory does not exist yet")

    present, missing, total = survey(args.data_dir)
    print(f"   {len(present)}/{len(present) + len(missing)} required files present")
    if present:
        print(f"   occupying {human_gb(total)}")
    if missing:
        shown = ", ".join(missing[:6])
        more = f" (+{len(missing) - 6} more)" if len(missing) > 6 else ""
        print(f"   missing: {shown}{more}")

    free = shutil.disk_usage(REPO_ROOT).free
    print(f"\n== disk: {human_gb(free)} free at {REPO_ROOT}")
    print(
        f"   the download is roughly {APPROX_DOWNLOAD_GB} GB compressed, and a warehouse "
        f"built from the WHOLE release lands nearer {APPROX_FULL_WAREHOUSE_GB} GB "
        f"(both approximate)."
    )
    if free < (APPROX_DOWNLOAD_GB + APPROX_FULL_WAREHOUSE_GB) * 1024**3:
        print(
            "   NOT enough room for a whole-release warehouse on this machine. "
            "Use --cohort-subjects to load a sample instead:"
        )
        print(
            "     python warehouse/build_duckdb.py --data-dir "
            f"{args.data_dir} \\\n"
            "         --db warehouse/mimic4_full.db --cohort-subjects 8000 --force"
        )
        print(
            "   Cohort size should be sized from ml/evaluation/reliability_report.md, "
            "not guessed."
        )

    if args.verify_checksums:
        print("\n== checksums")
        checked, problems = verify_checksums(args.data_dir)
        print(f"   verified {checked} files")
        for p in problems:
            print(f"   {p}")
        if problems:
            return 1

    if missing:
        print(f"\n== to fetch it yourself (credentialed access, {PHYSIONET_URL})")
        print("   Requires a PhysioNet account with CITI training and a signed DUA for")
        print("   MIMIC-IV. wget prompts for the password; nothing in this repo stores it.")
        print(f"\n     mkdir -p {args.data_dir}")
        print(
            f"     wget -r -N -c -np --user YOUR_PHYSIONET_USERNAME --ask-password \\\n"
            f"         -nH --cut-dirs=3 -P {args.data_dir} \\\n"
            f"         {PHYSIONET_URL}"
        )
        print("\n   Then re-run this script to confirm the landing zone is complete.")
        return 1

    print("\nLanding zone complete. Next:")
    print(
        f"   python warehouse/build_duckdb.py --data-dir {args.data_dir} \\\n"
        "       --db warehouse/mimic4_full.db --cohort-subjects N --force"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
