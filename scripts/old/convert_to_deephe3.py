import argparse
import sys
import os
from pathlib import Path

# Ensure src is in python path if running from scripts/
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.snapshot import Snapshot
from net.common import Config


def main():
    parser = argparse.ArgumentParser(
        description="Convert snapshots to DeepH-E3 format."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the input directory containing snapshots.",
    )
    parser.add_argument("output_path", type=Path, help="Path to the output directory.")
    parser.add_argument(
        "--num_snapshots", type=int, default=100, help="Number of snapshots to process."
    )
    parser.add_argument(
        "--start_snapshot", type=int, default=0, help="Starting snapshot index."
    )
    parser.add_argument(
        "--cutoff", type=float, default=7.5, help="Cutoff radius for the matrix."
    )

    args = parser.parse_args()

    cfg = Config(cutoff_radius=args.cutoff)

    if not args.output_path.exists():
        os.makedirs(args.output_path, exist_ok=True)

    for i in range(args.start_snapshot, args.start_snapshot + args.num_snapshots):
        # Input paths
        # Structure: <in_path>/<index>/info.dat and <in_path>/<index>/Si_DM
        snap_dir = args.input_path / str(i)
        matrix_path = snap_dir / "Si_DM"
        info_path = snap_dir / "info.dat"

        if not matrix_path.exists() or not info_path.exists():
            print(f"Skipping snapshot {i}: Files not found at {snap_dir}")
            continue

        print(f"Processing snapshot {i}...")
        try:
            snap = Snapshot.from_openmx(
                matrix_path, info_path, convention="openmx", cfg=cfg
            )

            # Output path
            # Structure: <out_path>/<formatted_index>/...
            out_dir = args.output_path / f"{i:02d}"

            snap.export_to_deephe3(out_dir)
            print(f"Saved to {out_dir}")

        except Exception as e:
            print(f"Failed to process snapshot {i}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
