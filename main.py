import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Optional


def run_pipeline():
    # Import lazily so --iterate can run without loading the default parameters.
    from modules import randomness, data_loading, data_processing, computations, plotting
    from parameters.data_parameters import rng_seed

    randomness(rng_seed)
    momenta = data_loading()
    batch, arrays = data_processing(momenta)
    production_arrays = computations(momenta, arrays)
    plotting(momenta, batch, production_arrays, arrays)


@dataclass(frozen=True)
class ParamRun:
    tag: str
    data_params_path: str
    exp_params_path: str


def _atlas_override_block(enabled: bool) -> str:
    # Keep this block self-contained so saved parameter files don't need to define
    # the ATLAS digitization arrays.
    lines: List[str] = []
    lines.append("\n# CLI override\n")
    lines.append(f"apply_atlas_track_reco = {bool(enabled)}\n")
    if enabled:
        lines.append("if 'large_data' in globals() and large_data:\n")
        lines.append("\traise ValueError('apply_atlas_track_reco requires large_data=False (needs r_lab)')\n")

        # Inject digitized points only if missing.
        lines.append("if 'atlas_track_reco_d0_points_mm' not in globals():\n")
        lines.append("\tatlas_track_reco_d0_points_mm = [\n")
        lines.append("\t\t0.440849, 0.750004, 2.24999, 4.00002, 6.00004, 8.5, 12.5,\n")
        lines.append("\t\t17.4999, 25.0001, 40.0, 62.5003, 87.5001, 125.001, 175.001,\n")
        lines.append("\t]\n")
        lines.append("\tatlas_track_reco_d0_eff = [\n")
        lines.append("\t\t1.07089, 0.934775, 0.92284, 0.907845, 0.825546, 0.823589,\n")
        lines.append("\t\t0.804646, 0.76969, 0.749444, 0.700954, 0.623784, 0.550713,\n")
        lines.append("\t\t0.406451, 0.383833,\n")
        lines.append("\t]\n")

        lines.append("if 'atlas_track_reco_rprod_points_mm' not in globals():\n")
        lines.append("\tatlas_track_reco_rprod_points_mm = [\n")
        lines.append("\t\t4.94069, 14.8222, 24.7034, 34.5848, 44.4662, 61.7591,\n")
        lines.append("\t\t86.4623, 111.166, 135.87, 172.925, 222.332, 271.739,\n")
        lines.append("\t]\n")
        lines.append("\tatlas_track_reco_rprod_eff = [\n")
        lines.append("\t\t0.936755, 0.911044, 0.886173, 0.868147, 0.836178, 0.81978,\n")
        lines.append("\t\t0.771877, 0.708359, 0.571463, 0.583806, 0.594192, 0.487514,\n")
        lines.append("\t]\n")

        lines.append("if 'atlas_track_reco_validation_mass_GeV' not in globals():\n")
        lines.append("\tatlas_track_reco_validation_mass_GeV = 10.0\n")
        lines.append("if 'atlas_track_reco_validation_mixing' not in globals():\n")
        lines.append("\tatlas_track_reco_validation_mixing = 1e-6\n")

    return "".join(lines)


def _iterate_output_block() -> str:
    # For --iterate: separate outputs per experimental parameter file/tag.
    # IMPORTANT: do NOT change `data_folder` (input lives there). Only override output paths.
    return (
        "\n# --iterate output isolation\n"
        "import os as _os\n"
        "_tag = _os.environ.get('W2HNL_ITER_TAG')\n"
        "if _tag:\n"
        "\toutput_folder = _os.path.join(data_folder, 'iter_outputs', _tag)\n"
        "else:\n"
        "\toutput_folder = None\n"
    )


def _iter_param_runs(saved_dir: str) -> List[ParamRun]:
    data_files = sorted(
        f
        for f in os.listdir(saved_dir)
        if f.endswith("_data_parameters.py") and os.path.isfile(os.path.join(saved_dir, f))
    )
    runs: List[ParamRun] = []
    for df in data_files:
        prefix = df[: -len("_data_parameters.py")]
        exp_files = sorted(
            f
            for f in os.listdir(saved_dir)
            if f.startswith(prefix + "_experimental_")
            and f.endswith("_parameters.py")
            and os.path.isfile(os.path.join(saved_dir, f))
        )
        if not exp_files:
            continue
        for ef in exp_files:
            tag = prefix + "__" + ef[: -len(".py")]
            runs.append(
                ParamRun(
                    tag=tag,
                    data_params_path=os.path.join(saved_dir, df),
                    exp_params_path=os.path.join(saved_dir, ef),
                )
            )
    return runs


def _run_one_with_params(repo_root: str, run: ParamRun) -> int:
    # Create a temporary "parameters" package that shadows the repo's parameters.
    # This avoids modifying the working tree while still letting existing imports
    # like `from parameters.data_parameters import *` resolve to the selected files.
    with tempfile.TemporaryDirectory(prefix="w2hnl_params_") as td:
        shadow_pkg = os.path.join(td, "parameters")
        os.makedirs(shadow_pkg, exist_ok=True)
        with open(os.path.join(shadow_pkg, "__init__.py"), "w", encoding="utf-8") as f:
            f.write("# auto-generated for --iterate\n")

        shutil.copyfile(run.data_params_path, os.path.join(shadow_pkg, "data_parameters.py"))
        shutil.copyfile(run.exp_params_path, os.path.join(shadow_pkg, "experimental_parameters.py"))

        with open(os.path.join(shadow_pkg, "data_parameters.py"), "a", encoding="utf-8") as f:
            f.write(_iterate_output_block())

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = td + os.pathsep + repo_root + (os.pathsep + existing if existing else "")
        env["W2HNL_ITER_TAG"] = run.tag

        cmd = [sys.executable, os.path.join(repo_root, "main.py")]
        print(f"\n[iterate] {run.tag}")
        return subprocess.run(cmd, cwd=repo_root, env=env).returncode


def _run_once_with_optional_atlas_override(repo_root: str, atlas_override: Optional[bool]) -> int:
    if atlas_override is None:
        run_pipeline()
        return 0

    with tempfile.TemporaryDirectory(prefix="w2hnl_params_") as td:
        shadow_pkg = os.path.join(td, "parameters")
        os.makedirs(shadow_pkg, exist_ok=True)
        with open(os.path.join(shadow_pkg, "__init__.py"), "w", encoding="utf-8") as f:
            f.write("# auto-generated for --atlas/--no-atlas\n")

        # Copy the default parameters and append an override.
        shutil.copyfile(
            os.path.join(repo_root, "parameters", "data_parameters.py"),
            os.path.join(shadow_pkg, "data_parameters.py"),
        )
        shutil.copyfile(
            os.path.join(repo_root, "parameters", "experimental_parameters.py"),
            os.path.join(shadow_pkg, "experimental_parameters.py"),
        )
        with open(os.path.join(shadow_pkg, "data_parameters.py"), "a", encoding="utf-8") as f:
            f.write(_atlas_override_block(bool(atlas_override)))

        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = td + os.pathsep + repo_root + (os.pathsep + existing if existing else "")
        cmd = [sys.executable, os.path.join(repo_root, "main.py")]
        return subprocess.run(cmd, cwd=repo_root, env=env).returncode


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--iterate",
        action="store_true",
        help="Iterate over parameter sets in parameters/saved_parameters.",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--atlas",
        action="store_true",
        help="Force-enable ATLAS track-reco efficiency cut (apply_atlas_track_reco=True).",
    )
    g.add_argument(
        "--no-atlas",
        action="store_true",
        help="Force-disable ATLAS track-reco efficiency cut (apply_atlas_track_reco=False).",
    )
    return p.parse_args(list(argv))


def main(argv: Iterable[str] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = os.path.dirname(os.path.abspath(__file__))

    atlas_override: Optional[bool]
    if args.atlas:
        atlas_override = True
    elif args.no_atlas:
        atlas_override = False
    else:
        atlas_override = None

    if not args.iterate:
        return _run_once_with_optional_atlas_override(repo_root, atlas_override)

    saved_dir = os.path.join(repo_root, "parameters", "saved_parameters")
    runs = _iter_param_runs(saved_dir)
    if not runs:
        print(f"No saved parameter runs found in: {saved_dir}")
        return 2

    rc = 0
    for run in runs:
        if atlas_override is None:
            rc = _run_one_with_params(repo_root, run)
        else:
            # Use the same shadow package mechanism, but append an override to the copied data parameters.
            with tempfile.TemporaryDirectory(prefix="w2hnl_params_") as td:
                shadow_pkg = os.path.join(td, "parameters")
                os.makedirs(shadow_pkg, exist_ok=True)
                with open(os.path.join(shadow_pkg, "__init__.py"), "w", encoding="utf-8") as f:
                    f.write("# auto-generated for --iterate + --atlas/--no-atlas\n")

                shutil.copyfile(run.data_params_path, os.path.join(shadow_pkg, "data_parameters.py"))
                shutil.copyfile(run.exp_params_path, os.path.join(shadow_pkg, "experimental_parameters.py"))
                with open(os.path.join(shadow_pkg, "data_parameters.py"), "a", encoding="utf-8") as f:
                    f.write(_iterate_output_block())
                    f.write(_atlas_override_block(bool(atlas_override)))

                env = os.environ.copy()
                existing = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = td + os.pathsep + repo_root + (os.pathsep + existing if existing else "")
                env["W2HNL_ITER_TAG"] = run.tag
                cmd = [sys.executable, os.path.join(repo_root, "main.py")]
                print(f"\n[iterate] {run.tag}")
                rc = subprocess.run(cmd, cwd=repo_root, env=env).returncode
        if rc != 0:
            print(f"[iterate] Failed: {run.tag} (exit={rc})")
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())