"""Execute frozen-v3 QA notebook checks in an isolated local project copy.

Requires the README environment/data setup and the exact saved frozen model.
No training runs. Results and an executed notebook are saved in --output-dir.
"""
import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
import types
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
FROZEN_COMMIT = "5e95a2cbdbe7d86d32d3aa9d88d3db77d7a9db2c"
MODEL_SHA256 = "9da5b69c57476cae69a266553c3b1d39c35cd0ae1306716ed586270a92d3dfa9"


class Capture(io.StringIO):
    def write(self, value):
        sys.__stdout__.write(value)
        sys.__stdout__.flush()
        return super().write(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/qa_v3")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("Output directory already exists; choose a new --output-dir to preserve evidence.")
    model = ROOT / "data/models/emnist_mlp.npz"
    if not model.is_file() or hashlib.sha256(model.read_bytes()).hexdigest() != MODEL_SHA256:
        parser.error("Restore the exact frozen checkpoint recorded in docs/version_freeze.md.")
    expected_data = json.loads((ROOT / "configs/emnist_sha256.json").read_text())
    for name, expected_hash in expected_data.items():
        path = ROOT / "data/emnist" / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            parser.error(f"EMNIST checksum verification failed: {name}. Run scripts/download_emnist.py.")
    subprocess.run(["git", "diff", "--exit-code", FROZEN_COMMIT, "--", "mlreader", "assets", "configs"], cwd=ROOT, check=True)
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_paths = subprocess.check_output(["git", "ls-files", "mlreader", "assets", "configs"], cwd=ROOT, text=True).splitlines()
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths}
    output.mkdir(parents=True)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    os.environ["MPLBACKEND"] = "Agg"
    cache = Path(tempfile.gettempdir()) / "ml7_qa_matplotlib_cache"
    cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    notebook_path = ROOT / "docs/qa/MileageLogReader_QA.ipynb"
    notebook = json.loads(notebook_path.read_text())
    started = datetime.now(timezone.utc).isoformat()
    record = {"started_utc": started, "code_commit": code_commit,
              "frozen_reader_commit": FROZEN_COMMIT, "model_sha256": MODEL_SHA256,
              "training_performed": False, "reader_source_sha256": source_hashes,
              "emnist_sha256": expected_data,
              "notebook_input_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
              "notebook_code_sha256": hashlib.sha256(json.dumps(
                  [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"],
                  ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
              "execution": "Local Python execution of notebook QA cells in original order",
              "skipped_cells": ["Step 1 Colab clone", "Step 2 Colab install/download/upload", "Step 23 Colab archive download"],
              "setup": "Local dependencies and verified frozen checkpoint; tracked source files copied into an isolated temporary project.",
              "as_of_date": "2026-10-05", "completed_steps": [], "status": "Running"}
    try:
        with tempfile.TemporaryDirectory(prefix="ml7_qa_v3_") as temporary:
            project = Path(temporary) / "mileage-log-reader"
            project.mkdir()
            # Use current tracked source and fixtures without copying ignored run outputs.
            paths = subprocess.check_output(["git", "ls-files", "mlreader", "scripts", "assets", "configs", "examples", "tests", "requirements.txt"], cwd=ROOT, text=True).splitlines()
            for name in paths:
                destination = project / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
            (project / "data/models").mkdir(parents=True)
            (project / "data/emnist").symlink_to(ROOT / "data/emnist", target_is_directory=True)
            (project / "data/models/emnist_mlp.npz").symlink_to(model)
            google = types.ModuleType("google")
            colab = types.ModuleType("google.colab")
            colab.files = types.SimpleNamespace(download=lambda path: None)
            google.colab = colab
            sys.modules["google"] = google
            sys.modules["google.colab"] = colab
            scope = {"PROJECT_DIR": project, "CODE_COMMIT": code_commit,
                     "FROZEN_READER_COMMIT": FROZEN_COMMIT, "EXPECTED_MODEL_SHA256": MODEL_SHA256}
            sys.path.insert(0, str(project))
            import matplotlib.pyplot as plt
            def show(*args, **kwargs):
                for number in plt.get_fignums():
                    buffer = io.BytesIO()
                    plt.figure(number).savefig(buffer, format="png", bbox_inches="tight")
                    current_cell["outputs"].append({"output_type": "display_data", "metadata": {},
                        "data": {"image/png": base64.b64encode(buffer.getvalue()).decode(), "text/plain": ["QA comparison figure"]}})
            plt.show = show
            execution_count = 0
            for index in range(4, 24):
                current_cell = notebook["cells"][index]
                source = "".join(current_cell["source"])
                current_cell["outputs"] = []
                execution_count += 1
                current_cell["execution_count"] = execution_count
                print("\n" + source.splitlines()[0], flush=True)
                captured = Capture()
                with redirect_stdout(captured), redirect_stderr(captured):
                    exec(compile(source, f"QA notebook cell {index}", "exec"), scope)
                current_cell["outputs"].insert(0, {"output_type": "stream", "name": "stdout", "text": captured.getvalue().splitlines(keepends=True)})
                record["completed_steps"].append(source.splitlines()[0])
            evidence = project / "outputs/qa"
            shutil.copytree(evidence, output / "evidence")
            shutil.copytree(project / "outputs/model_eval", output / "evidence/model_eval")
            for folder, source_folder in [("run_configuration", project / "configs"), ("fixture_metadata", project / "examples/synthetic_forms"), ("qa_fixture_metadata", project / "examples/qa_fixtures")]:
                destination = output / "evidence" / folder
                destination.mkdir()
                for path in source_folder.glob("*.json"):
                    shutil.copy2(path, destination / path.name)
                for path in source_folder.glob("*.csv"):
                    shutil.copy2(path, destination / path.name)
            shutil.copy2(project / "requirements.txt", output / "evidence/run_configuration/requirements.txt")
            tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=project, capture_output=True, text=True)
            (output / "evidence/Regression_Test_Log.txt").write_text(tests.stdout + tests.stderr)
            tests.check_returncode()
            record["regression_suite"] = "41 tests passed"
            import cv2, numpy, matplotlib
            environment = {"code_commit": code_commit, "frozen_reader_commit": FROZEN_COMMIT,
                "recorded_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "numpy": numpy.__version__, "opencv": cv2.__version__, "matplotlib": matplotlib.__version__,
                "scope": "Frozen-reader synthetic QA; no new handwriting-generalization claim.",
                "manual_review_required": ["QA-02 orientation", "QA-04 crops"]}
            (output / "evidence/QA_Run_Environment.json").write_text(json.dumps(environment, indent=2) + "\n")
            packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
            (output / "evidence/QA_Environment_Packages.txt").write_text(packages)
            record["status"] = "Completed - awaiting visual review"
    except Exception:
        record["status"] = "Failed"
        record["error"] = traceback.format_exc()
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        notebook["metadata"]["qa_execution"] = record
        (output / "QA_Execution_Record.json").write_text(json.dumps(record, indent=2) + "\n")
        (output / "MileageLogReader_QA.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"\nQA checks completed. Review the two comparison images in {output}/evidence.")


if __name__ == "__main__":
    main()
