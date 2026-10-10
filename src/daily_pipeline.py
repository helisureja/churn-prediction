
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.validation import clean_data, validate_data

ROOT = Path(__file__).resolve().parents[1]
INCOMING = ROOT / "data" / "incoming"
PROCESSED = ROOT / "data" / "processed"
ARCHIVE = ROOT / "data" / "archive"
REPORTS = ROOT / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def process_file(csv_path, training_command=None):
    for folder in (PROCESSED, ARCHIVE, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)

    name = csv_path.stem
    report = {
        "input_file": csv_path.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL",
        "trained": False,
        "quality_issues": [],
    }

    try:
        logger.info("Reading %s", csv_path.name)
        raw = pd.read_csv(csv_path)
        cleaned = clean_data(raw)
        errors = validate_data(cleaned)

        report["input_rows"] = len(raw)
        report["cleaned_rows"] = len(cleaned)
        report["quality_issues"] = errors

        if errors:
            logger.error("Validation failed: %s", errors)
        else:
            output = PROCESSED / f"{name}_clean.csv"
            cleaned.to_csv(output, index=False)
            report["processed_file"] = str(output)

            logger.info("Validation passed")

            if training_command is None:
                raise RuntimeError(
                    "Training command is not configured"
                )

            logger.info("Starting model training")
            subprocess.run(
                training_command + [str(output)],
                cwd=ROOT,
                check=True,
            )

            report["status"] = "PASS"
            report["trained"] = True
            logger.info("Training completed successfully")

    except Exception as exc:
        logger.exception("Pipeline failed")
        report["quality_issues"].append(str(exc))

    finally:
        report_path = REPORTS / f"{name}_report.json"
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        # Archive the original only after successful processing.
        if report["status"] == "PASS":
            destination = ARCHIVE / csv_path.name
            shutil.move(str(csv_path), str(destination))

    return report


def main():
    INCOMING.mkdir(parents=True, exist_ok=True)

    files = sorted(INCOMING.glob("*.csv"))

    if not files:
        logger.info("No new CSV files found")
        return

    # This must point to the compatible training adapter
    # created in the next step.
    command = [
        sys.executable,
        "-m",
        "src.train_daily",
    ]

    failed = False

    for csv_file in files:
        report = process_file(csv_file, command)
        print(json.dumps(report, indent=2))
        if report["status"] != "PASS":
            failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
