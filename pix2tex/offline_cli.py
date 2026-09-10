"""Command-line interface for local, offline formula recognition."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from pix2tex.offline_ocr import (
    FormulaRecognitionError,
    OfflineFormulaOCR,
    download_model,
    looks_like_formula,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize a formula image locally with PP-FormulaNet.",
    )
    parser.add_argument("image", nargs="?", type=Path, help="formula image to recognize")
    parser.add_argument("--model-dir", type=Path, help="directory containing local model files")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="download the model once for later offline recognition",
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="recognize even when the image does not resemble a formula crop",
    )
    parser.add_argument(
        "--minder",
        action="store_true",
        help="wrap output in Minder's $$ delimiters",
    )
    parser.add_argument("--copy", action="store_true", help="copy output to the Wayland clipboard")
    parser.add_argument(
        "--worker",
        action="store_true",
        help="serve newline-delimited JSON requests on standard input",
    )
    return parser


def _format_output(formula: str, minder: bool) -> str:
    return f"$${formula}$$" if minder else formula


def _copy_to_clipboard(text: str) -> None:
    wl_copy = shutil.which("wl-copy")
    if wl_copy is None:
        raise FormulaRecognitionError("wl-copy is required for --copy")
    try:
        subprocess.run([wl_copy], input=text, text=True, check=True)
    except subprocess.CalledProcessError as error:
        raise FormulaRecognitionError("Unable to copy the formula to the clipboard") from error


def _serve_worker(model: OfflineFormulaOCR, *, classify: bool) -> int:
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("id")
            image = request["image"]
            formula = model.recognize(image, classify=classify)
            response = {"id": request_id, "ok": True, "latex": formula}
        except (FormulaRecognitionError, KeyError, TypeError, json.JSONDecodeError) as error:
            response = {"id": request_id, "ok": False, "error": str(error)}
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.download_model:
            if args.worker or args.image is not None:
                raise FormulaRecognitionError("--download-model cannot be combined with recognition")
            device = "cpu" if args.device == "auto" else args.device
            print(download_model(device))
            return 0

        if not args.worker and args.image is None:
            raise FormulaRecognitionError("provide an image or use --worker")

        if not args.worker and not args.no_classify and not looks_like_formula(args.image):
            raise FormulaRecognitionError("The pasted image does not look like a formula crop")

        model = OfflineFormulaOCR(model_dir=args.model_dir, device=args.device)
        if args.worker:
            return _serve_worker(model, classify=not args.no_classify)

        formula = model.recognize(args.image, classify=False)
        output = _format_output(formula, args.minder)
        if args.copy:
            _copy_to_clipboard(output)
        print(output)
        return 0
    except FormulaRecognitionError as error:
        print(f"formulaocr-offline: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
