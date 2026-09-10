from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pix2tex import offline_cli
from pix2tex.offline_ocr import (
    FormulaRecognitionError,
    extract_formula,
    looks_like_formula,
    normalize_formula,
    require_model_dir,
    validate_formula,
)


def test_normalize_formula_removes_only_outer_presentation_wrappers():
    assert normalize_formula(r"$$\frac{a}{b}$$") == r"\frac{a}{b}"
    assert normalize_formula("```latex\nx^2 + y^2\n```") == "x^2 + y^2"


def test_validate_formula_rejects_malformed_or_unsafe_tex():
    with pytest.raises(FormulaRecognitionError, match="unbalanced braces"):
        validate_formula(r"\frac{a}{b")
    with pytest.raises(FormulaRecognitionError, match="mismatched environments"):
        validate_formula(r"\begin{matrix}x\end{array}")
    with pytest.raises(FormulaRecognitionError, match="disallowed"):
        validate_formula(r"\input{/etc/passwd}")


def test_extract_formula_accepts_paddle_result_shape():
    result = SimpleNamespace(json={"res": {"rec_formula": r"\begin{bmatrix}a\\b\end{bmatrix}"}})
    assert extract_formula(result) == r"\begin{bmatrix}a\\b\end{bmatrix}"


def test_require_model_dir_never_downloads(tmp_path):
    with pytest.raises(FormulaRecognitionError, match="--download-model"):
        require_model_dir(tmp_path)
    for name in ("inference.json", "inference.pdiparams", "inference.yml"):
        (tmp_path / name).touch()
    assert require_model_dir(tmp_path) == tmp_path.resolve()


def test_formula_crop_classifier_rejects_blank_and_color_photos(tmp_path):
    blank_path = tmp_path / "blank.png"
    Image.new("RGB", (240, 80), "white").save(blank_path)
    assert not looks_like_formula(blank_path)

    photo_path = tmp_path / "photo.png"
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (80, 240, 3), dtype=np.uint8)).save(photo_path)
    assert not looks_like_formula(photo_path)


def test_formula_crop_classifier_accepts_dark_text_on_light_background(tmp_path):
    formula_path = tmp_path / "formula.png"
    image = Image.new("RGB", (240, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 24, 55, 29), fill="black")
    draw.rectangle((72, 18, 78, 52), fill="black")
    draw.rectangle((96, 32, 155, 37), fill="black")
    draw.rectangle((175, 20, 181, 55), fill="black")
    image.save(formula_path)
    assert looks_like_formula(formula_path)


def test_cli_rejects_non_formula_before_loading_model(tmp_path, monkeypatch, capsys):
    image_path = tmp_path / "not-formula.png"
    Image.new("RGB", (120, 80), "white").save(image_path)

    monkeypatch.setattr(offline_cli, "looks_like_formula", lambda _path: False)

    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("the model should not be loaded for a non-formula image")

    monkeypatch.setattr(offline_cli, "OfflineFormulaOCR", fail_if_loaded)

    assert offline_cli.main([str(image_path)]) == 1
    assert "does not look like a formula" in capsys.readouterr().err
