import importlib
import io
import os
import tarfile
from shutil import which

import pytest
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from PIL import Image

from pix2tex.__main__ import build_parser
from pix2tex.api.app import ContentSizeLimitMiddleware, decode_image
from pix2tex.cli import resize_to_width
from pix2tex.dataset.arxiv import _safe_extract
from pix2tex.dataset.dataset import Im2LatexDataset
from pix2tex.dataset.latex2png import Latex
from pix2tex.dataset.preprocessing.preprocess_formulas import main as preprocess_formulas
from pix2tex.gui import App, load_capture, prediction_page, prepare_capture, screenshot_tool
from pix2tex.models.transformer import sample_next_token
from pix2tex.utils.utils import pad, post_process


def test_safe_extract_accepts_regular_files(tmp_path):
    payload = b'\\documentclass{article}'
    archive_path = tmp_path / 'paper.tar'
    with tarfile.open(archive_path, 'w') as archive:
        info = tarfile.TarInfo('source/main.tex')
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    destination = tmp_path / 'unpacked'
    with tarfile.open(archive_path) as archive:
        _safe_extract(archive, destination)

    assert (destination / 'source/main.tex').read_bytes() == payload


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / 'hostile.tar'
    with tarfile.open(archive_path, 'w') as archive:
        info = tarfile.TarInfo('../../escaped.txt')
        info.size = 1
        archive.addfile(info, io.BytesIO(b'x'))

    with tarfile.open(archive_path) as archive, pytest.raises(ValueError, match='escapes'):
        _safe_extract(archive, tmp_path / 'unpacked')
    assert not (tmp_path.parent / 'escaped.txt').exists()


def test_blank_images_report_a_clear_error():
    with pytest.raises(ValueError, match='blank'):
        pad(Image.new('RGB', (64, 32), 'white'))


def test_final_resizing_does_not_accumulate_aspect_ratio_drift():
    source = Image.new('L', (640, 128), 'white')

    intermediate = resize_to_width(source, 416)
    final = resize_to_width(source, 320)

    assert intermediate.size == (416, 83)
    assert final.size == (320, 64)


def test_zero_temperature_decoding_is_deterministic():
    logits = torch.tensor([[0.1, 4.0, 2.0]])

    samples = [sample_next_token(logits, temperature=0).item() for _ in range(10)]

    assert samples == [1] * 10


def test_post_process_repairs_safe_model_command_artifacts():
    prediction = (
        r'\mathrm{t}_{13}=\frac{\mathrm{exp}(-j\varphi)}'
        r'{\mathrm{1}-\mathrm{r}_{21}\mathrm{exp}\l(-j2\varphi)}'
    )

    assert post_process(prediction) == (
        r'\mathrm{t}_{13}=\frac{\exp(-j\varphi)}'
        r'{1-\mathrm{r}_{21}\exp(-j2\varphi)}'
    )


def test_command_line_default_does_not_override_model_temperature():
    arguments = build_parser().parse_args([])

    assert arguments.temperature is None
    assert build_parser().parse_args(['--temperature', '0.4']).temperature == 0.4


def test_prediction_page_escapes_model_output():
    hostile = r'</div><img src=x onerror=alert(1)>'
    page = prediction_page(hostile)

    assert hostile not in page
    assert '&lt;/div&gt;&lt;img src=x onerror=alert(1)&gt;' in page
    assert 'Content-Security-Policy' in page


def test_wayland_prefers_grim_and_slurp(monkeypatch):
    monkeypatch.delenv('SCREENSHOT_TOOL', raising=False)
    monkeypatch.setenv('XDG_SESSION_TYPE', 'wayland')
    monkeypatch.setenv('XDG_CURRENT_DESKTOP', 'Hyprland')
    monkeypatch.setattr(
        'pix2tex.gui.which', lambda command: f'/bin/{command}' if command in {'grim', 'slurp'} else None
    )

    assert screenshot_tool() == 'grim'


def test_capture_is_detached_from_closed_source():
    source = io.BytesIO()
    Image.new('RGB', (7, 5), 'red').save(source, format='PNG')

    capture = load_capture(source)
    source.close()

    assert capture.size == (7, 5)
    assert capture.getpixel((0, 0)) == (255, 0, 0)


def test_small_capture_is_not_resampled_before_model_preprocessing():
    capture = Image.new('L', (40, 20), 'white')

    prepared = prepare_capture(capture)

    assert prepared is capture
    assert prepared.size == (40, 20)


def test_retry_slot_does_not_forward_button_checked_state():
    class Receiver:
        received = 'not-called'

        def returnSnip(self, image=None):
            self.received = image

    receiver = Receiver()
    App.retryPrediction(receiver)

    assert receiver.received is None


def test_qt_checked_state_is_not_treated_as_an_image():
    class Field:
        value = None

        def setText(self, value):
            self.value = value

        def setEnabled(self, value):
            self.value = value

    class Receiver:
        model = type('Model', (), {'last_pic': None})()
        error = Field()
        retryButton = Field()
        was_shown = False

        def show(self):
            self.was_shown = True

    receiver = Receiver()
    App.returnSnip(receiver, False)

    assert receiver.error.value == "No captured image is available to retry."
    assert receiver.retryButton.value is False
    assert receiver.was_shown


def test_dataset_state_is_not_shared_between_instances():
    first = Im2LatexDataset()
    second = Im2LatexDataset()
    first.data['shape'].append('sample')

    assert second.data == {}


def test_latex_renderer_rejects_shell_metacharacters_in_font_names():
    with pytest.raises(ValueError, match='Unsafe font name'):
        Latex([r'\[x\]'], font='Latin Modern Math; touch injected')


@pytest.mark.skipif(which('node') is None, reason='Node.js is not installed')
def test_preprocessor_handles_metacharacters_in_paths(tmp_path):
    working_directory = tmp_path / 'spaces;and-metacharacters'
    working_directory.mkdir()
    source = working_directory / 'input formulas.txt'
    output = working_directory / 'output formulas.txt'
    source.write_text(r'a + b + c + d + e + f' + '\n', encoding='utf-8')

    preprocess_formulas(
        ['--mode', 'tokenize', '--input-file', str(source), '--output-file', str(output)]
    )

    assert output.is_file()
    assert 'a' in output.read_text(encoding='utf-8')


def test_decode_image_rejects_non_images():
    with pytest.raises(HTTPException) as error:
        decode_image(b'not an image')
    assert error.value.status_code == 422


def test_api_prediction_does_not_touch_server_clipboard(monkeypatch):
    api_module = importlib.import_module('pix2tex.api.app')
    calls = []

    def fake_model(image, **kwargs):
        calls.append((image.size, kwargs))
        return r'x^2'

    image_data = io.BytesIO()
    Image.new('RGB', (20, 10), 'white').save(image_data, format='PNG')
    monkeypatch.setattr(api_module, 'model', fake_model)

    with TestClient(api_module.app) as client:
        response = client.post(
            '/predict/',
            files={'file': ('equation.png', image_data.getvalue(), 'image/png')},
        )

    assert response.status_code == 200
    assert response.json() == r'x^2'
    assert calls == [((20, 10), {'resize': True, 'copy_to_clipboard': False})]


def test_api_key_is_enforced_when_configured(monkeypatch):
    api_module = importlib.import_module('pix2tex.api.app')
    monkeypatch.setenv('PIX2TEX_API_KEY', 'correct-secret')

    with pytest.raises(HTTPException) as error:
        api_module.require_api_key('wrong-secret')
    assert error.value.status_code == 401


def test_request_body_limit_applies_before_endpoint():
    limited_app = FastAPI()
    limited_app.add_middleware(ContentSizeLimitMiddleware, max_bytes=4)

    @limited_app.post('/')
    async def consume(request: Request):
        return {'size': len(await request.body())}

    with TestClient(limited_app) as client:
        response = client.post('/', content=b'12345')
    assert response.status_code == 413


@pytest.mark.skipif(not os.environ.get('RUN_LATEX_INTEGRATION'), reason='opt-in TeX integration')
def test_xelatex_rendering_round_trip():
    from pix2tex.dataset.latex2png import tex2pil

    images = tex2pil([r'\[E=mc^2\]'])
    assert len(images) == 1
    assert images[0].width > 0
    assert images[0].height > 0
