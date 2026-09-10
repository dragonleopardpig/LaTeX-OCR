import hashlib

import pytest

from pix2tex.model.checkpoints import get_latest_checkpoint as checkpoints


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content
        self.headers = {'content-length': str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield from (
            self.content[offset : offset + chunk_size]
            for offset in range(0, len(self.content), chunk_size)
        )


def test_checkpoint_download_is_verified_and_cached(tmp_path, monkeypatch):
    content = b'test checkpoint contents'
    monkeypatch.setattr(
        checkpoints,
        'CHECKPOINTS',
        {
            'test.pth': {
                'url': 'https://example.invalid/test.pth',
                'size': len(content),
                'sha256': hashlib.sha256(content).hexdigest(),
            }
        },
    )
    calls = []

    def fake_get(*_args, **_kwargs):
        calls.append(True)
        return FakeResponse(content)

    monkeypatch.setattr(checkpoints.requests, 'get', fake_get)
    destination = tmp_path / 'test.pth'

    assert checkpoints.download_checkpoint('test.pth', destination) == destination
    assert destination.read_bytes() == content
    assert destination.stat().st_mode & 0o777 == 0o600
    assert len(calls) == 1

    monkeypatch.setattr(
        checkpoints.requests,
        'get',
        lambda *_args, **_kwargs: pytest.fail('valid cached files must not be downloaded'),
    )
    assert checkpoints.download_checkpoint('test.pth', destination) == destination


def test_checkpoint_hash_mismatch_is_not_installed(tmp_path, monkeypatch):
    content = b'tampered'
    monkeypatch.setattr(
        checkpoints,
        'CHECKPOINTS',
        {
            'test.pth': {
                'url': 'https://example.invalid/test.pth',
                'size': len(content),
                'sha256': '0' * 64,
            }
        },
    )
    monkeypatch.setattr(
        checkpoints.requests,
        'get',
        lambda *_args, **_kwargs: FakeResponse(content),
    )
    destination = tmp_path / 'test.pth'

    with pytest.raises(RuntimeError, match='SHA-256'):
        checkpoints.download_checkpoint('test.pth', destination)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
