"""Dataset downloads report progress even below ten MiB."""

from hashlib import md5
from io import BytesIO
import urllib.request

from bdbench.datasets import download_checked


def test_small_download_reports_final_total(tmp_path, monkeypatch, capsys):
    data = b'x' * (1024 * 1024)
    class Response(BytesIO):
        headers = {'Content-Length': str(len(data))}
    monkeypatch.setattr(urllib.request, 'urlopen', lambda url, timeout: Response(data))
    destination = tmp_path / 'archive.bin'
    download_checked('https://example.invalid/archive', destination, md5(data).hexdigest())
    assert destination.read_bytes() == data
    assert '1.0/1.0 MiB (100%)' in capsys.readouterr().out


def test_reused_archive_does_not_claim_selected_download_source(tmp_path):
    data = b'already present'
    destination = tmp_path / 'archive.bin'
    destination.write_bytes(data)
    info = download_checked('https://example.invalid/unused', destination, md5(data).hexdigest())
    assert info['url'] is None
    assert info['archive_status'] == 'reused'
