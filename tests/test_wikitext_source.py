"""WikiText preparation accepts the pinned raw archive layout."""

import hashlib
import json
import zipfile

from bdbench import datasets


def test_raw_wikitext_archive_is_tokenized(tmp_path, monkeypatch):
    archive = tmp_path / 'raw.zip'
    with zipfile.ZipFile(archive, 'w') as output:
        for split in ('train', 'valid', 'test'):
            output.writestr(f'wikitext-2-raw/wiki.{split}.raw', 'alpha beta alpha')
    def fake_download(url, destination, expected_md5):
        destination.write_bytes(archive.read_bytes())
        return {'url': url, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'path': str(destination)}
    monkeypatch.setattr(datasets, 'download_checked', fake_download)
    result = datasets.prepare_wikitext(tmp_path)
    assert result['variant'] == 'raw-v1'
    assert json.loads((tmp_path / '.bdbench-data/datasets/wikitext2-raw-v1/train.json').read_text()) == [1, 2, 1]
