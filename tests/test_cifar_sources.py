"""CIFAR-10 sources resolve to checksum-compatible archives."""

import pytest

from bdbench import datasets
from bdbench.datasets import cifar_source_url


def test_source_choices_have_distinct_urls():
    urls = {cifar_source_url(name) for name in ('toronto', 'huggingface', 'zenodo')}
    assert len(urls) == 3
    assert all(url.startswith('https://') and url.endswith('cifar-10-python.tar.gz') or 'cifar-10-python.tar.gz?' in url for url in urls)


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match='Unknown CIFAR-10 source'):
        cifar_source_url('other')


def test_prepare_uses_chosen_source(tmp_path, monkeypatch):
    selected = []
    def fake_download(url, destination, expected_md5):
        selected.append(url)
        raise RuntimeError('stop before download')
    monkeypatch.setattr(datasets, 'download_checked', fake_download)
    with pytest.raises(RuntimeError, match='stop before download'):
        datasets.prepare_cifar10(tmp_path, source='huggingface')
    assert selected == [cifar_source_url('huggingface')]
