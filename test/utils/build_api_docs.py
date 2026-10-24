import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


root = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as temporary:
    source = Path(temporary)
    for name in ('torcheeg.utils.rst', 'torcheeg.visualizers.rst'):
        shutil.copyfile(root / 'docs/source' / name, source / name)
    shutil.copytree(root / 'docs/source/_static', source / '_static')
    (source / 'index.rst').write_text(
        'TorchEEG APIs\n=============\n\n.. toctree::\n\n'
        '   torcheeg.utils\n   torcheeg.visualizers\n')
    (source / 'conf.py').write_text(
        "project = 'TorchEEG'\n"
        "extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']\n"
        "html_theme = 'alabaster'\n"
        "master_doc = 'index'\n")
    subprocess.run(
        [sys.executable, '-m', 'sphinx', '-W', '--keep-going', '-b', 'html',
         str(source), str(root / 'docs/_build/api')],
        cwd=root, env=dict(os.environ, PYTHONPATH=str(root), MPLBACKEND='Agg'),
        check=True)
