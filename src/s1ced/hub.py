"""Find a release folder: a local path, or a Hugging Face repo id that is downloaded once into the HF cache."""
import os, re, sys
from pathlib import Path

DEFAULT_MODEL = 'syedazeez/s1ced-v3-deep'
REQUIRED = ('config.json', 'tokenizer.json', 'model.safetensors')
OPTIONAL = ('calibration.json',)           # temperature + conformal threshold fitted for these exact weights
REPO_ID = re.compile(r'^[\w.-]+/[\w.-]+$')


def resolve_release(model=None, revision=None):
    """model: a release folder, a Hugging Face repo id, or None for $S1CED_MODEL / the default release."""
    model = str(model or os.environ.get('S1CED_MODEL') or DEFAULT_MODEL)
    path = Path(model).expanduser()
    if path.is_dir():
        missing = [f for f in REQUIRED if not (path / f).exists()]
        if missing: raise FileNotFoundError(f'{path} is not an S1-CED release folder (missing {", ".join(missing)})')
        return path
    if not REPO_ID.match(model):
        raise FileNotFoundError(f'{model!r} is neither a release folder nor a Hugging Face repo id like {DEFAULT_MODEL}')
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    if not isinstance(try_to_load_from_cache(model, 'model.safetensors', revision=revision), str):
        print(f'Downloading {model} from Hugging Face (about 155 MB, first run only)...', file=sys.stderr, flush=True)
    return Path(snapshot_download(model, revision=revision, allow_patterns=[*REQUIRED, *OPTIONAL]))
