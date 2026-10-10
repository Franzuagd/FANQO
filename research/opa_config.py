"""Original branch settings with the user's supplied OPA lattice."""
from pathlib import Path
import runpy
_settings=runpy.run_path(str(Path(__file__).resolve().parents[1]/'user'/'general_config.py'))
globals().update({k:v for k,v in _settings.items() if not k.startswith('__')})
LATTICE_FILE='lattice_opa.py'
