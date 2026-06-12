# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from importlib.util import find_spec



datas = [('config.json', '.')]
hiddenimports = []
hikrobot_spec = find_spec('hikrobot')
if hikrobot_spec is not None:
    hiddenimports.append('hikrobot')
    hikrobot_locations = list(hikrobot_spec.submodule_search_locations or [])
    if hikrobot_spec.origin:
        hikrobot_locations.append(str(Path(hikrobot_spec.origin).resolve().parent))
    if hikrobot_locations:
        hikrobot_dir = Path(hikrobot_locations[0]).resolve()
        datas += [(str(path), str(Path('hikrobot') / 'MvImport')) for path in (hikrobot_dir / 'MvImport').glob('*.py')]


a = Analysis(
    ['stereo_capture_only.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MVSS_Capture',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
