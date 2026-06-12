# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from importlib.util import find_spec

MVS_RUNTIME_DIR = Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64")

datas = [('config.json', '.')]
hiddenimports = []

# --- Hikrobot Python package ---
hikrobot_spec = find_spec('hikrobot')
if hikrobot_spec is not None:
    hiddenimports.append('hikrobot')
    hikrobot_locations = list(hikrobot_spec.submodule_search_locations or [])
    if hikrobot_spec.origin:
        hikrobot_locations.append(str(Path(hikrobot_spec.origin).resolve().parent))
    if hikrobot_locations:
        hikrobot_dir = Path(hikrobot_locations[0]).resolve()
        for py_file in (hikrobot_dir).glob('*.py'):
            datas.append((str(py_file), 'hikrobot'))
        mvimport_dir = hikrobot_dir / 'MvImport'
        if mvimport_dir.exists():
            for py_file in mvimport_dir.glob('*.py'):
                datas.append((str(py_file), str(Path('hikrobot') / 'MvImport')))
            for cache_file in mvimport_dir.rglob('*.pyc'):
                rel = cache_file.relative_to(mvimport_dir)
                datas.append((str(cache_file), str(Path('hikrobot') / 'MvImport' / rel.parent)))

# --- MVS Runtime DLLs ---
binaries = []
if MVS_RUNTIME_DIR.exists():
    for dll_file in MVS_RUNTIME_DIR.glob('*.dll'):
        binaries.append((str(dll_file), '.'))
    for dll_file in MVS_RUNTIME_DIR.glob('*.ax'):
        binaries.append((str(dll_file), '.'))
    for cti_file in MVS_RUNTIME_DIR.glob('*.cti'):
        binaries.append((str(cti_file), '.'))
    ini_path = MVS_RUNTIME_DIR / 'CommonParameters.ini'
    if ini_path.exists():
        datas.append((str(ini_path), '.'))
    manifest_dir = MVS_RUNTIME_DIR
    for manifest_file in manifest_dir.glob('*.manifest'):
        datas.append((str(manifest_file), '.'))
    third_party_dir = MVS_RUNTIME_DIR / 'ThirdParty'
    if third_party_dir.exists():
        for dll_file in third_party_dir.glob('*.dll'):
            binaries.append((str(dll_file), '.'))


a = Analysis(
    ['stereo_capture_only_v2.py'],
    pathex=[],
    binaries=binaries,
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
    name='MVSS_Capture_建筑健康监测',
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
