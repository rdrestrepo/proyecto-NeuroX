import os

block_cipher = None 

excludes_list = [
    'pandas',
    'numba',
    'llvmlite',
    'sphinx',
    'docutils',
    'nbconvert',
    'IPython',
    'jupyter',
    'notebook',
    'tornado',
    'zmq',
    'cryptography',
    'urllib3',
    'requests',
    'certifi',
    'win32com',
    'win32',
    'win32api',
    'setuptools',
    'babel',
    'lxml',
    'pytest',
]

a = Analysis(
    ['Interfaz_neuroX.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets/logo_neurox.ico', 'assets'), 
    ],
    hiddenimports=[
        'scipy.signal',
        'scipy.ndimage',
        'scipy.interpolate',
        'scipy.integrate',  
        'sklearn.decomposition',
        'pywt',
        'fpdf',
        'PIL',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes_list,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NeuroX',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/logo_neurox.ico',  
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NeuroX',
)