import os

from PyInstaller.utils.hooks import collect_data_files

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

# ---- Datos empaquetados -------------------------------------------------
# 'assets' completo: el informe PDF y la interfaz buscan logos .png y la
# animacion de carga, no solo el .ico. Antes solo se copiaba el icono y el
# ejecutable arrancaba sin logo.
datas = [
    ('assets', 'assets'),
    # Umbrales del diagnostico de adquisicion. Va empaquetado como default;
    # el loader busca primero config.json junto a NeuroX.exe para que el
    # tecnico pueda ajustarlo sin recompilar.
    ('config.json', '.'),
]

# mne-icalabel incluye la red neuronal ICLabelNet.onnx (y megnet.onnx). Sin
# esos archivos ICLabel falla al clasificar componentes en el .exe.
datas += collect_data_files('mne_icalabel')

# MNE necesita sus montajes 10-10/10-20 en runtime para set_montage().
# collect_data_files sin filtro recoge todos los .fif/.elc/.txt de channels/data.
datas += collect_data_files('mne')

a = Analysis(
    ['Interfaz_neuroX.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'scipy.signal',
        'scipy.ndimage',
        'scipy.interpolate',
        'scipy.integrate',
        'pywt',
        'fpdf',
        'PIL',
        'mne',
        'mne_icalabel',
        'onnxruntime',
        'openpyxl',
        'pypdf',
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
