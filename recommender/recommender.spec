# -*- mode: python -*-

block_cipher = None


a = Analysis(['main.py'],
             pathex=['/Users/miketempleman/git/python/ContentRecommender/recommender'],
             binaries=None,
             datas=None,
             hiddenimports=['rabbit_consumer.py'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='recommender',
          debug=False,
          strip=False,
          upx=True,
          console=True )
