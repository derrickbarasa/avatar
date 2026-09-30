"""Build a standalone app:  pip install pyinstaller && python build_exe.py"""
import PyInstaller.__main__

PyInstaller.__main__.run([
    "avatar.py", "--name", "AvatarStudio", "--windowed", "--noconfirm",
    "--collect-all", "OpenGL", "--collect-submodules", "avatarkit",
])
