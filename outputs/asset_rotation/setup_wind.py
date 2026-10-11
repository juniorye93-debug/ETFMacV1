"""Attach the official macOS Wind API SDK to the current Python environment."""
from pathlib import Path
import site

sdk = Path('/Applications/Wind API.app/Contents/python/WindPy.py')
state_source = Path.home() / 'Library/Containers/com.wind.mac.api/Data/.Wind'
if not sdk.is_file() or not state_source.is_dir():
    raise SystemExit('请先安装并打开官方 Wind API 应用。')
for link, target in [(Path(site.getsitepackages()[0]) / 'WindPy.py', sdk),
                     (Path.home() / '.Wind', state_source)]:
    if link.exists() or link.is_symlink():
        if link.resolve() != target.resolve():
            raise SystemExit(f'保留已有路径，未覆盖: {link}；请检查其配置。')
    else:
        link.symlink_to(target, target_is_directory=target.is_dir())
    print('Ready:', link)
print('SDK配置完成。请在Wind API应用完成登录，再运行wind_fetch.py。')
