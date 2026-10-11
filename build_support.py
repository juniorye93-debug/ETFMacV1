"""Only reviewed runtime modules belong in the installable engine packages."""
from setuptools.command.build_py import build_py

MODULES = {
    'riskparity_legacy': {'__init__', 'rotation', 'cash_rates', 'rotation_v2', 'rotation_flat',
                         'forecast_adaptive', 'rotation_adaptive', 'wind_fetch', 'setup_wind'},
    'riskparity_single': {'__init__', 'model', 'fetch'},
}


class EngineBuild(build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if package in MODULES:
            modules = [row for row in modules if row[1] in MODULES[package]]
        return modules
