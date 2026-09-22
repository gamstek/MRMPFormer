import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from packaging.requirements import Requirement

spec = importlib.util.spec_from_file_location(
    'windows_package', Path(__file__).resolve().parents[1] / 'cpp/tools/build_windows_package.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


class PackageDependencyTests(unittest.TestCase):
    def test_file_is_the_source_of_roots_and_honors_markers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runtime.txt'
            path.write_text('# runtime\nnew-package>=2 # note\nignored; python_version < "2"\n', encoding='utf-8')
            self.assertEqual([str(r) for r in package.read_runtime_requirements(path)], ['new-package>=2'])

    def test_root_version_mismatch_fails(self):
        dist = SimpleNamespace(metadata={'Name': 'root'}, version='1', requires=[])
        with patch.object(package.metadata, 'distribution', return_value=dist):
            with self.assertRaisesRegex(RuntimeError, 'root>=2'):
                package.resolve_runtime_distributions([Requirement('root>=2')])

    def test_transitive_extras_cycles_and_unrelated_packages(self):
        def dist(name, dependencies):
            return SimpleNamespace(metadata={'Name': name}, version='2', requires=dependencies)
        installed = {'root': dist('Root', ['base>=1', 'optional; extra == "feature"']),
                     'base': dist('base', ['root>=1']), 'optional': dist('optional', []),
                     'build-only': dist('build-only', [])}
        with patch.object(package.metadata, 'distribution', side_effect=installed.__getitem__):
            resolved = package.resolve_runtime_distributions([Requirement('root'), Requirement('root[feature]')])
        self.assertEqual(set(resolved), {'root', 'base', 'optional'})

    def test_transitive_version_mismatch_fails(self):
        installed = {'root': SimpleNamespace(metadata={'Name': 'root'}, version='1', requires=['dep>=2']),
                     'dep': SimpleNamespace(metadata={'Name': 'dep'}, version='1', requires=[])}
        with patch.object(package.metadata, 'distribution', side_effect=installed.__getitem__):
            with self.assertRaisesRegex(RuntimeError, 'dep>=2'):
                package.resolve_runtime_distributions([Requirement('root')])


if __name__ == '__main__':
    unittest.main()
