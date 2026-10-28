"""Verify the published module graph without loading model libraries or weights."""
import ast
import importlib
from pathlib import Path
import subprocess
import sys
import unittest

import support
from twinpast.runtime import catalog
from twinpast.upstream import sources


class DistributionTests(unittest.TestCase):
    def test_catalog_targets_and_local_import_closure_exist(self):
        package = Path(importlib.import_module('twinpast').__file__).parent
        adapters = package / 'adapters'
        local_names = {p.stem for p in adapters.rglob('*.py')}
        for condition, spec in catalog().items():
            if 'factory' in spec:
                module, name = spec['factory'].split(':')
                target = package.parent / (module.replace('.', '/') + '.py')
            else:
                name = 'Adapter'
                target = adapters / spec['folder'] / (spec['module'] + '.py')
            self.assertTrue(target.is_file(), condition)
            tree = ast.parse(target.read_text())
            self.assertIn(name, {node.name for node in tree.body if isinstance(node, ast.ClassDef)}, condition)
        for path in adapters.rglob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in local_names:
                    options = [path.parent / (node.module + '.py'), adapters / 'eval_streaming_shared' / (node.module + '.py')]
                    self.assertTrue(any(p.is_file() for p in options), str(path) + ': ' + node.module)

    def test_factory_modules_import_without_model_dependencies(self):
        factories = sorted({spec['factory'].split(':')[0] for spec in catalog().values() if 'factory' in spec})
        code = ('import importlib, sys; '
                'modules=' + repr(factories) + '; '
                '[importlib.import_module(name) for name in modules]; '
                'assert not set(["torch", "transformers", "decord"]) & set(sys.modules)')
        result = subprocess.run([sys.executable, '-B', '-c', code], env={
            'PYTHONPATH': str(Path(importlib.import_module('twinpast').__file__).parents[1]),
            'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_source_registry_has_https_and_pinned_revisions(self):
        for source in sources().values():
            self.assertTrue(source['repository'].startswith('https://github.com/'))
            self.assertRegex(source['revision'], r'^[0-9a-f]{40}$')


if __name__ == '__main__':
    unittest.main()
