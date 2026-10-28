"""Regression checks for credential leakage and publication boundaries."""
import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import quote

import support
from twinpast.data import DataError
from twinpast.security import redact, validate_configuration

CODE_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_checker', CODE_ROOT / 'tools/check_release.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class CredentialTests(unittest.TestCase):
    def test_rejects_nested_and_alternative_credential_spellings(self):
        for value in ({'apiKey': 'example'}, {'generation': {'access-token': 'example'}},
                      {'headers': {'Authorization': 'example'}}, {'nested': [{'Token': 'example'}]}):
            with self.assertRaises(DataError):
                validate_configuration(value)
        validate_configuration({'generation': {'max_tokens': 128, 'max_output_tokens': 128}, 'checkpoint': 'example'})

    def test_redacts_encoded_values_dictionary_keys_and_header_fields(self):
        value = 'not-a-live-key/with+encoding=1'
        encoded = base64.b64encode(value.encode()).decode()
        source = {value: 'normal', 'items': [value, quote(value, safe=''), encoded],
                  'nested': {'Authorization': 'not-a-live-key', 'apiKey': 'not-a-live-key'}}
        with patch.dict('os.environ', {'TWINPAST_API_KEY': value}):
            cleaned = redact(source)
        serialized = json.dumps(cleaned)
        for secret in (value, quote(value, safe=''), encoded):
            self.assertNotIn(secret, serialized)
        self.assertEqual(cleaned['nested']['Authorization'], '[REDACTED]')
        self.assertEqual(cleaned['items'], ['[REDACTED]'] * 3)


class ReleaseChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'README.md').write_text('Public code.\n')
        (self.root / 'public-files.txt').write_text('README.md\npublic-files.txt\n')

    def test_accepts_clean_source_and_rejects_unknown_files(self):
        self.assertEqual(checker.audit(self.root), [])
        (self.root / '.env').write_text('local settings\n')
        self.assertIn('unlisted-file', {issue[2] for issue in checker.audit(self.root)})

    def test_rejects_links_and_caches_even_when_ignored(self):
        cache = self.root / '__pycache__'
        cache.mkdir()
        (cache / 'module.pyc').write_bytes(b'cache')
        (self.root / 'linked').symlink_to(self.root / 'README.md')
        kinds = {issue[2] for issue in checker.audit(self.root)}
        self.assertIn('unlisted-directory', kinds)
        self.assertIn('non-regular-file', kinds)

    def test_rejects_secret_contents_without_returning_secret_values(self):
        secret = 'sk-' + 'aB1cD2eF3gH4iJ5kL6mN7oP8'
        (self.root / 'README.md').write_text(secret)
        issues = checker.audit(self.root)
        self.assertIn('credential-pattern', {issue[2] for issue in issues})
        self.assertNotIn(secret, repr(issues))

    def test_detects_current_environment_credentials_without_known_prefix(self):
        value = 'not-a-live-key-for-local-fixture'
        (self.root / 'README.md').write_text(value)
        with patch.dict('os.environ', {'TWINPAST_API_KEY': value}):
            issues = checker.audit(self.root)
        self.assertIn('current-environment-credential', {issue[2] for issue in issues})

    def test_rejects_files_still_tracked_despite_gitignore(self):
        import subprocess
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        (self.root / 'local.log').write_text('local runtime log')
        subprocess.run(['git', '-C', str(self.root), 'add', 'local.log'], check=True)
        (self.root / 'local.log').unlink()
        issues = checker.audit(self.root)
        self.assertIn('unlisted-tracked-file', {issue[2] for issue in issues})


if __name__ == '__main__':
    unittest.main()
