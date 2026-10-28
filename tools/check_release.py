#!/usr/bin/env python3
"""Read-only publication check. Prints finding locations, never secret values."""
import argparse
from collections import Counter
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys

PREFIX = re.compile(
    r"sk-(?:proj-|ant-|or-v1-)?[A-Za-z0-9_-]{16,}|"
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|"
    r"hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|ASIA[A-Z0-9]{16}|"
    r"AIza[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{15,}|"
    r"gsk_[A-Za-z0-9]{20,}|nvapi-[A-Za-z0-9_-]{20,}|glpat-[A-Za-z0-9_-]{20,})")
ASSIGNMENT = re.compile(
    r'''(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key|auth[_-]?token|authorization|password|credential|token|secret)["']?\s*[:=]\s*(?:[bru]{0,2})?["']([^"'\r\n]{12,1024})["']''')
BEARER = re.compile(r'(?i)\bBearer\s+([A-Za-z0-9_+/.=-]{16,1024})')
PRIVATE_KEY = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----')
PERSONAL_PATH = re.compile(r'(?<![A-Za-z0-9])/(?:home/[^/\s"\']+|Users/[^/\s"\']+|data[0-9]*/[^/\s"\']+|root)/')
PLACEHOLDER = re.compile(r'(?i)(?:example|placeholder|dummy|not[-_ ]?a[-_ ]?live|your[-_ ]?(?:api|key|token|secret)|test[-_ ]?(?:key|token|secret)|redacted|changeme|should[-_ ]?not[-_ ]?appear)')


def plausible(value):
    return (16 <= len(value) <= 1024 and len(set(value)) >= 8 and not PLACEHOLDER.search(value)
            and not any(c in value for c in '{}()[] \\')
            and not value.startswith(('https://', 'http://', '${', '<', 'os.environ', 'os.getenv')))


def scan_text(text, env_secrets=()):
    """Return line numbers and finding types without retaining matched contents."""
    findings = set()
    for pattern, kind, group in [(PREFIX, 'credential-pattern', 0),
                                 (ASSIGNMENT, 'credential-assignment', 1),
                                 (BEARER, 'bearer-token', 1)]:
        for match in pattern.finditer(text):
            if plausible(match[group]):
                findings.add((text.count('\n', 0, match.start()) + 1, kind))
    for pattern, kind in [(PRIVATE_KEY, 'private-key'), (PERSONAL_PATH, 'personal-filesystem-path')]:
        for match in pattern.finditer(text):
            findings.add((text.count('\n', 0, match.start()) + 1, kind))
    for value in env_secrets:
        start = text.find(value)
        if start >= 0:
            findings.add((text.count('\n', 0, start) + 1, 'current-environment-credential'))
    return sorted(findings)


def audit(root):
    root = Path(root).resolve()
    issues = []
    def flag(name, kind, line=None):
        issues.append((name, line, kind))
    manifest = root / 'public-files.txt'
    if manifest.is_symlink() or not manifest.is_file():
        return [('public-files.txt', None, 'missing-or-unsafe-manifest')]
    expected = set()
    for name in manifest.read_text(encoding='utf-8').splitlines():
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or '\\' in name
                or any(part in ('', '.', '..') for part in name.split('/')) or name in expected):
            flag('public-files.txt', 'invalid-manifest-entry')
        else:
            expected.add(name)
    # Only this repository's administrative Git directory is excluded. Other
    # unlisted files, including ignored local secrets and caches, block release.
    actual = set()
    allowed_dirs = {str(parent) for name in expected for parent in PurePosixPath(name).parents if str(parent) != '.'}
    env_secrets = [value for name, value in os.environ.items()
                   if any(word in name.upper() for word in ['API_KEY', 'ACCESS_TOKEN', 'SECRET_KEY', 'AUTH_TOKEN', 'PASSWORD'])
                   and len(value) >= 16]
    for base, dirs, files in os.walk(root, followlinks=False):
        base = Path(base)
        for name in dirs[:]:
            path = base / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                flag(relative, 'symbolic-link')
                dirs.remove(name)
            elif base == root and name == '.git':
                dirs.remove(name)
            elif relative not in allowed_dirs:
                flag(relative, 'unlisted-directory')
                dirs.remove(name)
        for name in files:
            path = base / name
            relative = path.relative_to(root).as_posix()
            if base == root and name == '.git' and path.is_file() and not path.is_symlink():
                continue  # Git worktree administrative pointer, not published content.
            actual.add(relative)
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    flag(relative, 'non-regular-file')
                    continue
                if info.st_nlink > 1:
                    flag(relative, 'hard-linked-file')
                if info.st_mode & (stat.S_ISUID | stat.S_ISGID):
                    flag(relative, 'setid-file')
                if relative not in expected:
                    flag(relative, 'unlisted-file')
                    continue
                if info.st_size > 2 * 1024 * 1024:
                    flag(relative, 'oversize-source-file')
                    continue
                text = path.read_text(encoding='utf-8')
                for line, kind in scan_text(text, env_secrets):
                    flag(relative, kind, line)
            except (OSError, UnicodeError):
                flag(relative, 'unreadable-or-binary-source-file')
    for name in sorted(expected - actual):
        flag(name, 'missing-listed-file')
    if (root / '.git').exists():
        # Read the Git index as well: ignored credentials already tracked in Git
        # are not protected by adding a new .gitignore rule.
        tracked = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'], capture_output=True)
        if tracked.returncode:
            flag('.git', 'cannot-read-git-index')
        else:
            for raw in tracked.stdout.split(b'\0'):
                if raw and os.fsdecode(raw) not in expected:
                    flag(os.fsdecode(raw), 'unlisted-tracked-file')
    return sorted(set(issues), key=lambda value: (value[0], value[1] or 0, value[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    issues = audit(args.root)
    if issues:
        # Omit even filenames from console output: a credential could have been
        # used as a filename. Location details can be inspected via audit(root).
        print('Release check failed: ' + ', '.join('{}={}'.format(k, v) for k, v in sorted(Counter(i[2] for i in issues).items())), file=sys.stderr)
        return 1
    print('Release check passed: only listed public source files; no detected credentials or local artifacts.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
