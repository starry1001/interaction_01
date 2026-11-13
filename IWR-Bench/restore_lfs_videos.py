#!/usr/bin/env python3
"""Restore dataset MP4s from Git LFS, using the local cache first."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pull', action='store_true', help='Download missing LFS objects from origin')
    parser.add_argument('--direct', action='store_true', help='Bypass proxy settings for this invocation')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    root = Path(subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel'], cwd=project, text=True
    ).strip())
    data = project / 'data' / 'data'
    relative = data.relative_to(root).as_posix()
    pattern = f'{relative}/**/*.mp4'
    env = os.environ.copy()
    git = ['git']
    if args.direct:
        for name in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
            env.pop(name, None)
        git += ['-c', 'http.proxy=', '-c', 'https.proxy=']

    def run(*command):
        subprocess.run(git + list(command), cwd=root, env=env, check=True)

    def unresolved():
        paths = subprocess.check_output(
            ['git', 'ls-files', '-z', '--', pattern], cwd=root
        ).decode().split('\0')
        pending = []
        for name in filter(None, paths):
            path = root / name
            if not path.is_file():
                pending.append(name)
                continue
            with path.open('rb') as stream:
                if stream.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
                    pending.append(name)
        return pending

    print('Restoring dataset videos from the local LFS cache...', flush=True)
    run('lfs', 'checkout', pattern)
    pending = unresolved()
    if pending and args.pull:
        print(f'Downloading missing LFS objects for {len(pending)} videos...', flush=True)
        run('lfs', 'pull', f'--include={pattern}', '--exclude=', 'origin')
        pending = unresolved()
    if pending:
        print(f'{len(pending)} videos are still missing or contain LFS pointers:', file=sys.stderr)
        for name in pending:
            print(f'  {name}', file=sys.stderr)
        print('Run again with --pull; add --direct only if direct access works.', file=sys.stderr)
        return 1
    print('All tracked dataset MP4s are present and contain file content instead of LFS pointers.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
