# modified from https://github.com/soskek/arxiv_leaks

import argparse
import glob
import logging
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

import requests
from tqdm import tqdm

from pix2tex.dataset.demacro import *
from pix2tex.dataset.extract_latex import find_math
from pix2tex.dataset.scraping import recursive_search

# logging.getLogger().setLevel(logging.INFO)
arxiv_id = re.compile(r'(?<!\d)(\d{4}\.\d{5})(?!\d)')
arxiv_base = 'https://export.arxiv.org/e-print/'
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 50_000
NETWORK_TIMEOUT = (10, 120)


def get_all_arxiv_ids(text):
    '''returns all arxiv ids present in a string `text`'''
    ids = []
    for id in arxiv_id.findall(text):
        ids.append(id)
    return list(set(ids))


def download(url, dir_path='./'):
    idx = os.path.split(url)[-1]
    if re.fullmatch(r"[A-Za-z0-9._-]+", idx) is None:
        raise ValueError(f"Unsafe archive identifier: {idx!r}")
    file_name = idx + '.tar.gz'
    directory = Path(dir_path)
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / file_name
    if file_path.exists():
        return str(file_path)
    logging.info('\tdownload {}'.format(url) + '\n')
    temporary_path = None
    try:
        with requests.get(url, stream=True, timeout=NETWORK_TIMEOUT) as response:
            response.raise_for_status()
            advertised_size = int(response.headers.get('content-length', 0))
            if advertised_size > MAX_ARCHIVE_BYTES:
                raise ValueError(f"Archive is too large: {advertised_size} bytes")
            with tempfile.NamedTemporaryFile(
                mode='wb', dir=directory, prefix=f'.{file_name}.', delete=False
            ) as output:
                temporary_path = Path(output.name)
                downloaded = 0
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > MAX_ARCHIVE_BYTES:
                        raise ValueError("Archive download exceeded the size limit")
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        os.replace(temporary_path, file_path)
        return str(file_path)
    except requests.RequestException as error:
        logging.info('Could not download %s: %s', url, error)
        return 0
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _safe_extract(archive: tarfile.TarFile, destination: str) -> None:
    """Extract regular files and directories without path or link traversal."""
    root = Path(destination).resolve()
    members = archive.getmembers()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ValueError("Archive contains too many members")
    if sum(member.size for member in members) > MAX_EXTRACTED_BYTES:
        raise ValueError("Expanded archive exceeds the size limit")

    for member in members:
        target = (root / member.name).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Archive member escapes extraction directory: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"Archive links are not accepted: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"Unsupported archive member type: {member.name}")

    archive.extractall(root, members=members, filter='data')


def read_tex_files(file_path:str, demacro:bool=False)->str:
    """Read all tex files in the latex source at `file_path`. If it is not a `tar.gz` file try to read it as text file.

    Args:
        file_path (str): Path to latex source
        demacro (bool, optional): Deprecated. Call external `de-macro` program. Defaults to False.

    Returns:
        str: All Latex files concatenated into one string.
    """
    tex = ''
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                with tarfile.open(file_path, 'r') as tf:
                    _safe_extract(tf, tempdir)
                texfiles = [os.path.abspath(x) for x in glob.glob(os.path.join(tempdir, '**', '*.tex'), recursive=True)]
            except tarfile.ReadError:
                texfiles = [file_path]  # [os.path.join(tempdir, file_path+'.tex')]
            if demacro:
                ret = subprocess.run(['de-macro', *texfiles], cwd=tempdir, capture_output=True)
                if ret.returncode == 0:
                    texfiles = glob.glob(os.path.join(tempdir, '**', '*-clean.tex'), recursive=True)
            for texfile in texfiles:
                try:
                    ct = open(texfile, 'r', encoding='utf-8').read()
                    tex += ct
                except UnicodeDecodeError as e:
                    logging.debug(e)
                    pass
    except Exception as e:
        logging.debug('Could not read %s: %s' % (file_path, str(e)))
        raise e
    tex = pydemacro(tex)
    return tex


def download_paper(arxiv_id, dir_path='./'):
    url = arxiv_base + arxiv_id
    return download(url, dir_path)


def read_paper(targz_path, delete=False, demacro=False):
    paper = ''
    if targz_path != 0:
        paper = read_tex_files(targz_path, demacro=demacro)
        if delete:
            os.remove(targz_path)
    return paper


def parse_arxiv(id, save=None, demacro=True):
    if save is None:
        dir = tempfile.gettempdir()
    else:
        dir = save
    text = read_paper(download_paper(id, dir), delete=save is None, demacro=demacro)

    return find_math(text, wiki=False), []


if __name__ == '__main__':
    # logging.getLogger().setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser(description='Extract math from arxiv')
    parser.add_argument('-m', '--mode', default='top100', choices=['top', 'ids', 'dirs'],
                        help='Where to extract code from. top: current 100 arxiv papers (-m top int for any other number of papers), id: specific arxiv ids. \
                              Usage: `python arxiv.py -m ids id001 [id002 ...]`, dirs: a folder full of .tar.gz files. Usage: `python arxiv.py -m dirs directory [dir2 ...]`')
    parser.add_argument(nargs='*', dest='args', default=[])
    parser.add_argument('-o', '--out', default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data'), help='output directory')
    parser.add_argument('-d', '--demacro', dest='demacro', action='store_true',
                        help='Deprecated - Use de-macro (Slows down extraction, may but improves quality). Install https://www.ctan.org/pkg/de-macro')
    parser.add_argument('-s', '--save', default=None, type=str, help='When downloading files from arxiv. Where to save the .tar.gz files. Default: Only temporary')
    args = parser.parse_args()
    if '.' in args.out:
        args.out = os.path.dirname(args.out)
    skips = os.path.join(args.out, 'visited_arxiv.txt')
    if os.path.exists(skips):
        skip = open(skips, 'r', encoding='utf-8').read().split('\n')
    else:
        skip = []
    if args.save is not None:
        os.makedirs(args.save, exist_ok=True)
    try:
        if args.mode == 'ids':
            visited, math = recursive_search(parse_arxiv, args.args, skip=skip, unit='paper', save=args.save, demacro=args.demacro)
        elif args.mode == 'top':
            num = 100 if len(args.args) == 0 else int(args.args[0])
            url = 'https://arxiv.org/list/physics/pastweek?skip=0&show=%i' % num  # 'https://arxiv.org/list/hep-th/2203?skip=0&show=100'
            response = requests.get(url, timeout=NETWORK_TIMEOUT)
            response.raise_for_status()
            ids = get_all_arxiv_ids(response.text)
            math, visited = [], ids
            for id in tqdm(ids):
                try:
                    m, _ = parse_arxiv(id, save=args.save, demacro=args.demacro)
                    math.extend(m)
                except ValueError:
                    pass
        elif args.mode == 'dirs':
            files = []
            for folder in args.args:
                files.extend([os.path.join(folder, p) for p in os.listdir(folder)])
            math, visited = [], []
            for f in tqdm(files):
                try:
                    text = read_paper(f, delete=False, demacro=args.demacro)
                    math.extend(find_math(text, wiki=False))
                    visited.append(os.path.basename(f))
                except DemacroError as e:
                    logging.debug(f + str(e))
                    pass
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logging.debug(e)
                    raise e
        else:
            raise NotImplementedError
    except KeyboardInterrupt:
        pass
    print('Found %i instances of math latex code' % len(math))
    # print('\n'.join(math))
    # sys.exit(0)
    for entries, name in zip(
        [visited, math], ['visited_arxiv.txt', 'math_arxiv.txt'], strict=True
    ):
        f = os.path.join(args.out, name)
        if not os.path.exists(f):
            open(f, 'w').write('')
        f = open(f, 'a', encoding='utf-8')
        for element in entries:
            f.write(element)
            f.write('\n')
        f.close()
