# mostly taken from http://code.google.com/p/latexmath2png/
# install preview.sty
import glob
import io
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from shutil import which

from PIL import Image


class Latex:
    BASE = r'''
\documentclass[varwidth]{standalone}
\usepackage{fontspec,unicode-math}
\usepackage[active,tightpage,displaymath,textmath]{preview}
\setmathfont{%s}
\begin{document}
\thispagestyle{empty}
%s
\end{document}
'''

    def __init__(self, math, dpi=250, font='Latin Modern Math'):
        '''takes list of math code. `returns each element as PNG with DPI=`dpi`'''
        if re.fullmatch(r"[A-Za-z0-9 ._+-]+", font) is None:
            raise ValueError(f"Unsafe font name: {font!r}")
        self.math = math
        self.dpi = dpi
        self.font = font
        self.prefix_line = self.BASE.split("\n").index(
            "%s")  # used for calculate error formula index

    def write(self, return_bytes=False):
        # inline = bool(re.match('^\$[^$]*\$$', self.math)) and False
        texfile = None
        try:
            workdir = tempfile.gettempdir()
            fd, texfile = tempfile.mkstemp('.tex', 'eq', workdir, True)
            # print(self.BASE % (self.font, self.math))
            with os.fdopen(fd, 'w+') as f:
                document = self.BASE % (self.font, '\n'.join(self.math))
                # print(document)
                f.write(document)

            png, error_index = self.convert_file(
                texfile, workdir, return_bytes=return_bytes)
            return png, error_index

        finally:
            if texfile is not None and os.path.exists(texfile):
                try:
                    os.remove(texfile)
                except PermissionError:
                    pass

    def convert_file(self, infile, workdir, return_bytes=False):
        infile_path = Path(infile).resolve()
        workdir_path = Path(workdir).resolve()
        try:
            # Generate the PDF file
            #  not stop on error line, but return error line index,index start from 1
            command = [
                'xelatex',
                '-no-shell-escape',
                '-interaction=nonstopmode',
                '-file-line-error',
                f'-output-directory={workdir_path}',
                infile_path.name,
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    'openin_any': 'p',
                    'openout_any': 'p',
                    'TEXMFOUTPUT': str(workdir_path),
                }
            )
            process = subprocess.run(
                command,
                cwd=workdir_path,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=300,
                env=environment,
            )
            sout, serr = process.stdout, process.stderr
            # extract error line from sout
            error_index, _ = extract(
                text=sout,
                expression=rf"{re.escape(infile_path.name)}:(\d+)",
            )
            # extract success rendered equation
            if error_index != []:
                # offset index start from 0, same as self.math
                error_index = [int(_)-self.prefix_line-1 for _ in error_index]
            # Convert the PDF file to PNG's
            pdffile = workdir_path / infile_path.with_suffix('.pdf').name
            result, _ = extract(
                text=sout,
                expression=r"Output written on .*? \((\d+) pages?",
            )
            if not result:
                raise RuntimeError(f"XeLaTeX did not produce a PDF:\n{sout}\n{serr}")
            if int(result[0]) != len(self.math):
                raise Exception('xelatex rendering error, generated %d formula\'s page, but the total number of formulas is %d.' % (
                    int(result[0]), len(self.math)))
            pngfile = workdir_path / infile_path.with_suffix('.png').name

            image_magick = which('magick') or which('convert')
            if image_magick is None:
                raise RuntimeError('ImageMagick is required to render a LaTeX preview')
            command = [
                image_magick,
                '-density',
                str(self.dpi),
                '-colorspace',
                'gray',
                str(pdffile),
                '-quality',
                '90',
                str(pngfile),
            ]
            process = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=300,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"ImageMagick failed: {process.stderr.decode(errors='replace')}"
                )
            if return_bytes:
                if len(self.math) > 1:
                    png = [
                        pngfile.with_name(f'{pngfile.stem}-{i}.png').read_bytes()
                        for i in range(len(self.math))
                    ]
                else:
                    png = [pngfile.read_bytes()]
            else:
                # return path
                if len(self.math) > 1:
                    png = [
                        str(pngfile.with_name(f'{pngfile.stem}-{i}.png'))
                        for i in range(len(self.math))
                    ]
                else:
                    png = [str(pngfile)]
            return png, error_index
        finally:
            # Cleanup temporaries
            basefile = workdir_path / infile_path.stem
            tempext = ['.aux', '.pdf', '.log', '.xdv']
            if return_bytes:
                ims = glob.glob(str(basefile)+'*.png')
                for im in ims:
                    os.remove(im)
            for te in tempext:
                temporary_file = Path(str(basefile) + te)
                if temporary_file.exists():
                    temporary_file.unlink()


__cache = {}


def tex2png(eq, **kwargs):
    if eq not in __cache:
        __cache[eq] = Latex(eq, **kwargs).write(return_bytes=True)
    return __cache[eq]


def tex2pil(tex, return_error_index=False, **kwargs):
    pngs, error_index = Latex(tex, **kwargs).write(return_bytes=True)
    images = [Image.open(io.BytesIO(d)) for d in pngs]
    return (images, error_index) if return_error_index else images


def extract(text, expression=None):
    """extract text from text by regular expression

    Args:
        text (str): input text
        expression (str, optional): regular expression. Defaults to None.

    Returns:
        str: extracted text
    """
    try:
        pattern = re.compile(expression)
        results = re.findall(pattern, text)
        return results, True if len(results) != 0 else False
    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = r'\begin{equation}\mathcal{ L}\nonumber\end{equation}'

    print('Equation is: %s' % src)
    print(Latex([src]).write())
