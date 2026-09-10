# taken and modified from https://github.com/harvardnlp/im2markup
# tokenize latex formulas
import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def process_args(args):
    parser = argparse.ArgumentParser(description='Preprocess (tokenize or normalize) latex formulas')

    parser.add_argument('--mode', '-m', dest='mode',
                        choices=['tokenize', 'normalize'], default='normalize',
                        help=('Tokenize (split to tokens seperated by space) or normalize (further translate to an equivalent standard form).'
                              ))
    parser.add_argument('--input-file', '-i', dest='input_file',
                        type=str, required=True,
                        help=('Input file containing latex formulas. One formula per line.'
                              ))
    parser.add_argument('--output-file', '-o', dest='output_file',
                        type=str, required=True,
                        help=('Output file.'
                              ))
    parser.add_argument('-n', '--num-threads', dest='num_threads',
                        type=int, default=4,
                        help=('Number of threads, default=4.'))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default=None,
                        help=('Log file path, default=log.txt'))
    parameters = parser.parse_args(args)
    return parameters


def main(args):
    parameters = process_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Script being executed: %s' % __file__)

    input_path = Path(parameters.input_file).resolve()
    output_path = Path(parameters.output_file).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    operators = r'\s?'.join('|'.join(['arccos', 'arcsin', 'arctan', 'arg', 'cos', 'cosh', 'cot', 'coth', 'csc', 'deg', 'det', 'dim', 'exp', 'gcd', 'hom', 'inf',
                                     'injlim', 'ker', 'lg', 'lim', 'liminf', 'limsup', 'ln', 'log', 'max', 'min', 'Pr', 'projlim', 'sec', 'sin', 'sinh', 'sup', 'tan', 'tanh']))
    ops = re.compile(r'\\operatorname {(%s)}' % operators)
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=output_path.parent,
            prefix=f'.{output_path.name}.',
            suffix='.input.tmp',
            delete=False,
        ) as normalized_output:
            normalized_input = input_path.read_text(encoding='utf-8').replace('\r', ' ')
            # replace split and align environments with aligned
            normalized_input = re.sub(r'\\begin{(split|align|alignedat|alignat|eqnarray)\*?}(.+?)\\end{\1\*?}', r'\\begin{aligned}\2\\end{aligned}', normalized_input, flags=re.S)
            normalized_input = re.sub(r'\\begin{(smallmatrix)\*?}(.+?)\\end{\1\*?}', r'\\begin{matrix}\2\\end{matrix}', normalized_input, flags=re.S)
            normalized_output.write(normalized_input)
            normalized_path = Path(normalized_output.name)
            temporary_paths.append(normalized_path)

        with tempfile.NamedTemporaryFile(
            mode='w+b',
            dir=output_path.parent,
            prefix=f'.{output_path.name}.',
            suffix='.node.tmp',
            delete=False,
        ) as node_output, normalized_path.open('rb') as node_input:
            node_path = Path(node_output.name)
            temporary_paths.append(node_path)
            result = subprocess.run(
                [
                    'node',
                    str(Path(__file__).with_name('preprocess_latex.js')),
                    parameters.mode,
                ],
                stdin=node_input,
                stdout=node_output,
                stderr=subprocess.PIPE,
                timeout=300,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"LaTeX preprocessing failed: {result.stderr.decode(errors='replace')}"
            )

        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=output_path.parent,
            prefix=f'.{output_path.name}.',
            suffix='.output.tmp',
            delete=False,
        ) as final_output, node_path.open('r', encoding='utf-8') as processed_input:
            final_path = Path(final_output.name)
            temporary_paths.append(final_path)
            for line in processed_input:
                tokens = line.strip().split()
                if len(tokens) > 5:
                    post = ' '.join(tokens)
                    # use \sin instead of \operatorname{sin}
                    names = ['\\' + value.replace(' ', '') for value in re.findall(ops, post)]
                    post = re.sub(
                        ops,
                        lambda _match, replacements=names: replacements.pop(0),
                        post,
                    ).replace(
                        r'\\ \end{array}', r'\end{array}'
                    )
                    final_output.write(post + '\n')
            final_output.flush()
            os.fsync(final_output.fileno())

        os.replace(final_path, output_path)
        temporary_paths.remove(final_path)
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
