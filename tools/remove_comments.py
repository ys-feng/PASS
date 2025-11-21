#!/usr/bin/env python3
import io
import os
import sys
import tokenize

ROOT = os.path.join(os.path.dirname(__file__), '..', 'code')
TARGET_EXT = '.py'

def remove_comments_from_source(src: str) -> str:
    out = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(src).readline)
    except Exception as e:
        return src
    prev_end = (1,0)
    for tok_type, tok_str, start, end, line in tokens:
        if tok_type == tokenize.COMMENT:
            continue
        out.append((tok_type, tok_str))
    try:
        new_src = tokenize.untokenize(out)
    except Exception:
        return src
    return new_src


def process_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    new_src = remove_comments_from_source(src)
    if new_src != src:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_src)
        print(f"Stripped comments: {path}")
    else:
        print(f"No change: {path}")


def main():
    if not os.path.isdir(ROOT):
        print(f"Code folder not found: {ROOT}")
        sys.exit(1)
    for fname in os.listdir(ROOT):
        if fname.endswith(TARGET_EXT):
            process_file(os.path.join(ROOT, fname))

if __name__ == '__main__':
    main()
