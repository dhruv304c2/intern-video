"""Top-level argument parser: main.py {recall} {yt,local} ..."""

import argparse

from cli.recall import add_recall_parser


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="main.py recall {yt,local} - build a content-similarity retriever and run RDRecallTest."
    )
    subcommands = parser.add_subparsers(
        dest="command", required=True
    )
    add_recall_parser(subcommands)
    return parser
