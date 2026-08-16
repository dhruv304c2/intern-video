"""CLI entry point: main.py recall {yt,local} - build a content-similarity retriever and run RDRecallTest."""

from cli import build_arg_parser


def main() -> None:
    args = build_arg_parser().parse_args()
    score = args.run(args)
    print(f"RDRecallTest mean score: {score:.4f}")


if __name__ == "__main__":
    main()
