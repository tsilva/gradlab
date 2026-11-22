"""Prepare a verified checkpoint directory for Hugging Face publication."""

from pathlib import Path
from shutil import copyfile

from gradlab.cli_parser import ExactArgumentParser
from gradlab.policy_bundle import load_policy_bundle, write_huggingface_recipe


def main() -> None:
    parser = ExactArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Verified checkpoint bundle directory")
    parser.add_argument("output", type=Path, help="New directory for the public bundle")
    args = parser.parse_args()

    bundle = load_policy_bundle(args.source)
    args.output.mkdir(mode=0o700)
    copyfile(bundle.checkpoint_path, args.output / "model.zip")
    write_huggingface_recipe(bundle.model, bundle.recipe, args.output)
    for name in ("README.md", "replay.mp4"):
        path = args.source / name
        if path.is_file():
            copyfile(path, args.output / name)
    load_policy_bundle(args.output)


if __name__ == "__main__":
    main()
