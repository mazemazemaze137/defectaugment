import argparse
import json

from src.augment.cgan_256 import export_generated_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Export class-folder GAN samples from a cGAN checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_latest.pth")
    parser.add_argument("--output-dir", required=True, help="Output folder for generated class subdirectories")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--class-names", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = export_generated_samples(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        class_names=args.class_names,
        samples_per_class=args.samples_per_class,
        nz=args.nz,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
