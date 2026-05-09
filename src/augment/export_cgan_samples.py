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
    parser.add_argument("--truncation", type=float, default=1.0, help="Noise scale. Lower values usually make samples smoother and more stable.")
    parser.add_argument("--oversample-factor", type=int, default=1, help="Generate N times more candidates per class before quality selection.")
    parser.add_argument("--quality-select", action="store_true", help="Select generated samples by sharpness, contrast and brightness statistics.")
    parser.add_argument("--min-mean", type=float, default=15.0)
    parser.add_argument("--max-mean", type=float, default=240.0)
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
        truncation=args.truncation,
        oversample_factor=args.oversample_factor,
        quality_select=args.quality_select,
        min_mean=args.min_mean,
        max_mean=args.max_mean,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
