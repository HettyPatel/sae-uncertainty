"""Pre-download and cache all SAEs for a model."""

import argparse
from sae_lens import SAE


def main():
    parser = argparse.ArgumentParser(description="Download and cache SAEs")
    parser.add_argument('--release', type=str, default='llama_scope_lxr_8x')
    parser.add_argument('--id-template', type=str, default='l{layer}r_8x')
    parser.add_argument('--layers', type=str, default='0-31')
    args = parser.parse_args()

    layers = []
    for part in args.layers.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))

    print(f"Downloading {len(layers)} SAEs from {args.release}")
    for layer in layers:
        sae_id = args.id_template.format(layer=layer)
        print(f"  Layer {layer}: {sae_id}...", end=" ", flush=True)
        try:
            SAE.from_pretrained(release=args.release, sae_id=sae_id, device="cpu")
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    print("All done.")


if __name__ == '__main__':
    main()
