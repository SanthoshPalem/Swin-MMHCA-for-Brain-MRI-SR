"""
Extract paper-matching brain slices from preprocessed .pt files in processed_data/.

Each .pt file is a [3, 256, 256] tensor with channels [T1, T2, PD].
Filenames follow: IXIxxx-Site-xxxx_slice_NNN.pt

Based on anatomical calibration from preview_images:
  - img1 (below/at ventricle level):      Guys ~55-66, HH/IOP ~68-82
  - img2 (lateral ventricle bodies):      Guys ~65-76, HH/IOP ~80-94
  - img3 (above ventricles, semiovale):   Guys ~78-88, HH/IOP ~95-108
"""

import os
import glob
import argparse
import numpy as np
import torch
from PIL import Image


# Modality channel mapping in the .pt tensor
MODALITY_INDEX = {'T1': 0, 'T2': 1, 'PD': 2}

# Slice target ranges per site per paper image
TARGETS = {
    'Guys': {
        'img1_lower_mid': (55, 66),
        'img2_ventricle': (65, 76),
        'img3_upper':     (78, 88),
    },
    'HH': {
        'img1_lower_mid': (68, 82),
        'img2_ventricle': (80, 94),
        'img3_upper':     (95, 108),
    },
    'IOP': {
        'img1_lower_mid': (68, 82),
        'img2_ventricle': (80, 94),
        'img3_upper':     (95, 108),
    },
}


def get_site(subject_name):
    """Extract site from subject name, e.g. IXI076-Guys-0753 → Guys."""
    parts = subject_name.split('-')
    return parts[1] if len(parts) >= 2 else 'Guys'


def get_slice_number(filename):
    """Extract slice number from filename like IXI076-Guys-0753_slice_061.pt → 61."""
    base = os.path.splitext(os.path.basename(filename))[0]
    try:
        return int(base.split('_slice_')[-1])
    except ValueError:
        return -1


def get_subject_id(filename):
    """Extract subject ID from filename like IXI076-Guys-0753_slice_061.pt → IXI076-Guys-0753."""
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.split('_slice_')[0]


def tensor_to_image(channel_tensor):
    """Convert a [H, W] float tensor (0-1) to a uint8 PIL Image."""
    arr = channel_tensor.numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode='L')


def scan_processed_data(processed_dir):
    """
    Scan processed_data for all .pt files.
    Returns a dict: {subject_id: [(slice_num, filepath), ...]}
    """
    all_files = sorted(glob.glob(os.path.join(processed_dir, '*.pt')))
    if not all_files:
        print(f"WARNING: No .pt files found in {processed_dir}")
        return {}

    subject_dict = {}
    for f in all_files:
        sid = get_subject_id(f)
        snum = get_slice_number(f)
        if sid not in subject_dict:
            subject_dict[sid] = []
        subject_dict[sid].append((snum, f))

    # Sort slices per subject
    for sid in subject_dict:
        subject_dict[sid].sort(key=lambda x: x[0])

    print(f"Found {len(all_files)} .pt files across {len(subject_dict)} subjects.")
    return subject_dict


def extract_target_ranges(subject_dict, output_dir, modality='T2', top_n=3):
    """
    For each top subject, extract slices in the 3 target ranges.
    """
    mod_idx = MODALITY_INDEX[modality]

    # Score subjects by how many slices they have in the mid-range (quality proxy)
    scored = []
    for sid, slices in subject_dict.items():
        site = get_site(sid)
        targets = TARGETS.get(site, TARGETS['Guys'])
        all_range_start = min(r[0] for r in targets.values())
        all_range_end = max(r[1] for r in targets.values())
        count_in_range = sum(1 for snum, _ in slices if all_range_start <= snum <= all_range_end)
        scored.append((sid, slices, count_in_range))

    scored.sort(key=lambda x: x[2], reverse=True)
    print(f"\nTop {top_n} subjects by coverage of target slice ranges:")
    for sid, _, cnt in scored[:top_n]:
        print(f"  {sid} ({get_site(sid)}): {cnt} slices in target ranges")

    for sid, slices, _ in scored[:top_n]:
        site = get_site(sid)
        targets = TARGETS.get(site, TARGETS['Guys'])
        slice_map = {snum: fpath for snum, fpath in slices}

        print(f"\n{'='*60}")
        print(f"Extracting from {sid} (site: {site})")

        subject_out = os.path.join(output_dir, sid)
        os.makedirs(subject_out, exist_ok=True)

        for target_name, (start, end) in targets.items():
            target_out = os.path.join(subject_out, target_name)
            os.makedirs(target_out, exist_ok=True)
            saved = 0

            print(f"\n  [{target_name}] slices {start}-{end}:")
            for snum in range(start, end + 1):
                if snum not in slice_map:
                    continue
                fpath = slice_map[snum]
                tensor = torch.load(fpath, weights_only=True)  # [3, 256, 256]
                channel = tensor[mod_idx]                       # [256, 256]
                img = tensor_to_image(channel)
                fname = f"{sid}_slice_{snum:03d}_{modality}.png"
                img.save(os.path.join(target_out, fname))
                print(f"    Saved: {fname}")
                saved += 1

            if saved == 0:
                print(f"    (no slices found in this range for subject)")


def extract_specific_subject(subject_dict, subject_id, output_dir, modality='T2'):
    """Extract all target range slices for a specific subject."""
    if subject_id not in subject_dict:
        print(f"Subject '{subject_id}' not found in processed_data.")
        print("Available subjects (first 20):")
        for sid in list(subject_dict.keys())[:20]:
            print(f"  {sid}")
        return

    slices = subject_dict[subject_id]
    extract_target_ranges({subject_id: slices}, output_dir, modality, top_n=1)


def extract_exact_slices(subject_dict, subject_id, slice_indices, output_dir, modality='T2'):
    """Extract exact slice numbers from a specific subject."""
    mod_idx = MODALITY_INDEX[modality]

    if subject_id not in subject_dict:
        print(f"Subject '{subject_id}' not found.")
        return

    slice_map = {snum: fpath for snum, fpath in subject_dict[subject_id]}
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nExtracting exact slices {slice_indices} from {subject_id}:")
    for snum in slice_indices:
        if snum not in slice_map:
            print(f"  Slice {snum} not found (not in processed_data).")
            continue
        fpath = slice_map[snum]
        tensor = torch.load(fpath, weights_only=True)
        channel = tensor[mod_idx]
        img = tensor_to_image(channel)
        fname = f"{subject_id}_slice_{snum:03d}_{modality}.png"
        img.save(os.path.join(output_dir, fname))
        print(f"  Saved: {fname}")


def main():
    parser = argparse.ArgumentParser(description="Extract paper-matching slices from processed_data .pt files")
    parser.add_argument('--dataset_root', type=str, default='processed_data',
                        help='Path to processed_data folder containing .pt files')
    parser.add_argument('--output_dir', type=str, default='paper_slices',
                        help='Directory to save output PNG images')
    parser.add_argument('--modality', type=str, default='T2', choices=['T1', 'T2', 'PD'],
                        help='Which modality channel to extract')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'specific', 'exact'],
                        help='auto: find best subjects & extract target ranges. '
                             'specific: extract from --subject. '
                             'exact: extract --slices from --subject.')
    parser.add_argument('--subject', type=str, default=None,
                        help='Subject ID, e.g. IXI076-Guys-0753')
    parser.add_argument('--slices', type=str, default=None,
                        help='Comma-separated exact slice numbers, e.g. 60,70,82')
    parser.add_argument('--top_n', type=int, default=3,
                        help='Number of best subjects to process in auto mode')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    subject_dict = scan_processed_data(args.dataset_root)
    if not subject_dict:
        return

    if args.mode == 'exact' and args.subject and args.slices:
        slice_indices = [int(s.strip()) for s in args.slices.split(',')]
        extract_exact_slices(subject_dict, args.subject, slice_indices, args.output_dir, args.modality)

    elif args.mode == 'specific' and args.subject:
        extract_specific_subject(subject_dict, args.subject, args.output_dir, args.modality)

    else:  # auto
        extract_target_ranges(subject_dict, args.output_dir, args.modality, top_n=args.top_n)


if __name__ == '__main__':
    main()
