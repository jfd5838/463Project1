import h5py
import numpy as np
import random
from collections import defaultdict

file = 'VitalDB_CalBased_Test_Subset.mat'

path = f'{'VitalDB_CalBased_Test_Subset.mat'}' #You need to set path such that it can be compatible with your environment

all_abp_segments = []  # List of (10, samples) per selected subject
all_sbp_segments = []  # List of 10 scalars per selected subject
all_dbp_segments = []  # List of 10 scalars per selected subject
all_demographics = []  # List of (5,) per selected subject
subject_ids = []  # List of subject IDs or demog keys

with h5py.File(path, 'r') as f:
    subset = f['Subset']
    signals = subset['Signals']
    num_segments_raw = signals.shape[0]

    # ABP (channel 2: ABP_F)
    abp_all = signals[:, 2, :]

    # Per-segment labels
    sbp_raw = subset['SBP'][:]
    dbp_raw = subset['DBP'][:]
    sbp_per_seg = np.squeeze(sbp_raw)
    dbp_per_seg = np.squeeze(dbp_raw)

    # Limit to labelled segments
    num_labelled = len(sbp_per_seg)
    num_segments = min(num_segments_raw, num_labelled)
    abp_all = abp_all[:num_segments]  # Trim ABP if needed
    sbp_per_seg = sbp_per_seg[:num_segments]
    dbp_per_seg = dbp_per_seg[:num_segments]

    print(f"Num segments (labelled): {num_segments}")
    print(f"Signals shape: {signals.shape} (trimmed to {num_segments} for labels)")
    print(f"ABP shape: {abp_all.shape}")
    print(f"SBP/DBP per seg shapes: {sbp_per_seg.shape}, {dbp_per_seg.shape}")

    # Demographics per segment
    age_raw = subset['Age'][:]
    bmi_raw = subset['BMI'][:]
    gender_raw = subset['Gender'][:]
    height_raw = subset['Height'][:]
    weight_raw = subset['Weight'][:]

    print(f"Age raw shape: {age_raw.shape}")
    print(f"BMI raw shape: {bmi_raw.shape}")
    print(f"Gender raw shape: {gender_raw.shape}")
    print(f"Height raw shape: {height_raw.shape}")
    print(f"Weight raw shape: {weight_raw.shape}")

    # Extract per-segment for numerics
    if age_raw.shape[0] == 1:
        age_per_seg = np.full(num_segments, np.mean(age_raw))
    else:
        age_per_seg = np.squeeze(age_raw)[:num_segments]

    if bmi_raw.shape[0] == 1:
        bmi_per_seg = np.full(num_segments, np.mean(bmi_raw))
    else:
        bmi_per_seg = np.squeeze(bmi_raw)[:num_segments]

    if height_raw.shape[0] == 1:
        height_per_seg = np.full(num_segments, np.mean(height_raw))
    else:
        height_per_seg = np.squeeze(height_raw)[:num_segments]

    if weight_raw.shape[0] == 1:
        weight_per_seg = np.full(num_segments, np.mean(weight_raw))
    else:
        weight_per_seg = np.squeeze(weight_raw)[:num_segments]

    # Gender per-segment
    if gender_raw.shape[0] == 1:
        # Single: deref first
        first_ref = gender_raw[0, 0]
        if isinstance(first_ref, np.ndarray):
            first_ref = first_ref.item()
        gender_group = f[first_ref]
        gender_bytes = gender_group[()]
        gender_str = gender_bytes.tobytes().decode('utf-8').rstrip('\x00').strip()
        gender_numeric = 1 if gender_str == 'M' else 0
        gender_per_seg = np.full(num_segments, gender_numeric)
        print(f"Extracted Gender (single): '{gender_str}' ({gender_numeric})")
    else:
        # Per-segment: assume bytes or object
        gender_per_seg = np.zeros(num_segments)
        if gender_raw.dtype == object:
            for i in range(num_segments):
                ref = gender_raw[i]
                if isinstance(ref, np.ndarray):
                    ref = ref.item()
                gender_group = f[ref]
                gender_bytes = gender_group[()]
                gender_str = gender_bytes.tobytes().decode('utf-8').rstrip('\x00').strip()
                gender_per_seg[i] = 1 if gender_str == 'M' else 0
            print("Extracted Gender (per-segment refs)")
        else:
            gender_per_seg = np.squeeze(gender_raw == b'M').astype(float)[:num_segments]
            print("Extracted Gender (bytes array)")

    # Stack per-segment demographics: (num_segments, 5)
    demographics_per_seg = np.column_stack((age_per_seg, bmi_per_seg, gender_per_seg, height_per_seg, weight_per_seg))
    print(f"Demographics per seg shape: {demographics_per_seg.shape}")
    print(f"Sample row: Age={demographics_per_seg[0,0]:.0f}, BMI={demographics_per_seg[0,1]:.1f}, Gender={demographics_per_seg[0,2]}, Height={demographics_per_seg[0,3]:.0f}, Weight={demographics_per_seg[0,4]:.0f}")

    # Group segments by unique demographic profile (subjects)
    subject_groups = defaultdict(list)
    for i in range(num_segments):
        demog_key = tuple(demographics_per_seg[i])  # Hashable key
        subject_groups[demog_key].append(i)

    unique_subjects = list(subject_groups.keys())
    num_subjects = len(unique_subjects)
    print(f"Number of unique subjects: {num_subjects}")

    # Select up to 100 subjects
    if num_subjects > 100:
        selected_subjects = random.sample(unique_subjects, 100)
    else:
        selected_subjects = unique_subjects

    print(f"Selecting {len(selected_subjects)} subjects for extraction.")

    # Subject ID (sample first; extend if per-subject)
    sample_subject_id = 'unknown'
    if 'Subject' in subset:
        subject_ds = subset['Subject']
        subject_refs = np.squeeze(subject_ds[:])
        first_subject_ref_raw = subject_refs[0].item() if isinstance(subject_refs[0], np.ndarray) else subject_refs[0]
        subject_group = f[first_subject_ref_raw]
        subject_bytes = subject_group[()]
        sample_subject_id = subject_bytes.tobytes().decode('utf-8').rstrip('\x00').strip()
    print(f"Sample Subject ID: '{sample_subject_id}'")

    # Extract 10 segments per selected subject
    for subj_idx, subj_key in enumerate(selected_subjects, 1):
        seg_indices = subject_groups[subj_key]
        demog = np.array(subj_key)  # The demog tuple as array
        all_demographics.append(demog)
        subject_ids.append(f"{sample_subject_id}_{subj_idx}")  # Approximate ID

        print(f"\n--- Subject {subj_idx}/{len(selected_subjects)} (demog key: {demog}) ---")
        print(f"Available segments: {len(seg_indices)}")

        num_to_select = min(1000, len(seg_indices))
        selected_segs = random.sample(seg_indices, num_to_select)

        subject_abp = []
        subject_sbp = []
        subject_dbp = []
        for i in selected_segs:
            abp_seg = abp_all[i][:625]
            sbp_val = sbp_per_seg[i]
            dbp_val = dbp_per_seg[i]
            subject_abp.append(abp_seg)
            subject_sbp.append(sbp_val)
            subject_dbp.append(dbp_val)
            print(f"  Seg {i}: ABP shape {abp_seg.shape}, mean {np.mean(abp_seg):.0f} mmHg, SBP/DBP {sbp_val:.0f}/{dbp_val:.0f}")

        all_abp_segments.append(np.array(subject_abp))  # (num_selected, samples)
        all_sbp_segments.append(np.array(subject_sbp))  # (num_selected,)
        all_dbp_segments.append(np.array(subject_dbp))   # (num_selected,)

    print(f"\nExtraction complete: {len(selected_subjects)} subjects, ~{len(selected_subjects)*10} segments.")

# Final summary
total_subjects = len(all_abp_segments)
print(f"\n--- Summary ---")
print(f"Extracted data from {total_subjects} subjects.")
print(f"Total segments: {sum(len(abp) for abp in all_abp_segments)}")
print(f"ABP segments shape per subject: {all_abp_segments[0].shape if all_abp_segments else 'N/A'}")
print(f"Demographics shape: {np.array(all_demographics).shape} (subjects, 5)")

output_file = "VitalDB_Subset_Processed.npz"

np.savez_compressed(
    output_file,
    abp_segments=all_abp_segments,   # list of (num_selected, samples)
    sbp_segments=all_sbp_segments,   # list of (num_selected,)
    dbp_segments=all_dbp_segments,   # list of (num_selected,)
    demographics=np.array(all_demographics),  # (num_subjects, 5)
    subject_ids=np.array(subject_ids)  # list of subject IDs
)

print(f"\nSaved processed data to '{output_file}'")
print(f"  Contains: {len(all_abp_segments)} subjects")
print(f"  Example subject ABP shape: {all_abp_segments[0].shape if all_abp_segments else 'N/A'}")
print(f"  File size: ~{round(sum(abp.nbytes for abp in all_abp_segments)/1e6, 2)} MB (approx ABP only)")
