import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from fastdtw import fastdtw
from itertools import combinations
from collections import Counter

# load data (we got this file from segmentScraper)
data = np.load("VitalDB_Subset_Processed.npz", allow_pickle=True)
all_abp_segments = data["abp_segments"]
subject_ids = data["subject_ids"]

# Flatten all segments into a single list
all_segments = np.concatenate(all_abp_segments, axis=0)  # (total_segments, samples)
print(f"Total segments: {len(all_segments)}") # proof of 1000 segments in program


# Functions to identify relevant comparisons
def correlation_distance_matrix(X):
    corr = np.corrcoef(X)
    corr = np.nan_to_num(corr, nan=1.0)  # Handle any null values
    return 1 - corr


def dtw_distance_matrix(X, sample_size=100):
    n = len(X)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist, _ = fastdtw(X[i], X[j])
            dist_mat[i, j] = dist_mat[j, i] = dist
    return dist_mat


# recursive partition function to find closest pairs
def recursive_partition(indices, X, depth=0, method="correlation", threshold=0.4, max_depth=4):
    if len(indices) <= 5 or depth >= max_depth:
        return [indices]

    # Select subset of data
    subset = X[indices]

    # Compute distance matrix
    if method == "correlation":
        D = correlation_distance_matrix(subset)
    elif method == "dtw":
        D = dtw_distance_matrix(subset)
    else:
        raise ValueError("Unknown method")

    # Compute avg distance for pairs
    avg_dist = np.mean(D[np.triu_indices_from(D, 1)])
    print("  " * depth + f"Level {depth}: {len(indices)} segs, avg dist = {avg_dist:.3f}")

    # Stop splitting if already similar enough
    if avg_dist < threshold:
        return [indices]

    # Prepare matrix for clustering
    D = np.nan_to_num(D, nan=0.0, posinf=1.0, neginf=1.0)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)

    # Condense and then cluster
    try:
        linkage_mat = linkage(squareform(D), method='average')
        cluster_labels = fcluster(linkage_mat, 2, criterion='maxclust')

        # Split into two recursive branches
        cluster_indices = [np.where(cluster_labels == k)[0] for k in [1, 2]]
        result_clusters = []
        for c_idx in cluster_indices:
            sub_indices = [indices[i] for i in c_idx]
            if len(sub_indices) == 0:
                continue
            result_clusters.extend(recursive_partition(sub_indices, X, depth + 1, method, threshold, max_depth))
        return result_clusters
    except Exception as e:
        print("  " * depth + f"Warning: clustering failed, returning as single cluster")
        return [indices]


# Recursive clustering
print("\nStarting recursive clustering...")
clusters = recursive_partition(list(range(len(all_segments))), all_segments, method="correlation", threshold=0.4)
print(f"\nRecursive clustering complete: {len(clusters)} final clusters")

# Cluster summary
cluster_sizes = [len(c) for c in clusters]
print(f"Cluster sizes: {cluster_sizes}")
print(f"Mean cluster size: {np.mean(cluster_sizes):.2f}")

# Save cluster assignments
cluster_labels = np.zeros(len(all_segments), dtype=int)
for i, cluster in enumerate(clusters):
    for idx in cluster:
        cluster_labels[idx] = i


# closest pair function
def closest_pair_in_cluster(X, indices, method="correlation"):
    if len(indices) < 2:
        return (None, None, np.inf)

    best_pair = (None, None)
    best_dist = np.inf

    # Compare all  pairs
    for i, j in combinations(indices, 2):
        if method == "correlation":
            corr = np.corrcoef(X[i], X[j])[0, 1]
            dist = 1 - corr  # correlation distance
        elif method == "dtw":
            dist, _ = fastdtw(X[i], X[j])
        else:
            raise ValueError("Unknown method")

        if dist < best_dist:
            best_dist = dist
            best_pair = (i, j)

    return (*best_pair, best_dist)


# evaluate the clusters
closest_pairs = []
cluster_cohesion = []

print("\nFinding closest pairs within each cluster...")
for cluster_id, cluster_indices in enumerate(clusters):
    if len(cluster_indices) < 2:
        continue

    i1, i2, dist = closest_pair_in_cluster(all_segments, cluster_indices, method="correlation")
    closest_pairs.append((cluster_id, i1, i2, dist))
    cluster_cohesion.append(dist)

    print(f"Cluster {cluster_id:03d}: size={len(cluster_indices):3d}, closest pair=({i1},{i2}), dist={dist:.4f}")

mean_cohesion = np.mean(cluster_cohesion) if cluster_cohesion else np.nan
print(f"\nClosest-pair analysis complete for {len(closest_pairs)} clusters")
print(f"Average within-cluster distance (lower = tighter): {mean_cohesion:.4f}")


# Kadane function
def kadane_max_subarray(arr):
    max_sum = -np.inf
    cur_sum = 0.0
    start = 0
    best_start = 0
    best_end = 0

    for i, val in enumerate(arr):
        if cur_sum <= 0:
            cur_sum = val
            start = i
        else:
            cur_sum += val

        if cur_sum > max_sum:
            max_sum = cur_sum
            best_start = start
            best_end = i

    return float(max_sum), int(best_start), int(best_end)


# run Kadane on all segments
N, S = all_segments.shape  # total segments, samples
kadane_results = np.zeros((N, 3), dtype=float)  # columns: max_sum, start, end

for idx in range(N):
    seg = all_segments[idx]
    max_sum, s_idx, e_idx = kadane_max_subarray(seg)
    kadane_results[idx, 0] = max_sum
    kadane_results[idx, 1] = s_idx
    kadane_results[idx, 2] = e_idx

print(f"\nKadane run complete for {N} segments (each length {S}).")

# small function to help with finding center and length of intervals
def interval_center_length(row):
    s = int(row[1])
    e = int(row[2])
    length = max(1, e - s + 1)
    center = (s + e) / 2.0
    return center, length


centers = np.array([interval_center_length(kadane_results[i])[0] for i in range(N)])
lengths = np.array([interval_center_length(kadane_results[i])[1] for i in range(N)])
sums = kadane_results[:, 0]

# aggregate the stats before clustering
cluster_kadane_summary = []  # list of dict per cluster
print("\nCluster Kadane summaries:")
for cid, cluster_indices in enumerate(clusters):
    if len(cluster_indices) == 0:
        continue

    idxs = np.array(cluster_indices, dtype=int)
    cluster_sums = sums[idxs]
    cluster_centers = centers[idxs]
    cluster_lengths = lengths[idxs]

    mean_sum = float(np.mean(cluster_sums))
    median_sum = float(np.median(cluster_sums))
    mean_len = float(np.mean(cluster_lengths))
    median_len = float(np.median(cluster_lengths))

    # find the center bins of the modal
    center_bins = np.round(cluster_centers).astype(int)
    most_common_center, freq = Counter(center_bins).most_common(1)[0]

    # define modal interval as a representative window
    # use median length to define window
    half = int(np.ceil(median_len / 2.0))
    modal_start = max(0, most_common_center - half)
    modal_end = min(S - 1, most_common_center + half)

    # compute overlap fraction
    overlap_count = 0
    for idx in idxs:
        s = int(kadane_results[idx, 1])
        e = int(kadane_results[idx, 2])
        # if there is an overlap on modal_start and modal_end
        if not (e < modal_start or s > modal_end):
            overlap_count += 1
    overlap_frac = overlap_count / len(idxs)

    summary = {
        "cluster_id": cid,
        "size": len(idxs),
        "mean_kadane_sum": mean_sum,
        "median_kadane_sum": median_sum,
        "mean_kadane_length": mean_len,
        "median_kadane_length": median_len,
        "modal_center_bin": int(most_common_center),
        "modal_window": (int(modal_start), int(modal_end)),
        "overlap_fraction": float(overlap_frac)
    }
    cluster_kadane_summary.append(summary)

    print(f"Cluster {cid:03d}: size={summary['size']:3d}, mean_sum={mean_sum:.3f}, "
          f"median_len={median_len:.1f}, modal_center={summary['modal_center_bin']}, "
          f"modal_window={summary['modal_window']}, overlap={overlap_frac:.3f}")


# create plot of top K clusters to show reasoning
def plot_cluster_kadane_examples(top_k=4, examples_per_cluster=5, save_fig=False):
    import random

    sorted_clusters = sorted(cluster_kadane_summary, key=lambda x: x["size"], reverse=True)[:top_k]

    for idx, csum in enumerate(sorted_clusters):
        cid = csum["cluster_id"]
        cluster_indices = clusters[cid]
        if len(cluster_indices) == 0:
            continue

        # pick examples: closest pair + some random values for better variety
        picks = []

        # include closest pair if available
        try:
            pick_pair = next((p for p in closest_pairs if p[0] == cid), None)
            if pick_pair is not None:
                _, p1, p2, _ = pick_pair
                picks.extend([p1, p2])
        except Exception:
            pass

        # add random picks
        remaining = [i for i in cluster_indices if i not in picks]
        random.shuffle(remaining)
        picks.extend(remaining[:max(0, examples_per_cluster - len(picks))])
        picks = [p for p in picks if p is not None][:examples_per_cluster]

        # Plot
        plt.figure(figsize=(10, 2.4 * len(picks)))
        for row, seg_idx in enumerate(picks, start=1):
            seg = all_segments[seg_idx]
            s = int(kadane_results[seg_idx, 1])
            e = int(kadane_results[seg_idx, 2])
            ax = plt.subplot(len(picks), 1, row)
            ax.plot(seg)
            ax.set_title(f"Cluster {cid} - Seg {seg_idx} - Kadane [{s}:{e}] sum={kadane_results[seg_idx, 0]:.2f}")
            # highlight Kadane interval
            ax.axvspan(s, e, alpha=0.2, color='orange')
            ax.set_xlim(0, S - 1)
        plt.tight_layout()
        if save_fig:
            plt.savefig(f"cluster_{cid:03d}_kadane_examples.png")
            print(f"Saved figure: cluster_{cid:03d}_kadane_examples.png")
        plt.show()

import numpy as np
import matplotlib.pyplot as plt

# cluster visualization
def plot_cluster_representatives(all_segments, clusters, method="mean", top_k=5):
    sorted_clusters = sorted(enumerate(clusters), key=lambda x: len(x[1]), reverse=True)[:top_k]
    plt.figure(figsize=(10, 2.5 * len(sorted_clusters)))

    for i, (cid, indices) in enumerate(sorted_clusters, start=1):
        segs = np.array([all_segments[idx] for idx in indices])
        mean_seg = np.mean(segs, axis=0)
        std_seg = np.std(segs, axis=0)

        plt.subplot(len(sorted_clusters), 1, i)
        plt.plot(mean_seg, color="blue", label="Mean waveform")
        plt.fill_between(range(len(mean_seg)),
                         mean_seg - std_seg,
                         mean_seg + std_seg,
                         color="lightblue", alpha=0.4, label="±1 std")

        plt.title(f"Cluster {cid} — size={len(indices)}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()

    plt.tight_layout()
    plt.show()


# closest pair visualization
def plot_closest_pairs(all_segments, closest_pairs, max_pairs=5):
    plt.figure(figsize=(10, 3 * min(len(closest_pairs), max_pairs)))

    for i, (cid, i1, i2, dist) in enumerate(closest_pairs[:max_pairs], start=1):
        seg1 = all_segments[int(i1)]
        seg2 = all_segments[int(i2)]

        plt.subplot(min(len(closest_pairs), max_pairs), 1, i)
        plt.plot(seg1, label=f"Seg {i1}")
        plt.plot(seg2, label=f"Seg {i2}")
        plt.title(f"Cluster {cid} | Closest pair ({i1},{i2}) | Dist={dist:.4f}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()

    plt.tight_layout()
    plt.show()


# Kadane Interval visualization
def plot_kadane_intervals(kadane_results, clusters, top_k=5):
    sorted_clusters = sorted(enumerate(clusters), key=lambda x: len(x[1]), reverse=True)[:top_k]
    plt.figure(figsize=(10, 2.5 * len(sorted_clusters)))

    for i, (cid, indices) in enumerate(sorted_clusters, start=1):
        plt.subplot(len(sorted_clusters), 1, i)
        for j, idx in enumerate(indices):
            s = int(kadane_results[idx, 1])
            e = int(kadane_results[idx, 2])
            plt.hlines(y=j, xmin=s, xmax=e, color="orange", linewidth=2)
        plt.title(f"Cluster {cid} — Max subarray intervals ({len(indices)} segs)")
        plt.xlabel("Time (samples)")
        plt.ylabel("Segment index")
        plt.ylim(-1, len(indices))
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_cluster_kadane_examples(top_k=4, examples_per_cluster=4, save_fig=False)

plot_cluster_representatives(all_segments, clusters, top_k=5)
plot_closest_pairs(all_segments, closest_pairs, max_pairs=5)
plot_kadane_intervals(kadane_results, clusters, top_k=5)
