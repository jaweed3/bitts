"""
Monotonic Alignment Search (MAS) — Viterbi on a similarity grid.

From Glow-TTS / JETS: finds the most probable monotonic path between
text encoder outputs and target mel frames, then derives per-character durations.

This replaces the "dummy duration" hack (mel_len // text_len) with real
acoustic-linguistic alignment, solving the #1 cause of high loss in BitJETS.
"""

import torch
import torch.nn.functional as F


def viterbi_monotonic(log_prob: torch.Tensor) -> torch.Tensor:
    """
    Viterbi algorithm for monotonic alignment grid (Optimized).
    One loop over T_mel, vectorized over T_text.
    """
    T_text, T_mel = log_prob.shape
    device = log_prob.device
    dtype = log_prob.dtype

    # Q[i, j] = max log-prob of path ending at (i, j)
    Q = torch.full((T_text, T_mel), -1e9, device=device, dtype=dtype)
    
    # Initialize first column
    Q[0, 0] = log_prob[0, 0]
    
    # Fill DP table column by column (time dimension)
    # This is the only way to vectorize in pure PyTorch without Cython
    for j in range(1, T_mel):
        # Path can come from (i, j-1) [stay] or (i-1, j-1) [move]
        # We can only reach text index 'i' if j >= i
        max_i = min(j + 1, T_text)
        
        # stay: Q[i, j-1]
        # move: Q[i-1, j-1]
        prev_stay = Q[:max_i, j-1]
        prev_move = torch.cat([torch.tensor([-1e9], device=device, dtype=dtype), Q[:max_i-1, j-1]])
        
        Q[:max_i, j] = torch.max(prev_stay, prev_move) + log_prob[:max_i, j]

    # Backtrack (this remains a loop but it's only T_mel steps)
    path = torch.zeros(T_mel, dtype=torch.long, device=device)
    i = T_text - 1
    for j in range(T_mel - 1, -1, -1):
        path[j] = i
        if i > 0 and j > 0:
            if Q[i - 1, j - 1] >= Q[i, j - 1]:
                i -= 1

    return path


def path_to_durations(path: torch.Tensor, T_text: int) -> torch.Tensor:
    """
    Convert alignment path to per-text-position durations.

    Args:
        path: [T_mel] — text index for each mel frame.
        T_text: number of text positions.

    Returns:
        durations: [T_text] — integer duration (≥0) per text position.
    """
    durations = torch.zeros(T_text, dtype=torch.long, device=path.device)
    unique, counts = path.unique(return_counts=True)
    for pos, count in zip(unique.tolist(), counts.tolist()):
        durations[pos] = count
    return durations


def compute_similarity(encoder_out: torch.Tensor, mel_target: torch.Tensor) -> torch.Tensor:
    """
    Compute Negative L2 distance similarity (Optimized).
    Higher = more similar (closer in Euclidean space).
    """
    # encoder_out: [T_text, H]
    # mel_target:  [T_mel, 80]
    
    # We assume H == 80 here because it's called with projected encoder outputs.
    # Formula: -(a^2 - 2ab + b^2)
    a2 = torch.sum(encoder_out**2, dim=1, keepdim=True)    # [T_text, 1]
    b2 = torch.sum(mel_target**2, dim=1, keepdim=True).T   # [1, T_mel]
    ab = encoder_out @ mel_target.T                        # [T_text, T_mel]
    
    # Negative squared L2 distance
    dist = a2 - 2 * ab + b2
    return -0.5 * dist  # Higher is better


def extract_durations(encoder_out: torch.Tensor, mel_target: torch.Tensor) -> torch.Tensor:
    """
    Full pipeline: similarity → MAS → durations.

    Args:
        encoder_out: [T_text, H] or [1, T_text, H] — encoder output.
        mel_target:  [T_mel, 80] or [1, T_mel, 80] — target mel.

    Returns:
        durations: [T_text] — integer duration per text position.
    """
    if encoder_out.dim() == 3:
        encoder_out = encoder_out.squeeze(0)
    if mel_target.dim() == 3:
        mel_target = mel_target.squeeze(0)

    T_text = encoder_out.shape[0]
    T_mel = mel_target.shape[0]

    log_prob = compute_similarity(encoder_out, mel_target)  # [T_text, T_mel]
    path = viterbi_monotonic(log_prob)                        # [T_mel]
    durations = path_to_durations(path, T_text)               # [T_text]

    # Safety: ensure every text position gets at least 1 frame
    durations = torch.clamp(durations, min=1)

    # Fix total length mismatch
    total = durations.sum()
    if total < T_mel:
        # Distribute remaining frames proportionally
        deficit = T_mel - total
        durations[-1] += deficit
    elif total > T_mel:
        # Trim from the end (least important)
        excess = total - T_mel
        durations[-1] = max(1, durations[-1] - excess)

    return durations


def batch_extract_durations(
    encoder_out: torch.Tensor,  # [B, T_text, H]
    mel_target: torch.Tensor,   # [B, T_mel, 80]
    text_mask: torch.Tensor,    # [B, T_text] — True = real token
    mel_lens: torch.Tensor,     # [B] — Actual mel lengths
) -> torch.Tensor:
    """
    Extract durations for a batch, masking padding positions.

    Returns:
        durations: [B, T_text] — integer durations, 0 for padding.
    """
    B = encoder_out.shape[0]
    durations_list = []

    for b in range(B):
        t_mask = text_mask[b]  # [T_text]
        n_text = t_mask.sum().item()
        n_mel = mel_lens[b].item()
        
        if n_text == 0 or n_mel == 0:
            durations_list.append(torch.zeros_like(t_mask, dtype=torch.long))
            continue

        # Slice to real lengths
        enc_real = encoder_out[b, t_mask]          # [n_text, H]
        mel_real = mel_target[b, :n_mel]           # [n_mel, 80]
        
        dur = extract_durations(enc_real, mel_real)  # [n_text]

        # Pad back to full text length
        full_dur = torch.zeros(t_mask.shape[0], dtype=torch.long, device=dur.device)
        full_dur[t_mask] = dur
        durations_list.append(full_dur)

    return torch.stack(durations_list)
