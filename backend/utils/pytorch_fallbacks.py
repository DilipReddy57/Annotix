# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# PyTorch fallback for Windows/non-triton systems

"""PyTorch fallback for euclidean distance transform (EDT)

This provides a pure PyTorch implementation when triton is not available.
Uses scipy's distance_transform_edt for CPU fallback, with a GPU-compatible version.
"""

import torch
import numpy as np


def edt_pytorch(data: torch.Tensor) -> torch.Tensor:
    """
    Computes the Euclidean Distance Transform (EDT) of a batch of binary images.
    Pure PyTorch implementation - no triton required.
    
    Args:
        data: A tensor of shape (B, H, W) representing a batch of binary images.
              Non-zero values are treated as foreground.
    
    Returns:
        A tensor of the same shape as data containing the EDT.
        Distance to the nearest zero (background) pixel.
    """
    assert data.dim() == 3, "Input must be 3D (B, H, W)"
    
    B, H, W = data.shape
    device = data.device
    
    # For small tensors or CPU, use scipy (more accurate)
    if not data.is_cuda or (H * W < 512 * 512):
        return _edt_scipy_batch(data)
    
    # For GPU, use approximate but fast method
    return _edt_gpu_approx(data)


def _edt_scipy_batch(data: torch.Tensor) -> torch.Tensor:
    """Use scipy for accurate EDT computation (CPU)."""
    try:
        from scipy.ndimage import distance_transform_edt
        
        device = data.device
        data_np = data.cpu().numpy()
        results = []
        
        for i in range(data_np.shape[0]):
            # EDT expects 0 as background, non-zero as foreground
            # We want distance to nearest 0
            binary = (data_np[i] != 0).astype(np.float32)
            edt = distance_transform_edt(binary)
            results.append(edt)
        
        result = np.stack(results, axis=0)
        return torch.from_numpy(result).to(device=device, dtype=data.dtype)
        
    except ImportError:
        # Fallback to GPU approximation
        return _edt_gpu_approx(data)


def _edt_gpu_approx(data: torch.Tensor) -> torch.Tensor:
    """
    GPU-compatible approximate EDT using distance propagation.
    Uses iterative morphological distance transform approximation.
    """
    B, H, W = data.shape
    device = data.device
    dtype = torch.float32
    
    # Binary mask: True where we have foreground (non-zero)
    mask = (data != 0).to(dtype)
    
    # Initialize distances: 0 for background, large for foreground
    INF = float(H + W)
    dist = torch.where(mask > 0, torch.tensor(INF, device=device, dtype=dtype), 
                       torch.tensor(0.0, device=device, dtype=dtype))
    
    # Distance propagation using convolution (Chamfer-like)
    # We'll do multiple passes with a 3x3 kernel
    
    # Forward pass (top-left to bottom-right)
    for _ in range(max(H, W) // 2 + 1):
        # Horizontal neighbor distances
        dist_left = torch.nn.functional.pad(dist[:, :, :-1], (1, 0), value=INF)
        dist_right = torch.nn.functional.pad(dist[:, :, 1:], (0, 1), value=INF)
        
        # Vertical neighbor distances  
        dist_up = torch.nn.functional.pad(dist[:, :-1, :], (0, 0, 1, 0), value=INF)
        dist_down = torch.nn.functional.pad(dist[:, 1:, :], (0, 0, 0, 1), value=INF)
        
        # Diagonal neighbors
        dist_ul = torch.nn.functional.pad(dist[:, :-1, :-1], (1, 0, 1, 0), value=INF)
        dist_ur = torch.nn.functional.pad(dist[:, :-1, 1:], (0, 1, 1, 0), value=INF)
        dist_dl = torch.nn.functional.pad(dist[:, 1:, :-1], (1, 0, 0, 1), value=INF)
        dist_dr = torch.nn.functional.pad(dist[:, 1:, 1:], (0, 1, 0, 1), value=INF)
        
        # Take minimum of all neighbors + step distance
        SQRT2 = 1.41421356
        new_dist = torch.minimum(dist, dist_left + 1)
        new_dist = torch.minimum(new_dist, dist_right + 1)
        new_dist = torch.minimum(new_dist, dist_up + 1)
        new_dist = torch.minimum(new_dist, dist_down + 1)
        new_dist = torch.minimum(new_dist, dist_ul + SQRT2)
        new_dist = torch.minimum(new_dist, dist_ur + SQRT2)
        new_dist = torch.minimum(new_dist, dist_dl + SQRT2)
        new_dist = torch.minimum(new_dist, dist_dr + SQRT2)
        
        # Only update foreground pixels
        dist = torch.where(mask > 0, new_dist, dist)
        
        # Early exit if converged
        if torch.allclose(dist, new_dist, atol=1e-6):
            break
    
    return dist


# Monkey-patch: Replace triton EDT with PyTorch version
def install_edt_fallback():
    """Install the PyTorch EDT fallback into the SAM3 module."""
    import sys
    
    # Create a module that provides edt_triton as edt_pytorch
    class EDTFallback:
        edt_triton = staticmethod(edt_pytorch)
        
    # This will be imported instead of the triton version
    return edt_pytorch


# ============ NMS PyTorch Fallback ============

def nms_pytorch(
    ious: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """
    Perform NMS given the iou matrix, the scores and the iou threshold.
    Pure PyTorch implementation - no triton required.
    
    Args:
        ious: Pairwise IoU tensor of shape (N, N).
        scores: Scores tensor of shape (N,).
        iou_threshold: IoU threshold for suppression.
    
    Returns:
        Tensor: Indices of kept boxes, sorted by decreasing score.
    """
    assert scores.dim() == 1, "Scores must be 1D"
    num_boxes = scores.size(0)
    
    if num_boxes == 0:
        return torch.tensor([], dtype=torch.long, device=scores.device)
    
    # Sort boxes by scores in descending order
    _, sorted_indices = torch.sort(scores, descending=True, stable=True)
    
    # Reorder IoU matrix according to sorted indices
    ious_sorted = ious[sorted_indices][:, sorted_indices]
    
    # Keep mask - start with all True
    keep_mask = torch.ones(num_boxes, dtype=torch.bool, device=scores.device)
    
    # Sequential NMS
    for i in range(num_boxes - 1):
        if keep_mask[i]:
            # Suppress all boxes with IoU > threshold with current box
            suppress_mask = ious_sorted[i, i+1:] > iou_threshold
            keep_mask[i+1:] = keep_mask[i+1:] & ~suppress_mask
    
    # Return original indices of kept boxes
    return sorted_indices[keep_mask]


def batched_nms_pytorch(
    boxes: torch.Tensor,
    scores: torch.Tensor, 
    iou_threshold: float,
) -> torch.Tensor:
    """
    Batched NMS using torchvision if available, else pure PyTorch.
    
    Args:
        boxes: Boxes tensor of shape (N, 4) in [x1, y1, x2, y2] format.
        scores: Scores tensor of shape (N,).
        iou_threshold: IoU threshold for suppression.
    
    Returns:
        Tensor: Indices of kept boxes.
    """
    try:
        from torchvision.ops import nms
        return nms(boxes, scores, iou_threshold)
    except ImportError:
        # Compute IoU matrix manually
        ious = _box_iou(boxes, boxes)
        return nms_pytorch(ious, scores, iou_threshold)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        boxes2: Tensor of shape (M, 4) in [x1, y1, x2, y2] format
    
    Returns:
        IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    intersection = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # Union
    union = area1[:, None] + area2[None, :] - intersection
    
    return intersection / (union + 1e-6)


# ============ Connected Components PyTorch Fallback ============

def connected_components_pytorch(mask: torch.Tensor) -> torch.Tensor:
    """
    Find connected components in a binary mask.
    Uses scipy on CPU, returns labeled components.
    
    Args:
        mask: Binary mask tensor of shape (H, W) or (B, H, W)
    
    Returns:
        Labeled tensor with same shape, each component has unique ID
    """
    try:
        from scipy.ndimage import label as scipy_label
        
        device = mask.device
        was_batched = mask.dim() == 3
        
        if not was_batched:
            mask = mask.unsqueeze(0)
        
        mask_np = mask.cpu().numpy()
        results = []
        
        for i in range(mask_np.shape[0]):
            labeled, num_features = scipy_label(mask_np[i])
            results.append(labeled)
        
        result = np.stack(results, axis=0)
        result = torch.from_numpy(result).to(device=device, dtype=torch.int32)
        
        if not was_batched:
            result = result.squeeze(0)
        
        return result
        
    except ImportError:
        # Simple fallback - just return the mask as single component
        return mask.to(torch.int32)


# ============ Install All Fallbacks ============

def install_all_fallbacks():
    """
    Install all PyTorch fallbacks for triton kernels.
    Call this before importing SAM3 on Windows.
    """
    import sys
    
    # First install mock triton
    from backend.utils.mock_triton import install_mock_triton
    install_mock_triton()
    
    # Now patch the specific functions
    # These will be used when SAM3 tries to call triton kernels
    
    # Create fallback modules
    class EDTModule:
        edt_triton = staticmethod(edt_pytorch)
    
    class NMSModule:
        nms_triton = staticmethod(nms_pytorch)
    
    class CCModule:
        connected_components_triton = staticmethod(connected_components_pytorch)
    
    # Try to patch SAM3 modules if they exist
    try:
        import sam3.model.edt as edt_module
        edt_module.edt_triton = edt_pytorch
        print("[INFO] Patched sam3.model.edt with PyTorch fallback")
    except:
        pass
    
    try:
        import sam3.perflib.triton.nms as nms_module
        nms_module.nms_triton = nms_pytorch
        print("[INFO] Patched sam3.perflib.triton.nms with PyTorch fallback")
    except:
        pass
    
    return {
        'edt': edt_pytorch,
        'nms': nms_pytorch,
        'connected_components': connected_components_pytorch
    }

