"""
Dataset classes for Gen3R training.

Supports multiple multi-view 3D datasets as described in the paper:
- RealEstate10K
- DL3DV-10K
- Co3Dv2
- WildRGB-D
- TartanAir
"""

import os
import json
import random
from typing import Optional, Dict, List, Tuple, Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from einops import rearrange


class MultiViewDataset(Dataset):
    """
    Dataset for multi-view 3D data loading.
    
    Supports loading video frames with camera parameters (extrinsics, intrinsics),
    depth maps, and text prompts for training Gen3R.
    
    Args:
        data_root: Root directory containing the dataset
        split: Dataset split ('train', 'val', 'test')
        num_frames: Number of frames to sample per sequence
        resolution: Target resolution (height, width) for images
        dataset_type: Type of dataset ('re10k', 'dl3dv', 'co3d', 'wildrgbd', 'tartanair')
        transform: Optional transform function for data augmentation
        load_depth: Whether to load depth maps
        load_text: Whether to load text prompts
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        num_frames: int = 25,
        resolution: Tuple[int, int] = (560, 560),
        dataset_type: str = "re10k",
        transform: Optional[Callable] = None,
        load_depth: bool = True,
        load_text: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.resolution = resolution
        self.dataset_type = dataset_type
        self.transform = transform
        self.load_depth = load_depth
        self.load_text = load_text
        
        # Load dataset index
        self.samples = self._load_dataset_index()
        
    def _load_dataset_index(self) -> List[Dict[str, Any]]:
        """Load and parse dataset index file."""
        index_path = os.path.join(self.data_root, f"{self.split}.json")
        
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                samples = json.load(f)
            return samples
        else:
            # Scan directory structure for samples
            return self._scan_directory()
    
    def _scan_directory(self) -> List[Dict[str, Any]]:
        """Scan directory to build dataset index."""
        samples = []
        split_dir = os.path.join(self.data_root, self.split)
        
        if not os.path.exists(split_dir):
            return samples
        
        try:
            scene_names = sorted(os.listdir(split_dir))
        except (PermissionError, OSError) as e:
            import logging
            logging.warning(f"Could not list directory {split_dir}: {e}")
            return samples
            
        for scene_name in scene_names:
            scene_path = os.path.join(split_dir, scene_name)
            try:
                if not os.path.isdir(scene_path):
                    continue
                    
                # Check for required files
                frames_dir = os.path.join(scene_path, "frames")
                cameras_file = os.path.join(scene_path, "cameras.json")
                
                if os.path.exists(frames_dir) and os.path.exists(cameras_file):
                    samples.append({
                        "scene_name": scene_name,
                        "scene_path": scene_path,
                        "frames_dir": frames_dir,
                        "cameras_file": cameras_file,
                    })
            except (PermissionError, OSError) as e:
                import logging
                logging.warning(f"Could not access scene {scene_name}: {e}")
                continue
                
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load scene data
        result = self._load_scene(sample)
        
        # Apply transforms if specified
        if self.transform is not None:
            result = self.transform(result)
            
        return result
    
    def _load_scene(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Load a single scene's data."""
        scene_path = sample.get("scene_path", "")
        frames_dir = sample.get("frames_dir", os.path.join(scene_path, "frames"))
        cameras_file = sample.get("cameras_file", os.path.join(scene_path, "cameras.json"))
        
        # Load camera parameters
        if os.path.exists(cameras_file):
            try:
                with open(cameras_file, 'r') as f:
                    cameras = json.load(f)
                
                # Validate camera data structure
                extrinsics_data = cameras.get("extrinsics", [])
                intrinsics_data = cameras.get("intrinsics", [])
                
                if isinstance(extrinsics_data, list) and len(extrinsics_data) > 0:
                    extrinsics = torch.tensor(extrinsics_data, dtype=torch.float32)
                else:
                    extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_frames, 1, 1)
                    
                if isinstance(intrinsics_data, list) and len(intrinsics_data) > 0:
                    intrinsics = torch.tensor(intrinsics_data, dtype=torch.float32)
                else:
                    intrinsics = self._get_default_intrinsics()
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                import logging
                logging.warning(f"Could not parse cameras.json for {scene_path}: {e}")
                extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_frames, 1, 1)
                intrinsics = self._get_default_intrinsics()
        else:
            # Create default camera parameters
            extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_frames, 1, 1)
            intrinsics = self._get_default_intrinsics()
        
        # Load frames
        frames = self._load_frames(frames_dir, len(extrinsics))
        
        # Sample frames if needed
        if len(frames) > self.num_frames:
            indices = self._sample_frame_indices(len(frames), self.num_frames)
            frames = frames[indices]
            extrinsics = extrinsics[indices]
            intrinsics = intrinsics[indices] if len(intrinsics) > 1 else intrinsics.repeat(self.num_frames, 1, 1)
        
        # Resize frames to target resolution
        if frames.shape[-2:] != self.resolution:
            frames = F.interpolate(
                frames, 
                size=self.resolution, 
                mode='bilinear', 
                align_corners=False
            )
        
        result = {
            "frames": frames,  # [F, 3, H, W], in [0, 1]
            "extrinsics": extrinsics,  # [F, 4, 4]
            "intrinsics": intrinsics,  # [F, 3, 3]
        }
        
        # Load depth maps if available and requested
        if self.load_depth:
            depth_dir = os.path.join(scene_path, "depth")
            if os.path.exists(depth_dir):
                depth_maps = self._load_depth_maps(depth_dir, len(frames))
                result["depth_maps"] = depth_maps  # [F, H, W, 1]
        
        # Load text prompt if available and requested
        if self.load_text:
            prompt_file = os.path.join(scene_path, "prompt.txt")
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    prompt = f.read().strip()
            else:
                prompt = ""
            result["prompt"] = prompt
            
        return result
    
    def _load_frames(self, frames_dir: str, num_available: int) -> torch.Tensor:
        """Load video frames from directory."""
        import imageio
        import logging
        
        frames = []
        if os.path.exists(frames_dir):
            frame_files = sorted([
                f for f in os.listdir(frames_dir) 
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ])
            
            for frame_file in frame_files[:num_available]:
                frame_path = os.path.join(frames_dir, frame_file)
                try:
                    frame = imageio.v2.imread(frame_path)
                    frame = torch.from_numpy(frame).float() / 255.0  # [H, W, C]
                    if frame.dim() == 2:
                        frame = frame.unsqueeze(-1).repeat(1, 1, 3)
                    if frame.shape[-1] > 3:
                        frame = frame[..., :3]  # Take only RGB channels
                    frame = frame.permute(2, 0, 1)  # [C, H, W]
                    frames.append(frame)
                except Exception as e:
                    logging.warning(f"Could not load frame {frame_path}: {e}")
                    continue
        
        if len(frames) == 0:
            # Return dummy frames if no frames found
            frames = [torch.zeros(3, *self.resolution) for _ in range(self.num_frames)]
            
        return torch.stack(frames)  # [F, 3, H, W]
    
    def _load_depth_maps(self, depth_dir: str, num_frames: int) -> torch.Tensor:
        """Load depth maps from directory."""
        import imageio
        
        depth_maps = []
        depth_files = sorted([
            f for f in os.listdir(depth_dir)
            if f.endswith(('.png', '.exr', '.npy'))
        ])
        
        for depth_file in depth_files[:num_frames]:
            depth_path = os.path.join(depth_dir, depth_file)
            if depth_file.endswith('.npy'):
                depth = np.load(depth_path)
            else:
                depth = imageio.v2.imread(depth_path)
            depth = torch.from_numpy(depth).float()
            if depth.dim() == 2:
                depth = depth.unsqueeze(-1)
            depth_maps.append(depth)
            
        if len(depth_maps) == 0:
            depth_maps = [torch.zeros(*self.resolution, 1) for _ in range(num_frames)]
            
        return torch.stack(depth_maps)  # [F, H, W, 1]
    
    def _sample_frame_indices(self, total_frames: int, num_to_sample: int) -> List[int]:
        """Sample frame indices uniformly."""
        if total_frames <= num_to_sample:
            return list(range(total_frames))
        
        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, num_to_sample, dtype=int)
        return list(indices)
    
    def _get_default_intrinsics(self) -> torch.Tensor:
        """Get default camera intrinsics for the target resolution."""
        h, w = self.resolution
        fx = fy = max(h, w)
        cx, cy = w / 2, h / 2
        K = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32)
        return K.unsqueeze(0).repeat(self.num_frames, 1, 1)


def get_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 4,
    num_frames: int = 25,
    resolution: Tuple[int, int] = (560, 560),
    dataset_type: str = "re10k",
    shuffle: bool = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for multi-view dataset.
    
    Args:
        data_root: Root directory containing the dataset
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size for the dataloader
        num_workers: Number of worker processes for data loading
        num_frames: Number of frames to sample per sequence
        resolution: Target resolution (height, width)
        dataset_type: Type of dataset
        shuffle: Whether to shuffle the dataset (defaults to True for train)
        **kwargs: Additional arguments passed to the Dataset
        
    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = (split == "train")
    
    dataset = MultiViewDataset(
        data_root=data_root,
        split=split,
        num_frames=num_frames,
        resolution=resolution,
        dataset_type=dataset_type,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        collate_fn=collate_fn,
    )
    
    return dataloader


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for multi-view data."""
    result = {}
    
    # Stack tensor fields
    tensor_keys = ["frames", "extrinsics", "intrinsics", "depth_maps"]
    for key in tensor_keys:
        if key in batch[0] and batch[0][key] is not None:
            result[key] = torch.stack([item[key] for item in batch])
    
    # Collect string fields
    if "prompt" in batch[0]:
        result["prompt"] = [item["prompt"] for item in batch]
    
    return result
