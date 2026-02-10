from typing import Dict
import torch
from torch import Tensor

class EpisodeBatch:
    """Simple episode data storage."""
    def __init__(self, scheme, groups, batch_size, max_seq_length, device):
        self.scheme = scheme
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.device = device
        
        self.data = {}
        self._setup_data()

    def to(self, device) -> "EpisodeBatch":
        """
        Move all stored tensors to a target device (in-place) and update self.device.

        This is critical for Stage1/2 supervised replay:
        - store replay batches on CPU to avoid unbounded GPU memory growth
        - move only the sampled mini-batch to CUDA for training
        """
        try:
            dev = torch.device(device) if not isinstance(device, torch.device) else device
        except Exception:
            dev = device
        try:
            if str(self.device) == str(dev):
                return self
        except Exception:
            pass
        self.device = dev
        for k, v in list(self.data.items()):
            if isinstance(v, torch.Tensor):
                self.data[k] = v.detach().to(dev, non_blocking=True)
        return self

    def pin_memory(self) -> "EpisodeBatch":
        """
        Pin CPU tensors to speed up CPU->GPU transfers with non_blocking=True.
        Safe no-op if tensors are not on CPU.
        """
        for k, v in list(self.data.items()):
            if isinstance(v, torch.Tensor):
                try:
                    if v.device.type == "cpu":
                        self.data[k] = v.pin_memory()
                except Exception:
                    pass
        return self
        
    def _setup_data(self):
        """Initialize data storage structures."""
        for field_key, field_info in self.scheme.items():
            vshape = field_info["vshape"]
            dtype = field_info.get("dtype", torch.float32)
            
            if isinstance(vshape, int):
                vshape = (vshape,)
                
            if "group" in field_info:
                shape = (self.batch_size, self.max_seq_length, 
                        self.groups[field_info["group"]]) + vshape
            else:
                shape = (self.batch_size, self.max_seq_length) + vshape
                
            self.data[field_key] = torch.zeros(
                shape, dtype=dtype, device=self.device
            )
            
    def update(self, data, ts):
        """Update data at timestep."""
        for k, v in data.items():
            if k in self.data:
                if k in self.scheme and "group" in self.scheme[k]:
                    if isinstance(v, list):
                        if all(isinstance(item, torch.Tensor) for item in v):
                            v = torch.stack(v, dim=0)  # (n_agents, ...)
                        else:
                            v = torch.tensor(v, device=self.device)
                    
                    if isinstance(v, torch.Tensor):
                        v = v.to(self.device)
                        expected_agents = self.groups[self.scheme[k]["group"]]
                        if v.shape[0] != expected_agents:
                            raise ValueError(f"Expected {expected_agents} agents for field '{k}', got {v.shape[0]}")
                        self.data[k][:, ts] = v
                    else:
                        raise TypeError(f"Expected tensor or list of tensors for grouped field '{k}', got {type(v)}")
                else:
                    if isinstance(v, list):
                        if len(v) == 1:
                            v = v[0]
                        else:
                            v = torch.tensor(v, device=self.device)
                    
                    if isinstance(v, torch.Tensor):
                        v = v.to(self.device)
                    
                    self.data[k][:, ts] = v
                
    def __getitem__(self, item):
        """Get data by key or slice."""
        if isinstance(item, str):
            return self.data[item]
        elif isinstance(item, slice):
            try:
                indices = range(*item.indices(self.batch_size))
                new_bs = len(list(indices))
            except Exception:
                new_bs = None
            sliced_batch = EpisodeBatch(
                self.scheme, self.groups, 
                int(new_bs) if isinstance(new_bs, int) and new_bs > 0 else self.batch_size,
                self.max_seq_length, self.device
            )
            for key, tensor in self.data.items():
                sliced_batch.data[key] = tensor[item]
            try:
                if isinstance(new_bs, int) and new_bs > 0:
                    sliced_batch.batch_size = int(new_bs)
                else:
                    any_k = next(iter(sliced_batch.data.keys()))
                    sliced_batch.batch_size = int(sliced_batch.data[any_k].shape[0])
            except Exception:
                pass
            return sliced_batch
        else:
            indexed_batch = EpisodeBatch(
                self.scheme, self.groups, 
                1 if isinstance(item, int) else self.batch_size, 
                self.max_seq_length, self.device
            )
            for key, tensor in self.data.items():
                v = tensor[item]
                if isinstance(item, int) and isinstance(v, torch.Tensor) and v.ndim >= 1:
                    v = v.unsqueeze(0)
                indexed_batch.data[key] = v
            try:
                indexed_batch.batch_size = 1 if isinstance(item, int) else int(indexed_batch.batch_size)
            except Exception:
                pass
            return indexed_batch