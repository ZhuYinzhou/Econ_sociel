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
                # 处理有组的数据
                if k in self.scheme and "group" in self.scheme[k]:
                    # 如果v是列表，将其转换为张量
                    if isinstance(v, list):
                        # 检查列表中的元素是否都是张量
                        if all(isinstance(item, torch.Tensor) for item in v):
                            # 堆叠张量
                            v = torch.stack(v, dim=0)  # (n_agents, ...)
                        else:
                            # 如果不是张量，转换为张量
                            v = torch.tensor(v, device=self.device)
                    
                    # 确保v是张量并且在正确的设备上
                    if isinstance(v, torch.Tensor):
                        v = v.to(self.device)
                        # 确保张量形状正确 (n_agents, ...)
                        expected_agents = self.groups[self.scheme[k]["group"]]
                        if v.shape[0] != expected_agents:
                            raise ValueError(f"Expected {expected_agents} agents for field '{k}', got {v.shape[0]}")
                        self.data[k][:, ts] = v
                    else:
                        raise TypeError(f"Expected tensor or list of tensors for grouped field '{k}', got {type(v)}")
                else:
                    # 非组数据，直接赋值
                    if isinstance(v, list):
                        # 对于非组数据，如果是列表，取第一个元素或转换为张量
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
            # 字符串键，返回对应的数据
            return self.data[item]
        elif isinstance(item, slice):
            # 切片，返回一个新的EpisodeBatch对象包含切片后的数据
            # IMPORTANT: batch_size must match the sliced first dimension; otherwise downstream
            # code that relies on batch.batch_size will mismatch tensor shapes.
            try:
                indices = range(*item.indices(self.batch_size))
                new_bs = len(list(indices))
            except Exception:
                # fallback: infer from first tensor after slicing
                new_bs = None
            sliced_batch = EpisodeBatch(
                self.scheme, self.groups, 
                int(new_bs) if isinstance(new_bs, int) and new_bs > 0 else self.batch_size,
                self.max_seq_length, self.device
            )
            # 对所有数据应用切片
            for key, tensor in self.data.items():
                sliced_batch.data[key] = tensor[item]
            # Ensure metadata is consistent with tensors
            try:
                if isinstance(new_bs, int) and new_bs > 0:
                    sliced_batch.batch_size = int(new_bs)
                else:
                    # infer from any tensor
                    any_k = next(iter(sliced_batch.data.keys()))
                    sliced_batch.batch_size = int(sliced_batch.data[any_k].shape[0])
            except Exception:
                pass
            return sliced_batch
        else:
            # 其他索引类型，直接应用到所有数据
            indexed_batch = EpisodeBatch(
                self.scheme, self.groups, 
                1 if isinstance(item, int) else self.batch_size, 
                self.max_seq_length, self.device
            )
            for key, tensor in self.data.items():
                v = tensor[item]
                # If selecting a single element, keep a batch dimension of 1
                if isinstance(item, int) and isinstance(v, torch.Tensor) and v.ndim >= 1:
                    v = v.unsqueeze(0)
                indexed_batch.data[key] = v
            try:
                indexed_batch.batch_size = 1 if isinstance(item, int) else int(indexed_batch.batch_size)
            except Exception:
                pass
            return indexed_batch