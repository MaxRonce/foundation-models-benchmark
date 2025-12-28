
import torch
import sys

try:
    data = torch.load('/n03data/ronceray/embeddings/astropt_embeddings.pt', map_location='cpu')
    print(f"Loaded data type: {type(data)}")
    if isinstance(data, list) and len(data) > 0:
        print(f"First record keys: {data[0].keys()}")
        for key, value in data[0].items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
    elif isinstance(data, dict):
        print(f"Keys: {data.keys()}")
except Exception as e:
    print(f"Error loading file: {e}")
