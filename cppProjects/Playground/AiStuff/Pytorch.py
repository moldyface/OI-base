import torch

# Create two tensors (PyTorch's main data structure)
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])

# Perform operations
z = x + y
print("Tensor x:\n", x)
print("\nTensor y:\n", y)
print("\nSum (x + y):\n", z)

# Check tensor properties
print("\nTensor shape:", z.shape)
print("Tensor dtype:", z.dtype)
print("Device:", z.device)  # Will show 'cpu'
print("Cooked")