# Temporary utility functions for Balinski and Gomory algorithm implementation

To get the files
```
git clone https://github.com/cestwc/bgutils.git
```

Test examples

```python
from bgutils import generate_cost, perturb_sparse, perturb_l0

C = generate_cost(n, 1, 100)

# Example
C = torch.randint(1, 10, (5, 5))
C_perturbed = perturb_sparse(C, ratio=0.2, max_change=5)

print("Original Matrix:\n", C)
print("Perturbed Matrix:\n", C_perturbed)

# Example
C = torch.randint(1, 10, (5, 5))
X = torch.randint(0, 2, (5, 5))  # Random 0-1 mask

C_perturbed = perturb_l0(C, X, ratio=0.2, num_perturb=1)

print("Original Matrix:\n", C)
print("Perturbed Matrix:\n", C_perturbed)
```
