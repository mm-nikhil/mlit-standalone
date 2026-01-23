## CPU INFO

```
CPU: Intel Core i5-13420H (13th gen, Raptor Lake)
Cores: 8 cores, 12 threads (4 P-cores + 4 E-cores)

Cache per core:
  L1d: 320 KiB / 8 = 40 KiB per core
  L1i: 384 KiB / 8 = 48 KiB per core
  L2:  7 MiB / 5 = 1.4 MiB per instance
  L3:  12 MiB (shared)

SIMD: AVX2 (256-bit = 8 × float32)
```

## Mathematical Tile Size Calculation
```
Strategy 1: Conservative (Guaranteed to work)
m2 = 256    # L2 outer tile (M dimension)
n2 = 256    # L2 outer tile (N dimension)  
k2 = 8      # K reduction blocking (CRITICAL!)

m1 = 64     # L1 inner tile (M dimension)
n1 = 64     # L1 inner tile (N dimension)
k1 = 8      # K inner tile (matches k2)

Rationale:

m2/n2=256: Fits comfortably in 1.4MB L2, divides 512 evenly
k2=8: Optimal for C reuse
m1/n1=64: Fits in 40KB L1 with 50% headroom
k1=8: Matches k2 (no additional K tiling needed)
```

```
Strategy 2: Aggressive (Push limits)
m2 = 256
n2 = 256
k2 = 8

m1 = 80     # Slightly larger L1 tiles
n1 = 80
k1 = 8
Verification [80, 80, 8]:
A: 80 × 8 × 4 = 2,560 bytes
B: 8 × 80 × 4 = 2,560 bytes
C: 80 × 80 × 4 = 25,600 bytes
Total: 30,720 bytes = 30 KB ✓ (fits in 40 KB)
```

```
Strategy 3: Asymmetric (Theory-optimal)
m2 = 256
n2 = 256
k2 = 8

m1 = 64
n1 = 128    # Larger N (better for memory layout)
k1 = 8
Verification [64, 128, 8]:
A: 64 × 8 × 4 = 2,048 bytes
B: 8 × 128 × 4 = 4,096 bytes
C: 64 × 128 × 4 = 32,768 bytes
Total: 38,912 bytes = 38 KB ✓ (fits snugly in 40 KB)
```