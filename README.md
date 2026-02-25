# The 4-State Holographic Gate

**Neural network gates aren't binary switches — they're 4-state holographic encoders where "dead" channels carry up to 42% of output energy.**

## The Discovery

Every analysis of neural network activation functions (SiLU, GELU) treats channels as **binary**: alive or dead. Channels with negative gate values are "dead" — zero output, skip them, save compute.

**This is wrong.** Dead channels carry structured information via negative leakage, and the sign at near-zero magnitude ("negative zero") carries **4× more information than the magnitude itself**.

Validated independently on two architectures:
- **DDColor** (ConvNeXt, GELU): "dead" channels contribute 31.6% of output energy
- **Qwen2-7B** (Transformer, SiLU): "dead" channels contribute up to 42.4% of output energy

## The 4-State Gate

The SiLU activation `SiLU(x) = x · σ(x)` creates four operating regimes, with boundaries at **±log(φ) ≈ ±0.481** (where φ is the golden ratio):

```
     SiLU(x)
       │
  1.0  │                              ╱
       │                           ╱
  0.5  │                        ╱        +1 EXPAND: SiLU(x) ≈ x
       │                     ╱           Full fire, channels contribute proportionally
       │                  ╱
  log(φ)│- - - - - - - ╱ - - - - - - - - boundary
       │            ╱                    +0 PRESERVE+: SiLU(x) ≈ x/2
       │         ╱                       Linear regime, positive side
  0.0  │──────╳─────────────────────
       │    ╱                            -0 PRESERVE-: SiLU(x) ≈ x/2
       │  ╱                              Linear regime, NEGATIVE side
 -log(φ)│╱- - - - - - - - - - - - - - - boundary
       │                                 -1 CONTRACT: SiLU(x) ≈ x·exp(x)
 -0.5  │                                 Deep negative leakage
       │
      ─┼──────────────────────────── x
      -5  -3  -1   0   1   3   5
```

### 2 bits per channel

| State | Sign | Magnitude | SiLU behavior | Holographic role |
|-------|------|-----------|---------------|-----------------|
| **+1** (EXPAND) | + | high | ≈ x (identity) | Bright fringe |
| **+0** (PRESERVE+) | + | low | ≈ x/2 (linear) | Bright fringe (dim) |
| **-0** (PRESERVE-) | - | low | ≈ x/2 (linear, neg) | Dark fringe |
| **-1** (CONTRACT) | - | high | ≈ x·exp(x) (leakage) | Dark fringe (deep) |

The **sign bit is more important than the magnitude bit** — this is the key finding.

## Key Results

### 1. Dead Channels Carry Energy

Energy decomposition of MLP output by gate region (Qwen2-7B):

| Layer | EXPAND | PRESERVE | **CONTRACT** |
|-------|--------|----------|-------------|
| 0 | 60.7% | 10.8% | **6.9%** |
| 7 | 69.2% | 4.4% | **20.2%** |
| **14** | 52.2% | 8.4% | **42.4%** |
| 21 | 74.4% | 2.3% | **24.7%** |
| 27 | 91.9% | 0.04% | **3.6%** |

At layer 14, **42.4% of output energy comes from "dead" channels**. The sum exceeds 100% because positive and negative contributions have negative cross-terms — **destructive interference**, exactly like a hologram.

### 2. Sign > Magnitude

In the PRESERVE region (|gate| ≤ log(φ)), two ablations:
- **Remove sign**: replace SiLU(g) with |SiLU(g)|
- **Keep only sign**: replace magnitude with constant, preserve sign

| Layer | Remove sign | Keep ONLY sign | Sign advantage |
|-------|------------|----------------|---------------|
| 0 | 0.869 | **0.965** | 4.0× |
| 7 | 0.929 | **0.981** | 2.7× |
| 14 | 0.914 | **0.976** | 2.5× |
| 21 | 0.975 | **0.993** | 2.6× |

**The sign at zero carries ~4× more information than the magnitude.** In IEEE floating point, -0 == +0. In neural networks, they are fundamentally different signals.

### 3. Including Negative Zero Recovers Output

| Layer | Binary (skip dead) | Ternary (no neg zero) | **With negative zero** |
|-------|-------------------|----------------------|----------------------|
| 0 | 0.960 | 0.955 | **0.994** |
| 7 | 0.833 | 0.827 | **0.989** |
| **14** | **0.751** | 0.745 | **0.986** |
| 21 | 0.863 | 0.856 | **0.993** |
| 27 | 0.952 | 0.953 | **0.999** |

End-to-end: **4/5 same argmax WITH negative zero, 0/5 WITHOUT.**

## The Holographic Interpretation

A hologram encodes information in BOTH bright and dark fringes. Bright fringes alone give half the picture — you need the dark fringes to reconstruct the full wavefront.

The MLP gate operates identically:

```
HOLOGRAPHIC PLATE (the gate field)
══════════════════════════════════

Bright fringes (+1, +0):  Where the gate FIRES
  → Positive contribution to output
  → Carries the "what to say" signal

Dark fringes (-0, -1):    Where the gate LEAKS
  → Negative contribution to output
  → Carries the "what NOT to say" signal
  → Anti-correlated with bright fringes (r ≈ -0.10)

Together:
  → Complete interference pattern
  → Full reconstruction of the intended output
```

## Why This Matters

1. **You cannot skip dead channels.** Binary sparse MLP drops correlation to 0.75 at layer 14. Standard "activation sparsity" approaches destroy information.

2. **The sign is the cheap part.** For PRESERVE channels, you only need 1 bit (which side of zero), not the exact magnitude. The bias predicts sign 98-100% of the time.

3. **The 4-state encoding maps to 2 bits per channel.** For 18,944 intermediate channels: 4,736 bytes per token per layer. A tiny "gate code" that tells any receiver how to reconstruct the output.

4. **The φ-boundary is not arbitrary.** `sigmoid(log(φ)) = 1/φ` exactly. The golden ratio defines the natural transition point between the gate's operating regimes.

## Running the Demo

```bash
pip install -r requirements.txt
python holographic_gate_demo.py
```

The demo reproduces the key findings using synthetic MLP layers:
- 4-state classification with φ-boundaries
- Energy decomposition showing CONTRACT contribution
- Sign vs magnitude ablation
- Binary vs ternary vs negative-zero approximation quality

For full Qwen2-7B reproduction (requires ~16GB RAM + HuggingFace model):
```bash
python holographic_gate_demo.py --model qwen2
```

## The φ Connection

The boundaries at ±log(φ) come from an exact identity:

```
sigmoid(log(φ))  = 1/φ     EXACTLY
sigmoid(0)       = 1/2      EXACTLY
sigmoid(-log(φ)) = 1/φ²    EXACTLY
```

This defines three natural regions on the real line. For SiLU (`x · σ(x)`), these become the four states when combined with the sign of x.

## Files

| File | Description |
|------|-------------|
| `holographic_gate_demo.py` | Self-contained demo (synthetic + optional Qwen2) |
| `README.md` | This document |
| `requirements.txt` | Dependencies |

## References

- Finding 57 in [TruthSpace-LCM](https://github.com/lostdemeter/truthspace-lcm) FINDINGS.md
- Design Consideration 253: Negative Zero as the Fourth Dimension
- Phase 17C-D: Push-pull architecture and binary code discovery (DDColor)
- Related: [phi_gate](https://github.com/lostdemeter/phi_gate) — GELU ≈ x·σ(φ·x)

## License

MIT
