# symplectic-fleet

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Language: Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![SuperInstance](https://img.shields.io/badge/part%20of-SuperInstance-purple.svg)](https://github.com/SuperInstance)

Fleet state modeled as a symplectic manifold with Hamiltonian evolution, Noether conservation laws, and structure-preserving integrators.

## What It Does

`symplectic-fleet` treats a fleet of agents as a Hamiltonian system. Agent configurations are generalized coordinates q, resource fluxes are conjugate momenta p, and the fleet evolves via Hamilton's equations:

```
dq/dt =  ∂H/∂p    (velocity from momentum)
dp/dt = -∂H/∂q    (force from configuration)
```

The symplectic 2-form ω = Σ dqᵢ ∧ dpᵢ is preserved exactly by the integrators, so energy doesn't drift over long times. Noether's theorem provides a precise correspondence: every continuous fleet symmetry ↔ a conserved quantity.

The conservation law **γ + η = C** is foundational here: the fleet Hamiltonian H(q, p) is the conserved quantity C. Productive energy γ (directed agent movement) and entropy η (dissipation/spreading) sum to H, which symplectic integration preserves to machine precision.

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                    Phase Space (q, p)                  │
│  PhasePoint::new(q: Vec<f64>, p: Vec<f64>)            │
├───────────┬───────────────────┬───────────────────────┤
│Symplectic │   Hamiltonian     │      Noether          │
│  Form ω   │   H = T(p)+V(q)  │  Symmetry ↔ Conserved │
│           │                   │       Quantity         │
│ apply()   │ energy()          │ NoetherPair::          │
│ matrix()  │ equations_of_     │   compute_noether_    │
│ pfaffian()│   motion()        │   pair()              │
│ verify_   │ Potential::{Zero, │ translation_symmetry() │
│  antisym, │   Harmonic,       │ rotation_symmetry()   │
│  nondeg,  │   ShiftedHarmonic}│                       │
│  bilinear │ Kinetic::{Standard│                       │
│           │   , UnitMass}     │                       │
├───────────┴───────────────────┴───────────────────────┤
│               Integrators (symplectic)                 │
│  ┌──────────────────┐  ┌────────────────────────────┐ │
│  │ Störmer-Verlet   │  │ Implicit Midpoint          │ │
│  │ (leapfrog)       │  │ (fixed-point iteration)    │ │
│  │ 2nd-order,       │  │ 2nd-order, symplectic,     │ │
│  │ symplectic,      │  │ energy-exact for quadratic │ │
│  │ time-reversible  │  │ H                          │ │
│  └──────────────────┘  └────────────────────────────┘ │
├───────────────────────────────────────────────────────┤
│  Canonical Transformations    │   Poisson Brackets     │
│  (q,p) → (Q,P) preserves ω  │   {f,g} = Σ(∂f/∂qᵢ    │
│  identity, point_reflection,  │     ∂g/∂pᵢ - ∂f/∂pᵢ  │
│  fourier_transform, scaling,  │     ∂g/∂qᵢ)           │
│  phase_rotation               │   LiePoissonStructure   │
│  verify_canonical()           │   (so(3)× rigid body)  │
└───────────────────────────────────────────────────────┘
```

## Installation

```toml
[dependencies]
symplectic-fleet = { git = "https://github.com/SuperInstance/symplectic-fleet" }
```

## Usage

### Define a fleet Hamiltonian and integrate

```rust
use symplectic_fleet::*;

// Harmonic oscillator: H = p²/(2m) + ½k q²
// Two agents with masses [1.0, 1.0], stiffnesses [1.0, 1.0]
let h = Hamiltonian::harmonic(&[1.0, 1.0], &[1.0, 1.0]);
let initial = PhasePoint::new(vec![1.0, 0.0], vec![0.0, 1.0]);

// Störmer-Verlet: 10000 steps of size 0.01
let cfg = IntegratorConfig::new(0.01, 10000, IntegrationMethod::StormerVerlet);
let trajectory = integrate(&h, &initial, &cfg);

// Energy is conserved along the trajectory (bounded error, no drift)
let max_deviation = h.verify_energy_conservation(&trajectory);
assert!(max_deviation < 1e-3);
```

### Noether's theorem: symmetry ↔ conserved quantity

```rust
// Translation symmetry of free-particle Hamiltonian → momentum is conserved
let h = Hamiltonian::free(&[1.0, 1.0]);
let sym = translation_symmetry(2, 0); // translate along q₀

// Automatically compute the Noether conserved quantity
let pair = NoetherPair::compute_noether_pair(sym, &h);
let pt = PhasePoint::new(vec![1.0, 2.0], vec![5.0, 3.0]);
assert!((pair.evaluate(&pt) - 5.0).abs() < 1e-10); // = p₀

// Verify conservation along an actual trajectory
let cfg = IntegratorConfig::new(0.01, 1000, IntegrationMethod::StormerVerlet);
let traj = integrate(&h, &pt, &cfg);
let max_dev = pair.verify_conservation(&traj);
assert!(max_dev < 1e-4);
```

### Verify the symplectic form is preserved

```rust
let h = Hamiltonian::harmonic(&[1.0], &[1.0]);
let pt = PhasePoint::new(vec![1.0], vec![0.0]);
let cfg = IntegratorConfig::new(0.01, 10, IntegrationMethod::StormerVerlet);

// Checks MᵀJM = J where M is the flow Jacobian
let is_symplectic = verify_symplecticity(&h, &pt, &cfg);
assert!(is_symplectic);

// Phase space volume is exactly preserved (det(M) = 1)
let volume = compute_volume_change(&h, &pt, &cfg);
assert!((volume - 1.0).abs() < 1e-4);
```

### Canonical transformations

```rust
let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);

// Scaling: (q, p) → (λq, p/λ) preserves ω
let t = scaling(2.0);
assert!(t.verify_canonical(&pt));   // Jacobian satisfies MᵀJM = J
assert!(t.verify_inverse(&pt));     // T⁻¹(T(x)) = x

// Fourier-like swap: (q, p) → (p, -q)
let ft = fourier_transform();
assert!(ft.verify_canonical(&pt));

// Phase rotation by π/4
let rot = phase_rotation(std::f64::consts::PI / 4.0);
assert!(rot.verify_canonical(&pt));
```

### Poisson brackets and Lie-Poisson structures

```rust
let pb = PoissonBracket::new(2);
let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);

let q0: Observable = |pt: &PhasePoint| pt.q[0];
let p0: Observable = |pt: &PhasePoint| pt.p[0];

// Canonical bracket: {q₀, p₀} = 1
assert!((pb.apply(q0, p0, &pt) - 1.0).abs() < 1e-6);

// Verify algebraic identities
let f: Observable = |pt| pt.q[0] * pt.q[0];
let g: Observable = |pt| pt.q[0] * pt.p[1];
let h_fn: Observable = |pt| pt.q[1] * pt.p[0];
assert!(pb.verify_jacobi(f, g, h_fn, &pt));     // {f,{g,h}} + cyclic = 0
assert!(pb.verify_antisymmetry(f, g, &pt));      // {f,g} = -{g,f}

// so(3)× Lie-Poisson (rigid body dynamics)
let lp = LiePoissonStructure::so3();
let mu = vec![1.0, 2.0, 3.0];
let bracket_val = lp.bracket(|m| m[1], |m| m[2], &mu);
```

## API Reference

### Core Types

| Type | Module | Description |
|------|--------|-------------|
| `PhasePoint` | `symplectic` | Point in 2n-dim phase space (q, p) |
| `SymplecticForm` | `symplectic` | Canonical ω with `apply()`, `matrix()`, `verify_*` |
| `Hamiltonian` | `hamiltonian` | H = T(p) + V(q) with `energy()`, `equations_of_motion()` |
| `Symmetry` | `noether` | One-parameter family of transformations |
| `NoetherPair` | `noether` | Symmetry + conserved quantity |
| `CanonicalTransformation` | `canonical` | (q,p) → (Q,P) preserving ω |
| `PoissonBracket` | `poisson` | Canonical {f, g} and Jacobi identity |
| `LiePoissonStructure` | `poisson` | Bracket on dual Lie algebra g* |

### Integrators

| Method | Order | Properties |
|--------|-------|------------|
| `StormerVerlet` | 2nd | Symplectic, time-reversible, explicit |
| `ImplicitMidpoint` | 2nd | Symplectic, energy-exact for quadratic H |

### Built-in Symmetries

| Function | Symmetry | Conserved Quantity |
|----------|----------|--------------------|
| `translation_symmetry(dim, axis)` | q → q + s·ê | Linear momentum p_axis |
| `rotation_symmetry(dim, i, j)` | Rotation in (qᵢ, qⱼ) plane | Angular momentum Lᵢⱼ |

### Built-in Transformations

| Function | Map | Note |
|----------|-----|------|
| `identity()` | (q, p) → (q, p) | Trivial |
| `point_reflection()` | (q, p) → (-q, -p) | Involution |
| `fourier_transform()` | (q, p) → (p, -q) | Q↔P swap |
| `scaling(λ)` | (q, p) → (λq, p/λ) | Volume-preserving |
| `phase_rotation(θ)` | Rotation by θ in (q, p) plane | 1 DOF |

## Related Crates (SuperInstance Ecosystem)

- **ternary-mud** — Ternary algebra MUD rooms with Hodge lostness detection
- **meta-agent** — Task dispatch and agent coordination (uses this for fleet dynamics)
- **ternary-energy** — Energy conservation in ternary systems
- **ternary-thermodynamics** — Heat engines and Carnot cycles in ternary
- **ternary-kuramoto** — Oscillator synchronization over ternary coupling
- **forgemaster** — GPU fleet orchestration backend
