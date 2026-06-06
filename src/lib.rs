//! # symplectic-fleet
//!
//! Fleet state modeled as a symplectic manifold (q, p) where q = agent configurations
//! and p = conjugate momenta (resource fluxes). The fleet evolves via Hamilton's equations,
//! guaranteeing the symplectic 2-form is preserved. Noether's theorem provides a precise
//! correspondence: every continuous fleet symmetry ↔ a conserved quantity.
//!
//! ## Modules
//!
//! - [`symplectic`] — Symplectic form ω, Darboux coordinates, symplectic linear algebra
//! - [`hamiltonian`] — Fleet Hamiltonian H(q,p), Hamilton's equations, energy conservation
//! - [`noether`] — Automatic Noether pair computation: symmetry ↔ conserved quantity
//! - [`integrator`] — Symplectic integrators (Störmer-Verlet, implicit midpoint)
//! - [`canonical`] — Canonical transformations and generating functions
//! - [`poisson`] — Poisson brackets and Lie-Poisson structures

pub mod symplectic;
pub mod hamiltonian;
pub mod noether;
pub mod integrator;
pub mod canonical;
pub mod poisson;

pub use symplectic::{PhasePoint, SymplecticForm};
pub use hamiltonian::Hamiltonian;
pub use noether::{Symmetry, NoetherPair};
pub use integrator::{IntegratorConfig, IntegrationMethod};
pub use canonical::CanonicalTransformation;
pub use poisson::PoissonBracket;
