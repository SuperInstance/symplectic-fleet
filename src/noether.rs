//! # Noether's Theorem
//!
//! Every continuous symmetry of the Hamiltonian corresponds to a conserved quantity.
//! If a transformation Φₛ: (q, p) → (Q, P) leaves H invariant for all s,
//! then I(q, p) = p · (∂Q/∂s)|_{s=0} is conserved.

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::symplectic::PhasePoint;
use crate::hamiltonian::Hamiltonian;

/// A continuous symmetry: a one-parameter family of phase space transformations.
#[derive(Clone)]
pub struct Symmetry {
    /// Human-readable name of the symmetry.
    pub name: String,
    /// Generator of the transformation: returns (dq/ds, dp/ds).
    pub generator: Arc<dyn Fn(&PhasePoint) -> PhasePoint + Send + Sync>,
}

impl std::fmt::Debug for Symmetry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Symmetry").field("name", &self.name).finish()
    }
}

impl Serialize for Symmetry {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.name)
    }
}

impl<'de> Deserialize<'de> for Symmetry {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let name = String::deserialize(d)?;
        Ok(Symmetry {
            name,
            generator: Arc::new(|_| PhasePoint::new(vec![], vec![])),
        })
    }
}

impl Symmetry {
    /// Create a named symmetry from its generator.
    pub fn new<F>(name: &str, generator: F) -> Self
    where
        F: Fn(&PhasePoint) -> PhasePoint + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            generator: Arc::new(generator),
        }
    }

    /// Apply the symmetry transformation at parameter value s.
    pub fn transform(&self, point: &PhasePoint, s: f64) -> PhasePoint {
        let g = (self.generator)(point);
        PhasePoint::new(
            point.q.iter().zip(&g.q).map(|(qi, gi)| qi + s * gi).collect(),
            point.p.iter().zip(&g.p).map(|(pi, gi)| pi + s * gi).collect(),
        )
    }

    /// Verify the symmetry preserves H to numerical precision.
    pub fn preserves_hamiltonian(&self, h: &Hamiltonian, point: &PhasePoint, s: f64) -> bool {
        let e_before = h.energy(point);
        let transformed = self.transform(point, s);
        let e_after = h.energy(&transformed);
        (e_before - e_after).abs() < s.abs() * s.abs() * 10.0
    }
}

/// A Noether pair: a symmetry and its corresponding conserved quantity.
#[derive(Clone)]
pub struct NoetherPair {
    /// The continuous symmetry.
    pub symmetry: Symmetry,
    /// The conserved quantity I(q, p).
    pub conserved_quantity: Arc<dyn Fn(&PhasePoint) -> f64 + Send + Sync>,
}

impl std::fmt::Debug for NoetherPair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoetherPair").field("symmetry", &self.symmetry).finish()
    }
}

impl Serialize for NoetherPair {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.symmetry.name)
    }
}

impl<'de> Deserialize<'de> for NoetherPair {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let name = String::deserialize(d)?;
        Ok(NoetherPair {
            symmetry: Symmetry::new(&name, |_| PhasePoint::new(vec![], vec![])),
            conserved_quantity: Arc::new(|_| 0.0),
        })
    }
}

impl NoetherPair {
    /// Create a new Noether pair.
    pub fn new<F, G>(
        name: &str,
        generator: F,
        conserved: G,
    ) -> Self
    where
        F: Fn(&PhasePoint) -> PhasePoint + Send + Sync + 'static,
        G: Fn(&PhasePoint) -> f64 + Send + Sync + 'static,
    {
        Self {
            symmetry: Symmetry::new(name, generator),
            conserved_quantity: Arc::new(conserved),
        }
    }

    /// Evaluate the conserved quantity.
    pub fn evaluate(&self, point: &PhasePoint) -> f64 {
        (self.conserved_quantity)(point)
    }

    /// Verify conservation along a trajectory.
    pub fn verify_conservation(&self, trajectory: &[PhasePoint]) -> f64 {
        if trajectory.len() < 2 {
            return 0.0;
        }
        let i0 = self.evaluate(&trajectory[0]);
        if i0.abs() < 1e-15 {
            return trajectory
                .iter()
                .map(|p| self.evaluate(p).abs())
                .fold(0.0_f64, f64::max);
        }
        trajectory
            .iter()
            .map(|p| ((self.evaluate(p) - i0) / i0).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Automatically compute the Noether conserved quantity from a symmetry.
    /// The conserved quantity is I(q, p) = Σᵢ pᵢ · ξᵢ(q) where ξ is the
    /// configuration-space part of the generator.
    pub fn compute_noether_pair(
        symmetry: Symmetry,
        _h: &Hamiltonian,
    ) -> Self {
        let gen_fn = symmetry.generator.clone();
        let conserved = move |point: &PhasePoint| -> f64 {
            let g = gen_fn(point);
            point.p.iter().zip(&g.q).map(|(pi, gi)| pi * gi).sum()
        };
        Self {
            symmetry,
            conserved_quantity: Arc::new(conserved),
        }
    }
}

/// Built-in translational symmetry in the i-th coordinate direction.
pub fn translation_symmetry(dim: usize, axis: usize) -> Symmetry {
    assert!(axis < dim);
    Symmetry::new(
        &format!("translation_q{}", axis),
        move |_: &PhasePoint| {
            let mut dq = vec![0.0; dim];
            let dp = vec![0.0; dim];
            dq[axis] = 1.0;
            PhasePoint::new(dq, dp)
        },
    )
}

/// Built-in rotational symmetry in the (i, j) plane.
pub fn rotation_symmetry(dim: usize, i: usize, j: usize) -> Symmetry {
    assert!(i < dim && j < dim && i != j);
    Symmetry::new(
        &format!("rotation_q{}_q{}", i, j),
        move |point: &PhasePoint| {
            let mut dq = vec![0.0; dim];
            let dp = vec![0.0; dim];
            dq[i] = -point.q[j];
            dq[j] = point.q[i];
            PhasePoint::new(dq, dp)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::{IntegratorConfig, IntegrationMethod, integrate};

    #[test]
    fn test_translation_preserves_free_hamiltonian() {
        let h = Hamiltonian::free(&[1.0, 1.0]);
        let sym = translation_symmetry(2, 0);
        let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);
        assert!(sym.preserves_hamiltonian(&h, &pt, 0.1));
    }

    #[test]
    fn test_momentum_conserved_free_particle() {
        let h = Hamiltonian::free(&[1.0, 1.0]);
        let pt = PhasePoint::new(vec![0.0, 0.0], vec![3.0, 4.0]);
        let cfg = IntegratorConfig::new(0.01, 1000, IntegrationMethod::StormerVerlet);
        let traj = integrate(&h, &pt, &cfg);
        let p_total_0: f64 = traj[0].p.iter().sum();
        let p_total_end: f64 = traj.last().unwrap().p.iter().sum();
        assert!((p_total_0 - p_total_end).abs() < 1e-6);
    }

    #[test]
    fn test_rotation_symmetry_angular_momentum() {
        let h = Hamiltonian::harmonic(&[1.0, 1.0], &[1.0, 1.0]);
        let pt = PhasePoint::new(vec![1.0, 0.0], vec![0.0, 1.0]);
        let sym = rotation_symmetry(2, 0, 1);
        assert!(sym.preserves_hamiltonian(&h, &pt, 0.1));

        let compute_l = |p: &PhasePoint| p.q[0] * p.p[1] - p.q[1] * p.p[0];
        let l0 = compute_l(&pt);
        assert!((l0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_noether_pair_conservation() {
        let h = Hamiltonian::harmonic(&[1.0, 1.0], &[1.0, 1.0]);
        let pt = PhasePoint::new(vec![1.0, 0.0], vec![0.0, 1.0]);
        let pair = NoetherPair::new(
            "angular_momentum_01",
            |pt: &PhasePoint| PhasePoint::new(vec![-pt.q[1], pt.q[0]], vec![0.0, 0.0]),
            |pt: &PhasePoint| pt.q[0] * pt.p[1] - pt.q[1] * pt.p[0],
        );
        assert!((pair.evaluate(&pt) - 1.0).abs() < 1e-10);

        let cfg = IntegratorConfig::new(0.01, 1000, IntegrationMethod::StormerVerlet);
        let traj = integrate(&h, &pt, &cfg);
        let max_dev = pair.verify_conservation(&traj);
        assert!(max_dev < 1e-4, "Angular momentum deviation: {}", max_dev);
    }

    #[test]
    fn test_auto_noether_computation() {
        let h = Hamiltonian::free(&[1.0, 1.0]);
        let sym = translation_symmetry(2, 0);
        let pair = NoetherPair::compute_noether_pair(sym, &h);
        let pt = PhasePoint::new(vec![1.0, 2.0], vec![5.0, 3.0]);
        assert!((pair.evaluate(&pt) - 5.0).abs() < 1e-10);
    }
}

