//! # Canonical Transformations
//!
//! A canonical transformation (q, p) → (Q, P) preserves the symplectic form ω.
//! Equivalently, it preserves Poisson brackets.

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::symplectic::{PhasePoint, SymplecticForm};

/// A canonical transformation (q, p) → (Q, P).
pub struct CanonicalTransformation {
    /// Human-readable name.
    pub name: String,
    /// Forward map: (q, p) → (Q, P).
    pub forward: Arc<dyn Fn(&PhasePoint) -> PhasePoint + Send + Sync>,
    /// Inverse map: (Q, P) → (q, p).
    pub inverse: Arc<dyn Fn(&PhasePoint) -> PhasePoint + Send + Sync>,
}

impl std::fmt::Debug for CanonicalTransformation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CanonicalTransformation").field("name", &self.name).finish()
    }
}

impl Clone for CanonicalTransformation {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            forward: self.forward.clone(),
            inverse: self.inverse.clone(),
        }
    }
}

impl Serialize for CanonicalTransformation {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.name)
    }
}

impl<'de> Deserialize<'de> for CanonicalTransformation {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let name = String::deserialize(d)?;
        Ok(CanonicalTransformation {
            name,
            forward: Arc::new(|pt| pt.clone()),
            inverse: Arc::new(|pt| pt.clone()),
        })
    }
}

impl CanonicalTransformation {
    /// Create a named canonical transformation.
    pub fn new<F, G>(name: &str, forward: F, inverse: G) -> Self
    where
        F: Fn(&PhasePoint) -> PhasePoint + Send + Sync + 'static,
        G: Fn(&PhasePoint) -> PhasePoint + Send + Sync + 'static,
    {
        Self { name: name.to_string(), forward: Arc::new(forward), inverse: Arc::new(inverse) }
    }

    /// Apply the forward transformation.
    pub fn apply(&self, point: &PhasePoint) -> PhasePoint {
        (self.forward)(point)
    }

    /// Apply the inverse transformation.
    pub fn apply_inverse(&self, point: &PhasePoint) -> PhasePoint {
        (self.inverse)(point)
    }

    /// Verify the transformation is canonical by checking the Jacobian is symplectic.
    pub fn verify_canonical(&self, point: &PhasePoint) -> bool {
        let omega = SymplecticForm::new(point.dof());
        let n = point.dof();
        let eps = 1e-7;
        let mut jacobian = vec![vec![0.0; 2 * n]; 2 * n];
        for i in 0..2 * n {
            let mut q_plus = point.q.clone();
            let mut p_plus = point.p.clone();
            let mut q_minus = point.q.clone();
            let mut p_minus = point.p.clone();

            if i < n {
                q_plus[i] += eps;
                q_minus[i] -= eps;
            } else {
                p_plus[i - n] += eps;
                p_minus[i - n] -= eps;
            }

            let f_plus = (self.forward)(&PhasePoint::new(q_plus, p_plus));
            let f_minus = (self.forward)(&PhasePoint::new(q_minus, p_minus));

            for j in 0..n {
                jacobian[j][i] = (f_plus.q[j] - f_minus.q[j]) / (2.0 * eps);
                jacobian[n + j][i] = (f_plus.p[j] - f_minus.p[j]) / (2.0 * eps);
            }
        }
        omega.is_symplectic_matrix(&jacobian)
    }

    /// Verify the inverse is correct: T⁻¹(T(x)) = x.
    pub fn verify_inverse(&self, point: &PhasePoint) -> bool {
        let transformed = self.apply(point);
        let recovered = self.apply_inverse(&transformed);
        let diff = PhasePoint::new(
            point.q.iter().zip(&recovered.q).map(|(a, b)| a - b).collect(),
            point.p.iter().zip(&recovered.p).map(|(a, b)| a - b).collect(),
        );
        diff.norm() < 1e-8
    }
}

/// Identity transformation (trivially canonical).
pub fn identity() -> CanonicalTransformation {
    CanonicalTransformation::new("identity", |pt| pt.clone(), |pt| pt.clone())
}

/// Point reflection: (q, p) → (-q, -p).
pub fn point_reflection() -> CanonicalTransformation {
    CanonicalTransformation::new(
        "point_reflection",
        |pt| PhasePoint {
            q: pt.q.iter().map(|x| -x).collect(),
            p: pt.p.iter().map(|x| -x).collect(),
        },
        |pt| PhasePoint {
            q: pt.q.iter().map(|x| -x).collect(),
            p: pt.p.iter().map(|x| -x).collect(),
        },
    )
}

/// Swap q and p (with sign): (q, p) → (p, -q).
pub fn fourier_transform() -> CanonicalTransformation {
    CanonicalTransformation::new(
        "fourier_transform",
        |pt| PhasePoint {
            q: pt.p.clone(),
            p: pt.q.iter().map(|x| -x).collect(),
        },
        |pt| PhasePoint {
            q: pt.p.iter().map(|x| -x).collect(),
            p: pt.q.clone(),
        },
    )
}

/// Scaling: (q, p) → (λq, p/λ).
pub fn scaling(lambda: f64) -> CanonicalTransformation {
    let inv = 1.0 / lambda;
    CanonicalTransformation::new(
        &format!("scaling_{}", lambda),
        move |pt| PhasePoint {
            q: pt.q.iter().map(|x| x * lambda).collect(),
            p: pt.p.iter().map(|x| x * inv).collect(),
        },
        move |pt| PhasePoint {
            q: pt.q.iter().map(|x| x * inv).collect(),
            p: pt.p.iter().map(|x| x * lambda).collect(),
        },
    )
}

/// Phase rotation by angle θ (1 DOF).
pub fn phase_rotation(theta: f64) -> CanonicalTransformation {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    CanonicalTransformation::new(
        &format!("phase_rotation_{}", theta),
        move |pt| PhasePoint {
            q: pt.q.iter().zip(&pt.p).map(|(qi, pi)| qi * cos_t - pi * sin_t).collect(),
            p: pt.q.iter().zip(&pt.p).map(|(qi, pi)| qi * sin_t + pi * cos_t).collect(),
        },
        move |pt| PhasePoint {
            q: pt.q.iter().zip(&pt.p).map(|(qi, pi)| qi * cos_t + pi * sin_t).collect(),
            p: pt.q.iter().zip(&pt.p).map(|(qi, pi)| -qi * sin_t + pi * cos_t).collect(),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonian::Hamiltonian;

    #[test]
    fn test_identity_canonical() {
        let t = identity();
        let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);
        assert!(t.verify_canonical(&pt));
        assert!(t.verify_inverse(&pt));
    }

    #[test]
    fn test_point_reflection_canonical() {
        let t = point_reflection();
        let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);
        assert!(t.verify_canonical(&pt));
        assert!(t.verify_inverse(&pt));
    }

    #[test]
    fn test_fourier_transform_canonical() {
        let t = fourier_transform();
        let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);
        assert!(t.verify_canonical(&pt));
        assert!(t.verify_inverse(&pt));
    }

    #[test]
    fn test_scaling_canonical() {
        let t = scaling(2.0);
        let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);
        assert!(t.verify_canonical(&pt));
        assert!(t.verify_inverse(&pt));
    }

    #[test]
    fn test_phase_rotation_canonical() {
        let t = phase_rotation(std::f64::consts::PI / 4.0);
        let pt = PhasePoint::new(vec![1.0], vec![0.0]);
        assert!(t.verify_canonical(&pt));
        assert!(t.verify_inverse(&pt));
    }

    #[test]
    fn test_rotation_preserves_energy() {
        let h = Hamiltonian::harmonic(&[1.0], &[1.0]);
        let pt = PhasePoint::new(vec![1.0], vec![0.0]);
        let e_before = h.energy(&pt);
        let t = phase_rotation(std::f64::consts::PI / 3.0);
        let pt_new = t.apply(&pt);
        let e_after = h.energy(&pt_new);
        assert!((e_before - e_after).abs() < 1e-10);
    }
}

