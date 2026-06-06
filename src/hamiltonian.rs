//! # Hamiltonian Dynamics
//!
//! The fleet Hamiltonian H(q, p) governs evolution via Hamilton's equations:
//!   dq/dt =  ∂H/∂p   (velocity from momentum)
//!   dp/dt = -∂H/∂q   (force from configuration)
//!
//! Energy H is conserved along trajectories.

use serde::{Deserialize, Serialize};
use crate::symplectic::PhasePoint;

/// The type of Hamiltonian potential V(q).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Potential {
    /// V(q) = 0 — free particle.
    Zero,
    /// V(q) = Σ ½ kᵢ qᵢ² — harmonic, with stiffnesses k.
    Harmonic { stiffnesses: Vec<f64> },
    /// V(q) = Σ ½ kᵢ (qᵢ - cᵢ)² — harmonic about center c.
    ShiftedHarmonic { stiffnesses: Vec<f64>, centers: Vec<f64> },
}

impl Potential {
    /// Evaluate V(q).
    pub fn eval(&self, q: &[f64]) -> f64 {
        match self {
            Potential::Zero => 0.0,
            Potential::Harmonic { stiffnesses } => q
                .iter()
                .zip(stiffnesses)
                .map(|(qi, ki)| 0.5 * ki * qi * qi)
                .sum(),
            Potential::ShiftedHarmonic { stiffnesses, centers } => q
                .iter()
                .zip(stiffnesses)
                .zip(centers)
                .map(|((qi, ki), ci)| 0.5 * ki * (qi - ci).powi(2))
                .sum(),
        }
    }

    /// Compute -∂V/∂q using central finite differences.
    pub fn neg_grad(&self, q: &[f64]) -> Vec<f64> {
        let h = 1e-7;
        q.iter()
            .enumerate()
            .map(|(i, _)| {
                let mut q_p = q.to_vec();
                let mut q_m = q.to_vec();
                q_p[i] += h;
                q_m[i] -= h;
                -(self.eval(&q_p) - self.eval(&q_m)) / (2.0 * h)
            })
            .collect()
    }
}

/// The type of kinetic energy T(p).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Kinetic {
    /// T(p) = Σ pᵢ²/(2mᵢ) — standard, with masses m.
    Standard { masses: Vec<f64> },
    /// T(p) = Σ pᵢ²/2 — unit masses.
    UnitMass,
}

impl Kinetic {
    /// Evaluate T(p).
    pub fn eval(&self, p: &[f64]) -> f64 {
        match self {
            Kinetic::Standard { masses } => {
                p.iter().zip(masses).map(|(pi, mi)| pi * pi / (2.0 * mi)).sum()
            }
            Kinetic::UnitMass => p.iter().map(|pi| pi * pi / 2.0).sum(),
        }
    }

    /// Compute ∂T/∂p using central finite differences.
    pub fn grad(&self, p: &[f64]) -> Vec<f64> {
        let h = 1e-7;
        p.iter()
            .enumerate()
            .map(|(i, _)| {
                let mut p_p = p.to_vec();
                let mut p_m = p.to_vec();
                p_p[i] += h;
                p_m[i] -= h;
                (self.eval(&p_p) - self.eval(&p_m)) / (2.0 * h)
            })
            .collect()
    }
}

/// A fleet Hamiltonian H(q, p) = T(p) + V(q).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Hamiltonian {
    /// Potential energy V(q).
    pub potential: Potential,
    /// Kinetic energy T(p).
    pub kinetic: Kinetic,
}

impl Hamiltonian {
    /// Create a new separable Hamiltonian.
    pub fn new(potential: Potential, kinetic: Kinetic) -> Self {
        Self { potential, kinetic }
    }

    /// Evaluate H(q, p) = V(q) + T(p).
    pub fn energy(&self, point: &PhasePoint) -> f64 {
        self.potential.eval(&point.q) + self.kinetic.eval(&point.p)
    }

    /// Compute ∂H/∂p = ∂T/∂p.
    pub fn grad_p(&self, point: &PhasePoint) -> Vec<f64> {
        self.kinetic.grad(&point.p)
    }

    /// Compute -∂H/∂q = -∂V/∂q.
    pub fn neg_grad_q(&self, point: &PhasePoint) -> Vec<f64> {
        self.potential.neg_grad(&point.q)
    }

    /// Hamilton's equations: returns (dq/dt, dp/dt).
    pub fn equations_of_motion(&self, point: &PhasePoint) -> (Vec<f64>, Vec<f64>) {
        (self.grad_p(point), self.neg_grad_q(point))
    }

    /// Verify energy conservation along a trajectory.
    /// Returns max relative energy deviation from initial.
    pub fn verify_energy_conservation(&self, trajectory: &[PhasePoint]) -> f64 {
        if trajectory.len() < 2 {
            return 0.0;
        }
        let e0 = self.energy(&trajectory[0]);
        if e0.abs() < 1e-15 {
            return trajectory.iter().map(|p| self.energy(p).abs()).fold(0.0_f64, f64::max);
        }
        trajectory
            .iter()
            .map(|p| ((self.energy(p) - e0) / e0).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Free-particle Hamiltonian: H = Σ pᵢ²/(2m).
    pub fn free(masses: &[f64]) -> Self {
        Self::new(Potential::Zero, Kinetic::Standard { masses: masses.to_vec() })
    }

    /// Harmonic oscillator: H = Σ pᵢ²/(2m) + Σ ½k qᵢ².
    pub fn harmonic(masses: &[f64], stiffnesses: &[f64]) -> Self {
        Self::new(
            Potential::Harmonic { stiffnesses: stiffnesses.to_vec() },
            Kinetic::Standard { masses: masses.to_vec() },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_energy() {
        let h = Hamiltonian::harmonic(&[1.0, 1.0], &[1.0, 1.0]);
        let pt = PhasePoint::new(vec![1.0, 0.0], vec![0.0, 1.0]);
        assert!((h.energy(&pt) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_equations_of_motion_harmonic() {
        let h = Hamiltonian::harmonic(&[1.0, 1.0], &[1.0, 1.0]);
        let pt = PhasePoint::new(vec![1.0, 0.0], vec![0.0, 1.0]);
        let (dqdt, dpdt) = h.equations_of_motion(&pt);
        assert!((dqdt[0] - 0.0).abs() < 1e-5);
        assert!((dqdt[1] - 1.0).abs() < 1e-5);
        assert!((dpdt[0] - (-1.0)).abs() < 1e-5);
        assert!((dpdt[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_free_particle_zero_force() {
        let h = Hamiltonian::free(&[1.0, 1.0]);
        let pt = PhasePoint::new(vec![0.0, 0.0], vec![3.0, 4.0]);
        let (_, dpdt) = h.equations_of_motion(&pt);
        for dp in &dpdt {
            assert!(dp.abs() < 1e-6);
        }
    }

    #[test]
    fn test_grad_p_accuracy() {
        let h = Hamiltonian::free(&[1.0, 1.0, 1.0]);
        let pt = PhasePoint::new(vec![0.0; 3], vec![2.0, 3.0, 5.0]);
        let grad = h.grad_p(&pt);
        assert!((grad[0] - 2.0).abs() < 1e-5);
        assert!((grad[1] - 3.0).abs() < 1e-5);
        assert!((grad[2] - 5.0).abs() < 1e-5);
    }
}
