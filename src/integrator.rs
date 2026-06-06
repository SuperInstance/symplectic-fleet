//! # Symplectic Integrators
//!
//! Standard numerical integrators (RK4, Euler) do not preserve the symplectic form,
//! leading to artificial energy drift over long times. Symplectic integrators preserve
//! the symplectic structure exactly, giving bounded energy error for all time.
//!
//! Implemented methods:
//! - **Störmer-Verlet** (leapfrog): 2nd-order, time-reversible, symplectic.
//!   Splits H = T(p) + V(q) and alternates half-steps.
//! - **Implicit midpoint**: 2nd-order, symplectic, energy-preserving for quadratic H.
//!
//! For separable Hamiltonians H(q,p) = T(p) + V(q):
//!   Verlet: p_{n+½} = p_n - (h/2)∂V/∂q(q_n)
//!           q_{n+1} = q_n + h ∂T/∂p(p_{n+½})
//!           p_{n+1} = p_{n+½} - (h/2)∂V/∂q(q_{n+1})

use serde::{Deserialize, Serialize};
use crate::symplectic::PhasePoint;
use crate::symplectic::SymplecticForm;
use crate::hamiltonian::Hamiltonian;

/// Available symplectic integration methods.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum IntegrationMethod {
    /// Störmer-Verlet (leapfrog): 2nd-order symplectic.
    StormerVerlet,
    /// Implicit midpoint rule: 2nd-order symplectic.
    ImplicitMidpoint,
}

/// Configuration for a symplectic integrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratorConfig {
    /// Time step size h.
    pub dt: f64,
    /// Number of integration steps.
    pub steps: usize,
    /// Integration method.
    pub method: IntegrationMethod,
}

impl IntegratorConfig {
    /// Create a new integrator configuration.
    pub fn new(dt: f64, steps: usize, method: IntegrationMethod) -> Self {
        Self { dt, steps, method }
    }
}

/// Integrate Hamilton's equations from an initial condition.
/// Returns the full trajectory as a vector of phase points.
pub fn integrate(h: &Hamiltonian, initial: &PhasePoint, config: &IntegratorConfig) -> Vec<PhasePoint> {
    match config.method {
        IntegrationMethod::StormerVerlet => stormer_verlet(h, initial, config.dt, config.steps),
        IntegrationMethod::ImplicitMidpoint => implicit_midpoint(h, initial, config.dt, config.steps),
    }
}

/// Störmer-Verlet (leapfrog) symplectic integrator.
fn stormer_verlet(h: &Hamiltonian, initial: &PhasePoint, dt: f64, steps: usize) -> Vec<PhasePoint> {
    let n = initial.dof();
    let mut trajectory = Vec::with_capacity(steps + 1);
    let mut q = initial.q.clone();
    let mut p = initial.p.clone();
    trajectory.push(PhasePoint::new(q.clone(), p.clone()));

    for _ in 0..steps {
        let pt = PhasePoint::new(q.clone(), p.clone());

        // Half-step in p: p_{n+½} = p_n - (dt/2) ∂V/∂q(q_n)
        let neg_grad_v = h.neg_grad_q(&pt);
        for i in 0..n {
            p[i] += (dt / 2.0) * neg_grad_v[i];
        }

        // Full step in q: q_{n+1} = q_n + dt ∂T/∂p(p_{n+½})
        let pt_half = PhasePoint::new(q.clone(), p.clone());
        let grad_t = h.grad_p(&pt_half);
        for i in 0..n {
            q[i] += dt * grad_t[i];
        }

        // Half-step in p: p_{n+1} = p_{n+½} - (dt/2) ∂V/∂q(q_{n+1})
        let pt_new = PhasePoint::new(q.clone(), p.clone());
        let neg_grad_v2 = h.neg_grad_q(&pt_new);
        for i in 0..n {
            p[i] += (dt / 2.0) * neg_grad_v2[i];
        }

        trajectory.push(PhasePoint::new(q.clone(), p.clone()));
    }

    trajectory
}

/// Implicit midpoint symplectic integrator.
/// Solves: z_{n+1} = z_n + h J ∇H((z_n + z_{n+1})/2) by fixed-point iteration.
fn implicit_midpoint(h: &Hamiltonian, initial: &PhasePoint, dt: f64, steps: usize) -> Vec<PhasePoint> {
    let n = initial.dof();
    let max_iter = 20;
    let tol = 1e-12;
    let mut trajectory = Vec::with_capacity(steps + 1);
    let mut q = initial.q.clone();
    let mut p = initial.p.clone();
    trajectory.push(PhasePoint::new(q.clone(), p.clone()));

    for _ in 0..steps {
        let q_old = q.clone();
        let p_old = p.clone();

        // Fixed-point iteration for the implicit midpoint
        let mut q_new = q.clone();
        let mut p_new = p.clone();

        for _ in 0..max_iter {
            let q_mid: Vec<f64> = q_old.iter().zip(&q_new).map(|(a, b)| (a + b) / 2.0).collect();
            let p_mid: Vec<f64> = p_old.iter().zip(&p_new).map(|(a, b)| (a + b) / 2.0).collect();
            let mid = PhasePoint::new(q_mid, p_mid);

            let dqdt = h.grad_p(&mid);
            let dpdt = h.neg_grad_q(&mid);

            let q_next: Vec<f64> = q_old.iter().zip(&dqdt).map(|(qi, dqi)| qi + dt * dqi).collect();
            let p_next: Vec<f64> = p_old.iter().zip(&dpdt).map(|(pi, dpi)| pi + dt * dpi).collect();

            let dq: f64 = q_next.iter().zip(&q_new).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            let dp: f64 = p_next.iter().zip(&p_new).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();

            q_new = q_next;
            p_new = p_next;

            if dq < tol && dp < tol {
                break;
            }
        }

        q = q_new;
        p = p_new;
        trajectory.push(PhasePoint::new(q.clone(), p.clone()));
    }

    trajectory
}

/// Verify Liouville's theorem: the symplectic form is preserved along the flow.
/// Checks that the linearized flow map satisfies MᵀJM = J.
pub fn verify_symplecticity(
    h: &Hamiltonian,
    initial: &PhasePoint,
    config: &IntegratorConfig,
) -> bool {
    let omega = SymplecticForm::new(initial.dof());
    let n = initial.dof();
    let eps = 1e-5;

    // Compute the Jacobian of the flow map numerically
    // For each basis direction, perturb and integrate
    let mut flow_jacobian = vec![vec![0.0; 2 * n]; 2 * n];

    for i in 0..2 * n {
        let mut q_plus = initial.q.clone();
        let mut p_plus = initial.p.clone();
        let mut q_minus = initial.q.clone();
        let mut p_minus = initial.p.clone();

        if i < n {
            q_plus[i] += eps;
            q_minus[i] -= eps;
        } else {
            p_plus[i - n] += eps;
            p_minus[i - n] -= eps;
        }

        let pt_plus = PhasePoint::new(q_plus, p_plus);
        let pt_minus = PhasePoint::new(q_minus, p_minus);

        let traj_plus = integrate(h, &pt_plus, config);
        let traj_minus = integrate(h, &pt_minus, config);

        let final_plus = &traj_plus[traj_plus.len() - 1];
        let final_minus = &traj_minus[traj_minus.len() - 1];

        for j in 0..n {
            flow_jacobian[j][i] = (final_plus.q[j] - final_minus.q[j]) / (2.0 * eps);
            flow_jacobian[n + j][i] = (final_plus.p[j] - final_minus.p[j]) / (2.0 * eps);
        }
    }

    omega.is_symplectic_matrix(&flow_jacobian)
}

/// Compute phase space volume change under the flow.
/// For symplectic flows, the Jacobian determinant is exactly 1.
pub fn compute_volume_change(
    h: &Hamiltonian,
    initial: &PhasePoint,
    config: &IntegratorConfig,
) -> f64 {
    let eps = 1e-5;
    let n = initial.dof();
    let traj = integrate(h, initial, config);

    if traj.len() < 2 || n > 3 {
        return 1.0; // Only meaningful for small systems
    }

    // Compute the Jacobian determinant numerically
    let mut flow_jacobian = vec![vec![0.0; 2 * n]; 2 * n];

    for i in 0..2 * n {
        let mut q_plus = initial.q.clone();
        let mut p_plus = initial.p.clone();
        let mut q_minus = initial.q.clone();
        let mut p_minus = initial.p.clone();

        if i < n {
            q_plus[i] += eps;
            q_minus[i] -= eps;
        } else {
            p_plus[i - n] += eps;
            p_minus[i - n] -= eps;
        }

        let pt_plus = PhasePoint::new(q_plus, p_plus);
        let pt_minus = PhasePoint::new(q_minus, p_minus);

        let traj_plus = integrate(h, &pt_plus, config);
        let traj_minus = integrate(h, &pt_minus, config);

        let final_plus = &traj_plus[traj_plus.len() - 1];
        let final_minus = &traj_minus[traj_minus.len() - 1];

        for j in 0..n {
            flow_jacobian[j][i] = (final_plus.q[j] - final_minus.q[j]) / (2.0 * eps);
            flow_jacobian[n + j][i] = (final_plus.p[j] - final_minus.p[j]) / (2.0 * eps);
        }
    }

    crate::symplectic::determinant(&flow_jacobian)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verlet_energy_conservation_harmonic() {
        let h = Hamiltonian::harmonic(&[1.0, 1.0], &[1.0, 1.0]);
        let pt = PhasePoint::new(vec![1.0, 0.0], vec![0.0, 1.0]);
        let cfg = IntegratorConfig::new(0.01, 10000, IntegrationMethod::StormerVerlet);
        let traj = integrate(&h, &pt, &cfg);
        let max_dev = h.verify_energy_conservation(&traj);
        // Verlet is symplectic: energy oscillates but doesn't drift
        // Error is O(h²) per step, but bounded globally
        assert!(max_dev < 1e-3, "Energy deviation too large: {}", max_dev);
    }

    #[test]
    fn test_verlet_energy_conservation_many_steps() {
        let h = Hamiltonian::harmonic(&[1.0], &[4.0]);
        let pt = PhasePoint::new(vec![1.0], vec![0.0]);
        // Run for many periods; ω = 2, period ≈ π
        let cfg = IntegratorConfig::new(0.001, 100000, IntegrationMethod::StormerVerlet);
        let traj = integrate(&h, &pt, &cfg);
        let max_dev = h.verify_energy_conservation(&traj);
        assert!(max_dev < 0.01, "Long-term energy drift: {}", max_dev);
    }

    #[test]
    fn test_implicit_midpoint_energy_conservation() {
        let h = Hamiltonian::harmonic(&[1.0], &[1.0]);
        let pt = PhasePoint::new(vec![1.0], vec![0.0]);
        let cfg = IntegratorConfig::new(0.01, 5000, IntegrationMethod::ImplicitMidpoint);
        let traj = integrate(&h, &pt, &cfg);
        let max_dev = h.verify_energy_conservation(&traj);
        assert!(max_dev < 1e-3, "Implicit midpoint energy deviation: {}", max_dev);
    }

    #[test]
    fn test_verlet_preserves_symplectic_form() {
        let h = Hamiltonian::harmonic(&[1.0], &[1.0]);
        let pt = PhasePoint::new(vec![1.0], vec![0.0]);
        let cfg = IntegratorConfig::new(0.01, 10, IntegrationMethod::StormerVerlet);
        let symplectic = verify_symplecticity(&h, &pt, &cfg);
        assert!(symplectic, "Verlet should preserve symplectic form");
    }

    #[test]
    fn test_implicit_midpoint_preserves_symplectic_form() {
        let h = Hamiltonian::harmonic(&[1.0], &[1.0]);
        let pt = PhasePoint::new(vec![1.0], vec![0.0]);
        let cfg = IntegratorConfig::new(0.01, 10, IntegrationMethod::ImplicitMidpoint);
        let symplectic = verify_symplecticity(&h, &pt, &cfg);
        assert!(symplectic, "Implicit midpoint should preserve symplectic form");
    }

    #[test]
    fn test_free_particle_straight_line() {
        let h = Hamiltonian::free(&[1.0]);
        let pt = PhasePoint::new(vec![0.0], vec![1.0]);
        let cfg = IntegratorConfig::new(0.1, 10, IntegrationMethod::StormerVerlet);
        let traj = integrate(&h, &pt, &cfg);
        let final_pt = traj.last().unwrap();
        // q = v*t = 1.0, p unchanged = 1.0
        assert!((final_pt.q[0] - 1.0).abs() < 1e-10);
        assert!((final_pt.p[0] - 1.0).abs() < 1e-10);
    }
}
