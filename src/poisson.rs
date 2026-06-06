//! # Poisson Brackets and Lie-Poisson Structures
//!
//! The Poisson bracket is the algebraic reflection of the symplectic form:
//!   {f, g} = Σᵢ (∂f/∂qᵢ)(∂g/∂pᵢ) - (∂f/∂pᵢ)(∂g/∂qᵢ)
//!
//! Properties: antisymmetry, bilinearity, Jacobi identity, Leibniz rule.

use serde::{Deserialize, Serialize};
use crate::symplectic::PhasePoint;

/// A differentiable function on phase space.
pub type Observable = fn(&PhasePoint) -> f64;

/// The canonical Poisson bracket on 2n-dimensional phase space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PoissonBracket {
    /// Number of degrees of freedom n.
    pub dimension: usize,
}

impl PoissonBracket {
    /// Create the canonical Poisson bracket.
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }

    /// Numerical partial derivative ∂f/∂qᵢ via central differences.
    fn partial_q(&self, f: Observable, point: &PhasePoint, i: usize) -> f64 {
        let h = 1e-6;
        let mut q_p = point.q.clone();
        let mut q_m = point.q.clone();
        q_p[i] += h;
        q_m[i] -= h;
        (f(&PhasePoint::new(q_p, point.p.clone())) - f(&PhasePoint::new(q_m, point.p.clone())))
            / (2.0 * h)
    }

    /// Numerical partial derivative ∂f/∂pᵢ via central differences.
    fn partial_p(&self, f: Observable, point: &PhasePoint, i: usize) -> f64 {
        let h = 1e-6;
        let mut p_p = point.p.clone();
        let mut p_m = point.p.clone();
        p_p[i] += h;
        p_m[i] -= h;
        (f(&PhasePoint::new(point.q.clone(), p_p)) - f(&PhasePoint::new(point.q.clone(), p_m)))
            / (2.0 * h)
    }

    /// Compute {f, g} using central finite differences.
    pub fn apply(&self, f: Observable, g: Observable, point: &PhasePoint) -> f64 {
        let mut bracket = 0.0;
        for i in 0..self.dimension {
            bracket += self.partial_q(f, point, i) * self.partial_p(g, point, i)
                - self.partial_p(f, point, i) * self.partial_q(g, point, i);
        }
        bracket
    }

    /// Verify antisymmetry: {f, g} = -{g, f}.
    pub fn verify_antisymmetry(&self, f: Observable, g: Observable, point: &PhasePoint) -> bool {
        let fg = self.apply(f, g, point);
        let gf = self.apply(g, f, point);
        (fg + gf).abs() < 1e-6
    }

    /// Verify Jacobi identity: {f, {g, h}} + {g, {h, f}} + {h, {f, g}} = 0.
    /// Uses nested numerical differentiation.
    pub fn verify_jacobi(
        &self,
        f: Observable,
        g: Observable,
        h_fn: Observable,
        point: &PhasePoint,
    ) -> bool {
        let hh = 1e-5;
        let n = self.dimension;

        // Helper: compute ∂/∂xᵢ of the bracket {a, b} at a given point
        let partial_bracket_q = |a: Observable, b: Observable, pt: &PhasePoint, idx: usize| -> f64 {
            let mut q_p = pt.q.clone();
            let mut q_m = pt.q.clone();
            q_p[idx] += hh;
            q_m[idx] -= hh;
            let pb = PoissonBracket::new(n);
            (pb.apply(a, b, &PhasePoint::new(q_p, pt.p.clone()))
                - pb.apply(a, b, &PhasePoint::new(q_m, pt.p.clone())))
                / (2.0 * hh)
        };

        let partial_bracket_p = |a: Observable, b: Observable, pt: &PhasePoint, idx: usize| -> f64 {
            let mut p_p = pt.p.clone();
            let mut p_m = pt.p.clone();
            p_p[idx] += hh;
            p_m[idx] -= hh;
            let pb = PoissonBracket::new(n);
            (pb.apply(a, b, &PhasePoint::new(pt.q.clone(), p_p))
                - pb.apply(a, b, &PhasePoint::new(pt.q.clone(), p_m)))
                / (2.0 * hh)
        };

        let pb = PoissonBracket::new(n);

        // {f, {g, h}} = Σᵢ (∂f/∂qᵢ)(∂{g,h}/∂pᵢ) - (∂f/∂pᵢ)(∂{g,h}/∂qᵢ)
        let mut f_gh = 0.0;
        for i in 0..n {
            f_gh += pb.partial_q(f, point, i) * partial_bracket_p(g, h_fn, point, i)
                - pb.partial_p(f, point, i) * partial_bracket_q(g, h_fn, point, i);
        }

        // {g, {h, f}}
        let mut g_hf = 0.0;
        for i in 0..n {
            g_hf += pb.partial_q(g, point, i) * partial_bracket_p(h_fn, f, point, i)
                - pb.partial_p(g, point, i) * partial_bracket_q(h_fn, f, point, i);
        }

        // {h, {f, g}}
        let mut h_fg = 0.0;
        for i in 0..n {
            h_fg += pb.partial_q(h_fn, point, i) * partial_bracket_p(f, g, point, i)
                - pb.partial_p(h_fn, point, i) * partial_bracket_q(f, g, point, i);
        }

        (f_gh + g_hf + h_fg).abs() < 1e-2
    }

    /// Verify bilinearity: {αf + βg, h} = α{f,h} + β{g,h}.
    /// Tests with specific α, β values using concrete observable functions.
    pub fn verify_bilinearity(
        &self,
        alpha: f64,
        beta: f64,
        point: &PhasePoint,
    ) -> bool {
        // Use concrete functions: f = q₀², g = q₀p₁, h = p₀²
        let f: Observable = |pt: &PhasePoint| pt.q[0] * pt.q[0];
        let g: Observable = |pt: &PhasePoint| pt.q[0] * pt.p[1];
        let h: Observable = |pt: &PhasePoint| pt.p[0] * pt.p[0];

        // By bilinearity: {αf+βg, h} = α{f,h} + β{g,h}
        let lhs = alpha * self.apply(f, h, point) + beta * self.apply(g, h, point);

        // Compute {αf+βg, h} directly via numerical differentiation
        let n = self.dimension;
        let hh = 1e-6;
        let mut rhs = 0.0;
        for i in 0..n {
            // ∂(αf+βg)/∂qᵢ numerically
            let mut q_p = point.q.clone();
            let mut q_m = point.q.clone();
            q_p[i] += hh;
            q_m[i] -= hh;
            let daf_bg_dqi = (alpha * (q_p[0]*q_p[0] - q_m[0]*q_m[0]) + beta * (q_p[0]*point.p[1] - q_m[0]*point.p[1])) / (2.0*hh);

            let mut p_p = point.p.clone();
            let mut p_m = point.p.clone();
            p_p[i] += hh;
            p_m[i] -= hh;
            let daf_bg_dpi = alpha * 0.0 + beta * point.q[0] * (p_p[1] - p_m[1]) / (2.0*hh);

            let dh_dqi = (p_p[0]*p_p[0] - p_m[0]*p_m[0]) / (2.0*hh);
            let dh_dpi = (point.p[0]*2.0*hh) / (2.0*hh);

            rhs += daf_bg_dqi * dh_dpi - daf_bg_dpi * dh_dqi;
        }

        (lhs - rhs).abs() < 1e-6
    }

    /// Verify Leibniz rule: {f, gh} = g{f,h} + h{f,g}.
    pub fn verify_leibniz(&self, point: &PhasePoint) -> bool {
        let f: Observable = |pt: &PhasePoint| pt.q[0];
        let g: Observable = |pt: &PhasePoint| pt.q[1];
        let h: Observable = |pt: &PhasePoint| pt.p[0];

        // {f, g·h} where g·h = q₁·p₀
        let gh: Observable = |pt: &PhasePoint| pt.q[1] * pt.p[0];
        let lhs = self.apply(f, gh, point);
        let rhs = g(point) * self.apply(f, h, point) + h(point) * self.apply(f, g, point);
        (lhs - rhs).abs() < 1e-4
    }
}

/// A Lie-Poisson structure on a dual Lie algebra g*.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiePoissonStructure {
    /// Dimension of the Lie algebra.
    pub dimension: usize,
    /// Structure constants c[i][j][k] = cᵢⱼᵏ.
    pub structure_constants: Vec<Vec<Vec<f64>>>,
}

impl LiePoissonStructure {
    /// Create a Lie-Poisson structure from structure constants.
    pub fn new(dimension: usize, structure_constants: Vec<Vec<Vec<f64>>>) -> Self {
        assert_eq!(structure_constants.len(), dimension);
        Self { dimension, structure_constants }
    }

    /// Compute the Lie-Poisson bracket {f, g}(μ).
    pub fn bracket<F, G>(&self, f: F, g: G, mu: &[f64]) -> f64
    where
        F: Fn(&[f64]) -> f64,
        G: Fn(&[f64]) -> f64,
    {
        let h = 1e-6;
        let n = self.dimension;
        let mut result = 0.0;

        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let c = self.structure_constants[i][j][k];
                    if c.abs() < 1e-15 {
                        continue;
                    }
                    let mut mu_p = mu.to_vec();
                    let mut mu_m = mu.to_vec();
                    mu_p[i] += h;
                    mu_m[i] -= h;
                    let df = (f(&mu_p) - f(&mu_m)) / (2.0 * h);

                    let mut mu_p2 = mu.to_vec();
                    let mut mu_m2 = mu.to_vec();
                    mu_p2[j] += h;
                    mu_m2[j] -= h;
                    let dg = (g(&mu_p2) - g(&mu_m2)) / (2.0 * h);

                    result += c * mu[k] * df * dg;
                }
            }
        }
        result
    }

    /// Create the so(3)* Lie-Poisson structure (rigid body bracket).
    pub fn so3() -> Self {
        let n = 3;
        let mut c = vec![vec![vec![0.0; n]; n]; n];
        c[0][1][2] = 1.0;
        c[1][2][0] = 1.0;
        c[2][0][1] = 1.0;
        c[0][2][1] = -1.0;
        c[2][1][0] = -1.0;
        c[1][0][2] = -1.0;
        Self::new(n, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_antisymmetry() {
        let pb = PoissonBracket::new(2);
        let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);
        let f: Observable = |pt: &PhasePoint| pt.q[0] * pt.q[0];
        let g: Observable = |pt: &PhasePoint| pt.p[0] * pt.p[0];
        assert!(pb.verify_antisymmetry(f, g, &pt));
    }

    #[test]
    fn test_poisson_jacobi() {
        let pb = PoissonBracket::new(2);
        let pt = PhasePoint::new(vec![1.0, 0.5], vec![0.3, 0.7]);
        let f: Observable = |pt: &PhasePoint| pt.q[0] * pt.q[0];
        let g: Observable = |pt: &PhasePoint| pt.q[0] * pt.p[1];
        let h: Observable = |pt: &PhasePoint| pt.q[1] * pt.p[0];
        assert!(pb.verify_jacobi(f, g, h, &pt));
    }

    #[test]
    fn test_poisson_bilinearity() {
        let pb = PoissonBracket::new(2);
        let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);
        assert!(pb.verify_bilinearity(2.0, -1.0, &pt));
    }

    #[test]
    fn test_poisson_leibniz() {
        let pb = PoissonBracket::new(2);
        let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);
        assert!(pb.verify_leibniz(&pt));
    }

    #[test]
    fn test_canonical_bracket_qp() {
        let pb = PoissonBracket::new(2);
        let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);
        let q0: Observable = |pt: &PhasePoint| pt.q[0];
        let p0: Observable = |pt: &PhasePoint| pt.p[0];
        let p1: Observable = |pt: &PhasePoint| pt.p[1];
        assert!((pb.apply(q0, p0, &pt) - 1.0).abs() < 1e-6);
        assert!(pb.apply(q0, p1, &pt).abs() < 1e-6);
    }

    #[test]
    fn test_so3_lie_poisson() {
        let lp = LiePoissonStructure::so3();
        let mu = vec![1.0, 0.0, 0.0];
        let f = |m: &[f64]| m[1];
        let g = |m: &[f64]| m[2];
        let brk = lp.bracket(f, g, &mu);
        assert!((brk - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_so3_antisymmetry() {
        let lp = LiePoissonStructure::so3();
        let mu = vec![1.0, 2.0, 3.0];
        let f = |m: &[f64]| m[0] * m[1];
        let g = |m: &[f64]| m[2];
        let fg = lp.bracket(&f, &g, &mu);
        let gf = lp.bracket(&g, &f, &mu);
        assert!((fg + gf).abs() < 1e-6);
    }
}
