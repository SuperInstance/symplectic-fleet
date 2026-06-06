//! # Symplectic Form and Linear Algebra
//!
//! The symplectic form ω is the fundamental structure of a symplectic manifold.
//! In Darboux coordinates, ω = Σ dqᵢ ∧ dpᵢ, represented as the block matrix J = [[0, I], [-I, 0]].
//!
//! Properties verified:
//! - Antisymmetry: ω(u, v) = -ω(v, u)
//! - Non-degeneracy: ω(u, v) = 0 for all v ⟹ u = 0
//! - Bilinearity
//! - Closedness (dω = 0, automatic in the canonical case)

use serde::{Deserialize, Serialize};

/// A point in 2n-dimensional phase space (q, p).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhasePoint {
    pub q: Vec<f64>,
    pub p: Vec<f64>,
}

impl PhasePoint {
    /// Create a new phase point from configuration q and momentum p.
    pub fn new(q: Vec<f64>, p: Vec<f64>) -> Self {
        assert_eq!(q.len(), p.len(), "q and p must have the same dimension");
        Self { q, p }
    }

    /// The number of degrees of freedom (half the phase space dimension).
    pub fn dof(&self) -> usize {
        self.q.len()
    }

    /// Total phase space dimension (2n).
    pub fn dim(&self) -> usize {
        2 * self.dof()
    }

    /// Concatenate into a single 2n-vector: [q₁,...,qₙ, p₁,...,pₙ].
    pub fn to_vec(&self) -> Vec<f64> {
        self.q.iter().chain(self.p.iter()).copied().collect()
    }

    /// Construct from a 2n-vector.
    pub fn from_vec(v: &[f64]) -> Self {
        assert!(v.len() % 2 == 0, "vector must have even length");
        let n = v.len() / 2;
        Self {
            q: v[..n].to_vec(),
            p: v[n..].to_vec(),
        }
    }

    /// Add two phase points component-wise.
    pub fn add(&self, other: &PhasePoint) -> PhasePoint {
        PhasePoint {
            q: self.q.iter().zip(&other.q).map(|(a, b)| a + b).collect(),
            p: self.p.iter().zip(&other.p).map(|(a, b)| a + b).collect(),
        }
    }

    /// Scale a phase point by a scalar.
    pub fn scale(&self, s: f64) -> PhasePoint {
        PhasePoint {
            q: self.q.iter().map(|x| x * s).collect(),
            p: self.p.iter().map(|x| x * s).collect(),
        }
    }

    /// Euclidean inner product in phase space.
    pub fn dot(&self, other: &PhasePoint) -> f64 {
        self.q
            .iter()
            .zip(&other.q)
            .chain(self.p.iter().zip(&other.p))
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Euclidean norm.
    pub fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }
}

/// The canonical symplectic form ω on a 2n-dimensional phase space.
///
/// In Darboux coordinates: ω(u, v) = uᵀJv where J = [[0, Iₙ], [-Iₙ, 0]].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SymplecticForm {
    /// Number of degrees of freedom n (phase space is 2n-dimensional).
    pub dimension: usize,
}

impl SymplecticForm {
    /// Create the canonical symplectic form on 2n-dimensional phase space.
    pub fn new(dimension: usize) -> Self {
        assert!(dimension > 0, "dimension must be positive");
        Self { dimension }
    }

    /// Apply ω to two phase vectors: ω(u, v) = uᵀJv.
    ///
    /// In component form: Σᵢ (u_qᵢ v_pᵢ - u_pᵢ v_qᵢ).
    pub fn apply(&self, u: &PhasePoint, v: &PhasePoint) -> f64 {
        assert_eq!(u.dof(), self.dimension);
        assert_eq!(v.dof(), self.dimension);
        let mut result = 0.0;
        for i in 0..self.dimension {
            result += u.q[i] * v.p[i] - u.p[i] * v.q[i];
        }
        result
    }

    /// Verify antisymmetry: ω(u, v) = -ω(v, u).
    pub fn verify_antisymmetry(&self, u: &PhasePoint, v: &PhasePoint) -> bool {
        let uv = self.apply(u, v);
        let vu = self.apply(v, u);
        (uv + vu).abs() < 1e-12
    }

    /// Verify non-degeneracy: if ω(u, v) = 0 for all v, then u = 0.
    /// We check against the standard basis vectors.
    pub fn verify_nondegeneracy(&self, u: &PhasePoint) -> bool {
        if u.norm() < 1e-12 {
            return true; // zero vector trivially satisfies the condition
        }
        // For each basis vector eᵢ, check that ω(u, eᵢ) is nonzero for some i
        let mut all_zero = true;
        for i in 0..self.dimension {
            // Basis vector for qᵢ
            let mut eq = vec![0.0; self.dimension];
            let ep = vec![0.0; self.dimension];
            eq[i] = 1.0;
            let e_q = PhasePoint::new(eq, ep);
            if self.apply(u, &e_q).abs() > 1e-12 {
                all_zero = false;
                break;
            }
            // Basis vector for pᵢ
            let eq2 = vec![0.0; self.dimension];
            let mut ep2 = vec![0.0; self.dimension];
            ep2[i] = 1.0;
            let e_p = PhasePoint::new(eq2, ep2);
            if self.apply(u, &e_p).abs() > 1e-12 {
                all_zero = false;
                break;
            }
        }
        !all_zero
    }

    /// Verify bilinearity: ω(αu + βv, w) = αω(u,w) + βω(v,w).
    pub fn verify_bilinearity(
        &self,
        u: &PhasePoint,
        v: &PhasePoint,
        w: &PhasePoint,
        alpha: f64,
        beta: f64,
    ) -> bool {
        let lhs = self.apply(&u.scale(alpha).add(&v.scale(beta)), w);
        let rhs = alpha * self.apply(u, w) + beta * self.apply(v, w);
        (lhs - rhs).abs() < 1e-10
    }

    /// Build the full 2n × 2n matrix representation J of the symplectic form.
    pub fn matrix(&self) -> Vec<Vec<f64>> {
        let n = self.dimension;
        let d = 2 * n;
        let mut mat = vec![vec![0.0; d]; d];
        for i in 0..n {
            mat[i][n + i] = 1.0;
            mat[n + i][i] = -1.0;
        }
        mat
    }

    /// Check if a 2n × 2n matrix M satisfies MᵀJM = J (symplectic condition).
    pub fn is_symplectic_matrix(&self, m: &[Vec<f64>]) -> bool {
        let j = self.matrix();
        let d = 2 * self.dimension;
        let mtjm = mat_mul(&mat_transpose(m), &mat_mul(&j, m));
        for i in 0..d {
            for j_idx in 0..d {
                if (mtjm[i][j_idx] - j[i][j_idx]).abs() > 1e-6 {
                    return false;
                }
            }
        }
        true
    }
}

/// Multiply two matrices.
pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b[0].len();
    let inner = b.len();
    let mut c = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            for k in 0..inner {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Transpose a matrix.
pub fn mat_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return vec![];
    }
    let rows = a.len();
    let cols = a[0].len();
    let mut t = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Gaussian elimination solver: solve Ax = b.
/// Uses partial pivoting for numerical stability.
pub fn gaussian_elimination(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        // Swap rows
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        assert!(pivot.abs() > 1e-14, "Singular matrix in Gaussian elimination");

        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    x
}

/// Compute the determinant using Gaussian elimination.
pub fn determinant(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut m = a.to_vec();
    let mut det = 1.0;

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = m[col][col].abs();
        for row in (col + 1)..n {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }
        if max_row != col {
            m.swap(col, max_row);
            det *= -1.0;
        }
        let pivot = m[col][col];
        if pivot.abs() < 1e-14 {
            return 0.0;
        }
        det *= pivot;
        for row in (col + 1)..n {
            let factor = m[row][col] / pivot;
            for j in (col + 1)..n {
                m[row][j] -= factor * m[col][j];
            }
        }
    }
    det
}

/// Compute the inverse of a matrix using Gaussian elimination.
pub fn matrix_inverse(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Forward elimination
    for col in 0..n {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        assert!(pivot.abs() > 1e-14, "Singular matrix");

        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    inv
}

/// Compute the Pfaffian of a 2n × 2n antisymmetric matrix.
/// For the canonical symplectic matrix J, pf(J) = 1.
pub fn pfaffian(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    assert_eq!(n % 2, 0, "Pfaffian requires even-dimensional matrix");
    let m = n / 2;

    if m == 0 {
        return 1.0;
    }
    if m == 1 {
        return a[0][1];
    }

    // Recursive expansion along first row
    let mut pf = 0.0;
    let mut sign = 1.0;
    for j in 1..n {
        if a[0][j].abs() < 1e-15 {
            continue;
        }
        // Build (n-2) × (n-2) submatrix excluding rows/cols 0 and j
        let mut sub = vec![];
        for i in 1..n {
            if i == j {
                continue;
            }
            let mut row = vec![];
            for k in 1..n {
                if k == j {
                    continue;
                }
                row.push(a[i][k]);
            }
            sub.push(row);
        }
        pf += sign * a[0][j] * pfaffian(&sub);
        sign *= -1.0;
    }
    pf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symplectic_form_antisymmetry() {
        let omega = SymplecticForm::new(3);
        let u = PhasePoint::new(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);
        let v = PhasePoint::new(vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]);
        assert!(omega.verify_antisymmetry(&u, &v));
    }

    #[test]
    fn test_symplectic_form_nondegeneracy() {
        let omega = SymplecticForm::new(2);
        let u = PhasePoint::new(vec![1.0, 0.0], vec![0.0, 0.0]);
        assert!(omega.verify_nondegeneracy(&u));
    }

    #[test]
    fn test_symplectic_form_bilinearity() {
        let omega = SymplecticForm::new(2);
        let u = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);
        let v = PhasePoint::new(vec![5.0, 6.0], vec![7.0, 8.0]);
        let w = PhasePoint::new(vec![9.0, 10.0], vec![11.0, 12.0]);
        assert!(omega.verify_bilinearity(&u, &v, &w, 2.5, -1.3));
    }

    #[test]
    fn test_symplectic_form_value() {
        let omega = SymplecticForm::new(2);
        let u = PhasePoint::new(vec![1.0, 0.0], vec![0.0, 1.0]);
        let v = PhasePoint::new(vec![0.0, 1.0], vec![1.0, 0.0]);
        // ω(u,v) = 1*1 - 0*0 + 0*0 - 1*1 = 1 - 1 = 0
        assert!((omega.apply(&u, &v)).abs() < 1e-12);
    }

    #[test]
    fn test_symplectic_matrix_is_symplectic() {
        let omega = SymplecticForm::new(2);
        let j = omega.matrix();
        assert!(omega.is_symplectic_matrix(&j));
    }

    #[test]
    fn test_gaussian_elimination() {
        let a = vec![vec![2.0, 1.0], vec![5.0, 3.0]];
        let b = vec![4.0, 7.0];
        let x = gaussian_elimination(&a, &b);
        assert!((x[0] - 5.0).abs() < 1e-10);
        assert!((x[1] - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn test_determinant_symplectic_matrix() {
        let omega = SymplecticForm::new(3);
        let j = omega.matrix();
        let det = determinant(&j);
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_inverse_symplectic() {
        let omega = SymplecticForm::new(2);
        let j = omega.matrix();
        let inv = matrix_inverse(&j);
        // J⁻¹ = -J
        for i in 0..4 {
            for k in 0..4 {
                assert!((inv[i][k] - (-j[i][k])).abs() < 1e-10);
            }
        }
    }
}
