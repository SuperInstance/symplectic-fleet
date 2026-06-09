//! # Symplectic Fleet Tutorial
//!
//! A progressive walkthrough of fleet dynamics modeled as a symplectic manifold.
//! The fleet evolves via Hamilton's equations, guaranteeing the symplectic 2-form
//! is preserved. Noether's theorem provides a precise correspondence between
//! continuous symmetries and conserved quantities.
//!
//! ## Lessons
//!
//! 1. Phase space & symplectic form — the geometric foundation
//! 2. Hamiltonian dynamics — energy and equations of motion
//! 3. Symplectic integrators — structure-preserving time evolution
//! 4. Noether's theorem — symmetries and conservation laws
//! 5. Canonical transformations — coordinate changes that preserve structure
//! 6. Poisson brackets — the algebraic structure of classical mechanics

use symplectic_fleet::symplectic::{PhasePoint, SymplecticForm};
use symplectic_fleet::symplectic;
use symplectic_fleet::hamiltonian::{Hamiltonian, Potential, Kinetic};
use symplectic_fleet::integrator::{self, IntegratorConfig, IntegrationMethod};
use symplectic_fleet::noether::{self, NoetherPair};
use symplectic_fleet::canonical::{self, CanonicalTransformation};
use symplectic_fleet::poisson::{PoissonBracket, Observable};

fn main() {
    println!("════════════════════════════════════════════════════════");
    println!("  SYMPLECTIC FLEET TUTORIAL");
    println!("  Fleet Dynamics as a Symplectic Manifold");
    println!("════════════════════════════════════════════════════════\n");

    lesson_1_phase_space();
    lesson_2_hamiltonian();
    lesson_3_integrators();
    lesson_4_noether();
    lesson_5_canonical_transformations();
    lesson_6_poisson_brackets();

    println!("\n✅ Tutorial complete! The fleet evolves in perfect symplectic harmony.");
}

// ─── Lesson 1: Phase Space & Symplectic Form ──────────────────────────────

fn lesson_1_phase_space() {
    println!("━━━ Lesson 1: Phase Space & Symplectic Form ━━━\n");
    println!("Phase space has coordinates (q, p) — configurations and momenta.");
    println!("The symplectic form ω is the fundamental geometric structure.\n");

    // Create a point in 3-DOF phase space
    let point = PhasePoint::new(
        vec![1.0, 2.0, 3.0],  // configuration q
        vec![4.0, 5.0, 6.0],  // momentum p
    );
    println!("  Point in phase space:");
    println!("    q = {:?}", point.q);
    println!("    p = {:?}", point.p);
    println!("    DOF = {}, dim = {}", point.dof(), point.dim());

    // Convert to/from vector representation
    let v = point.to_vec();
    println!("    As vector: {:?}", v);
    let recovered = PhasePoint::from_vec(&v);
    assert_eq!(point, recovered);

    // The canonical symplectic form
    let omega = SymplecticForm::new(3);
    let u = PhasePoint::new(vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]);
    let v2 = PhasePoint::new(vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]);

    let omega_uv = omega.apply(&u, &v2);
    println!("\n  ω(u, v) = {}", omega_uv);
    println!("  (where u = (1,0,0; 0,1,0), v = (0,1,0; 1,0,0))");

    // Verify properties
    let a = PhasePoint::new(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);
    let b = PhasePoint::new(vec![7.0, 8.0, 9.0], vec![10.0, 11.0, 12.0]);
    let c = PhasePoint::new(vec![2.0, 1.0, 0.5], vec![0.3, 0.7, 1.1]);

    println!("\n  Verifying symplectic form properties:");
    println!("    Antisymmetry ω(a,b) = -ω(b,a): {}", omega.verify_antisymmetry(&a, &b));
    println!("    Non-degeneracy:                     {}", omega.verify_nondegeneracy(&a));
    println!("    Bilinearity:                        {}", omega.verify_bilinearity(&a, &b, &c, 2.5, -1.3));

    // The matrix representation J
    let j = omega.matrix();
    println!("\n  Symplectic matrix J ({}×{}):", j.len(), j[0].len());
    for row in &j {
        println!("    {:?}", row.iter().map(|x| format!("{:5.1}", x)).collect::<Vec<_>>());
    }

    // Determinant of J is always 1
    let det = symplectic::determinant(&j);
    println!("  det(J) = {:.1} (always 1 for symplectic form)", det);

    // Pfaffian
    let pf = symplectic::pfaffian(&j);
    println!("  pf(J) = {:.1}", pf);

    println!();
}

// ─── Lesson 2: Hamiltonian Dynamics ──────────────────────────────────────

fn lesson_2_hamiltonian() {
    println!("━━━ Lesson 2: Hamiltonian Dynamics ━━━\n");
    println!("The Hamiltonian H(q,p) = T(p) + V(q) governs fleet evolution.");
    println!("Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q.\n");

    // Harmonic oscillator: H = p²/2 + q²/2
    let h = Hamiltonian::harmonic(&[1.0], &[1.0]);
    let pt = PhasePoint::new(vec![1.0], vec![0.0]);

    println!("  Harmonic oscillator: H = p²/2 + q²/2");
    println!("  Initial state: q=1, p=0");
    println!("  Energy: H = {:.1}", h.energy(&pt));

    let (dqdt, dpdt) = h.equations_of_motion(&pt);
    println!("  dq/dt = {:.1} (velocity = p/m = 0)", dqdt[0]);
    println!("  dp/dt = {:.1} (force = -kq = -1)", dpdt[0]);

    // Free particle
    let h_free = Hamiltonian::free(&[1.0, 1.0]);
    let pt_free = PhasePoint::new(vec![0.0, 0.0], vec![3.0, 4.0]);
    println!("\n  Free particle: H = (p₁² + p₂²)/2");
    println!("  Energy: H = {:.1}", h_free.energy(&pt_free));

    let (dq, dp) = h_free.equations_of_motion(&pt_free);
    println!("  dq/dt = {:?} (constant velocity)", dq);
    println!("  dp/dt = {:?} (zero force)", dp);

    // 2D harmonic oscillator (coupled)
    let h_2d = Hamiltonian::harmonic(&[1.0, 1.0], &[1.0, 4.0]);
    let pt_2d = PhasePoint::new(vec![1.0, 0.0], vec![0.0, 1.0]);
    println!("\n  2D harmonic: H = (p₁² + p₂²)/2 + q₁²/2 + 2q₂²");
    println!("  Energy: H = {:.1}", h_2d.energy(&pt_2d));

    // Shifted harmonic (attractor at center)
    let h_shifted = Hamiltonian::new(
        Potential::ShiftedHarmonic {
            stiffnesses: vec![1.0, 1.0],
            centers: vec![3.0, -1.0],
        },
        Kinetic::UnitMass,
    );
    let pt_shifted = PhasePoint::new(vec![3.0, -1.0], vec![0.0, 0.0]);
    println!("\n  Shifted harmonic at equilibrium (3, -1):");
    println!("  Energy at center: H = {:.1} (should be 0)", h_shifted.energy(&pt_shifted));

    println!();
}

// ─── Lesson 3: Symplectic Integrators ──────────────────────────────────────

fn lesson_3_integrators() {
    println!("━━━ Lesson 3: Symplectic Integrators ━━━\n");
    println!("Standard integrators (RK4, Euler) cause energy drift over time.");
    println!("Symplectic integrators preserve the geometric structure exactly.\n");

    let h = Hamiltonian::harmonic(&[1.0], &[1.0]);
    let initial = PhasePoint::new(vec![1.0], vec![0.0]);

    // Störmer-Verlet (leapfrog)
    let cfg_verlet = IntegratorConfig::new(0.01, 10000, IntegrationMethod::StormerVerlet);
    let traj_verlet = integrator::integrate(&h, &initial, &cfg_verlet);
    let max_dev_verlet = h.verify_energy_conservation(&traj_verlet);

    println!("  Störmer-Verlet (10,000 steps, dt=0.01):");
    println!("    Energy deviation: {:.2e}", max_dev_verlet);
    println!("    (Bounded — no drift!)");

    // Implicit midpoint
    let cfg_midpoint = IntegratorConfig::new(0.01, 5000, IntegrationMethod::ImplicitMidpoint);
    let traj_midpoint = integrator::integrate(&h, &initial, &cfg_midpoint);
    let max_dev_midpoint = h.verify_energy_conservation(&traj_midpoint);

    println!("\n  Implicit midpoint (5,000 steps, dt=0.01):");
    println!("    Energy deviation: {:.2e}", max_dev_midpoint);

    // Long-term behavior — the key advantage
    let cfg_long = IntegratorConfig::new(0.001, 100000, IntegrationMethod::StormerVerlet);
    let traj_long = integrator::integrate(&h, &initial, &cfg_long);
    let max_dev_long = h.verify_energy_conservation(&traj_long);
    println!("\n  Long run (100,000 steps, dt=0.001):");
    println!("    Energy deviation: {:.2e}", max_dev_long);
    println!("    (Still bounded — symplectic magic!)");

    // Verify symplecticity of the flow map
    let cfg_short = IntegratorConfig::new(0.01, 10, IntegrationMethod::StormerVerlet);
    let is_symplectic = integrator::verify_symplecticity(&h, &initial, &cfg_short);
    println!("\n  Flow map is symplectic: {}", is_symplectic);

    // Phase space volume (Liouville's theorem)
    let vol = integrator::compute_volume_change(&h, &initial, &cfg_short);
    println!("  Phase space volume change: {:.6} (should be 1.0)", vol);

    // Show a trajectory
    println!("\n  First 5 steps of Verlet trajectory:");
    let cfg_show = IntegratorConfig::new(0.1, 5, IntegrationMethod::StormerVerlet);
    let traj_show = integrator::integrate(&h, &initial, &cfg_show);
    for (i, pt) in traj_show.iter().enumerate() {
        println!("    step {}: q={:.4}, p={:.4}, H={:.4}",
            i, pt.q[0], pt.p[0], h.energy(pt));
    }

    println!();
}

// ─── Lesson 4: Noether's Theorem ──────────────────────────────────────

fn lesson_4_noether() {
    println!("━━━ Lesson 4: Noether's Theorem ━━━\n");
    println!("Every continuous symmetry ↔ a conserved quantity.");
    println!("Translation → momentum, Rotation → angular momentum.\n");

    // Translation symmetry → linear momentum conservation
    let h_free = Hamiltonian::free(&[1.0, 1.0]);
    let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);

    let trans = noether::translation_symmetry(2, 0);
    println!("  Translation in q₀:");
    println!("    Preserves H: {}", trans.preserves_hamiltonian(&h_free, &pt, 0.5));

    // Compute Noether pair automatically
    let pair = NoetherPair::compute_noether_pair(trans, &h_free);
    println!("    Conserved quantity (p₀): {:.1}", pair.evaluate(&pt));

    // Verify momentum conservation along trajectory
    let cfg = IntegratorConfig::new(0.01, 1000, IntegrationMethod::StormerVerlet);
    let traj = integrator::integrate(&h_free, &pt, &cfg);
    let max_dev = pair.verify_conservation(&traj);
    println!("    Max deviation over 1000 steps: {:.2e}", max_dev);

    // Rotation symmetry → angular momentum conservation
    let h_harm = Hamiltonian::harmonic(&[1.0, 1.0], &[1.0, 1.0]);
    let pt_rot = PhasePoint::new(vec![1.0, 0.0], vec![0.0, 1.0]);

    let rot = noether::rotation_symmetry(2, 0, 1);
    println!("\n  Rotation in (q₀, q₁) plane:");
    println!("    Preserves H: {}", rot.preserves_hamiltonian(&h_harm, &pt_rot, 0.1));

    let angular_momentum = NoetherPair::new(
        "angular_momentum_01",
        |pt: &PhasePoint| PhasePoint::new(vec![-pt.q[1], pt.q[0]], vec![0.0, 0.0]),
        |pt: &PhasePoint| pt.q[0] * pt.p[1] - pt.q[1] * pt.p[0],
    );
    let l0 = angular_momentum.evaluate(&pt_rot);
    println!("    Angular momentum L = q₀p₁ - q₁p₀ = {:.1}", l0);

    let traj_harm = integrator::integrate(&h_harm, &pt_rot, &cfg);
    let l_max_dev = angular_momentum.verify_conservation(&traj_harm);
    println!("    Max deviation: {:.2e}", l_max_dev);

    // Custom Noether pair
    let total_momentum = NoetherPair::new(
        "total_momentum",
        |_pt: &PhasePoint| PhasePoint::new(vec![1.0, 1.0], vec![0.0, 0.0]),
        |pt: &PhasePoint| pt.p.iter().sum(),
    );
    println!("\n  Custom conserved quantity (total momentum):");
    println!("    Value: {:.1}", total_momentum.evaluate(&pt));
    let max_dev_custom = total_momentum.verify_conservation(&traj);
    println!("    Max deviation: {:.2e}", max_dev_custom);

    println!();
}

// ─── Lesson 5: Canonical Transformations ──────────────────────────────

fn lesson_5_canonical_transformations() {
    println!("━━━ Lesson 5: Canonical Transformations ━━━\n");
    println!("Canonical transformations (q,p)→(Q,P) preserve the symplectic form.");
    println!("They're the 'safe' coordinate changes in Hamiltonian mechanics.\n");

    let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);

    // Identity
    let id = canonical::identity();
    let id_result = id.apply(&pt);
    assert_eq!(id_result, pt);
    println!("  Identity: trivially canonical = {}", id.verify_canonical(&pt));

    // Point reflection: (q, p) → (-q, -p)
    let reflect = canonical::point_reflection();
    let reflected = reflect.apply(&pt);
    println!("\n  Point reflection (q,p) → (-q,-p):");
    println!("    q = {:?} → {:?}", pt.q, reflected.q);
    println!("    Canonical: {}, Inverse correct: {}",
        reflect.verify_canonical(&pt), reflect.verify_inverse(&pt));

    // Fourier transform: (q, p) → (p, -q)
    let fourier = canonical::fourier_transform();
    let ft_result = fourier.apply(&pt);
    println!("\n  Fourier transform (q,p) → (p,-q):");
    println!("    q = {:?} → {:?}", pt.q, ft_result.q);
    println!("    Canonical: {}, Inverse correct: {}",
        fourier.verify_canonical(&pt), fourier.verify_inverse(&pt));

    // Scaling: (q, p) → (λq, p/λ)
    let scale = canonical::scaling(2.0);
    let scaled = scale.apply(&pt);
    println!("\n  Scaling (q,p) → (2q, p/2):");
    println!("    q = {:?} → {:?}", pt.q, scaled.q);
    println!("    p = {:?} → {:?}", pt.p, scaled.p);
    println!("    Canonical: {}, Inverse correct: {}",
        scale.verify_canonical(&pt), scale.verify_inverse(&pt));

    // Phase rotation (1 DOF)
    let rotation = canonical::phase_rotation(std::f64::consts::PI / 4.0);
    let pt_1d = PhasePoint::new(vec![1.0], vec![0.0]);
    let rotated = rotation.apply(&pt_1d);
    println!("\n  Phase rotation by π/4:");
    println!("    (1, 0) → ({:.4}, {:.4})", rotated.q[0], rotated.p[0]);

    // Verify rotation preserves harmonic energy
    let h = Hamiltonian::harmonic(&[1.0], &[1.0]);
    let e_before = h.energy(&pt_1d);
    let e_after = h.energy(&rotated);
    println!("    Energy before: {:.4}, after: {:.4}", e_before, e_after);
    assert!((e_before - e_after).abs() < 1e-10);

    // Custom canonical transformation
    let custom = CanonicalTransformation::new(
        "swap_and_scale",
        |pt: &PhasePoint| PhasePoint::new(
            pt.q.iter().map(|x| x * 3.0).collect(),
            pt.p.iter().map(|x| x / 3.0).collect(),
        ),
        |pt: &PhasePoint| PhasePoint::new(
            pt.q.iter().map(|x| x / 3.0).collect(),
            pt.p.iter().map(|x| x * 3.0).collect(),
        ),
    );
    println!("\n  Custom scaling (q,p) → (3q, p/3):");
    println!("    Canonical: {}, Inverse correct: {}",
        custom.verify_canonical(&pt), custom.verify_inverse(&pt));

    println!();
}

// ─── Lesson 6: Poisson Brackets ──────────────────────────────────────

fn lesson_6_poisson_brackets() {
    println!("━━━ Lesson 6: Poisson Brackets ━━━\n");
    println!("Poisson brackets are the algebraic reflection of the symplectic form:");
    println!("  {{f, g}} = Σᵢ (∂f/∂qᵢ)(∂g/∂pᵢ) - (∂f/∂pᵢ)(∂g/∂qᵢ)\n");

    let pb = PoissonBracket::new(2);
    let pt = PhasePoint::new(vec![1.0, 2.0], vec![3.0, 4.0]);

    // Fundamental bracket: {q₀, p₀} = 1
    let q0: Observable = |pt: &PhasePoint| pt.q[0];
    let p0: Observable = |pt: &PhasePoint| pt.p[0];
    let bracket_qp = pb.apply(q0, p0, &pt);
    println!("  {{q₀, p₀}} = {:.1} (fundamental, should be 1)", bracket_qp);

    // Independent coordinates: {q₀, p₁} = 0
    let p1: Observable = |pt: &PhasePoint| pt.p[1];
    let bracket_q0p1 = pb.apply(q0, p1, &pt);
    println!("  {{q₀, p₁}} = {:.1} (independent, should be 0)", bracket_q0p1);

    // Verify properties
    let f: Observable = |pt: &PhasePoint| pt.q[0] * pt.q[0];
    let g: Observable = |pt: &PhasePoint| pt.p[0] * pt.p[0];
    let h_fn: Observable = |pt: &PhasePoint| pt.q[1] * pt.p[0];

    println!("\n  Verifying Poisson bracket axioms:");
    println!("    Antisymmetry {{f,g}} = -{{g,f}}: {}", pb.verify_antisymmetry(f, g, &pt));
    println!("    Jacobi identity:                  {}", pb.verify_jacobi(f, g, h_fn, &pt));
    println!("    Bilinearity:                      {}", pb.verify_bilinearity(2.0, -1.0, &pt));
    println!("    Leibniz rule:                     {}", pb.verify_leibniz(&pt));

    // Angular momentum as Poisson bracket
    let l_z: Observable = |pt: &PhasePoint| pt.q[0] * pt.p[1] - pt.q[1] * pt.p[0];
    let pt_circ = PhasePoint::new(vec![1.0, 0.0], vec![0.0, 1.0]);
    let l_val = l_z(&pt_circ);
    println!("\n  Angular momentum L_z = q₀p₁ - q₁p₀ = {:.1}", l_val);

    // {L_z, q₀} = p₁ (angular momentum generates rotations)
    let bracket_lz_q0 = pb.apply(l_z, q0, &pt_circ);
    println!("  {{L_z, q₀}} = {:.4} (should be p₁ = {:.1})", bracket_lz_q0, pt_circ.p[1]);

    println!();
}
