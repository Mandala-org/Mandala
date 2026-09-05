# Descriptor completeness obligations (v1)

This note fixes the claims tested by the Stage-3 geometry-only batch. It does
not use Hamiltonian targets, and numerical collision searches are evidence,
not proofs.

## Domain

An environment is a finite, species-labelled point measure in the open ball
of radius `R_D`. The center is excluded. For finite-domain statements we also
require at most `N_max` points, a positive point separation, and a positive
margin from the cutoff. Coordinates are expressed in the fixed global frame;
permutations within a species are immaterial. Full O(3), including inversion,
acts covariantly on the descriptor.

## D1: complete-basis injectivity

For each species, D1 contains the coefficients of convolution of the point
measure with a fixed kernel in a complete radial-angular basis on the ball.
The coefficient sequence therefore determines the convolved density. For the
delta-kernel limit, its labelled support is the neighborhood. For a fixed
Gaussian whose Fourier transform is nowhere zero, convolution is injective
and can be deconvolved in principle. The known cutoff window is strictly
positive in the open ball and can be divided out after support recovery.
Consequently the infinite D1 sequence is injective on the stated domain.

This is a basis-limit statement. No finite `(n_max, l_max)` truncation is
declared injective without the reconstruction, rank, and collision evidence
reported by Stage 3.

## D2: finite moment route

D2 is a fixed invertible irreducible change of basis of the retained
species-resolved polynomial moment collection. A finite atomic measure with a
known upper support bound can generically be recovered from sufficiently many
multivariate moments by an annihilating-ideal/Prony construction. A checkable
global theorem for the exact degree sequence used here still requires an
explicit unisolvence bound and stability assumptions in terms of `N_max`,
minimum separation, and boundary margin. Until that reduction is completed,
the valid status is “theorem-oriented with local/generic numerical evidence,”
not “proved complete.”

## D3: finite Fourier sampling route

The complete characteristic function determines a finite point measure.
Finite exponential samples can support Prony recovery under appropriate
sampling and separation conditions, but the present spherical shell/angular
truncations do not automatically satisfy such a theorem. D3 is therefore
classified as complete in the infinite sampling limit and empirically tested
at finite resolution.

## D4: inheritance

D4 is the direct sum of D1 and deterministic polynomial covariants:
`D4 = D1 + A2 + A3`. Projection onto the retained raw summand recovers D1
exactly. Every injectivity property established for a particular D1 base is
therefore inherited by the corresponding D4 configuration. The added lifts
do not contain new information; they change accessibility and conditioning
for shallow downstream maps.

## Numerical interpretation

The Stage-3 suite records fixed-count/species multistart inversion, normalized
descriptor residual, species-aware assignment RMSD, pair-distance error,
Jacobian rank and singular values, continuation from the weakest local
direction, and constrained collision searches. Optimizer failure is not proof
of incompleteness. A verified geometrically distinct match is evidence of a
finite-truncation collision. Failure to find one is not proof of injectivity.
