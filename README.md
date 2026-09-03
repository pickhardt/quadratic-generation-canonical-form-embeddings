# Quadratic generation for the canonical-form embeddings of X_cf(3,6) and Y(3,6)

Scripts and certificates accompanying the paper *"Quadratic generation for the
canonical-form embeddings of X_cf(3,6) and Y(3,6), and Koszulness for del Pezzo
complements."*

This repository holds both halves of the following:

* **scripts** -- 23 Python programs and one C program, which generate and check
  every certificate; and
* **certificates** -- the exact finite-field artifacts they produce (`.npz`,
  `.json`, `.npy`), which are what the paper's claims actually rest on.

Everything needed to replay every computation behind Theorems A and B is here,
except the one external input noted below. Theorem C is proved by hand and needs
no computation at all.

**Keep the layout flat.** Every script resolves its imports and inputs from the
working directory, so run them from the top of this repository. (Moving files into
subdirectories will break the chain.)

## External input

One file is required that is not reproduced here: Schock's Chow-ring
presentation of his small resolution of M(3,6), which is parsed by
`x36_chow_degree.py` and `x36_c2_blowup.py` to produce every intersection
number and Chern number used for X_cf(3,6). Fetch it with

    ./fetch_chow.sh

which downloads `ChowM36.sage` from

    https://github.com/NSchock/IntersectionM36
    commit  282e0f3feee68b695f0a6cb255ef3b47425ae5fd
    sha256  613a38de9b93c85b3bf61166782c6937ab101af13a22f03ea8da3a8d0a0f2885

and verifies its hash. It is fetched rather than vendored because that
repository carries no license grant. Everything else needed to replay every
computation behind Theorems A and B is in this bundle.

Note on provenance versus dependency: the canonical functions themselves --
the four S_6-orbits of rational functions defining the two embeddings -- were
taken from the supplementary code of Hollering-Pavlov-Pratt
(doi:10.5281/zenodo.21624557), and that is the correct citation for the
*choice* of orbits. It is not a runtime dependency: the scripts here
reconstruct those functions from scratch out of 3x3 Plucker minors and S_6
permutations (`y36_cubic_sieve.py`), so the certificates replay without it.

## Environment

    Python 3.14.5, python-flint 0.9.0 (FLINT 3.6.0), NumPy 2.5.0, SciPy
    prime p = 1000003, master seed 260728368

Install the Python dependencies with

    python -m pip install -r requirements.txt

SciPy is used only by `x36_chow_degree.py` and `x36_c2_blowup.py`, for sparse
assembly and least squares while parsing the Chow-ring presentation.
Everything else the scripts import is from the standard library.

Auxiliary deterministic seeds: quartic projection 271828182 and quintic
projection 161803399 for X_cf; quartic evaluation-point seed 314159265 for X_cf
and 271828182 for Y.

## Shared module

`y36_cubic_sieve.py` defines the canonical functions, the deterministic point
sampler, and the graded reverse lexicographic order. Every script below imports
it, in both chains.

## Order of execution

Y(3,6) / Naruki:

    y36_cubic_rank_certificate.py
    y36_cubic_basis_dense.py
    y36_quartic_rank_certificate.py
    y36_quartic_full_profiles.py
    y36_orient_quartic_rules.py
    y36_quartic_profiles_quintic_shadow.py
    y36_quartic_standard4.py            # standard quartics + column selection
    y36_quartic_evaluation_matrix.py    # 23730 x 23730 evaluation matrix
    y36_quartic_determinant.py          # det = 92947 != 0  (degree-four surjectivity)
    y36_quintic_residual5_certificate.py

X_cf(3,6):

    x36_chow_degree.py
    x36_c2_blowup.py
    x36_cubic_rank_certificate.py
    x36_cubic_basis_dense.py
    x36_quartic_rank_certificate.py
    x36_quartic_full_profiles.py
    x36_orient_quartic_rules.py
    x36_quartic_evaluation_matrix.py    # 18695 x 18695 evaluation matrix
    x36_nmod_det.c                      # det = 942745 != 0
    x36_quintic_residual_certificate.py

Not on the certificate path, shipped for provenance: `y36_lex_search.py` and
`y36_termorder_search.py` (the term-order searches reported in the paper's
limitations section), and `y36_quintic_rank_certificate.py` /
`y36_quintic_rank_certificate_from_raw.py` (superseded variants of the
degree-five computation). No claim depends on them.

## The two large matrices are not archived here

The quartic evaluation matrices are 1.4 GB (X_cf, 18695^2) and 2.25 GB
(Y, 23730^2) as uint32. They are deterministically regenerated from the recorded
point seeds fairly quickly. Their SHA-256 hashes are:

    98156c7e8e60b07f5fe6bbb3adfe7e43f7931404bc6f371297389548f3be0948
      x36_quartic_eval_18695_u32.bin
    8641090a44c97a0ebf1111432d64fa26877b83cc84f26b5528ec6b7694ffca82
      y36_quartic_eval_23730_u32.bin

## Determinants

The X_cf determinant uses `x36_nmod_det.c`, a thin wrapper over FLINT's
`nmod_mat_det`. It declares the handful of `nmod_mat` symbols it needs
directly, so no FLINT headers are required -- only a shared FLINT to link
against, plus pthreads for the progress heartbeat:

    cc -O2 -o x36_nmod_det x36_nmod_det.c -lflint -lpthread

Add `-L<dir>` (and `-Wl,-rpath,<dir>` on Linux, or `DYLD_LIBRARY_PATH` on
macOS) if libflint is not on the default search path; the copy bundled inside
the installed `python-flint` package works. Then

    ./x36_nmod_det x36_quartic_eval_18695_u32.bin 18695 18695

which prints `DETERMINANT_MOD_P=942745`. The Y determinant
uses `y36_quartic_determinant.py`, a blocked LU over F_p in exact double
precision: entries stay reduced in [0,p) and each block product has inner
dimension 512, so every accumulation is bounded by 512*(p-1)^2 < 2^53. It was
checked against FLINT's `nmod_mat_det` on random matrices and on the 600x600 and
1500x1500 leading submatrices of the actual matrix before use.
