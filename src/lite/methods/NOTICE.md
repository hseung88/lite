# Third-party baseline implementations

DSoftKI and DDSVGP include code distributed under Apache-2.0. The license is
retained in `LICENSE` in this directory.

DSoftKI: https://arxiv.org/abs/2505.09134.
DDSVGP: Padidar et al., "Scaling Gaussian Processes with Derivative Information
Using Variational Inference," NeurIPS 2021.

Modifications include package-relative imports, compatibility with
`linear_operator`, shared in-memory data loading, deterministic seeding,
dtype/device handling, training-curve recording, and removal of external
experiment tracking. DDSVGP's variational objective receives the latent
distribution; its likelihood is applied once. DDSVGP uses an isotropic RBF
kernel. DSoftKI supports RBF and Matérn kernels and uses float32.
