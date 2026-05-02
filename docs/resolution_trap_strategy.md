# Strategy: The Resolution Trap (planned)

## The Concept
Standard Convolutional Neural Networks (CNNs/U-Nets) are spatially biased. They learn patterns in **voxels**, not in **meters**. This means a model trained on a 32x32x32 grid ($dx=62\mu m$) cannot be deployed on a 128x128x128 grid ($dx=15\mu m$) without serious physical distortion.

## The Narrative (LinkedIn Storytelling)
> "We built a perfect surrogate... until we changed the resolution. In the worlds of physics and AI, scaling isn't just about 'adding more pixels.' It's about preserving the dimensionless constants of nature."

## The "Resolution Trap" Problem
The Laplacian $\nabla^2 T$ in our PDE residual depends on $1/dx^2$. If the AI doesn't understand that the spatial relationship has changed, the "Physics Loss" will diverge, and the surrogate will behave like a different material entirely.

## Logical Future Improvement
After we settle the **Material Zoo** and **Adaptive Weighting**, we will solve the Resolution Trap by moving to **Neural Operators**.
*   **Fourier Neural Operators (FNO)** or **Continuous Neural Operators (CNO)**.
*   These architectures perform operations in the frequency domain or use integral kernels that are mathematically **resolution-invariant**.
*   *Result*: Train on a cheap 32x32 simulation; Deploy on a high-fidelity 512x512 digital twin.
