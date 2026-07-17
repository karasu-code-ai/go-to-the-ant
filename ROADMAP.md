# Go to the Ant — the CUDA edition

Every system here coordinates the same way: through a shared, decaying field — local reads and writes, no
messaging, no global state. That idea is language-agnostic, and each edition is the same faithful, bit-identical
recreation seen through one language's lens — free to grow its own way from a common foundation.

**The CUDA lens** — the field *is* device memory and the agents *are* threads; deposits are the atomic-add
race made literal. It is distinct-by-design (parallel update order breaks bit-identity — that is the point).
Natural directions lean into parallelism: much larger swarms, the field operations as kernels, and studying
how the emergence survives when the update order no longer does.
