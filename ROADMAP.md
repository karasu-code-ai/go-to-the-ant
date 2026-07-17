# Go to the Ant — the C edition

Every system here coordinates the same way: through a shared, decaying field — local reads and writes, no
messaging, no global state. That idea is language-agnostic, and each edition is the same faithful, bit-identical
recreation seen through one language's lens — free to grow its own way from a common foundation.

**The C lens** — the field is *literally a block of memory*: manual allocation, a hand-rolled PRNG, the
mechanism laid utterly bare. Natural directions lean into that bare-metal character — tighter memory and
per-agent cost, lower-level field operations, and running the same swarms where compute is scarce.
