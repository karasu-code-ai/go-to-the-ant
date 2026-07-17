# Go to the Ant — the Rust edition

Every system here coordinates the same way: through a shared, decaying field — local reads and writes, no
messaging, no global state. That idea is language-agnostic, and each edition is the same faithful, bit-identical
recreation seen through one language's lens — free to grow its own way from a common foundation.

**The Rust lens** — ownership and the borrow checker turn "who may touch the field, and when" into a property
the compiler enforces rather than a convention. Natural directions lean into that: making the field's access
rules type-level, and exploring how much of a swarm's correctness can be *proven* rather than tested.
