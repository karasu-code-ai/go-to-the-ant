# Go to the Ant — the Java edition

Every system here coordinates the same way: through a shared, decaying field — local reads and writes, no
messaging, no global state. That idea is language-agnostic, and each edition is the same faithful, bit-identical
recreation seen through one language's lens — free to grow its own way from a common foundation.

**The Java lens** — the classical agent-based-modeling lineage (MASON, Repast, NetLogo): a scheduler steps a
population of individually-instantiated agents over a shared store. Natural directions lean into that heritage —
a pluggable step/schedule contract, and using the JVM's newer concurrency (virtual threads) to scale the population.
