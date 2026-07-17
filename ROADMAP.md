# Go to the Ant — the Go edition

Every system here coordinates the same way: through a shared, decaying field — local reads and writes, no
messaging, no global state. That idea is language-agnostic, and each edition is the same faithful, bit-identical
recreation seen through one language's lens — free to grow its own way from a common foundation.

**The Go lens** — the actor framing lives in the naming today (agents over a shared store), kept sequential so
the result stays bit-identical. Natural directions lean into that heritage: an agent per goroutine, and the
field as something several concurrent — even networked — participants read and reinforce.
