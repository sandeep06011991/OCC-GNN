#  Proof that processes cannot communicate through shared threading module.
# Cmodule [N] = PyShell[1 Worker] == Trainer[4]
Issue PyShell can be a bottleneck
(CModule = PyShell) N == Trainer
