# Arena mathematics

For source representation `x`, candidate `x'`, and defender `f_t`, the implemented
attack score is `R(A,D_t)=|f_t(x')-f_t(x)|`. Query-normalized reward is descriptive
drift per counted target query. No unused penalty is claimed.

For label-preserving RandomSMILES only, the smoke defender adds `(x', y)` to its
ridge objective. Chemistry-changing attacks never receive guessed source labels.
The implemented defender gain is the ratio `G_D=R(A,D_t)/R(A,D_{t+1})`; values
above one mean reduced vulnerability. Cross-transfer stores the change in each
attacker column after the active curriculum update. The Thompson controller has
independent `Beta(alpha_k,beta_k)` arms and updates them using clipped
`mean_drift/(mean_drift+1)` evidence. This is a curriculum heuristic, not a
formal game equilibrium.

