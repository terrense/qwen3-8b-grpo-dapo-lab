# VAPO — NOT IMPLEMENTED

No config here yet, deliberately. VAPO does not exist in verl 1252cc71 or verl-recipe;
see [../../analysis/vapo_feasibility.md](../../analysis/vapo_feasibility.md).

The recommended first step is **plain PPO with GAE + critic** using VeRL's existing
maintained path (`algorithm.adv_estimator=gae` with the critic enabled), which delivers
the cross-paradigm value-function insight with zero new algorithm code.
