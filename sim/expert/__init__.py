"""Expert pipeline: scripted experts, 7-D delta actions, and the thermal judge.

Modules (import directly; this ``__init__`` deliberately re-exports nothing so
that ``sim.expert.thermal_judge`` stays on the pure-numpy import tier):

    expert.py         scripted expert state machines (policy-rate observations only)
    actions.py        7-D delta action pipeline (normalize -> DLS IK -> PD)
    pipeline.py       algorithm constants and the temperature target-kind helper
    thermal_judge.py  material statistics (pure numpy; no engine import)
"""
