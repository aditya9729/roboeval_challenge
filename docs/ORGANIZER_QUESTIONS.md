# Organizer clarification draft — NOT SENT

Contact listed by the challenge: yiruwang@cs.washington.edu. Source: S01/S02.

Subject: RoboEval Track 2 — training-only augmentation and RL clarification

We are developing an entry using the untouched official evaluation release and only declared camera/proprioceptive observations at inference. Could you clarify the following training-only cases?

1. May we generate demonstrations in a separately maintained training environment with modified object geometry, friction, mass, textures, or background assets, while preserving task intent and evaluating only on the official release? Which changes require prior approval?
2. May a training-only teacher or verifier use simulator object poses and contact information, provided those fields never reach the submitted policy or its inference process?
3. May training use intermediate-state resets or restoration of saved simulator states? Is ordinary `env.reset(seed=...)` sampling among a non-evaluation pool permitted without additional approval?
4. May an external learner use progress/success diagnostics returned in `info`, or use a training-only shaped reward, without modifying the benchmark code? If not, which reward signals are explicitly permitted?
5. For a model that requires a separate Python environment, is a packaged local inference subprocess with pinned dependencies acceptable for your held-out reruns?

Our default implementation disables cases 1, 2, 3's intermediate resets, and any non-native reward until clarified. We will report all data sources and modifications transparently.

## Approval record
Status: UNANSWERED. Do not treat this draft as permission.

For each answer record: date, organizer, exact response, allowed scope, affected configs, source location, reviewer, and whether any prior experiment must be excluded. Do not send this message automatically.
