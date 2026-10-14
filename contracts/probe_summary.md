# Contract probe summary

- family: LiftPot, reset seed 100000, MUJOCO_GL=egl
- gripper: open=0 closed=1 rounding={'0.49_acts_like': 'cmd_0.0', '0.51_acts_like': 'cmd_1.0'}
- speed clip: max per-step ctrl delta 0.10875 rad (limit 0.10875, within=True)
- determinism: same-seed bit-identical = True
- state_only: 7.8 ms/step, peak RSS 1785 MB
- three_cameras: 17.5 ms/step, peak RSS 2227 MB
