# Hardware session: deploying the G1 walking policy

Run in order. Each step gates the next. Camera open throughout.

| # | script | dev mode | publishes | purpose |
|---|--------|----------|-----------|---------|
| 01 | `01_log_lowstate.py` | no | **no** | telemetry + SDK introspection |
| 02 | `02_damp_test.py` | yes | yes (`--arm`) | prove the emergency stop |
| 03 | `03_dry_run.py` | either | **no** | policy on real data, sends nothing |
| 04 | `04_stand.py` | yes | yes (`--arm`) | closed loop, ZERO command |

All of 02 and 04 default to a dry run. `--arm` is required to publish.

## Pass criteria

**01** — lowstate ~500 Hz; `imu_state` is the PELVIS IMU (not torso); joint
order matches `deploy/constants.py::JOINT_NAMES`; waist roll/pitch not locked.

**02** — robot stays slack; Ctrl-C leaves it slack. Confirm **on camera**, not
in the terminal: a bad CRC is dropped silently, so "nothing moved" can mean
"the command never landed".

**03** — gravity ≈ `[0,0,-1]` upright; `|q − default|` small in a normal
standing pose; action deviation near **0.367 rad** (the sim golden value);
loop p99 well under 20 ms.

**04** — robot holds the default pose. It will not walk and must not be asked to.

## Abort immediately

* Robot vibrates → not in developer mode. Kill, then `robot status`.
* `|q − default|` large while standing normally → joint mapping is wrong.
* Any joint faults → damp, contact the admin, do not retry in a loop.

## Cleanup (required)

```bash
robot normal            # hand control back
pkill -f session/       # nothing left looping
rm -f *.jsonl           # shared disk is small
```
