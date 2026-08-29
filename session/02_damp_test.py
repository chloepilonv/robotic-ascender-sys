"""STEP 2 -- the emergency stop. Run this BEFORE any policy. No policy loaded.

Once you are in developer mode the built-in motion service is released, so
there is no stop command left to call: killing this process IS the stop. That
makes the shutdown path a safety device, and this script exists to prove it
works while nothing is at stake.

What it does: enters a loop that publishes a DAMPING command (kp=0, kd small,
zero torque) to all 29 joints. The robot should stay slack in the harness and
not move. Then you press Ctrl-C and confirm it stays slack.

    # with the robot ALREADY in developer mode and the face light GREEN:
    python session/02_damp_test.py --iface eth0            # dry: prints, sends nothing
    python session/02_damp_test.py --iface eth0 --arm      # actually publishes

CAUTION -- read before using --arm:
  * The publisher path below is now VERIFIED against unitree_sdk2_python's
    unitree_hg IDL and example/g1/low_level/g1_low_level_example.py. Three
    things it corrected, all of which would have failed on the robot:
      - LowCmd_ must be built by unitree_hg_msg_dds__LowCmd_(), not LowCmd_();
        the bare dataclass has no array defaults.
      - mode_machine MUST be copied from the incoming LowState_ or commands
        are rejected. That means we cannot publish until lowstate arrives.
      - mode_pr must be set (PR = series ankle control, what the policy wants).
  * A LowCmd_ with a wrong or missing CRC is typically rejected silently, so
    "nothing happened" does not mean "it is safe" -- it may mean the command
    never landed. Confirm on camera, not from the terminal.
  * Never run this while the built-in motion service is still active. Two
    controllers fighting makes the robot vibrate violently.
"""
import argparse
import signal
import sys
import time

N_MOTORS = 29
DAMP_KD = 2.0          # small damping; joints go slack, harness takes the load
CTRL_HZ = 50.0


class Bridge:
    """Thin LowCmd publisher. Field names are VERIFIED BY 01, not assumed."""

    def __init__(self, iface, armed):
        self.armed = armed
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                 ChannelPublisher,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        ChannelFactoryInitialize(0, iface)
        self._crc = CRC()
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()
        self._cmd = unitree_hg_msg_dds__LowCmd_()   # NOT LowCmd_()
        # mode_machine comes from the robot; we may not publish without it.
        self._mode_machine = None
        ChannelSubscriber("rt/lowstate", LowState_).Init(self._on_state, 1)
        print(f"publisher ready on rt/lowcmd (armed={armed}); "
              f"waiting for lowstate to learn mode_machine...")

    def _on_state(self, msg):
        if self._mode_machine is None:
            self._mode_machine = int(msg.mode_machine)
            print(f"  mode_machine = {self._mode_machine}")

    def ready(self):
        return self._mode_machine is not None

    def send_damping(self):
        """kp=0, kd=DAMP_KD, q/dq/tau = 0 on every joint -> slack."""
        if self._mode_machine is None:
            return False
        self._cmd.mode_pr = 0          # PR: series ankle control
        self._cmd.mode_machine = self._mode_machine
        for i in range(N_MOTORS):
            mc = self._cmd.motor_cmd[i]
            mc.mode = 1          # enable
            mc.q = 0.0
            mc.dq = 0.0
            mc.kp = 0.0
            mc.kd = DAMP_KD
            mc.tau = 0.0
        if not self.armed:
            return False
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--arm", action="store_true",
                    help="actually publish (default: dry run, sends nothing)")
    ap.add_argument("--seconds", type=float, default=30.0)
    a = ap.parse_args()

    if not a.arm:
        print("DRY RUN: nothing will be published. Add --arm to publish.\n")

    bridge = Bridge(a.iface, a.arm)
    stop = {"flag": False}

    def shutdown(signum, _frame):
        print(f"\nsignal {signum} -> sending damping and exiting")
        for _ in range(10):           # send several; DDS is best-effort
            bridge.send_damping()
            time.sleep(0.01)
        print("damping sent. CONFIRM ON CAMERA that the robot is slack.")
        stop["flag"] = True
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    print("waiting for lowstate (mode_machine)...")
    while not bridge.ready():
        time.sleep(0.2)
    print("holding damping. Press Ctrl-C to test the stop path.\n")

    t0 = time.time()
    n = 0
    try:
        while not stop["flag"] and time.time() - t0 < a.seconds:
            sent = bridge.send_damping()
            n += 1
            if n % int(CTRL_HZ) == 0:
                print(f"  t={time.time()-t0:5.1f}s  damping frames sent={n} "
                      f"{'(DRY)' if not sent else ''}")
            time.sleep(1.0 / CTRL_HZ)
    finally:
        for _ in range(10):
            bridge.send_damping()
            time.sleep(0.01)
        print("exited via damping path.")


if __name__ == "__main__":
    main()
