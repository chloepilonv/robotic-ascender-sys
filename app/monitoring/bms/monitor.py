"""D) Battery monitor: read the G1 BMS over DDS, raise an alarm when SOC is low.

Usage:
    python bms/monitor.py --iface eth0 --threshold 20
Replace `alarm()` later with "call the swap robot" / notify a human.
"""

import argparse
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


def alarm(soc: int):
    print(f"\a*** LOW BATTERY: {soc}% ***")


class BmsMonitor:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self.alarmed = False
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._on_state, 10)

    def _on_state(self, msg: LowState_):
        bms = msg.bms_state
        print(f"soc={bms.soc}%  current={bms.current}mA  temps={list(bms.bq_ntc)}", end="\r")
        if bms.soc < self.threshold and not self.alarmed:
            self.alarmed = True
            alarm(bms.soc)
        elif bms.soc >= self.threshold + 5:   # hysteresis so it doesn't spam around the threshold
            self.alarmed = False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iface", default="eth0")
    p.add_argument("--threshold", type=int, default=20, help="alarm below this SOC %")
    args = p.parse_args()
    ChannelFactoryInitialize(0, args.iface)
    BmsMonitor(args.threshold)
    while True:
        time.sleep(1)
