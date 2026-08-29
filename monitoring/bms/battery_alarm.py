"""D) Deterministic battery watchdog. Reads BMS from the G1 LowState and raises an alarm.

Usage:
    python bms/battery_alarm.py --iface eth0 --low 20 --critical 10

Alarm = print + write a JSON line to bms/alarms.jsonl. Swap `alarm()` for MQTT / HTTP
to notify another robot or a human later.
"""
import argparse
import json
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

LEVELS = {"critical": 2, "low": 1, "ok": 0}


def alarm(level: str, soc: int, voltage: float, current: float):
    rec = {"t": time.time(), "level": level, "soc": soc, "voltage": voltage, "current": current}
    print("ALARM", rec)
    with open("bms/alarms.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


class BatteryWatch:
    def __init__(self, low: int, critical: int, period: float):
        self.low, self.critical, self.period = low, critical, period
        self.last_level = "ok"
        self.last_alarm_t = 0.0
        ChannelSubscriber("rt/lowstate", LowState_).Init(self.on_state, 10)

    def on_state(self, msg: LowState_):
        bms = msg.bms_state
        soc, v, i = int(bms.soc), msg.power_v, msg.power_a
        level = "critical" if soc <= self.critical else "low" if soc <= self.low else "ok"
        # fire when level worsens, or repeat every `period` s while not ok
        worse = LEVELS[level] > LEVELS[self.last_level]
        due = level != "ok" and time.time() - self.last_alarm_t > self.period
        if worse or due:
            alarm(level, soc, v, i)
            self.last_alarm_t = time.time()
        self.last_level = level


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iface", default="eth0")
    p.add_argument("--low", type=int, default=20, help="%% SOC -> 'low' alarm")
    p.add_argument("--critical", type=int, default=10, help="%% SOC -> 'critical' alarm")
    p.add_argument("--period", type=float, default=60.0, help="repeat alarm every N s")
    a = p.parse_args()
    ChannelFactoryInitialize(0, a.iface)
    BatteryWatch(a.low, a.critical, a.period)
    while True:
        time.sleep(1)
