"""List every DDS topic the G1 is publishing (run this FIRST on the robot network).

    python app/bms/real/discover_topics.py --iface eth0 --seconds 5

Prints topic name + type name. Use it to confirm the BMS/mainboard topic names
used in monitor_battery.py (rt/lf/bmsstate, rt/lf/mainboardstate).
"""
import argparse, time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from cyclonedds.domain import DomainParticipant
from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication

p = argparse.ArgumentParser()
p.add_argument("--iface", default="eth0")
p.add_argument("--seconds", type=float, default=5.0)
a = p.parse_args()

ChannelFactoryInitialize(0, a.iface)          # sets up CycloneDDS config for that NIC
dr = BuiltinDataReader(DomainParticipant(0), BuiltinTopicDcpsPublication)
seen = {}
t0 = time.time()
while time.time() - t0 < a.seconds:
    for s in dr.take(N=100):
        seen[s.topic_name] = s.type_name
    time.sleep(0.2)
for name, typ in sorted(seen.items()):
    print(f"{name:35s} {typ}")
