"""Headless check of the BMS panel numbers, no browser, ~10 s.

    python -m app.bms_ui.selftest            # 3 s walk on free_0, prints the readout

Expected: P_elec roughly 150-300 W while walking, R_int ~0.13 ohm at 15 C,
SOC dropping a few hundredths of a percent per second.
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAME = "bms_selftest"
EPISODE = os.path.join(ROOT, "app", "harness", "episodes", NAME)


def main():
    shutil.rmtree(EPISODE, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "app.harness.runtime", "--world", "free_0",
                    "--duration", "3", "--hold-w", "--no-render", "--output-name", NAME],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    hud = json.load(open(os.path.join(EPISODE, "hud.json")))
    ok = True
    for key in ["bms_soc_pct", "bms_pack_V", "bms_current_A", "bms_power_W",
                "bms_mech_power_W", "bms_t_bat_C", "bms_r_int_ohm", "bms_max_abs_tau_Nm"]:
        series = hud.get(key)
        if not series:
            print(f"MISSING {key}"); ok = False; continue
        print(f"{key:22s} t=0 {series[0]:9.3f}   t=1s {series[50]:9.3f}   t=3s {series[-1]:9.3f}")
    p = hud["bms_power_W"][-1]
    if not 100 <= p <= 400:
        print(f"WARN power_W {p:.0f} outside the 100-400 W walking band"); ok = False
    shutil.rmtree(EPISODE, ignore_errors=True)
    subprocess.run(["git", "checkout", "--", "app/harness/fingerprint_slope_0.json"],
                   cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
