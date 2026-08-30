# Mounting the ascender on the real G1 — `right_wrist_yaw_link`

Joint: `right_wrist_yaw_joint`. Its child link `right_wrist_yaw_link` ends in the **hand socket** where Unitree's rubber hand / Dex3-1 plug in.
(29-DoF G1 = Advanced/Flagship/EDU wrist; the Basic/Standard has no wrist joints and a different 6×M3 clamp.)

## Stock interface (from Unitree's manuals)
| Item | Value | Source |
|---|---|---|
| Attachment | round **stem** on the hand, slides into the wrist socket | Prosthetic-hand manual p.5, Dex3-1 manual p.4 |
| Lock | U-shaped **clamp ("hoop")**, **2× M3, 10 mm thread** | same |
| Cover plate | 2× M2, 5 mm thread; hides the **power connector** (powered hands only) | same |
| Socket (wrist side) | round cup: **bore Ø ≈ 49–50 mm at the mouth, ≈ 20 mm deep**, wall ≈ 3 mm, outer Ø ≈ 56–60 mm (mouth at x = 41.5 mm from the wrist-yaw joint) | measured on Unitree's `right_wrist_yaw_link.STL` (menagerie), ±0.5 mm mesh resolution |
| Hand "stem" (plug) | the round base of the hand that fills that cup: Ø ≈ 49 mm × ~20 mm + a groove the clamp hooks into; **not in any public file** (the rubber-hand STL starts at the cup mouth) → caliper it | — |
| Warning | with a hand/tool mounted, avoid squatting and lying down | Dex3-1 manual p.7 |

Manuals: [G1 End Prosthetic Hand assembly guide](https://www.unitree.com/images/G1-End%20Prosthetic%20Hand%20Disassembly%20and%20Assembly%20Guide%20Manual.pdf),
[Dex3-1 assembly guide](https://www.unitree.com/images/G1-Flagship%20Version%20A%26B%20Terminal%20Three-Fingered%20Dexterous%20Hand%20Dex3-1%20Disassembly%20and%20Assembly%20Guide%20Manual%20V1.1.pdf).
CAD: no public G1 STEP ([unitree_cad](https://github.com/unitreerobotics/unitree_cad) has only A1/Aliengo/Laikago). EDU owners: support ticket at support.unitree.com → ask for `wrist_yaw_link` + hand-stem STEP.

## Where to see the socket
`python3 -m mujoco.viewer --mjcf $PWD/assets/robots/mujoco/g1_unitree.xml` → double-click the right wrist. Or open
`assets/robots/g1/_menagerie/unitree_g1/assets/right_wrist_yaw_link.STL` in Fusion 360 / MeshLab and use the measure tool.
Measure script: slice the STL at x = 40 mm → inner radius ≈ 24.5 mm (see git log of this file).

## Build plan
1. Remove the rubber hand: 2× M2 cover plate, 2× M3 clamp, pull the hand. Caliper the stem (Ø, length, groove position, any anti-rotation flat).
2. CAD the adapter (Fusion 360 / Onshape): stem copy → plate → boss with the ascender's carabiner hole (Ø 12–14 mm for an M8/M10 pin). Keep the tool's rope axis as in sim: mount pose is read from `assets/robots/g1_unitree_ascender.usd` (`tool_ascender` xform).
3. Make it: PA12-CF print (first fit) → 6061-T6 aluminium (field). PLA creeps under the clamp preload and in the cold.
4. Fit with the stock clamp + 2× M3 ×10; ascender on the boss with a shoulder bolt + nyloc.
5. Weigh it, then update the mass/pose in `assets/robots/g1/attach_tool.py` and rebuild USD + MJCF (`assets/robots/mujoco/build.py`).

## Limits to respect
Wrist yaw actuator torque limit ±5 N·m (menagerie MJCF `actuatorfrcrange`) — the rope load must go through the arm, not the wrist motor: rope axis through the wrist joint centre (that is why the mount pose centres the cam head on the joint).
