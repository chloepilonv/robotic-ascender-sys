# ascender_v2 — THE current end-effector (EOAT): Petzl Basic + printed wrist adapter

**v2 supersedes the bare `tool_ascender` mount**: the Petzl is now carried by a printed adapter that plugs into the G1 wrist socket and is held by Unitree's stock clamp + 2× M3. Tool pose moves +12.5 mm along wrist X vs v1 (clears the 6 mm flange).

Robot files that include it: `assets/robots/mujoco/g1_unitree_ascender_v2.xml` (MuJoCo) and `assets/robots/g1_unitree_ascender_v2.usd` (Isaac).

| File | What |
|---|---|
| `adapter.scad` + `wrist_outline_x34.scad` | parametric source (OpenSCAD + BOSL2), echo design checks |
| `adapter.stl` | print file (binary), 70 cm³ → 72 g PA12-CF |
| `adapter.py` / `adapter.step` | same part in CadQuery → editable STEP for Fusion 360 / Onshape |
| `drawing.png` / `.svg` | dimensioned schematic of the adapter + BOM |
| `wrist_interface_drawing.png` / `.svg` | the wrist end as measured on Unitree's STL (opening Ø38.95, D-outline, end wall) |

## Build
    OPENSCADPATH="$HOME/Library/Application Support/OpenSCAD/libraries" openscad -o adapter.stl adapter.scad
    openscad -o /dev/null --export-format=echo adapter.scad     # design checks
(needs OpenSCAD ≥ 2024 nightly on Apple Silicon: `brew install --cask openscad@snapshot`, BOSL2 in the library path.)

## Assembly (real robot, ~15 min)
1. Power off. Remove the wrist cover plate (2× M2) and the hand clamp (2× M3); pull the rubber hand.
2. Push the adapter plug (Ø 38.6) through the wrist opening; the flange sits on the end wall.
3. Refit Unitree's clamp + 2× M3 over the plug groove (**groove = TODO**, see MOUNT.md); tighten the 4× M3 ×10 collar bolts.
4. Sleeve Ø18/12 into the Petzl eye, slide the Petzl between the cheeks (pocket 6.2 mm for the 5.8 mm plate, offset +4.9 mm), Ø12 shoulder bolt + nyloc.

## Design checks (from `adapter.scad` echoes)
- plug/opening clearance 0.175 mm/side; collar clearance 0.3 mm/side; walls 4 mm (FDM ≥ 1.6)
- cheek shear at 350 N: 2 MPa (PA12-CF yield ~60 MPa)
- clearance check (mesh sampling): 0 adapter/Petzl overlap — cam head untouched, arm runs beside the plate, pin bore centred on the Ø18 eye
- rope load 350 N × 51 mm = **18 N·m on the wrist** (actuators 5 N·m) → forearm must align with the rope while climbing

## Open item
Plug retention inside the wrist's metal socket — not in any public file. Needs calipers on a real G1 hand plug or the Unitree drawing.
