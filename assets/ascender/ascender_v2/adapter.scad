// ── Ascender adapter for the Unitree G1 right wrist (right_wrist_yaw_link) ───────────────────
// Parametric OpenSCAD + BOSL2. Units: mm. Frame = wrist link frame: X toward the hand, Z up, Y sideways.
//
// Measured on Unitree's right_wrist_yaw_link.STL (menagerie) — the link is a 1.5 mm plastic shell:
//   * end wall at x = 39.5..41.5 with a round opening Ø 38.95 centred at (y = -3, z = 0)
//   * D-shaped outline behind the wall (wrist_outline_x34.json), y -30..+23.8, z -30.5..+30
// The hand's plug goes THROUGH the opening into a metal socket inside (not public). The plug is the load path;
// its internal retention feature is a TODO (needs a real G1 or a Unitree drawing).
//
// Parts (single print / single machined part):
//   PLUG      Ø 38.6 x 25 through the opening                          load path
//   FLANGE    6 mm, D-outline + rim, bears on the end wall
//   COLLAR    D-sleeve over the shell, split top/bottom, 4x M3 along Z (same axis as Unitree's stock clamp
//             screws); anti-rotation only — the shell is plastic
//   CRADLE    U-bracket on the Petzl's Ø18 attachment eye: 2 cheeks + Ø12 shoulder bolt with Ø18/12 sleeve
// Render (from this folder):  OPENSCADPATH="~/Library/Application Support/OpenSCAD/libraries" openscad -o adapter.stl adapter.scad
//          OPENSCADPATH="~/Library/Application Support/OpenSCAD/libraries" openscad -o /dev/null --export-format=echo adapter.scad   (design checks only)

include <BOSL2/std.scad>
include <BOSL2/screws.scad>

$fn = 96;

// ── Parameters: measured wrist ───────────────────────────────────────────────────────────────
wall_x1      = 41.5;      // outer face of the end wall
hole_d       = 38.95;     // opening in the end wall
hole_c       = [-3, 0];   // opening centre (y, z)
include <wrist_outline_x34.scad>
include <filler_origin.scad>   // conformal filler between the flange face and the curved Petzl back (filler.dat heightfield from the scan)
outline      = wrist_outline;   // D-outline of the shell at x = 34 (y, z)
collar_x0    = 30;        // straight part of the shell
clearance    = 0.3;       // per side, printed part on a plastic shell

// ── Parameters: adapter ──────────────────────────────────────────────────────────────────────
plug_d       = hole_d - 0.35;   // slip fit through the opening
plug_l       = 25;
flange_t     = 6;
rim          = 5;
collar_t     = 4;
split_gap    = 1.0;       // between the two collar halves so the bolts clamp
m3_clear     = 3.4;
process      = "FDM";     // "FDM" (PA12-CF / PETG-CF) or "CNC" (6061-T6)

// ── Parameters: Petzl interface (measured on the scan) ───────────────────────────────────────
frame_t      = 6.2;       // plate at the eye spans tool y +2.0..+7.8 (5.8 mm, measured on the scan) + clearance
eye_d        = 18;        // attachment eye
pin_d        = 12;        // shoulder bolt; sleeve Ø18/12 fills the eye
cheek_t      = 6;
cheek_w      = 26;
bar_t        = 8;
arm_y        = 1.2;       // arm centred on the outer cheek (wrist y +1.2); the Petzl plate occupies wrist y −7..−2
// tool pose in the wrist frame (from g1_unitree_ascender.usd, +8 mm X so the tool clears the flange)
tool_pos     = [50.33, 0, -51.43];   // USD pose + 11.72 mm X: the Petzl head sits 0.2 mm off the flange face
tool_axis    = [0.22453, 0, 0.97447];   // USD quat (w≈0, xyz=tool_axis) = 180° about this axis
tool_ang     = 180;
eye_tool     = [-9, 4.9, 19.4];   // eye centre in the tool frame (plate mid-plane at y = +4.9)

// ── Derived ──────────────────────────────────────────────────────────────────────────────────
function offset_poly(p, d) = let(c = [mean([for (q = p) q[0]]), mean([for (q = p) q[1]])])
    [for (q = p) let(n = unit(q - c)) q + d * n];
outline_in  = offset_poly(outline, clearance);
outline_out = offset_poly(outline, clearance + collar_t);
y_lo = min([for (q = outline_out) q[0]]);  y_hi = max([for (q = outline_out) q[0]]);
z_lo = min([for (q = outline_out) q[1]]);

// (y,z) polygon extruded along +X: BOSL2 "YZ" plane → rotate so the extrusion axis is X
module along_x(x0, len, pts) { translate([x0, 0, 0]) rotate([90, 0, 90]) linear_extrude(len) polygon(pts); }

// ── Modules ──────────────────────────────────────────────────────────────────────────────────
module plug()   { translate([wall_x1 - plug_l, hole_c[0], hole_c[1]]) rotate([0, 90, 0]) cyl(d = plug_d, l = plug_l + 0.5, anchor = BOTTOM); }
module flange() {
    along_x(wall_x1, flange_t, outline_out);
    // conformal filler: heightfield (mm) sampled from the Petzl scan, surface base 1 mm inside the flange
    translate([wall_x1 + flange_t - 1, filler_y0, filler_z0]) rotate([90, 0, 90]) surface(file = "filler.dat", convexity = 8);
}
module collar() {
    difference() {
        along_x(collar_x0, wall_x1 - collar_x0, outline_out);
        along_x(collar_x0 - 1, wall_x1 - collar_x0 + 2, outline_in);
    }
}
module clamp_bolts() {   // 4x M3 along Z through the collar walls, 2 per side (stock clamp screw axis)
    for (y = [y_lo + collar_t / 2 + 0.3, y_hi - collar_t / 2 - 0.3], x = [collar_x0 + 3.5, wall_x1 - 4])
        translate([x, y, 0]) cyl(d = m3_clear, l = 100);
}
module split()  { translate([(collar_x0 + wall_x1) / 2, 0, 0]) cuboid([wall_x1 - collar_x0 + 0.01, 200, split_gap]); }

module cradle() {   // in the tool frame, origin at the frame's bottom edge under the eye
    difference() {
        union() {
            translate([0, 0, -bar_t / 2]) cuboid([cheek_w, frame_t + 2 * cheek_t, bar_t]);
            for (s = [-1, 1]) translate([0, s * (frame_t + cheek_t) / 2, cheek_w / 2]) cuboid([cheek_w, cheek_t, cheek_w]);
        }
        translate([0, 0, eye_tool[2]]) rotate([90, 0, 0]) cyl(d = pin_d + 0.2, l = 60);   // cradle is centred on the eye
    }
}
module arm() {   // from the bottom of the flange to the OUTER cheek (beside the Petzl plate, never through it)
    a = [wall_x1 + flange_t / 2, arm_y, z_lo + rim / 2];
    b = tool_pos + rot(a = tool_ang, v = tool_axis, p = [eye_tool[0], eye_tool[1] - (frame_t + cheek_t) / 2, cheek_w / 2]);
    d = b - a;  L = norm(d);
    translate(a) rot(from = [0, 0, 1], to = d) translate([0, 0, -6]) cuboid([12, cheek_t, L + 12], anchor = BOTTOM, rounding = 2, edges = "Z");
}

module adapter() {
    difference() {
        union() {
            plug(); flange(); collar(); arm();
            translate(tool_pos) rotate(a = tool_ang, v = tool_axis) translate([eye_tool[0], eye_tool[1], 0]) cradle();
        }
        clamp_bolts();
        split();
    }
}

// ── Design checks (echo) ─────────────────────────────────────────────────────────────────────
F_rope   = 350;                                  // N (robot on the rope)
lever_z  = abs(tool_pos[2]);                     // mm, eye below the wrist axis
echo(str("Plug Ø", plug_d, " in Ø", hole_d, " opening: ", (hole_d - plug_d) / 2, " mm/side clearance (", process, ")"));
echo(str("Wrist torque from rope load: ", F_rope * lever_z / 1000, " N.m at ", F_rope, " N (wrist actuators 5 N.m) -> forearm must align with the rope"));
echo(str("Collar clearance ", clearance, " mm/side; FDM: walls ", collar_t, " mm >= 1.6 OK; M3 clearance ", m3_clear, " (holes shrink ~0.15) OK"));
echo(str("Cheek shear at ", F_rope, " N: ", F_rope / (2 * (cheek_w - pin_d) * cheek_t), " MPa (PA12-CF yield ~60, 6061 ~275)"));

// ── Main ─────────────────────────────────────────────────────────────────────────────────────
adapter();
