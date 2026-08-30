// The first-person camera: the view from the robot's own eyes.
//
// WHERE IT SITS. Exactly where the robot's stereo pair sits. `app/harness/
// guide.py::_add_eye_cameras` mounts `eye_left` / `eye_right` either side of the
// `d435i` camera already in `assets/robots/mujoco/g1_unitree_ascender.xml`,
// displaced +/- baseline/2 along the CAMERA's own x axis -- so the midpoint of
// the two eyes IS the d435i mount, and that is the one number this file needs:
//
//     <camera name="d435i" pos="0.0789635 0 0.386" xyaxes="0 -1 0  0 0 1"
//             fovy="58" />                       ... on body `torso_link`
//
// The G1 has no neck. The d435i hangs off `torso_link`, so the eyes turn with
// the waist and rock with the walk for free: this camera is welded to the
// torso body's pose off the wire (app/harness/pose_stream.py), with no lag, no
// spring and no collision. The robot's body IS the camera boom.
//
// THE AXES, read off the MJCF's `xyaxes` and confirmed against the compiled
// model (`model.cam_quat[d435i]` = (-0.5, -0.5, 0.5, 0.5) wxyz):
//
//     camera right (+x)  = torso -y      (a robot facing +x has its right at -y)
//     camera up    (+y)  = torso +z
//     camera back  (+z)  = torso -x      so it LOOKS along torso +x
//
// LOOKING AROUND IS CAMERA-ONLY. Nothing here is sent anywhere: the drag turns
// the picture, never the robot. The policy owns the waist joints, and a browser
// that could write to them would be driving the thing the RL policy is being
// judged on. What the clamps borrow from the waist is only its RANGE, because
// that is how far a real G1 could turn its eyes:
//
//     waist_yaw_joint    range="-2.618 2.618"   = +/- 150.00 deg
//     waist_pitch_joint  range="-0.52  0.52"    = +/-  29.79 deg
//     waist_roll_joint   range="-0.52  0.52"    ignored -- a rolled horizon is
//                                               nausea, not information
//
// SIGNS MATCH THE THIRD-PERSON CAMERA, deliberately, so one drag habit works in
// both views. There, `movementX` adds to the MuJoCo azimuth (where the camera
// SITS) and the viewing direction is azimuth + 180, so a rightward drag swings
// the view counterclockwise; `movementY` adds to elevation, and elevation is
// NEGATIVE when the camera is above looking down, so a downward drag tilts the
// view up. Both are "drag the world", which is what the 2-D page and MuJoCo's
// own viewer do. Same two degrees-per-pixel constants, same feel.
//
// NO AUTO-RECENTRE (the call, stated because the third-person camera does the
// opposite). There, the boom drifts back behind the robot after two idle
// seconds because a chase camera that does not is a chase camera you have to
// keep steering. Here the offset IS the head turn: a climber who looks left and
// keeps walking is still looking left, and having the view creep forward on its
// own while you watch something is the worse surprise. R -- which already
// recentres the third-person boom -- zeroes this offset too.
import * as THREE from './vendor/three.module.js';

// The d435i mount on `torso_link`, in the torso body's own frame. Metres.
export const EYE_MOUNT_IN_TORSO_METERS = new THREE.Vector3(0.0789635, 0, 0.386);
// The RealSense's own vertical field of view, so the picture is the lens the
// robot actually carries rather than a number picked to look nice.
export const FIELD_OF_VIEW_DEGREES = 58;
// Small enough that the snow immediately under the robot's chin is drawn. The
// third-person camera's 0.08 m is fine at four metres of boom and too coarse
// here, where the ground can be 30 cm from the lens on a steep face.
export const NEAR_PLANE_METERS = 0.04;

// waist_yaw_joint, range="-2.618 2.618" rad.
export const YAW_LIMIT_DEGREES = 150.00;
// waist_pitch_joint, range="-0.52 0.52" rad.
export const PITCH_LIMIT_DEGREES = 29.79;

const YAW_DEGREES_PER_PIXEL = 0.15;      // chase_camera.AZIMUTH_DEGREES_PER_PIXEL
const PITCH_DEGREES_PER_PIXEL = 0.12;    // chase_camera.ELEVATION_DEGREES_PER_PIXEL
// A breath of lag on the look offset only -- not on the mount. The mount is
// rigid on purpose (the view rocks with the gait, which is the whole point);
// this only takes the stair-step off a trackpad's chunky movementX.
const LOOK_LAG_SECONDS = 0.05;

function blend(lagSeconds, elapsedSeconds) {
  // Frame-rate independent exponential follow, same form as chase_camera's.
  if (lagSeconds <= 0 || elapsedSeconds <= 0) return 1;
  return 1 - Math.exp(-elapsedSeconds / lagSeconds);
}

export class FirstPersonCamera {
  constructor(camera) {
    this.camera = camera;
    // Where the drag has asked to look, and where the view actually is.
    this.yawDegrees = 0;
    this.pitchDegrees = 0;
    this.smoothedYawDegrees = 0;
    this.smoothedPitchDegrees = 0;

    // The fixed part of the orientation: torso frame -> camera frame, built
    // from the three axis vectors above rather than by retyping the compiled
    // quaternion, so the comment and the code cannot drift apart.
    const right = new THREE.Vector3(0, -1, 0);
    const up = new THREE.Vector3(0, 0, 1);
    const back = new THREE.Vector3(-1, 0, 0);
    this.mountQuaternion = new THREE.Quaternion().setFromRotationMatrix(
      new THREE.Matrix4().makeBasis(right, up, back));

    this._mountOffset = new THREE.Vector3();
    this._yawQuaternion = new THREE.Quaternion();
    this._pitchQuaternion = new THREE.Quaternion();
    this._forward = new THREE.Vector3();
    this._localUp = new THREE.Vector3(0, 1, 0);
    this._localRight = new THREE.Vector3(1, 0, 0);
  }

  // The same call the third-person camera gets, with the same numbers.
  look(movementX, movementY) {
    this.yawDegrees = Math.max(-YAW_LIMIT_DEGREES, Math.min(YAW_LIMIT_DEGREES,
      this.yawDegrees + movementX * YAW_DEGREES_PER_PIXEL));
    this.pitchDegrees = Math.max(-PITCH_LIMIT_DEGREES, Math.min(PITCH_LIMIT_DEGREES,
      this.pitchDegrees + movementY * PITCH_DEGREES_PER_PIXEL));
  }

  recentreNow() {
    this.yawDegrees = this.pitchDegrees = 0;
    this.smoothedYawDegrees = this.smoothedPitchDegrees = 0;
  }

  // `torsoPosition` / `torsoQuaternion` are the torso body's WORLD pose, the
  // same interpolated numbers the mesh is drawn at, so the eye never lags the
  // head it is bolted to by even one frame.
  update(elapsedSeconds, torsoPosition, torsoQuaternion) {
    const share = blend(LOOK_LAG_SECONDS, elapsedSeconds);
    this.smoothedYawDegrees += (this.yawDegrees - this.smoothedYawDegrees) * share;
    this.smoothedPitchDegrees += (this.pitchDegrees - this.smoothedPitchDegrees) * share;

    this._mountOffset.copy(EYE_MOUNT_IN_TORSO_METERS).applyQuaternion(torsoQuaternion);
    this.camera.position.copy(torsoPosition).add(this._mountOffset);

    // torso -> mount -> yaw about the camera's own up -> pitch about the yawed
    // right. Yaw first, pitch second, which is what a head does and what keeps
    // the horizon level at every yaw.
    this._yawQuaternion.setFromAxisAngle(
      this._localUp, this.smoothedYawDegrees * Math.PI / 180);
    this._pitchQuaternion.setFromAxisAngle(
      this._localRight, this.smoothedPitchDegrees * Math.PI / 180);
    this.camera.quaternion.copy(torsoQuaternion)
      .multiply(this.mountQuaternion)
      .multiply(this._yawQuaternion)
      .multiply(this._pitchQuaternion);

    // The chase camera writes `fov` every frame from its own speed ramp and it
    // still runs while we are in here (its azimuth is what steers the robot),
    // so the lens is re-asserted rather than set once.
    if (Math.abs(this.camera.fov - FIELD_OF_VIEW_DEGREES) > 0.01) {
      this.camera.fov = FIELD_OF_VIEW_DEGREES;
      this.camera.updateProjectionMatrix();
    }
  }

  // Where the view is looking, in the third-person camera's own convention:
  // the MuJoCo azimuth a boom would SIT at to show this direction, i.e. the
  // viewing heading plus half a turn. The wind ribbons are drawn from this, and
  // they must not care which camera is live.
  azimuthDegrees() {
    this._forward.set(0, 0, -1).applyQuaternion(this.camera.quaternion);
    const headingDegrees = Math.atan2(this._forward.y, this._forward.x) * 180 / Math.PI;
    return ((headingDegrees + 180) % 360 + 360) % 360;
  }
}
