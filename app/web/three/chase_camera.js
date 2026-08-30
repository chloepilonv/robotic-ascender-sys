// The third-person camera. Everything a walking simulator does and a MuJoCo
// orbit does not.
//
// The runtime's own `ChaseCamera` (app/harness/runtime.py) is a rigid orbit: it
// sits at an azimuth/elevation around the pelvis and snaps there instantly. That
// is fine for a recorded mp4 and reads as a tripod when you are driving. This is
// the game version -- a spring arm that lags, frames the robot off-centre, gets
// out of the way of the mountain, leans into the slope, looks where the robot is
// going, and drifts back behind it when you stop steering.
//
// IT STILL SPEAKS THE SERVER'S LANGUAGE. `azimuthDegrees` / `elevationDegrees`
// are exactly the two numbers app/web/index.html sends, in exactly MuJoCo's
// convention (azimuth is where the camera SITS, elevation is negative when it is
// above looking down), and they go up the socket unchanged. So the robot still
// steers toward the camera heading, the guide follower still agrees with the
// picture, and the recorded video still frames what the driver was looking at.
//
//   offset from the pelvis = distance * (cos(el)cos(az), cos(el)sin(az), -sin(el))
//
// which puts azimuth 180 / elevation -15 behind the robot and slightly above,
// looking uphill along +x, the same default the 2-D page opens on.
import * as THREE from './vendor/three.module.js';

export const DEFAULT_AZIMUTH_DEGREES = 180;
export const DEFAULT_ELEVATION_DEGREES = -15;
// MuJoCo elevation is negative when the camera is above the robot looking down.
// The ceiling is a few degrees above level on purpose: the face is 38.6 deg, and
// a camera allowed under the slope puts the climber behind the mountain.
export const ELEVATION_MINIMUM_DEGREES = -70;
export const ELEVATION_MAXIMUM_DEGREES = 8;
export const AZIMUTH_DEGREES_PER_PIXEL = 0.15;
export const ELEVATION_DEGREES_PER_PIXEL = 0.12;

const BOOM_LENGTH_METERS = 4.3;
const TARGET_HEIGHT_METERS = 0.80;        // above the pelvis: head, not hips
// Position lags harder than rotation, which is the whole trick: the arm swings
// smoothly while the framing stays locked on the subject.
const POSITION_LAG_SECONDS = 0.15;
const ROTATION_LAG_SECONDS = 0.34;
const AIM_LAG_SECONDS = 0.22;
// Off-centre framing: the robot sits left of centre and low, so the mountain it
// is climbing gets the rest of the frame.
const FRAME_LATERAL_METERS = 0.70;
const FRAME_VERTICAL_METERS = 0.30;
const LOOK_AHEAD_SECONDS = 0.55;
const LOOK_AHEAD_MAXIMUM_METERS = 1.5;
const FIELD_OF_VIEW_SLOW = 50;
const FIELD_OF_VIEW_FAST = 58;
const FIELD_OF_VIEW_FULL_SPEED = 1.6;     // m/s where the wide end is reached
const FIELD_OF_VIEW_LAG_SECONDS = 0.6;
// How close the camera may come to the ground before it is pushed up, and how
// much it pulls in off a wall it would otherwise be buried in.
const TERRAIN_CLEARANCE_METERS = 0.55;
const COLLISION_PULL_IN_METERS = 0.30;
const MINIMUM_BOOM_METERS = 1.6;
// How many points along the boom are tested against the height field. Twelve on
// a 4.3 m arm is a sample every 36 cm, which is finer than the camera's own
// clearance.
const COLLISION_SAMPLES = 12;
const RECENTRE_IDLE_SECONDS = 2.0;
const RECENTRE_LAG_SECONDS = 1.1;
const SWAY_AMPLITUDE_METERS = 0.035;
// The boom's own pitch range, which is wider than the mouse's because the slope
// bias above pushes past it. Terrain collision and the ride-up clamp are what
// keep the camera out of the mountain now, not this number.
const BOOM_ELEVATION_MINIMUM_DEGREES = -75;
const BOOM_ELEVATION_MAXIMUM_DEGREES = 38;

function blend(lagSeconds, elapsedSeconds) {
  // The frame-rate independent form of an exponential follow. A plain
  // `x += (target - x) * 0.1` is a different camera at 60 fps and at 144.
  if (lagSeconds <= 0 || elapsedSeconds <= 0) return 1;
  return 1 - Math.exp(-elapsedSeconds / lagSeconds);
}

function shortestAngleDegrees(from, to) {
  return ((to - from + 540) % 360) - 180;
}

export class ChaseCamera {
  constructor(camera) {
    this.camera = camera;
    this.azimuthDegrees = DEFAULT_AZIMUTH_DEGREES;
    this.elevationDegrees = DEFAULT_ELEVATION_DEGREES;
    // What the boom is ACTUALLY at, as opposed to what the mouse asked for.
    this.smoothedAzimuthDegrees = DEFAULT_AZIMUTH_DEGREES;
    this.smoothedElevationDegrees = DEFAULT_ELEVATION_DEGREES;
    this.followPosition = new THREE.Vector3();
    this.aimPosition = new THREE.Vector3();
    this.fieldOfView = FIELD_OF_VIEW_SLOW;
    this.secondsSinceMouse = 99;
    this.autoRecentre = true;
    this.seeded = false;
    this.swayPhase = Math.random() * 100;
    this.boomLength = BOOM_LENGTH_METERS;

    this._raycaster = new THREE.Raycaster();
    this._previousTarget = new THREE.Vector3();
    this._velocity = new THREE.Vector3();
    this._desired = new THREE.Vector3();
    this._offset = new THREE.Vector3();
    this._forward = new THREE.Vector3();
    this._right = new THREE.Vector3();
    this._down = new THREE.Vector3(0, 0, -1);
    this._up = new THREE.Vector3(0, 0, 1);
    this._aimWanted = new THREE.Vector3();
  }

  // The mouse. Same numbers, same clamps, same feel as the 2-D page, so a judge
  // who learned one has learned the other.
  look(movementX, movementY) {
    this.azimuthDegrees =
      ((this.azimuthDegrees + movementX * AZIMUTH_DEGREES_PER_PIXEL) % 360 + 360) % 360;
    this.elevationDegrees = Math.max(ELEVATION_MINIMUM_DEGREES,
      Math.min(ELEVATION_MAXIMUM_DEGREES,
               this.elevationDegrees + movementY * ELEVATION_DEGREES_PER_PIXEL));
    this.secondsSinceMouse = 0;
  }

  recentreNow() {
    this.azimuthDegrees = DEFAULT_AZIMUTH_DEGREES;
    this.elevationDegrees = DEFAULT_ELEVATION_DEGREES;
    this.secondsSinceMouse = 99;
  }

  // `target` is the pelvis, `headingRadians` its yaw about world z, `heightField`
  // the coarse world-XY height grid the boom is not allowed to go under (see
  // world.js -- raycasting a 300k-triangle terrain per frame cost two thirds of
  // the frame rate), `slopeDegrees` how steep the face is.
  update(elapsedSeconds, target, headingRadians, heightField, slopeDegrees) {
    this.secondsSinceMouse += elapsedSeconds;
    if (!this.seeded) {
      this.followPosition.copy(target);
      this._previousTarget.copy(target);
      this.aimPosition.copy(target);
      this.seeded = true;
    }

    // --- what the boom is trying to look at -----------------------------
    // Velocity is measured from the target itself, so it needs nothing from the
    // socket beyond the pose that is already arriving.
    const inverseElapsed = elapsedSeconds > 1e-4 ? 1 / elapsedSeconds : 0;
    this._velocity.copy(target).sub(this._previousTarget).multiplyScalar(inverseElapsed);
    this._previousTarget.copy(target);
    const speed = Math.min(this._velocity.length(), 4.0);

    this.followPosition.lerp(target, blend(POSITION_LAG_SECONDS, elapsedSeconds));

    // --- where the boom is pointing --------------------------------------
    // After two idle seconds the camera drifts back behind the robot, the way
    // every third-person game does it. It is a drift, not a snap: the target is
    // the mouse's own azimuth, nudged toward the robot's heading.
    if (this.autoRecentre && this.secondsSinceMouse > RECENTRE_IDLE_SECONDS) {
      // The runtime turns the browser's azimuth into a heading by adding half a
      // turn (runtime.BROWSER_AZIMUTH_OFFSET_DEGREES), i.e. the camera's VIEWING
      // direction is where the robot walks. So the azimuth that puts the camera
      // behind the robot is its yaw minus half a turn.
      const behindDegrees = headingRadians * 180 / Math.PI - 180;
      const share = blend(RECENTRE_LAG_SECONDS, elapsedSeconds);
      this.azimuthDegrees = ((this.azimuthDegrees
        + shortestAngleDegrees(this.azimuthDegrees, behindDegrees) * share) % 360 + 360) % 360;
    }
    // SLOPE-AWARE PITCH, and the sign of it is the whole difference between a
    // walking simulator and a map. MuJoCo elevation is NEGATIVE when the camera
    // is above the target looking down. On a 38.6 deg face the default -15 puts
    // the boom nearly perpendicular to the snow: the first build of this camera
    // framed a white sheet with no horizon in it at all. Biasing elevation
    // POSITIVE drops the camera down the fall line until it is looking UP the
    // slope past the climber, which is the shot the mountain is for.
    //
    // The bias is applied to the BOOM only. `this.elevationDegrees` -- the number
    // that goes up the socket and steers the runtime's own camera -- keeps the
    // 2-D page's [-70, 8] range untouched, so both views still agree about where
    // the driver is looking.
    const slopeBiasDegrees = Math.min(28, Math.max(0, slopeDegrees) * 0.62);
    const wantedElevation = Math.max(BOOM_ELEVATION_MINIMUM_DEGREES,
      Math.min(BOOM_ELEVATION_MAXIMUM_DEGREES, this.elevationDegrees + slopeBiasDegrees));

    const rotationBlend = blend(ROTATION_LAG_SECONDS, elapsedSeconds);
    this.smoothedAzimuthDegrees = this.smoothedAzimuthDegrees
      + shortestAngleDegrees(this.smoothedAzimuthDegrees, this.azimuthDegrees) * rotationBlend;
    this.smoothedElevationDegrees +=
      (wantedElevation - this.smoothedElevationDegrees) * rotationBlend;

    // --- the arm ----------------------------------------------------------
    const azimuth = this.smoothedAzimuthDegrees * Math.PI / 180;
    const elevation = this.smoothedElevationDegrees * Math.PI / 180;
    this._offset.set(Math.cos(elevation) * Math.cos(azimuth),
                     Math.cos(elevation) * Math.sin(azimuth),
                     -Math.sin(elevation));
    // The pivot is chest height, not the hips: a boom anchored at the pelvis on a
    // steep face frames a lot of snow and not much robot.
    const pivotZ = this.followPosition.z + TARGET_HEIGHT_METERS;
    this._desired.set(this.followPosition.x, this.followPosition.y, pivotZ)
      .addScaledVector(this._offset, this.boomLength);

    // Terrain collision: march out along the boom and stop at the first sample
    // that is inside the mountain, then pull in a little further so the near
    // plane is not scraping snow. Without this, one step onto the uphill side of
    // a fold buries the camera and the robot vanishes.
    let allowedLength = this.boomLength;
    if (heightField) {
      for (let step = 1; step <= COLLISION_SAMPLES; step++) {
        const distance = this.boomLength * step / COLLISION_SAMPLES;
        const x = this.followPosition.x + this._offset.x * distance;
        const y = this.followPosition.y + this._offset.y * distance;
        const z = pivotZ + this._offset.z * distance;
        const groundZ = heightField.heightAt(x, y);
        if (Number.isNaN(groundZ)) continue;          // past the patch edge
        if (z < groundZ + TERRAIN_CLEARANCE_METERS) {
          allowedLength = Math.max(MINIMUM_BOOM_METERS,
                                   distance - COLLISION_PULL_IN_METERS);
          break;
        }
      }
      this._desired.set(this.followPosition.x, this.followPosition.y, pivotZ)
        .addScaledVector(this._offset, allowedLength);
      // And ride up: a boom that cleared everything can still end up under a
      // rise once the sway is added, so hold a clearance over the ground the
      // camera actually ends up above.
      const groundZ = heightField.heightAt(this._desired.x, this._desired.y);
      if (!Number.isNaN(groundZ)) {
        this._desired.z = Math.max(this._desired.z, groundZ + TERRAIN_CLEARANCE_METERS);
      }
    }

    // A breath of sway, in camera-local axes so it never fights the arm.
    this.swayPhase += elapsedSeconds;
    this._forward.copy(this._offset).negate();
    this._right.crossVectors(this._forward, this._up).normalize();
    this._desired
      .addScaledVector(this._right, Math.sin(this.swayPhase * 0.62) * SWAY_AMPLITUDE_METERS)
      .addScaledVector(this._up, Math.sin(this.swayPhase * 0.83 + 1.7) * SWAY_AMPLITUDE_METERS);
    this.camera.position.copy(this._desired);

    // --- framing ----------------------------------------------------------
    // Off-centre by aiming beside the robot rather than by shifting the
    // projection: the aim point is what the look-ahead and the lag act on, and
    // one smoothed point is easier to reason about than a moving frustum.
    this._aimWanted.set(this.followPosition.x, this.followPosition.y,
                        this.followPosition.z + TARGET_HEIGHT_METERS * 0.7)
      .addScaledVector(this._right, FRAME_LATERAL_METERS)
      .addScaledVector(this._up, FRAME_VERTICAL_METERS);
    if (speed > 0.05) {
      this._aimWanted.addScaledVector(
        this._velocity,
        Math.min(LOOK_AHEAD_SECONDS, LOOK_AHEAD_MAXIMUM_METERS / Math.max(speed, 0.05)));
    }
    this.aimPosition.lerp(this._aimWanted, blend(AIM_LAG_SECONDS, elapsedSeconds));
    this.camera.lookAt(this.aimPosition);

    // Speed opens the lens, which is the cheapest way to make walking feel like
    // moving. Lagged hard so it breathes rather than pumps.
    const share = Math.min(1, speed / FIELD_OF_VIEW_FULL_SPEED);
    const wantedFieldOfView =
      FIELD_OF_VIEW_SLOW + (FIELD_OF_VIEW_FAST - FIELD_OF_VIEW_SLOW) * share;
    this.fieldOfView += (wantedFieldOfView - this.fieldOfView)
      * blend(FIELD_OF_VIEW_LAG_SECONDS, elapsedSeconds);
    if (Math.abs(this.camera.fov - this.fieldOfView) > 0.01) {
      this.camera.fov = this.fieldOfView;
      this.camera.updateProjectionMatrix();
    }
  }
}
