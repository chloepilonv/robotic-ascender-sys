// A red pennant on a short pole on top of the G1's head, to show the wind.
//
// WHAT IT IS FOR. The 3-D page already knows the wind -- the snow drifts with
// it and the sidebar prints it -- but neither tells you at a glance which way it
// blows on the mountain the robot is standing on. A flag does: it points
// DOWNWIND, it lifts from hanging limp to flying level as the wind gets up, and
// it ripples faster and harder the harder it blows. Red because everything else
// in frame is snow.
//
// IT IS DECORATION AND NOTHING ELSE. Nothing here reaches the simulator. There
// is no MuJoCo body, no mocap slot, no geom, no mass and no collision shape --
// the flag is a Three.js Group parented to the head's node in the GLB, built and
// animated entirely in the browser, and the pose stream it rides on is
// read-only. So the physics parity question does not arise: the model the
// solver integrates is byte for byte the model it integrated before this file
// existed, because this file cannot write to it.
//
// WHERE IT SITS. The G1 has no head BODY -- `head_link` is a GEOM hanging off
// `torso_link`, and so are the `d435i` camera and the `mid360` lidar site. So
// the mount is an offset in `torso_link`'s frame (the sidecar names that body as
// `torso_body`), taken from the lidar site's own height of 0.456 m with a
// centimetre and a half of clearance. Because the pole is parented to that node
// it TILTS WITH THE ROBOT, which is what a pole bolted to a head does -- while
// the cloth is counter-rotated back into the world frame every frame, because
// the wind does not care which way the robot is facing.
//
// The mount is also 7.5 cm BEHIND the `d435i` camera's own plane and 8.5 cm
// above it. That camera looks along +x with a 58 deg vertical field of view,
// which on the eyes' 4:3 frame is a 42.7 deg half DIAGONAL; the pole's base sits
// 131.4 deg off its optical axis and the pole's top 107.7 deg. Both are past
// 90 deg, i.e. behind the image plane, where no field of view can reach -- so a
// future in-model version of this flag could not get into the robot's eyes
// either. (Arithmetic from the MJCF's own numbers; nothing here is eyeballed.)
//
// COST. Two draw calls and 250 triangles against the world's ~780,000, one
// small vertex shader, and no per-frame allocation: the ripple is entirely a
// function of `uTime` on the GPU, and the JavaScript side writes four uniforms
// and one quaternion a frame.
//
// EVERYTHING HERE IS Z-UP, like the rest of this directory.
import * as THREE from './vendor/three.module.js';

// torso-local metres. x/y are the head geom's own centre (`head_link` sits at
// x = 0.0039635); z clears the `mid360` site at 0.456.
const MOUNT_IN_BODY = [0.0039635, 0.0, 0.471];

const POLE_LENGTH_METERS = 0.15;
const POLE_RADIUS_METERS = 0.004;
const PENNANT_LENGTH_METERS = 0.12;      // pole to tip, along the wind
const PENNANT_HEIGHT_METERS = 0.08;      // at the pole; it tapers to the tip
const PENNANT_TIP_SHARE = 0.22;          // the tip's height, as a share of the root's
const CLOTH_SEGMENTS_ALONG = 26;         // enough that the ripple is a curve, not a crease
const CLOTH_SEGMENTS_ACROSS = 3;

// Lift. 0 m/s hangs it straight down the pole; FULL_LIFT flies it level. 12 m/s
// is about where a real pennant this size stops climbing.
const FULL_LIFT_METERS_PER_SECOND = 12.0;

// The ripple. Amplitude is a share of the cloth's LENGTH, so the shape is scale
// free, and it grows from 0 at the pole to its full value at the tip (the root
// is nailed to the pole; the tip whips). Both amplitude and frequency are
// exactly zero at zero wind, which is what "hangs limp" means.
const FLUTTER_AMPLITUDE_SHARE = 0.30;
const FLUTTER_REFERENCE_METERS_PER_SECOND = 10.0;
const FLUTTER_WAVES_ALONG_CLOTH = 1.15;
const FLUTTER_HZ_PER_METER_PER_SECOND = 0.32;

const POLE_COLOUR = 0x22242a;
const PENNANT_COLOUR = 0xc41414;

// How fast the drawn wind chases the streamed wind. The stream gusts at 50 Hz
// and a flag has mass: without this the cloth snaps between angles.
const WIND_SMOOTHING_SECONDS = 0.25;

export function liftShare(speedMetersPerSecond) {
  return Math.min(1, Math.max(0, speedMetersPerSecond) / FULL_LIFT_METERS_PER_SECOND);
}

export function flutterAmplitudeMeters(speedMetersPerSecond) {
  const share = Math.min(1, Math.max(0, speedMetersPerSecond)
                            / FLUTTER_REFERENCE_METERS_PER_SECOND);
  return FLUTTER_AMPLITUDE_SHARE * PENNANT_LENGTH_METERS * share;
}

export function flutterFrequencyHertz(speedMetersPerSecond) {
  const speed = Math.max(0, speedMetersPerSecond);
  return speed <= 0 ? 0 : FLUTTER_HZ_PER_METER_PER_SECOND * speed;
}

// The cloth: a plane in the local XZ (length along +x, height along +z),
// tapered to a pennant point, with `uv.x` carrying the share of the way to the
// tip that the ripple is a function of.
function makeClothGeometry() {
  const geometry = new THREE.PlaneGeometry(
    PENNANT_LENGTH_METERS, PENNANT_HEIGHT_METERS,
    CLOTH_SEGMENTS_ALONG, CLOTH_SEGMENTS_ACROSS);
  // PlaneGeometry is built in XY with +z normal; stand it up into XZ with +y
  // normal, and shift it so x = 0 is the edge at the pole rather than the middle.
  geometry.rotateX(-Math.PI / 2);
  geometry.translate(PENNANT_LENGTH_METERS / 2, 0, 0);
  const positions = geometry.getAttribute('position');
  const uvs = geometry.getAttribute('uv');
  for (let index = 0; index < positions.count; index++) {
    const share = uvs.getX(index);          // 0 at the pole, 1 at the tip
    const taper = 1 - (1 - PENNANT_TIP_SHARE) * share;
    positions.setZ(index, positions.getZ(index) * taper);
  }
  positions.needsUpdate = true;
  geometry.computeVertexNormals();
  return geometry;
}

// The ripple, injected into a MeshStandardMaterial so it keeps the scene's
// lighting, shadows and fog. The normal is the ANALYTIC one -- the cross of the
// displaced tangent with the height direction -- so the cloth catches the sun
// along the wave instead of reading as a flat card with a wobbly outline.
function dressCloth(material, uniforms) {
  material.onBeforeCompile = shader => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>
        uniform float uTime, uAmplitude, uWaves, uFrequency, uLength;
        float clothSway(float share) {
          return uAmplitude * share
                 * sin(6.2831853 * (uWaves * share - uFrequency * uTime));
        }
        float clothSwaySlope(float share) {
          float k = 6.2831853 * uWaves;
          float p = 6.2831853 * (uWaves * share - uFrequency * uTime);
          return uAmplitude * (sin(p) + share * k * cos(p));
        }`)
      // The flat cloth's normal is +y everywhere, so the displaced one is the
      // cross of the swayed tangent with the height direction, which is this.
      // `side: DoubleSide` flips it per fragment for the back face already.
      .replace('#include <beginnormal_vertex>', `#include <beginnormal_vertex>
        objectNormal = normalize(vec3(
          -clothSwaySlope(uv.x) / max(uLength, 1e-6), 1.0, 0.0));`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        transformed.y += clothSway(uv.x);`);
  };
  material.needsUpdate = true;
}

export class WindFlag {
  // `mountNode` is the GLB node for the body the flag rides on (`torso_link`).
  constructor(mountNode) {
    this.mount = new THREE.Group();
    this.mount.position.set(MOUNT_IN_BODY[0], MOUNT_IN_BODY[1], MOUNT_IN_BODY[2]);

    // The pole. CylinderGeometry runs along +y, so stand it along +z and lift
    // it so it grows UP from the mount rather than straddling it.
    const poleGeometry = new THREE.CylinderGeometry(
      POLE_RADIUS_METERS, POLE_RADIUS_METERS * 1.25, POLE_LENGTH_METERS, 8, 1);
    poleGeometry.rotateX(Math.PI / 2);
    poleGeometry.translate(0, 0, POLE_LENGTH_METERS / 2);
    const pole = new THREE.Mesh(poleGeometry, new THREE.MeshStandardMaterial({
      color: POLE_COLOUR, roughness: 0.45, metalness: 0.6 }));
    pole.castShadow = true;
    this.mount.add(pole);

    // `world` is counter-rotated out of the robot's frame every update, so
    // everything inside it is in WORLD axes: +x east, +y north, +z up.
    this.world = new THREE.Group();
    this.world.position.set(0, 0, POLE_LENGTH_METERS);
    this.mount.add(this.world);

    // `heading` turns the cloth downwind; `hang` drops it from level (+x) to
    // straight down (-z) as the wind dies. Two nested groups rather than one
    // quaternion because they are two independent things and this way each one
    // is readable in a debugger.
    this.heading = new THREE.Group();
    this.world.add(this.heading);
    this.hang = new THREE.Group();
    this.heading.add(this.hang);

    this.uniforms = {
      uTime: { value: 0 },
      uAmplitude: { value: 0 },
      uWaves: { value: FLUTTER_WAVES_ALONG_CLOTH },
      uFrequency: { value: 0 },
      uLength: { value: PENNANT_LENGTH_METERS },
    };
    const clothMaterial = new THREE.MeshStandardMaterial({
      color: PENNANT_COLOUR, roughness: 0.82, metalness: 0.0,
      side: THREE.DoubleSide });
    dressCloth(clothMaterial, this.uniforms);
    this.cloth = new THREE.Mesh(makeClothGeometry(), clothMaterial);
    this.cloth.castShadow = true;
    this.cloth.frustumCulled = false;   // the shader moves it off its own bounds
    this.hang.add(this.cloth);

    // Scratch, allocated once: `update` runs at the display rate and a Vector3
    // per frame is garbage the collector has to chase.
    this._mountQuaternion = new THREE.Quaternion();
    this._scratchPosition = new THREE.Vector3();
    this._scratchScale = new THREE.Vector3();
    this._upAxis = new THREE.Vector3(0, 0, 1);
    this._smoothedEast = 0;
    this._smoothedNorth = 0;
    this._seeded = false;
    this.attached = false;
    if (mountNode) {
      mountNode.add(this.mount);
      this.attached = true;
    }
  }

  dispose() {
    this.mount.traverse(object => {
      if (object.geometry) object.geometry.dispose();
      if (object.material) object.material.dispose();
    });
    if (this.mount.parent) this.mount.parent.remove(this.mount);
  }

  set visible(value) { this.mount.visible = Boolean(value); }
  get visible() { return this.mount.visible; }

  // `windEast` / `windNorth` are the world-frame wind VELOCITY in m/s, exactly
  // what `stage.setWind` is handed from the state message's
  // `wind_velocity_world_meters_per_second` (or, when only the scalars are on
  // the wire, from `wind_speed_mps` and `wind_heading_degrees`, which
  // render3d.html reassembles into the same vector -- heading is CCW from east,
  // so east is the cosine).
  update(elapsedSeconds, windEast, windNorth) {
    if (!this.attached) return;
    if (!this._seeded) {
      this._smoothedEast = windEast; this._smoothedNorth = windNorth;
      this._seeded = true;
    } else if (elapsedSeconds > 0) {
      const blend = 1 - Math.exp(-elapsedSeconds / WIND_SMOOTHING_SECONDS);
      this._smoothedEast += (windEast - this._smoothedEast) * blend;
      this._smoothedNorth += (windNorth - this._smoothedNorth) * blend;
    }
    const east = this._smoothedEast, north = this._smoothedNorth;
    const speed = Math.hypot(east, north);

    // Out of the robot's frame and into the world's. `updateWorldMatrix` walks
    // the (three-deep) parent chain rather than trusting a matrix the renderer
    // has not refreshed yet this frame.
    this.mount.updateWorldMatrix(true, false);
    this.mount.matrixWorld.decompose(
      this._scratchPosition, this._mountQuaternion, this._scratchScale);
    this.world.quaternion.copy(this._mountQuaternion).invert();

    // A dead-calm heading is undefined, so keep the last one: a flag hanging
    // limp should not spin.
    if (speed > 1e-4) {
      this.heading.quaternion.setFromAxisAngle(
        this._upAxis, Math.atan2(north, east));
    }
    // Rotating about +y takes +x toward -z, so this angle IS the drop from
    // level: 0 at full lift, 90 deg with no wind at all.
    this.hang.rotation.y = (Math.PI / 2) * (1 - liftShare(speed));

    this.uniforms.uTime.value += elapsedSeconds;
    this.uniforms.uAmplitude.value = flutterAmplitudeMeters(speed);
    this.uniforms.uFrequency.value = flutterFrequencyHertz(speed);
  }
}

export { MOUNT_IN_BODY, POLE_LENGTH_METERS, PENNANT_LENGTH_METERS,
         PENNANT_HEIGHT_METERS, FULL_LIFT_METERS_PER_SECOND };
