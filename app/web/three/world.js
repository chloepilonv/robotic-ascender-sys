// The mountain: what gets loaded once and dressed, as opposed to what moves.
//
// `app/harness/export_scene.py` writes one GLB per world -- a node per MuJoCo
// body, each visible geom a child mesh in its body-local frame -- plus a JSON
// sidecar carrying the key from body INDEX (which is all the pose message has)
// to node NAME. This module turns that pair into a lit, fogged, snow-covered
// Three.js scene, and hands back a flat array of body objects the pose stream
// can drive.
//
// EVERYTHING HERE IS Z-UP. MuJoCo is Z-up, the GLB is written Z-up, and
// `THREE.Object3D.DEFAULT_UP` is set to (0,0,1) before anything is constructed,
// so no axis conversion happens anywhere between the solver and the screen. A
// glTF is conventionally Y-up, but nothing in the format requires it and
// rotating the world would mean rotating every streamed pose too.
import * as THREE from './vendor/three.module.js';
import { GLTFLoader } from './vendor/addons/loaders/GLTFLoader.js';
import { WindFlag, HIKER_MOUNT_IN_BODY } from './flag.js';

// The sun the recorded mp4 uses (app/harness/graphics.py). Repeating the same
// two numbers is what keeps the WebGL view and the JPEG view the same weather;
// the sidecar carries them per world, and these are only the fallback.
const SUN_ELEVATION_DEGREES = 16.0;
const SUN_AZIMUTH_DEGREES = 215.0;

// Footprints are painted into ONE canvas covering the terrain's world XY box
// and sampled by the terrain shader. No decal geometry, no z-fighting, no
// second draw call -- and a 1024 canvas over a 25 m patch is 2.4 cm a pixel,
// finer than a boot.
const FOOTPRINT_CANVAS_PIXELS = 1024;
const FOOTPRINT_LENGTH_METERS = 0.24;
const FOOTPRINT_WIDTH_METERS = 0.13;
// Prints fade so a long session does not end up a uniformly trampled sheet.
const FOOTPRINT_FADE_PER_SECOND = 0.018;
const FOOTPRINT_FADE_INTERVAL_SECONDS = 0.35;

const SNOW_PARTICLE_COUNT = 9000;
const SNOW_BOX_METERS = 44.0;          // the cube of snow that follows the camera
const SNOW_FALL_METERS_PER_SECOND = 1.6;

// Shadows: one directional light, its orthographic shadow camera fitted to a
// box this wide around the robot. Wide enough that the whole climber and the
// slope it stands on cast, tight enough that 2048 texels still resolve a boot.
const SHADOW_HALF_EXTENT_METERS = 9.0;
const SHADOW_MAP_PIXELS = 2048;
const SUN_DISTANCE_METERS = 60.0;

const FOG_COLOUR = 0xbfd0e2;           // cold blue-white, matching graphics.py's haze
const FOG_DENSITY_PER_METER = 0.0085;

// ------------------------------------------------------------- VISIBILITY
// HOW FAR YOU CAN SEE, IN METRES, AND NOTHING ELSE (user's ruling,
// 2026-08-30). It used to be a `storm` switch whose thickness was derived from
// the wind speed; it is now its own dial, and the wind has nothing to do with
// it. Left of the slider is 100 m -- CLEAR, the page's own fog untouched and
// the robot's eyes untouched. Right is 3 m, a white-out you cannot see the
// hiker across.
//
// THE SAME THREE NUMBERS LIVE IN app/harness/storm.py (CLEAR_VISIBILITY_METERS,
// MINIMUM_VISIBILITY_METERS, and this share). If one moves, move the other, or
// the picture and the robot stop being in the same weather.
//
// The SHARE is logarithmic because visibility is: the step from 100 m to 50 m
// is barely a haze and the step from 6 m to 3 m is the difference between
// navigating and not. `share = ln(clear / v) / ln(clear / minimum)` puts the
// midpoint at 17.3 m, which is why the slider's own middle reads about 15 m.
const CLEAR_VISIBILITY_METERS = 100.0;
const MINIMUM_VISIBILITY_METERS = 3.0;
const VISIBILITY_LOG_SPAN = Math.log(CLEAR_VISIBILITY_METERS
                                     / MINIMUM_VISIBILITY_METERS);

// -> 0 at 100 m (clear), 1 at 3 m (white-out). Clamped both ends.
export function whiteoutShare(visibilityMeters) {
  const visibility = Math.max(MINIMUM_VISIBILITY_METERS,
    Math.min(CLEAR_VISIBILITY_METERS, Number(visibilityMeters)
             || CLEAR_VISIBILITY_METERS));
  return Math.max(0, Math.min(1,
    Math.log(CLEAR_VISIBILITY_METERS / visibility) / VISIBILITY_LOG_SPAN));
}

// A small value-noise pair, shared by the terrain's colour and its roughness.
// Three octaves is enough for "the snow is not a flat sheet" and cheap enough
// to run per fragment on an integrated GPU.
const NOISE_GLSL = /* glsl */`
  float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
  }
  float valueNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0)), d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
  }
  float fbm(vec2 p) {
    float total = 0.0, amplitude = 0.5;
    for (int octave = 0; octave < 3; octave++) {
      total += valueNoise(p) * amplitude;
      p *= 2.03;
      amplitude *= 0.5;
    }
    return total;
  }
`;

function sunDirection(elevationDegrees, azimuthDegrees) {
  // graphics.py's own construction, so the shadows fall the same way as the
  // recorded video's: the light POINTS along this vector, the sun SITS along
  // its negation.
  const elevation = elevationDegrees * Math.PI / 180;
  const azimuth = azimuthDegrees * Math.PI / 180;
  return new THREE.Vector3(
    -Math.cos(elevation) * Math.cos(azimuth),
    -Math.cos(elevation) * Math.sin(azimuth),
    -Math.sin(elevation)).normalize();
}

// ---------------------------------------------------------------- footprints
class FootprintCanvas {
  constructor(bounds) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.canvas.height = FOOTPRINT_CANVAS_PIXELS;
    this.context = this.canvas.getContext('2d');
    this.context.fillStyle = '#000';
    this.context.fillRect(0, 0, FOOTPRINT_CANVAS_PIXELS, FOOTPRINT_CANVAS_PIXELS);
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.NoColorSpace;
    this.texture.wrapS = this.texture.wrapT = THREE.ClampToEdgeWrapping;
    this.originX = bounds.originX; this.originY = bounds.originY;
    this.sizeX = bounds.sizeX; this.sizeY = bounds.sizeY;
    this.dirty = false;
    this.secondsSinceFade = 0;
  }

  // world metres -> canvas pixels. +y in the world runs UP the canvas, so the
  // row is flipped; the shader samples with the same convention.
  stamp(x, y, yawRadians) {
    const u = (x - this.originX) / this.sizeX;
    const v = (y - this.originY) / this.sizeY;
    if (u < 0 || u > 1 || v < 0 || v > 1) return;
    const pixelsPerMeterX = FOOTPRINT_CANVAS_PIXELS / this.sizeX;
    const pixelsPerMeterY = FOOTPRINT_CANVAS_PIXELS / this.sizeY;
    const context = this.context;
    context.save();
    context.translate(u * FOOTPRINT_CANVAS_PIXELS, (1 - v) * FOOTPRINT_CANVAS_PIXELS);
    context.rotate(-yawRadians);
    context.scale(FOOTPRINT_LENGTH_METERS * pixelsPerMeterX,
                  FOOTPRINT_WIDTH_METERS * pixelsPerMeterY);
    const gradient = context.createRadialGradient(0, 0, 0, 0, 0, 1);
    gradient.addColorStop(0, 'rgba(255,255,255,0.95)');
    gradient.addColorStop(0.55, 'rgba(255,255,255,0.55)');
    gradient.addColorStop(1, 'rgba(255,255,255,0)');
    context.fillStyle = gradient;
    context.globalCompositeOperation = 'lighter';
    context.beginPath();
    context.arc(0, 0, 1, 0, Math.PI * 2);
    context.fill();
    context.restore();
    this.dirty = true;
  }

  update(elapsedSeconds) {
    this.secondsSinceFade += elapsedSeconds;
    if (this.secondsSinceFade >= FOOTPRINT_FADE_INTERVAL_SECONDS) {
      const share = FOOTPRINT_FADE_PER_SECOND * this.secondsSinceFade;
      this.secondsSinceFade = 0;
      this.context.save();
      this.context.globalCompositeOperation = 'source-over';
      this.context.fillStyle = `rgba(0,0,0,${share.toFixed(3)})`;
      this.context.fillRect(0, 0, FOOTPRINT_CANVAS_PIXELS, FOOTPRINT_CANVAS_PIXELS);
      this.context.restore();
      this.dirty = true;
    }
    if (this.dirty) { this.texture.needsUpdate = true; this.dirty = false; }
  }
}

// ------------------------------------------------------- the terrain, cheaply
// WHY THIS EXISTS, measured rather than assumed. The camera needs two answers
// every frame -- "is the mountain between me and the robot" and "how far above
// the ground am I" -- and the obvious way to get them is THREE.Raycaster
// against the terrain mesh. That mesh is 300,000 triangles with no BVH, so
// every ray is a linear scan of all of them: the frame rate went from 120 fps
// to 37 the moment the boom started actually hitting the slope.
//
// So the terrain is rasterised ONCE at load into a coarse world-XY height grid
// -- one pass over its vertices, keeping the highest z per cell -- and both
// questions become bilinear lookups. A camera does not need triangle precision;
// it needs to not be inside the mountain, and 12 cm cells on a 25 m patch are
// two orders of magnitude finer than that.
const HEIGHT_FIELD_CELLS = 256;

class TerrainHeightField {
  constructor(meshes) {
    this.cells = HEIGHT_FIELD_CELLS;
    this.grid = new Float32Array(this.cells * this.cells).fill(-Infinity);
    const box = new THREE.Box3();
    const point = new THREE.Vector3();
    for (const mesh of meshes) {
      mesh.updateWorldMatrix(true, false);
      const positions = mesh.geometry.getAttribute('position');
      for (let index = 0; index < positions.count; index++) {
        point.fromBufferAttribute(positions, index).applyMatrix4(mesh.matrixWorld);
        box.expandByPoint(point);
      }
    }
    this.minX = box.min.x; this.minY = box.min.y;
    this.sizeX = Math.max(box.max.x - box.min.x, 1e-6);
    this.sizeY = Math.max(box.max.y - box.min.y, 1e-6);
    this.minZ = box.min.z;
    for (const mesh of meshes) {
      const positions = mesh.geometry.getAttribute('position');
      for (let index = 0; index < positions.count; index++) {
        point.fromBufferAttribute(positions, index).applyMatrix4(mesh.matrixWorld);
        const column = Math.min(this.cells - 1, Math.max(0, Math.floor(
          (point.x - this.minX) / this.sizeX * this.cells)));
        const row = Math.min(this.cells - 1, Math.max(0, Math.floor(
          (point.y - this.minY) / this.sizeY * this.cells)));
        const cell = row * this.cells + column;
        if (point.z > this.grid[cell]) this.grid[cell] = point.z;
      }
    }
    // A cell no vertex landed in would read -Infinity and let the camera sink
    // through the ground there, so empty cells inherit their nearest filled
    // neighbour by two sweeps.
    this._fillHoles();
  }

  _fillHoles() {
    const cells = this.cells, grid = this.grid;
    for (let pass = 0; pass < 2; pass++) {
      const forward = pass === 0;
      for (let step = 0; step < cells * cells; step++) {
        const index = forward ? step : cells * cells - 1 - step;
        if (grid[index] !== -Infinity) continue;
        const row = Math.floor(index / cells), column = index % cells;
        let best = -Infinity;
        for (const [rowOffset, columnOffset] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
          const neighbourRow = row + rowOffset, neighbourColumn = column + columnOffset;
          if (neighbourRow < 0 || neighbourRow >= cells) continue;
          if (neighbourColumn < 0 || neighbourColumn >= cells) continue;
          const value = grid[neighbourRow * cells + neighbourColumn];
          if (value > best) best = value;
        }
        if (best !== -Infinity) grid[index] = best;
      }
    }
  }

  // Bilinear, and NaN outside the patch -- a sandbox world's edge is a real
  // cliff and the camera should be allowed past it rather than clamped to a
  // guessed height.
  heightAt(x, y) {
    const u = (x - this.minX) / this.sizeX * this.cells - 0.5;
    const v = (y - this.minY) / this.sizeY * this.cells - 0.5;
    if (u < 0 || v < 0 || u > this.cells - 1 || v > this.cells - 1) return NaN;
    const column = Math.floor(u), row = Math.floor(v);
    const fractionX = u - column, fractionY = v - row;
    const nextColumn = Math.min(column + 1, this.cells - 1);
    const nextRow = Math.min(row + 1, this.cells - 1);
    const a = this.grid[row * this.cells + column];
    const b = this.grid[row * this.cells + nextColumn];
    const c = this.grid[nextRow * this.cells + column];
    const d = this.grid[nextRow * this.cells + nextColumn];
    return (a * (1 - fractionX) + b * fractionX) * (1 - fractionY)
         + (c * (1 - fractionX) + d * fractionX) * fractionY;
  }
}

// ------------------------------------------------------------- snow, on the GPU
// Every flake's whole trajectory is one line of vertex shader: a fixed random
// seed advected by wind and gravity, wrapped into a box that follows the
// camera. Nothing per-flake ever touches JavaScript, so 9000 of them cost one
// draw call and no CPU at all.
function makeSnow() {
  const seeds = new Float32Array(SNOW_PARTICLE_COUNT * 3);
  const sizes = new Float32Array(SNOW_PARTICLE_COUNT);
  const phases = new Float32Array(SNOW_PARTICLE_COUNT);
  for (let index = 0; index < SNOW_PARTICLE_COUNT; index++) {
    seeds[index * 3 + 0] = Math.random();
    seeds[index * 3 + 1] = Math.random();
    seeds[index * 3 + 2] = Math.random();
    sizes[index] = 0.35 + Math.random() * 1.15;
    phases[index] = Math.random() * Math.PI * 2;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(seeds, 3));
  geometry.setAttribute('flakeSize', new THREE.BufferAttribute(sizes, 1));
  geometry.setAttribute('flakePhase', new THREE.BufferAttribute(phases, 1));
  // The bounding sphere is meaningless for a shader that relocates every vertex;
  // an infinite one stops the frustum culler throwing the whole cloud away.
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Infinity);

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uWind: { value: new THREE.Vector3() },
      uCentre: { value: new THREE.Vector3() },
      uBox: { value: SNOW_BOX_METERS },
      uPixels: { value: 1 },
      uOpacity: { value: 0.0 },
      uFogColour: { value: new THREE.Color(FOG_COLOUR) },
      uFogDensity: { value: FOG_DENSITY_PER_METER },
    },
    vertexShader: /* glsl */`
      attribute float flakeSize;
      attribute float flakePhase;
      uniform float uTime; uniform vec3 uWind; uniform vec3 uCentre;
      uniform float uBox; uniform float uPixels;
      varying float vFade;
      void main() {
        vec3 drift = uWind * uTime + vec3(0.0, 0.0, -${SNOW_FALL_METERS_PER_SECOND.toFixed(2)} * uTime);
        vec3 world = position * uBox + drift;
        // wrap into a box centred on the camera: a flake that leaves one face
        // re-enters through the opposite one, so the cloud is endless
        world = mod(world - uCentre + uBox * 0.5, uBox) - uBox * 0.5 + uCentre;
        world.x += sin(uTime * 1.3 + flakePhase) * 0.16 * flakeSize;
        world.y += cos(uTime * 1.1 + flakePhase * 1.7) * 0.16 * flakeSize;
        vec4 viewPosition = viewMatrix * vec4(world, 1.0);
        // fade out at the box edge so flakes pop in and out invisibly
        float edge = length(world - uCentre) / (uBox * 0.5);
        float depth = -viewPosition.z;
        // NOTHING ON THE LENS (user's ruling: the weather is FOG, and what it had
        // become was "particles slapping the camera"). A flake a hand's width
        // from the lens is a screen-filling white disc, and a field of them
        // reads as a windscreen rather than as weather. So the near fade starts
        // far further out than a physical camera's would -- flakes are
        // invisible inside 4 m and only reach full strength at 9 m -- and the
        // size cap is tight enough that no flake can ever be more than a speck.
        // The whiteout itself is the FOG; these are texture in the middle
        // distance, nothing more.
        vFade = (1.0 - smoothstep(0.55, 1.0, edge)) * smoothstep(4.0, 9.0, depth);
        gl_Position = projectionMatrix * viewPosition;
        gl_PointSize = min(flakeSize * uPixels / max(depth, 0.6), 3.5);
      }`,
    fragmentShader: /* glsl */`
      uniform float uOpacity; uniform vec3 uFogColour;
      varying float vFade;
      void main() {
        vec2 offset = gl_PointCoord - 0.5;
        float disc = 1.0 - smoothstep(0.18, 0.5, length(offset));
        float alpha = disc * vFade * uOpacity;
        if (alpha < 0.004) discard;
        gl_FragColor = vec4(mix(vec3(1.0), uFogColour, 0.25), alpha);
      }`,
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
  });
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  points.renderOrder = 5;
  return points;
}

// --------------------------------------------------------------------- world
export class World {
  constructor(scene) {
    this.scene = scene;
    this.root = null;
    this.sidecar = null;
    this.bodies = [];              // body index -> Object3D
    this.terrainMeshes = [];
    this.terrainUniforms = [];
    this.footprints = null;
    this.heightField = null;
    this.footBodies = [];
    this.lastContacts = null;
    this.pelvisIndex = 1;
    this.guideIndex = -1;      // the hiker's mocap root, for her own pennant
    this.flag = null;              // the robot head's wind pennant (flag.js)
    this.hikerFlag = null;         // the same pennant on the hiker's head
    // The body the eye cameras hang off. The G1 has no head body: `d435i` --
    // and so the stereo pair guide.py mounts beside it -- is a child of
    // `torso_link`, which is what the first-person camera rides.
    this.torsoIndex = -1;
    this.name = null;
    this.worldKey = 0;
    this.triangleCount = 0;

    // EXPOSURE. The first pass had sun 3.1 / sky 1.15 / fill 0.35 over a 0.93
    // snow albedo, which summed past 1.0 everywhere and came back a featureless
    // white sheet -- exactly the mistake graphics.py records making with
    // MuJoCo's own lights. What sells snow is CONTRAST between the lit and the
    // shaded side of the roughness, so the total sits near 1 and most of it is
    // directional.
    this.sun = new THREE.DirectionalLight(0xfff1de, 2.05);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(SHADOW_MAP_PIXELS, SHADOW_MAP_PIXELS);
    this.sun.shadow.camera.near = 1;
    this.sun.shadow.camera.far = SUN_DISTANCE_METERS * 2.2;
    this.sun.shadow.bias = -0.0006;
    this.sun.shadow.normalBias = 0.035;
    const shadowCamera = this.sun.shadow.camera;
    shadowCamera.left = -SHADOW_HALF_EXTENT_METERS;
    shadowCamera.right = SHADOW_HALF_EXTENT_METERS;
    shadowCamera.top = SHADOW_HALF_EXTENT_METERS;
    shadowCamera.bottom = -SHADOW_HALF_EXTENT_METERS;
    scene.add(this.sun);
    scene.add(this.sun.target);

    // Snow bounces a great deal of light back up; without a strong sky/ground
    // bounce the shaded side of every fold goes black and the face reads as a
    // paper cut-out.
    this.sky = new THREE.HemisphereLight(0x8fb4e0, 0xbcc6d2, 0.52);
    scene.add(this.sky);
    this.fill = new THREE.AmbientLight(0xa8bcd8, 0.14);
    scene.add(this.fill);

    // Clear by default: the far-field flakes only fall once the VISIBILITY
    // slider is off its clear end. Set from the page rather than passed through
    // update(), so Stage's call signature is untouched.
    this.visibilityMeters = CLEAR_VISIBILITY_METERS;
    this.snow = makeSnow();
    scene.add(this.snow);

    this.sunVector = sunDirection(SUN_ELEVATION_DEGREES, SUN_AZIMUTH_DEGREES);
    this._scratch = new THREE.Vector3();
    this._quaternion = new THREE.Quaternion();
  }

  get loaded() { return this.root !== null; }

  dispose() {
    if (!this.root) return;
    // The flags hang off GLB nodes, so they have to come off before the
    // traversal below disposes everything under the root -- otherwise the next
    // world would inherit a flag whose buffers are already freed.
    if (this.flag) { this.flag.dispose(); this.flag = null; }
    if (this.hikerFlag) { this.hikerFlag.dispose(); this.hikerFlag = null; }
    this.root.traverse(object => {
      if (object.geometry) object.geometry.dispose();
      if (object.material) {
        for (const material of [].concat(object.material)) material.dispose();
      }
    });
    this.scene.remove(this.root);
    this.root = null;
    this.bodies = [];
    this.torsoIndex = -1;
    this.guideIndex = -1;
    this.terrainMeshes = [];
    this.terrainUniforms = [];
    this.footprints = null;
    this.heightField = null;
  }

  async load(worldName) {
    const sidecar = await (await fetch(
      `/app/harness/scene_assets/${worldName}.json`, { cache: 'no-store' })).json();
    const buffer = await (await fetch(sidecar.glb)).arrayBuffer();
    const gltf = await new Promise((resolve, reject) =>
      new GLTFLoader().parse(buffer, '', resolve, reject));

    this.dispose();
    this.sidecar = sidecar;
    this.name = worldName;
    this.worldKey = fnv1a32(worldName);
    this.root = gltf.scene;
    this.root.matrixAutoUpdate = false;   // the root never moves; the bodies do
    this.scene.add(this.root);

    // Body index -> node, by NAME: the pose message is positional, the GLB is
    // not, and the sidecar is the only thing that knows both.
    this.bodies = new Array(sidecar.nbody).fill(null);
    const byName = new Map();
    this.root.traverse(object => { if (object.name) byName.set(object.name, object); });
    for (const body of sidecar.bodies) {
      const node = byName.get(body.node) || byName.get(sanitizeNodeName(body.node));
      if (!node) { console.warn('render3d: no node for body', body); continue; }
      node.matrixAutoUpdate = false;
      this.bodies[body.index] = node;
    }
    this.pelvisIndex = sidecar.pelvis_body ?? 1;
    // `torso_body` is written by app/harness/export_scene.py. The name fallback
    // exists because a sidecar exported before that field did would otherwise
    // silently leave the first-person camera parked at the origin.
    this.torsoIndex = sidecar.torso_body ?? -1;
    if (this.torsoIndex < 0) {
      const named = (sidecar.bodies || []).find(body => body.name === 'torso_link');
      this.torsoIndex = named ? named.index : -1;
    }
    this.footBodies = (sidecar.foot_bodies || []).map(foot => foot.index);
    this.lastContacts = null;

    // GLTFLoader strips every character outside [\w-] from a node name
    // (PropertyBinding.sanitizeNodeName), so the sidecar's names are matched
    // both raw and sanitised. The exporter writes `body__geom` for exactly that
    // reason; the belt and braces is here because a name that quietly stops
    // matching costs the terrain its shader and says nothing about it.
    const terrainNodes = new Set();
    for (const name of sidecar.terrain_nodes || []) {
      terrainNodes.add(name);
      terrainNodes.add(sanitizeNodeName(name));
    }
    this.triangleCount = sidecar.statistics ? sidecar.statistics.triangles : 0;
    this.footprints = sidecar.terrain ? new FootprintCanvas({
      originX: sidecar.terrain.center_world[0] - sidecar.terrain.half_extent_meters[0],
      originY: sidecar.terrain.center_world[1] - sidecar.terrain.half_extent_meters[1],
      sizeX: sidecar.terrain.half_extent_meters[0] * 2,
      sizeY: sidecar.terrain.half_extent_meters[1] * 2,
    }) : null;

    this.root.traverse(object => {
      if (!object.isMesh) return;
      object.castShadow = true;
      object.receiveShadow = true;
      const material = object.material;
      material.side = THREE.FrontSide;
      material.envMapIntensity = 0.6;
      if (terrainNodes.has(object.name)
          || terrainNodes.has(object.parent && object.parent.name ? object.parent.name : '')) {
        this.terrainMeshes.push(object);
        object.castShadow = false;      // the ground casting onto itself is all
                                        // acne and no picture at this scale
        this._dressTerrain(material);
      } else {
        // Everything else is the robot, its gear and the rope. MuJoCo's colours
        // arrive as flat baseColorFactors; a little metal on the darker parts
        // is what stops the whole machine reading as painted plastic.
        const colour = material.color;
        const brightness = (colour.r + colour.g + colour.b) / 3;
        material.roughness = Math.max(0.28, Math.min(0.92, material.roughness));
        material.metalness = brightness < 0.32 ? 0.55 : material.metalness;
      }
    });

    this.heightField = this.terrainMeshes.length
      ? new TerrainHeightField(this.terrainMeshes) : null;

    // THE WIND PENNANT (flag.js). The G1's head is a geom on `torso_link`, not
    // a body of its own, so the flag rides that node -- the sidecar names it,
    // and the node name is the fallback for a sidecar written before it did.
    //
    // BUILT AFTER THE TRAVERSAL ABOVE, DELIBERATELY. That loop walks everything
    // under the root and rewrites each material -- `side = FrontSide`, a
    // metalness lift on dark colours -- which would turn the cloth one-sided and
    // the pole into painted plastic. The flag dresses itself; it must not be in
    // the tree while the GLB's own dressing runs.
    const mountIndex = sidecar.torso_body ?? -1;
    const mountNode = (mountIndex >= 0 ? this.bodies[mountIndex] : null)
      || byName.get('torso_link') || null;
    this.flag = mountNode ? new WindFlag(mountNode) : null;
    if (!mountNode) console.warn('render3d: no torso node; the wind flag is off');

    // AND ONE ON THE HIKER (user's ruling, 2026-08-30). The guide is a single
    // MOCAP ROOT called `guide` carrying her torso, head, hat and pack, with
    // six welded limb bodies hanging off it (app/harness/guide.py), so her head
    // rides that root and the pole is an offset in its frame.
    //
    // IT NEEDS NO VISIBILITY RULE. `Guide.write` parks the whole mocap body at
    // z = -50 m for every tick the guide knob is off, and the flag is a CHILD of
    // that node, so it goes under the world with her and comes back with her.
    // The sidecar has no `guide_body` field -- the body is appended by the same
    // surgery that adds the eye cameras -- so it is found by NAME, which is the
    // name guide.py's `GUIDE_BODY_NAME` writes.
    const guideBody = (sidecar.bodies || []).find(body => body.name === 'guide');
    this.guideIndex = guideBody ? guideBody.index : -1;
    const hikerNode = (this.guideIndex >= 0 ? this.bodies[this.guideIndex] : null)
      || byName.get('guide') || null;
    this.hikerFlag = hikerNode
      ? new WindFlag(hikerNode, HIKER_MOUNT_IN_BODY) : null;
    if (!hikerNode) console.warn('render3d: no guide node; the hiker has no flag');

    const sun = sidecar.sun || {};
    this.sunVector = sun.direction
      ? new THREE.Vector3(sun.direction[0], sun.direction[1], sun.direction[2]).normalize()
      : sunDirection(SUN_ELEVATION_DEGREES, SUN_AZIMUTH_DEGREES);
    return sidecar;
  }

  // The one place the alpine look actually lives: snow on the flats, scoured
  // rock and blue ice where the face is steep, and the footprint canvas
  // sampled by world XY. Injected into MeshStandardMaterial rather than
  // replaced, so shadows, fog and the physically based lighting all still run.
  _dressTerrain(material) {
    const uniforms = {
      uSnow: { value: new THREE.Color(0.72, 0.77, 0.86) },
      uRock: { value: new THREE.Color(0.17, 0.17, 0.19) },
      uIce: { value: new THREE.Color(0.34, 0.44, 0.55) },
      uRockStart: { value: 0.30 },
      uRockEnd: { value: 0.62 },
      uDecal: { value: this.footprints ? this.footprints.texture : null },
      uDecalOrigin: { value: new THREE.Vector2(
        this.footprints ? this.footprints.originX : 0,
        this.footprints ? this.footprints.originY : 0) },
      uDecalSize: { value: new THREE.Vector2(
        this.footprints ? this.footprints.sizeX : 1,
        this.footprints ? this.footprints.sizeY : 1) },
      uHasDecal: { value: this.footprints ? 1.0 : 0.0 },
    };
    material.onBeforeCompile = shader => {
      Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>',
          '#include <common>\nvarying vec3 vTerrainWorld;\nvarying vec3 vTerrainNormal;')
        .replace('#include <begin_vertex>',
          `#include <begin_vertex>
           vTerrainWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;
           vTerrainNormal = normalize(mat3(modelMatrix) * objectNormal);`);
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>',
          `#include <common>
           varying vec3 vTerrainWorld;
           varying vec3 vTerrainNormal;
           uniform vec3 uSnow; uniform vec3 uRock; uniform vec3 uIce;
           uniform float uRockStart; uniform float uRockEnd;
           uniform sampler2D uDecal; uniform vec2 uDecalOrigin;
           uniform vec2 uDecalSize; uniform float uHasDecal;
           ${NOISE_GLSL}`)
        .replace('#include <color_fragment>',
          `#include <color_fragment>
           float terrainSlope = 1.0 - clamp(vTerrainNormal.z, 0.0, 1.0);
           float terrainGrain = fbm(vTerrainWorld.xy * 5.5);
           float terrainCoarse = fbm(vTerrainWorld.xy * 0.22);
           // the fall line is +x on these patches, so stretching the noise
           // across y reads as wind-scoured streaks running down the face
           float terrainStreak = fbm(vec2(vTerrainWorld.x * 3.1, vTerrainWorld.y * 0.4));
           float rockShare = smoothstep(uRockStart, uRockEnd,
                                        terrainSlope + (terrainCoarse - 0.5) * 0.22);
           vec3 snowColour = uSnow * (0.93 + 0.11 * terrainGrain)
                             * (0.96 + 0.07 * terrainStreak);
           vec3 rockColour = mix(uRock, uIce, smoothstep(0.28, 0.85, terrainGrain));
           vec3 terrainColour = mix(snowColour, rockColour, rockShare);
           if (uHasDecal > 0.5) {
             vec2 decalUv = (vTerrainWorld.xy - uDecalOrigin) / uDecalSize;
             float inside = step(0.0, decalUv.x) * step(decalUv.x, 1.0)
                          * step(0.0, decalUv.y) * step(decalUv.y, 1.0);
             float print = texture2D(uDecal, decalUv).r * inside * (1.0 - rockShare);
             terrainColour = mix(terrainColour, terrainColour * vec3(0.50, 0.56, 0.68),
                                 clamp(print, 0.0, 1.0));
           }
           diffuseColor.rgb = terrainColour;`)
        .replace('#include <roughnessmap_fragment>',
          `#include <roughnessmap_fragment>
           roughnessFactor = mix(0.97, 0.34,
             smoothstep(uRockStart, uRockEnd, terrainSlope));`);
      this.terrainUniforms.push(shader.uniforms);
    };
    material.needsUpdate = true;
  }

  // ------------------------------------------------------------- per tick
  // `poses` is [nbody x 7] world xyz + wxyz straight off the socket.
  applyPoses(poses, contacts) {
    if (!this.root) return;
    const bodies = this.bodies;
    for (let index = 0; index < bodies.length; index++) {
      const node = bodies[index];
      if (!node) continue;
      const base = index * 7;
      node.position.set(poses[base], poses[base + 1], poses[base + 2]);
      // MuJoCo quaternions are wxyz; Three's are xyzw.
      node.quaternion.set(poses[base + 4], poses[base + 5], poses[base + 6],
                          poses[base + 3]);
      node.updateMatrix();
    }
    this._stampFootprints(poses, contacts);
  }

  // A footprint is painted on the 0 -> 1 edge of that foot's contact byte,
  // which is MuJoCo's own contact list rather than a height threshold guessed
  // in JavaScript.
  _stampFootprints(poses, contacts) {
    if (!this.footprints || !contacts) return;
    const previous = this.lastContacts;
    for (let index = 0; index < contacts.length && index < this.footBodies.length; index++) {
      const down = contacts[index] === 1;
      const wasDown = previous ? previous[index] === 1 : false;
      if (!down || wasDown) continue;
      const bodyIndex = this.footBodies[index];
      const base = bodyIndex * 7;
      this._quaternion.set(poses[base + 4], poses[base + 5], poses[base + 6],
                           poses[base + 3]);
      this._scratch.set(1, 0, 0).applyQuaternion(this._quaternion);
      this.footprints.stamp(poses[base], poses[base + 1],
                            Math.atan2(this._scratch.y, this._scratch.x));
    }
    this.lastContacts = contacts.slice();
  }

  // ------------------------------------------------------------- per frame
  update(elapsedSeconds, cameraPosition, followPosition, windEast, windNorth,
         pixelHeight) {
    if (this.footprints) this.footprints.update(elapsedSeconds);
    // The same world wind vector the snow drifts with, so the flags and the
    // flakes can never disagree about which way it is blowing -- and the two
    // pennants get the SAME vector, so they always stream the same way.
    if (this.flag) this.flag.update(elapsedSeconds, windEast, windNorth);
    if (this.hikerFlag) this.hikerFlag.update(elapsedSeconds, windEast, windNorth);

    // The shadow camera is a 18 m box: parking it on the robot is what buys a
    // 2048 map enough texels to resolve a boot on a 25 m face.
    this.sun.target.position.copy(followPosition);
    this.sun.position.copy(followPosition)
      .addScaledVector(this.sunVector, -SUN_DISTANCE_METERS);
    this.sun.target.updateMatrixWorld();

    // THE FLAKES ARE NOT THE WEATHER -- the FOG is (user's ruling, and the page
    // sets that fog from the VISIBILITY slider). These are texture in the middle
    // distance, deliberately sparse and small: a flake's diameter in pixels is
    // flakeSize * uPixels / distance, and the vertex shader caps it at 3.5 px
    // and fades everything inside 4 m to nothing. Turning them up was what made
    // the whole thing read as particles on the lens.
    //
    // THEY FOLLOW THE VISIBILITY, NOT THE WIND (user's ruling, 2026-08-30):
    // how many there are and how strongly they show is a question about the
    // weather's thickness. What the WIND still owns is the direction they
    // drift, which is the whole reason the drift vector below is the wind's.
    const share = whiteoutShare(this.visibilityMeters);
    this.snow.visible = share > 0;
    const uniforms = this.snow.material.uniforms;
    uniforms.uTime.value += elapsedSeconds;
    uniforms.uCentre.value.copy(cameraPosition);
    uniforms.uPixels.value = pixelHeight * 0.012 * (1 + 0.35 * share);
    // Far fewer of them than there were, and the count barely grows: in a real
    // whiteout you see LESS, not more, because the fog gets there first.
    this.snow.geometry.setDrawRange(0,
      Math.round(SNOW_PARTICLE_COUNT * (0.10 + 0.25 * share)));
    // The horizontal drift is the WORLD wind vector, so orbiting the robot
    // swings the snow around with the scene rather than with the camera.
    uniforms.uWind.value.set(windEast, windNorth, 0);
    uniforms.uOpacity.value = Math.min(0.34, 0.08 + 0.26 * share) * (share > 0 ? 1 : 0);
  }
}

// GLTFLoader's own rule (PropertyBinding.sanitizeNodeName).
function sanitizeNodeName(name) {
  return name.replace(/\s/g, '_').replace(/[^\w-]/g, '');
}

// The same five lines as `pose_stream.world_key`. A pose frame carries the key
// of the world it was computed in, so the ~1.6 s a first-time world build takes
// cannot paint one map's poses onto another's mesh.
export function fnv1a32(text) {
  let value = 0x811C9DC5;
  for (let index = 0; index < text.length; index++) {
    value ^= text.charCodeAt(index);
    value = Math.imul(value, 0x01000193) >>> 0;
  }
  return value >>> 0;
}

export { FOG_COLOUR, FOG_DENSITY_PER_METER, SUN_ELEVATION_DEGREES,
         SUN_AZIMUTH_DEGREES, CLEAR_VISIBILITY_METERS,
         MINIMUM_VISIBILITY_METERS };
