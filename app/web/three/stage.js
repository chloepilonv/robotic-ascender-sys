// The renderer, the sky, the post chain and the frame loop -- everything that
// is the same whichever world is loaded.
//
// `app/web/render3d.html` owns the sidebar and the websocket; it hands this
// class poses and wind and asks it to draw. The split is historical and worth
// keeping: the page's controls were ported wholesale from the retired 2-D page,
// while none of this file has any counterpart there.
import * as THREE from './vendor/three.module.js';
import { EffectComposer } from './vendor/addons/postprocessing/EffectComposer.js';
import { RenderPass } from './vendor/addons/postprocessing/RenderPass.js';
import { ShaderPass } from './vendor/addons/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from './vendor/addons/postprocessing/UnrealBloomPass.js';
import { SSAOPass } from './vendor/addons/postprocessing/SSAOPass.js';
import { OutputPass } from './vendor/addons/postprocessing/OutputPass.js';
import { World, FOG_COLOUR, FOG_DENSITY_PER_METER } from './world.js';
import { ChaseCamera } from './chase_camera.js';
import { FirstPersonCamera, EYE_MOUNT_IN_TORSO_METERS,
         HIKER_EYE_IN_BODY_METERS, HIKER_YAW_LIMIT_DEGREES,
         NEAR_PLANE_METERS as FIRST_PERSON_NEAR_PLANE_METERS }
  from './first_person_camera.js';

// How far from the eye the head geom is allowed to be before we decide this
// model has no head to cull. The real gap is 0.07 m and the runner-up is
// 0.17 m, so 0.25 m separates them with room to spare and still refuses to
// blank a random chest plate on some future robot.
const HEAD_GEOM_SEARCH_RADIUS_METERS = 0.25;
// The hiker's head is not one geom but four -- head, neck, beanie, pompom -- all
// siblings on the same mocap root as her torso and her pack, so hers is a RADIUS
// and not a nearest-one. From the eye at (0.10, 0, 1.63) those four sit 0.05 to
// 0.19 m away and the next thing out (a shoulder at 0.30 m) is comfortably past
// this line.
const HIKER_HEAD_CULL_RADIUS_METERS = 0.24;

// WHERE THE BOOM LOOKS ON EACH SUBJECT, and why it is not the same number. The
// chase camera pivots TARGET_HEIGHT_METERS (0.80 m) above whatever point it is
// handed. The robot's is its PELVIS, which stands about 0.75 m off the snow, so
// the pivot lands at chest height. The guide's body origin is her GROUND
// CONTACT, so she needs that 0.75 m added back or the boom frames her boots.
const GUIDE_FOLLOW_LIFT_METERS = 0.75;

// Z-UP, ONCE, BEFORE ANYTHING IS CONSTRUCTED. Every camera's `lookAt`, every
// Object3D's default orientation and the whole GLB read this. Setting it here
// rather than per object is what keeps the MuJoCo frame unconverted end to end.
THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const NEAR_PLANE_METERS = 0.08;
const FAR_PLANE_METERS = 2400;
const MAXIMUM_PIXEL_RATIO = 2;

// ---------------------------------------------------------------- the sky
// A PHOTOGRAPH, NOT A GRADIENT. This page used to draw Sky.js's Preetham model:
// correct physics, empty picture. On a Himalayan face the thing that reads as
// altitude is the SKYLINE -- Nuptse, Changtse, the Khumbu ridge -- and no
// analytic sky has one. The texture is a real Kala Patthar panorama folded into
// an equirectangular sphere by app/harness/build_sky_texture.py; provenance,
// author and licence are in app/web/sky/README.md.
const SKY_TEXTURE_URL = new URL('../sky/everest_kala_patthar_4k.jpg', import.meta.url);
// Until it decodes, the page must not flash black behind the mountain.
const SKY_FALLBACK_COLOUR = 0x2a4a8c;

// TWO ROTATIONS, FOR TWO DIFFERENT REASONS.
//   tilt  -- Three.js samples an equirectangular map with `asin(direction.y)`,
//            i.e. it assumes Y is up. This whole app is Z-up (see world.js), so
//            without a quarter turn about X the panorama wraps round the world
//            sideways and the horizon runs vertically up the screen.
//   yaw   -- which way Everest faces. Three.js maps a world azimuth phi to
//            texture azimuth 180 + yaw - phi (derived from `equirectUv` and the
//            euler negation in WebGLBackground, then checked against renders).
//            The panorama's brightest sky sits at texture azimuth 93 deg, so
//            yaw 150 puts its bright quarter at world azimuth 237 -- 22 deg off
//            world.js's sun at 215, close enough that the glow and the shadows
//            agree, and chosen over the exact 128 because 150 is what swings
//            the Everest-Nuptse massif into the view the chase camera starts
//            in. A sky whose bright quarter disagrees with the shadows is the
//            tell that a backdrop is pasted on; 22 degrees does not show.
const SKY_TILT_RADIANS = Math.PI / 2;
const SKY_YAW_DEGREES = 150;

// Image-based lighting from the same photograph, on top of the three lights
// world.js already places. Kept LOW on purpose: world.js records that the first
// pass at this scene summed past 1.0 everywhere and came back a white sheet,
// and an environment map is one more additive term in that sum. It is here for
// the cold blue bounce on the robot's metal, not to light the scene.
const SKY_ENVIRONMENT_INTENSITY = 0.35;

// The grade. A cold shadow lift and a warm highlight roll is the entire look of
// a high-altitude photograph; the vignette and the grain stop the snow reading
// as a flat CG sheet.
const GradeShader = {
  uniforms: {
    tDiffuse: { value: null },
    uTime: { value: 0 },
    uVignette: { value: 0.42 },
    uGrain: { value: 0.028 },
    uContrast: { value: 1.14 },
    uSaturation: { value: 0.90 },
    uShadowTint: { value: new THREE.Color(0.62, 0.74, 0.95) },
    uHighlightTint: { value: new THREE.Color(1.0, 0.995, 0.985) },
  },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform float uTime, uVignette, uGrain, uContrast, uSaturation;
    uniform vec3 uShadowTint, uHighlightTint;
    varying vec2 vUv;
    float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
      vec3 colour = texture2D(tDiffuse, vUv).rgb;
      float luminance = dot(colour, vec3(0.2126, 0.7152, 0.0722));
      // split tone: shadows toward alpine blue, highlights toward low sun
      colour *= mix(uShadowTint, uHighlightTint, smoothstep(0.1, 0.72, luminance));
      colour = mix(vec3(luminance), colour, uSaturation);
      colour = (colour - 0.5) * uContrast + 0.5;
      vec2 centred = vUv - 0.5;
      colour *= 1.0 - uVignette * dot(centred, centred) * 1.35;
      colour += (hash(vUv * 917.0 + fract(uTime) * 311.0) - 0.5) * uGrain;
      gl_FragColor = vec4(clamp(colour, 0.0, 1.0), 1.0);
    }`,
};

export class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({
      canvas, antialias: true, powerPreference: 'high-performance', stencil: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAXIMUM_PIXEL_RATIO));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.92;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(FOG_COLOUR, FOG_DENSITY_PER_METER);
    this.camera = new THREE.PerspectiveCamera(50, 16 / 9, NEAR_PLANE_METERS, FAR_PLANE_METERS);
    this.camera.position.set(-4, 0, 2);

    this._buildSky();
    this.world = new World(this.scene);
    this.chase = new ChaseCamera(this.camera);
    // BOTH CAMERAS DRIVE THE SAME PerspectiveCamera, and the chase camera keeps
    // running even while the first-person view is live. That is not waste: its
    // `azimuthDegrees` is what render3d.html sends up the socket every 20 ms,
    // and the runtime turns that into the robot's steering heading. Freeze it
    // and W would walk wherever the boom happened to be pointing when V was
    // pressed. So the chase updates, its auto-recentre keeps the heading behind
    // the robot, and the first-person camera then OVERWRITES the position,
    // orientation and lens for the frame.
    this.firstPerson = new FirstPersonCamera(this.camera);
    // THE GUIDE IS A SUBJECT, NOT A MODE (user's ruling, 2026-08-30: "when the
    // guide is turned on the camera should be on the guide, not the bot").
    // Its own first-person camera, because the mount and the look limits differ
    // -- and its own instance rather than a re-pointed one, so switching back to
    // the robot finds the head exactly where it was left.
    this.firstPersonGuide = new FirstPersonCamera(this.camera, {
      mountInBody: HIKER_EYE_IN_BODY_METERS,
      yawLimitDegrees: HIKER_YAW_LIMIT_DEGREES,
    });
    this.followGuide = false;
    this.guideActive = false;   // the page's guide knob, for the rope-map camera lock
    this.ropeWorld = false;
    this._hikerHeadNodes = null;
    this.firstPersonEnabled = false;
    this._thirdPersonNearPlaneMeters = NEAR_PLANE_METERS;
    // The one geom culled in first person, found once per world load. `undefined`
    // means "not looked for yet"; `null` means "looked, this model has no head".
    this._headNode = undefined;

    this.composer = null;
    this.gradePass = null;
    this.bloomPass = null;
    this.ssaoPass = null;
    this.ambientOcclusion = true;
    this.postProcessing = false;   // post chain removed from the UI (user: looked worse, tanked fps); plain forward render

    this.windEast = 0; this.windNorth = 0;
    this.paused = false;
    this.frames = 0; this.fps = 0; this._fpsWindowStart = performance.now();
    this._lastFrameMilliseconds = performance.now();
    this._pelvis = new THREE.Vector3();
    this._forward = new THREE.Vector3();
    this._quaternion = new THREE.Quaternion();
    this._headingRadians = Math.PI;
    this._torsoPosition = new THREE.Vector3();
    this._torsoQuaternion = new THREE.Quaternion();

    // Poses arrive at 50 Hz and we draw at the display rate, so every frame
    // interpolates between the last two ticks. Without this the robot moves in
    // 20 ms steps while the camera moves smoothly, which reads as judder in the
    // robot alone -- the worst of both.
    this._posePrevious = null;
    this._poseLatest = null;
    this._poseBlend = null;
    this._poseArrivalMilliseconds = 0;
    this._poseIntervalMilliseconds = 20;

    this.resize();
  }

  _buildSky() {
    // The photograph has to decode before it can be a background, and the GLB
    // usually wins that race -- so paint the fallback FIRST and swap.
    this.scene.background = new THREE.Color(SKY_FALLBACK_COLOUR);
    this.skyTexture = null;
    this.skyReady = false;
    this._applySkyRotation();

    new THREE.TextureLoader().load(SKY_TEXTURE_URL.href, texture => {
      texture.mapping = THREE.EquirectangularReflectionMapping;
      texture.colorSpace = THREE.SRGBColorSpace;
      // Three re-projects this into a cube render target for the background and
      // a PMREM for the environment; both want a clean minification chain, and
      // ANISOTROPY is what keeps the ridge from crawling when the camera pans.
      texture.anisotropy = Math.min(
        8, this.renderer.capabilities.getMaxAnisotropy());
      texture.needsUpdate = true;
      this.skyTexture = texture;
      this.scene.background = texture;
      this.scene.environment = texture;
      this.scene.environmentIntensity = SKY_ENVIRONMENT_INTENSITY;
      this._applySkyRotation();
      this.skyReady = true;
    }, undefined, error => {
      // A missing texture must not take the page down: the fallback colour
      // stays, the scene still runs, and the console says why the sky is flat.
      console.error('[stage] sky texture failed to load', SKY_TEXTURE_URL.href, error);
    });
  }

  // Both rotations are set from the same two numbers so the lighting can never
  // drift out of step with the picture it is supposed to come from.
  _applySkyRotation() {
    const yaw = SKY_YAW_DEGREES * Math.PI / 180;
    this.scene.backgroundRotation.set(SKY_TILT_RADIANS, 0, yaw);
    this.scene.environmentRotation.set(SKY_TILT_RADIANS, 0, yaw);
  }

  _buildComposer() {
    if (this.composer) this.composer.dispose();
    // DEVICE pixels, not CSS pixels: EffectComposer sizes its own targets from
    // the renderer's drawing buffer, and a pass constructed at CSS size would
    // sample a half-resolution AO into a full-resolution frame.
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    const width = Math.max(2, size.x), height = Math.max(2, size.y);
    const composer = new EffectComposer(this.renderer);
    composer.addPass(new RenderPass(this.scene, this.camera));
    if (this.ambientOcclusion) {
      // TWO NON-OBVIOUS THINGS, both measured rather than guessed (a probe that
      // read one pixel back off the canvas per pass combination found them):
      //  * this SSAOPass does NOT render a beauty pass. It renders normals and
      //    depth, computes the occlusion, blurs it, and MULTIPLIES the result
      //    over the read buffer -- so it needs a RenderPass in front of it or it
      //    multiplies over nothing.
      //  * it leaves `needsSwap` true, which is only correct when it is the LAST
      //    pass. With bloom and the grade behind it the composer swaps, the next
      //    pass reads a buffer nothing ever wrote, and the whole picture comes
      //    back BLACK. Turning the swap off is the fix.
      const ssao = new SSAOPass(this.scene, this.camera, width, height);
      ssao.kernelRadius = 0.35;      // metres: contact shadow, not a halo
      ssao.minDistance = 0.0006;
      ssao.maxDistance = 0.06;
      ssao.needsSwap = false;
      composer.addPass(ssao);
      this.ssaoPass = ssao;
    } else {
      this.ssaoPass = null;
    }
    this.bloomPass = new UnrealBloomPass(
      new THREE.Vector2(width, height), 0.32, 0.62, 0.86);
    composer.addPass(this.bloomPass);
    composer.addPass(new OutputPass());
    this.gradePass = new ShaderPass(GradeShader);
    this.gradePass.renderToScreen = true;
    composer.addPass(this.gradePass);
    this.composer = composer;
  }

  setAmbientOcclusion(enabled) {
    if (enabled === this.ambientOcclusion) return;
    this.ambientOcclusion = enabled;
    this._buildComposer();
  }

  setPostProcessing(enabled) { this.postProcessing = enabled; }

  resize() {
    const width = Math.max(2, this.canvas.clientWidth);
    const height = Math.max(2, this.canvas.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAXIMUM_PIXEL_RATIO));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this._buildComposer();
  }

  async loadWorld(name) {
    const sidecar = await this.world.load(name);
    this.chase.seeded = false;
    // A world swap rebuilds every node, so a torso hidden before the swap is
    // visible again after it unless it is re-hidden here -- and the cached
    // culling lists point at nodes that no longer exist, so they are dropped
    // rather than re-tested. (An EMPTY cached list would pass the parent test
    // vacuously and never rebuild, which is the bug this line is for.)
    this._headNode = undefined;
    this._hikerHeadNodes = null;
    this._applyFirstPersonVisibility();
    this._posePrevious = this._poseLatest = this._poseBlend = null;
    this.slopeDegrees = sidecar.slope_degrees || 0;
    this.ropeWorld = Boolean(sidecar.rope_enabled);
    // Fog scaled to the map: a 25 m patch and a 120 m sandbox want very
    // different densities or one is soup and the other is glass.
    const diagonal = sidecar.terrain
      ? Math.hypot(sidecar.terrain.half_extent_meters[0] * 2,
                   sidecar.terrain.half_extent_meters[1] * 2)
      : 30;
    this.scene.fog.density = Math.min(0.010, 0.16 / Math.max(diagonal, 8));
    return sidecar;
  }

  // ------------------------------------------------------------ the stream
  notePoses(poses, contacts, arrivalMilliseconds) {
    if (this._poseLatest) {
      this._poseIntervalMilliseconds = Math.max(
        4, Math.min(60, arrivalMilliseconds - this._poseArrivalMilliseconds));
      this._posePrevious = this._poseLatest;
    } else {
      this._posePrevious = poses;
      this._poseBlend = new Float32Array(poses.length);
    }
    this._poseLatest = poses;
    this._poseArrivalMilliseconds = arrivalMilliseconds;
    this.world._stampFootprints(poses, contacts);
  }

  setWind(east, north) { this.windEast = east; this.windNorth = north; }
  setPaused(paused) { this.paused = paused; }

  // THE TWO NUMBERS THAT GO UP THE SOCKET. They are the CHASE camera's in both
  // views, always, because they are the robot's steering command and not a
  // description of the picture: `runtime.py` adds half a turn to the azimuth
  // and walks the robot that way. In first person the chase camera has drifted
  // to behind the robot, so W walks where the robot is facing -- which is what
  // W does in every first-person game -- and looking around steers nothing.
  get cameraAzimuthDegrees() { return this.chase.azimuthDegrees; }
  get cameraElevationDegrees() { return this.chase.elevationDegrees; }

  // Which first-person camera is live. One expression, so the four places that
  // ask cannot disagree.
  get activeFirstPerson() {
    return this.followGuide ? this.firstPersonGuide : this.firstPerson;
  }

  // WHO THE CAMERA IS ON. Called from the page whenever the guide switch moves
  // -- on the click AND on the runtime's echo, so a run started with `--guide`
  // frames her without being told twice.
  //
  // THE SPRING IS RE-SEEDED, NOT FLOWN (user's ruling: "make the switch
  // instantaneous on toggle"). `chase.seeded = false` makes the next update
  // place the follow point, the previous target and the aim point ON the new
  // subject instead of lerping toward it, so the boom snaps and then springs
  // normally from there -- rather than sailing across the map for a second and
  // a half at the position lag.
  setFollowGuide(enabled) {
    const wanted = Boolean(enabled);
    if (wanted === this.followGuide) return this.followGuide;
    this.followGuide = wanted;
    this.chase.seeded = false;
    this._applyFirstPersonVisibility();
    return this.followGuide;
  }
  // Where the PICTURE is pointing, in the same convention. The wind ribbons in
  // render3d.html are drawn in screen space from this and must follow whichever
  // camera is live, or a storm blows sideways the moment V is pressed.
  get viewAzimuthDegrees() {
    return this.firstPersonEnabled
      ? this.activeFirstPerson.azimuthDegrees() : this.chase.azimuthDegrees;
  }
  // WHERE THE ROBOT IS FACING, world radians. `_headingRadians` above is the
  // BOOM's subject and becomes the HIKER's the moment the guide is on, so
  // anything about the ROBOT's own body -- the sound-direction indicator, whose
  // bearing arrives in the robot's body frame -- must read the pelvis itself and
  // not the boom. Same yaw-about-world-z formula as the update (a pelvis on a
  // 38.6 deg face is pitched over, so its local +x is nothing like its heading).
  get robotHeadingRadians() {
    const pelvisNode = (this.world && this.world.bodies)
      ? this.world.bodies[this.world.pelvisIndex] : null;
    if (!pelvisNode) return this._headingRadians;
    const quaternion = pelvisNode.quaternion;
    const w = quaternion.w, x = quaternion.x, y = quaternion.y, z = quaternion.z;
    return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  }
  look(movementX, movementY) {
    if (this.firstPersonEnabled) this.activeFirstPerson.look(movementX, movementY);
    else this.chase.look(movementX, movementY);
  }
  // R recentres ALL of them, whichever is live: coming back to third person
  // after a reset should not find a head still turned 150 degrees from where it
  // was, and nor should coming back to the robot from the guide.
  recentreCamera() {
    this.chase.recentreNow();
    this.firstPerson.recentreNow();
    this.firstPersonGuide.recentreNow();
  }

  // ------------------------------------------------------- the view toggle
  setFirstPerson(enabled) {
    const wanted = Boolean(enabled);
    if (wanted === this.firstPersonEnabled) return this.firstPersonEnabled;
    this.firstPersonEnabled = wanted;
    this.camera.near = wanted ? FIRST_PERSON_NEAR_PLANE_METERS
                              : this._thirdPersonNearPlaneMeters;
    this.camera.updateProjectionMatrix();
    this._applyFirstPersonVisibility();
    return this.firstPersonEnabled;
  }

  toggleFirstPerson() { return this.setFirstPerson(!this.firstPersonEnabled); }

  // The camera sits INSIDE the head, so the head has to go -- and ONLY the head.
  //
  // The measurement that says it is needed: raycasting out of the eye in first
  // person hits the head mesh at 0.05 m to the left of frame at 90 degrees of
  // yaw and dead centre at 150, so a near plane alone (which only clips the
  // sliver straight ahead) does not save it. Turning your head must not fill
  // the screen with the back of your own head.
  //
  // WHY NOT HIDE THE WHOLE `torso_link` NODE, which was the first version: a
  // hidden node casts no shadow, and on this mountain the climber's own shadow
  // stretched up the slope is most of what the first-person frame contains.
  // Three.js tests shadow casters against the MAIN camera's layers, so the
  // usual layer trick cannot buy it back. Culling one geom instead keeps the
  // chest, the jacket and the logo plates in the shadow map, and seeing your
  // own chest when you look back over your shoulder is right, not a bug.
  //
  // WHICH CHILD IS THE HEAD: the one whose geom origin is NEAREST THE EYE, and
  // it is not close -- head_link sits 0.07 m from the d435i mount, the next
  // nearest torso geom 0.17 m. That is the honest test ("the mesh the camera is
  // inside of"), and it survives an export that renumbers geoms, which a node
  // name like `torso_link__geom_58` does not.
  _applyFirstPersonVisibility() {
    if (!this.world || !this.world.bodies) return;
    const insideRobot = this.firstPersonEnabled && !this.followGuide;
    const insideHiker = this.firstPersonEnabled && this.followGuide;

    const torsoNode = this.world.bodies[this.world.torsoIndex];
    if (torsoNode) {
      if (this._headNode === undefined || this._headNode?.parent !== torsoNode) {
        this._headNode = null;
        let nearest = Infinity;
        for (const child of torsoNode.children) {
          const distance = child.position.distanceTo(EYE_MOUNT_IN_TORSO_METERS);
          if (distance < nearest) { nearest = distance; this._headNode = child; }
        }
        if (nearest > HEAD_GEOM_SEARCH_RADIUS_METERS) this._headNode = null;
      }
      if (this._headNode) this._headNode.visible = !insideRobot;
    }

    // THE SAME RULE FOR HER, by radius rather than by nearest: her head, neck,
    // beanie and pompom are four separate geoms on one mocap root that also
    // carries her whole torso, so "the nearest child" would blank a hat and
    // leave a skull. Her PENNANT goes too -- a 36 cm flag on a 45 cm pole above
    // your own eyes fills the frame the moment you look up, which is the same
    // objection that culls the robot's head and not a new one.
    const hikerNode = this.world.guideIndex >= 0
      ? this.world.bodies[this.world.guideIndex] : null;
    if (hikerNode) {
      if (this._hikerHeadNodes === null
          || this._hikerHeadNodes.some(node => node.parent !== hikerNode)) {
        this._hikerHeadNodes = hikerNode.children.filter(child =>
          child.position.distanceTo(HIKER_EYE_IN_BODY_METERS)
            <= HIKER_HEAD_CULL_RADIUS_METERS);
      }
      for (const node of this._hikerHeadNodes) node.visible = !insideHiker;
    }
    if (this.world.hikerFlag) this.world.hikerFlag.visible = !insideHiker;
  }

  // Poses are 50 Hz, the display is not: blend the last two ticks by how far
  // through the interval the wall clock is, clamped so a stalled stream freezes
  // rather than extrapolating the robot through the mountain.
  _interpolatePoses(nowMilliseconds) {
    if (!this._poseLatest) return;
    const previous = this._posePrevious, latest = this._poseLatest;
    let share = (nowMilliseconds - this._poseArrivalMilliseconds)
      / this._poseIntervalMilliseconds;
    share = Math.max(0, Math.min(1, share));
    if (previous === latest || share >= 1) {
      this.world.applyPoses(latest, null);
      return;
    }
    const blended = this._poseBlend;
    for (let index = 0; index < latest.length; index += 7) {
      for (let axis = 0; axis < 3; axis++) {
        blended[index + axis] = previous[index + axis]
          + (latest[index + axis] - previous[index + axis]) * share;
      }
      // Quaternions get a nearest-neighbour slerp done by hand: four floats,
      // 33 bodies, 60 times a second is nothing, and importing a Quaternion
      // object per body per frame is not.
      let w0 = previous[index + 3], x0 = previous[index + 4],
          y0 = previous[index + 5], z0 = previous[index + 6];
      const w1 = latest[index + 3], x1 = latest[index + 4],
            y1 = latest[index + 5], z1 = latest[index + 6];
      let dot = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1;
      if (dot < 0) { w0 = -w0; x0 = -x0; y0 = -y0; z0 = -z0; dot = -dot; }
      const one = 1 - share;
      let w = w0 * one + w1 * share, x = x0 * one + x1 * share,
          y = y0 * one + y1 * share, z = z0 * one + z1 * share;
      const length = Math.hypot(w, x, y, z) || 1;
      blended[index + 3] = w / length; blended[index + 4] = x / length;
      blended[index + 5] = y / length; blended[index + 6] = z / length;
    }
    this.world.applyPoses(blended, null);
  }

  // ------------------------------------------------------------- one frame
  render(nowMilliseconds) {
    const wallSeconds = Math.min(0.1,
      Math.max(0, (nowMilliseconds - this._lastFrameMilliseconds) / 1000));
    this._lastFrameMilliseconds = nowMilliseconds;
    if (!this.world.loaded) return;

    this._interpolatePoses(nowMilliseconds);

    // WHO THE BOOM IS ON. The guide's mocap root when the guide is on, the
    // robot's pelvis otherwise -- and the guide's root sits at her ground
    // contact, hence the lift (see GUIDE_FOLLOW_LIFT_METERS). Her mocap
    // quaternion is a pure yaw about world z, so the same heading formula reads
    // both bodies correctly.
    const guideNode = (this.followGuide && this.world.guideIndex >= 0)
      ? this.world.bodies[this.world.guideIndex] : null;
    const pelvisNode = guideNode || this.world.bodies[this.world.pelvisIndex];
    if (pelvisNode) {
      this._pelvis.copy(pelvisNode.position);
      if (guideNode) this._pelvis.z += GUIDE_FOLLOW_LIFT_METERS;
      this._quaternion.copy(pelvisNode.quaternion);
      // YAW ABOUT WORLD Z, not "where the body's own +x happens to point".
      // On a 38.6 deg face the pelvis is pitched over by the slope, so its local
      // +x aims into the snow and its ground projection is nothing like the
      // heading -- which sent the auto-recentre 50 degrees off and framed the
      // climb sideways. This is runtime.root_yaw_radians, the same formula the
      // steering controller uses.
      const w = this._quaternion.w, x = this._quaternion.x;
      const y = this._quaternion.y, z = this._quaternion.z;
      this._headingRadians = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    }
    // A frozen mountain is a frozen mountain: no lag, no sway, no snow drift.
    const elapsedSeconds = this.paused ? 0 : wallSeconds;
    this.chase.update(elapsedSeconds, this._pelvis, this._headingRadians,
                      this.world.heightField, this.slopeDegrees || 0);
    // ...and then, in first person, the eye overwrites what the boom just
    // wrote. The chase camera's OWN state is untouched, so its azimuth is still
    // the steering command and V can be pressed back at any time.
    if (this.firstPersonEnabled) {
      // The body the eye is bolted to: her mocap root with the guide on, the
      // robot's `torso_link` otherwise. Both are x-forward / z-up, which is why
      // one camera class rides either.
      const mountNode = guideNode
        || this.world.bodies[this.world.torsoIndex];
      if (mountNode) {
        this._torsoPosition.copy(mountNode.position);
        this._torsoQuaternion.copy(mountNode.quaternion);
      }
      // ROPE MAP + GUIDE ON + robot's own eyes: stabilised orientation
      // (position and heading only -- the climb's body-rock stays out of the
      // lens; user's ruling 2026-08-30).
      const lockHeading = (this.ropeWorld && this.guideActive && !this.followGuide)
        ? this.robotHeadingRadians : null;
      this.activeFirstPerson.update(elapsedSeconds, this._torsoPosition,
                                    this._torsoQuaternion, lockHeading);
    }
    this.world.update(elapsedSeconds, this.camera.position, this._pelvis,
                      this.windEast, this.windNorth,
                      this.canvas.clientHeight * this.renderer.getPixelRatio());
    if (this.gradePass) this.gradePass.uniforms.uTime.value += wallSeconds;

    if (this.postProcessing && this.composer) this.composer.render(wallSeconds);
    else this.renderer.render(this.scene, this.camera);

    this.frames++;
    if (nowMilliseconds - this._fpsWindowStart >= 500) {
      this.fps = this.frames * 1000 / (nowMilliseconds - this._fpsWindowStart);
      this.frames = 0;
      this._fpsWindowStart = nowMilliseconds;
    }
  }
}
