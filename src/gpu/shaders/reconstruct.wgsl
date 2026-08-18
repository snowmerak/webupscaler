struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  motionSize: vec4<f32>,
  timeScale: vec4<f32>,
  thresholds: vec4<f32>,
  flags: vec4<f32>,
}

@group(0) @binding(0) var inputFrame: texture_2d<f32>;
@group(0) @binding(1) var motionMetaCurrent: texture_2d<f32>;
@group(0) @binding(2) var linearClamp: sampler;
@group(0) @binding(3) var<uniform> uniforms: FrameUniforms;
@group(0) @binding(4) var reconstruction: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var futureFrame: texture_2d<f32>;

fn luma(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn finiteVec3(value: vec3<f32>) -> bool {
  return all(value == value) && all(abs(value) <= vec3<f32>(65504.0));
}

fn finiteVec4(value: vec4<f32>) -> bool {
  return all(value == value) && all(abs(value) <= vec4<f32>(65504.0));
}

fn loadInput(position: vec2<i32>) -> vec3<f32> {
  let maximum = vec2<i32>(uniforms.inputSize.xy) - vec2<i32>(1);
  return textureLoad(
    inputFrame,
    clamp(position, vec2<i32>(0), maximum),
    0,
  ).rgb;
}

fn loadFuture(position: vec2<i32>) -> vec3<f32> {
  let maximum = vec2<i32>(uniforms.inputSize.xy) - vec2<i32>(1);
  return textureLoad(
    futureFrame,
    clamp(position, vec2<i32>(0), maximum),
    0,
  ).rgb;
}

fn lanczos2Weight(distance: f32) -> f32 {
  let x = abs(distance);
  if (x < 0.0001) {
    return 1.0;
  }
  if (x >= 2.0) {
    return 0.0;
  }
  let pix = 3.14159265359 * x;
  return sin(pix) * sin(pix * 0.5) / (pix * pix * 0.5);
}

// Build a bounded spatial value only for lattice sites that no real decoded
// sample observes. Integer source positions are handled separately below and
// never pass through this filter.
fn spatialTwoX(sourceCoordinate: vec2<f32>) -> vec3<f32> {
  let base = vec2<i32>(floor(sourceCoordinate));
  let phase = fract(sourceCoordinate);
  var result = vec3<f32>(0.0);
  var weightSum = 0.0;

  for (var y = -1; y <= 2; y += 1) {
    let weightY = lanczos2Weight(f32(y) - phase.y);
    for (var x = -1; x <= 2; x += 1) {
      let weight = lanczos2Weight(f32(x) - phase.x) * weightY;
      result += loadInput(base + vec2<i32>(x, y)) * weight;
      weightSum += weight;
    }
  }

  let filtered = result / max(weightSum, 0.0001);
  let a = loadInput(base);
  let b = loadInput(base + vec2<i32>(1, 0));
  let c = loadInput(base + vec2<i32>(0, 1));
  let d = loadInput(base + vec2<i32>(1, 1));
  let minimum = min(min(a, b), min(c, d));
  let maximum = max(max(a, b), max(c, d));
  let allowance = (maximum - minimum) * 0.04 + vec3<f32>(0.001);
  return clamp(filtered, minimum - allowance, maximum + allowance);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let outputDimensions = textureDimensions(reconstruction);
  if (any(id.xy >= outputDimensions)) {
    return;
  }

  let scale = uniforms.outputSize.xy * uniforms.inputSize.zw;
  let sourceCoordinate = vec2<f32>(id.xy) / scale;
  let sourceSample = vec2<i32>(round(sourceCoordinate));
  let sourcePhase = sourceCoordinate - round(sourceCoordinate);
  let exactTwoX = all(abs(scale - vec2<f32>(2.0)) < vec2<f32>(0.001));
  let currentObserved = exactTwoX
    && all(abs(sourcePhase) < vec2<f32>(0.001));

  // One of every four 2x pixels is an exact sample from the center frame. It
  // is copied verbatim and can never be softened by temporal blending.
  if (currentObserved) {
    textureStore(
      reconstruction,
      vec2<i32>(id.xy),
      vec4<f32>(loadInput(sourceSample), 1.0),
    );
    return;
  }

  let spatial = spatialTwoX(sourceCoordinate);
  let sourceUv = (sourceCoordinate + vec2<f32>(0.5)) * uniforms.inputSize.zw;
  let sampledMotion = textureSampleLevel(
    motionMetaCurrent,
    linearClamp,
    sourceUv,
    0.0,
  );
  let motion = select(
    vec4<f32>(0.0, 0.0, 0.0, 1.0),
    sampledMotion,
    finiteVec4(sampledMotion),
  );

  // motion.xy maps a center-frame coordinate to the next decoded frame. If
  // that mapped coordinate lands close to an integer LR sample, the future
  // frame contains a real observation for this otherwise-empty 2x site.
  let futureCoordinate = sourceCoordinate + motion.xy;
  let futureSample = vec2<i32>(round(futureCoordinate));
  let futurePhase = futureCoordinate - round(futureCoordinate);
  let futurePhaseCoverage = exp(-16.0 * dot(futurePhase, futurePhase));
  let futureInBounds = all(futureSample >= vec2<i32>(0))
    && all(futureSample < vec2<i32>(uniforms.inputSize.xy));
  let motionConfidence = clamp(motion.z, 0.0, 1.0);
  let matchTrust = 1.0 - smoothstep(0.18, 0.72, motion.w);
  let futureObservation = loadFuture(futureSample);
  let photoError = abs(luma(futureObservation) - luma(spatial));
  let photoConsistent = photoError < 0.32;

  // This is deliberately a selection, not an average: an accepted decoded
  // sample replaces the placeholder; otherwise Lanczos remains untouched.
  let futureObserved = futureInBounds
    && futurePhaseCoverage >= 0.35
    && motionConfidence >= 0.08
    && matchTrust >= 0.30
    && photoConsistent
    && finiteVec3(futureObservation);
  let resolved = select(spatial, futureObservation, futureObserved);
  let observationMask = select(0.0, 1.0, futureObserved);

  textureStore(
    reconstruction,
    vec2<i32>(id.xy),
    vec4<f32>(clamp(resolved, vec3<f32>(0.0), vec3<f32>(1.0)), observationMask),
  );
}
