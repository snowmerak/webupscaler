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

struct KernelAccumulation {
  premultiplied: vec3<f32>,
  weight: f32,
  minimum: vec3<f32>,
  maximum: vec3<f32>,
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

fn currentKernel(coordinate: vec2<f32>) -> KernelAccumulation {
  let base = vec2<i32>(floor(coordinate));
  let phase = fract(coordinate);
  var accumulation = KernelAccumulation(
    vec3<f32>(0.0),
    0.0,
    vec3<f32>(1.0),
    vec3<f32>(0.0),
  );

  for (var y = -1; y <= 2; y += 1) {
    let weightY = lanczos2Weight(f32(y) - phase.y);
    for (var x = -1; x <= 2; x += 1) {
      let weight = lanczos2Weight(f32(x) - phase.x) * weightY;
      let sample = loadInput(base + vec2<i32>(x, y));
      accumulation.premultiplied += sample * weight;
      accumulation.weight += weight;
      accumulation.minimum = min(accumulation.minimum, sample);
      accumulation.maximum = max(accumulation.maximum, sample);
    }
  }

  return accumulation;
}

fn futureKernel(coordinate: vec2<f32>) -> KernelAccumulation {
  let base = vec2<i32>(floor(coordinate));
  let phase = fract(coordinate);
  var accumulation = KernelAccumulation(
    vec3<f32>(0.0),
    0.0,
    vec3<f32>(1.0),
    vec3<f32>(0.0),
  );

  for (var y = -1; y <= 2; y += 1) {
    let weightY = lanczos2Weight(f32(y) - phase.y);
    for (var x = -1; x <= 2; x += 1) {
      let weight = lanczos2Weight(f32(x) - phase.x) * weightY;
      let sample = loadFuture(base + vec2<i32>(x, y));
      accumulation.premultiplied += sample * weight;
      accumulation.weight += weight;
      accumulation.minimum = min(accumulation.minimum, sample);
      accumulation.maximum = max(accumulation.maximum, sample);
    }
  }

  return accumulation;
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

  let current = currentKernel(sourceCoordinate);
  let spatial = current.premultiplied / max(current.weight, 0.0001);
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

  // motion.xy maps center-frame coordinates into the next decoded frame. The
  // future Lanczos footprint is therefore sampled around x + motion instead
  // of selecting only the rare position that lands exactly on an LR pixel.
  let futureCoordinate = sourceCoordinate + motion.xy;
  let future = futureKernel(futureCoordinate);
  let futureFiltered = future.premultiplied / max(future.weight, 0.0001);
  let futurePhase = futureCoordinate - round(futureCoordinate);
  let futurePhaseCoverage = exp(-12.0 * dot(futurePhase, futurePhase));
  let futureInBounds = all(futureCoordinate >= vec2<f32>(-0.5))
    && all(futureCoordinate <= uniforms.inputSize.xy - vec2<f32>(0.5));
  let motionConfidence = clamp(motion.z, 0.0, 1.0);
  let matchTrust = 1.0 - smoothstep(0.18, 0.72, motion.w);
  let photoError = abs(luma(futureFiltered) - luma(spatial));
  let photoTrust = 1.0 - smoothstep(0.055, 0.22, photoError);
  let colorError = length(futureFiltered - spatial) * 0.57735026919;
  let colorTrust = 1.0 - smoothstep(0.045, 0.18, colorError);
  let phaseGain = mix(0.12, 1.0, futurePhaseCoverage);
  let temporalValid = futureInBounds
    && finiteVec3(spatial)
    && finiteVec3(futureFiltered);
  let temporalScale = select(
    0.0,
    clamp(
      0.82 * motionConfidence * matchTrust * photoTrust * colorTrust * phaseGain,
      0.0,
      0.65,
    ),
    temporalValid,
  );

  // Current and warped-future decoded samples share one normalized Lanczos
  // accumulation. This uses temporal evidence to construct every missing 2x
  // lattice site while exact current-frame sites remain untouched above.
  let combinedWeight = current.weight + future.weight * temporalScale;
  let combined = (
    current.premultiplied + future.premultiplied * temporalScale
  ) / max(combinedWeight, 0.0001);
  // Never let future evidence expand the center frame's local color envelope.
  // That rejects disocclusions and coarse-motion errors before they can turn
  // into a soft colored trail.
  let localMinimum = current.minimum;
  let localMaximum = current.maximum;
  let allowance = (localMaximum - localMinimum) * 0.025 + vec3<f32>(0.00075);
  let bounded = clamp(combined, localMinimum - allowance, localMaximum + allowance);
  let resolved = select(spatial, bounded, finiteVec3(bounded));
  let temporalContribution = temporalScale / (1.0 + temporalScale);

  textureStore(
    reconstruction,
    vec2<i32>(id.xy),
    vec4<f32>(clamp(resolved, vec3<f32>(0.0), vec3<f32>(1.0)), temporalContribution),
  );
}
