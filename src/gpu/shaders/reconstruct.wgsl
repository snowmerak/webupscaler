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

  // A single 16x16 block match is not enough near silhouettes.  Compare it
  // with the four adjacent motion cells and collapse temporal weight where
  // the field is discontinuous.  Smooth camera motion survives, while
  // foreground/background boundaries no longer borrow from each other.
  let motionTexel = 1.0 / vec2<f32>(textureDimensions(motionMetaCurrent));
  let sampledMotionLeft = textureSampleLevel(
    motionMetaCurrent,
    linearClamp,
    sourceUv - vec2<f32>(motionTexel.x, 0.0),
    0.0,
  );
  let sampledMotionRight = textureSampleLevel(
    motionMetaCurrent,
    linearClamp,
    sourceUv + vec2<f32>(motionTexel.x, 0.0),
    0.0,
  );
  let sampledMotionUp = textureSampleLevel(
    motionMetaCurrent,
    linearClamp,
    sourceUv - vec2<f32>(0.0, motionTexel.y),
    0.0,
  );
  let sampledMotionDown = textureSampleLevel(
    motionMetaCurrent,
    linearClamp,
    sourceUv + vec2<f32>(0.0, motionTexel.y),
    0.0,
  );
  let motionNeighborsValid = finiteVec4(sampledMotionLeft)
    && finiteVec4(sampledMotionRight)
    && finiteVec4(sampledMotionUp)
    && finiteVec4(sampledMotionDown);
  let motionSpread = max(
    max(
      length(sampledMotionLeft.xy - motion.xy),
      length(sampledMotionRight.xy - motion.xy),
    ),
    max(
      length(sampledMotionUp.xy - motion.xy),
      length(sampledMotionDown.xy - motion.xy),
    ),
  );
  let motionCoherence = select(
    0.0,
    1.0 - smoothstep(0.22, 1.35, motionSpread),
    motionNeighborsValid,
  );
  let neighborConfidence = select(
    0.0,
    clamp(
      min(
        min(sampledMotionLeft.z, sampledMotionRight.z),
        min(sampledMotionUp.z, sampledMotionDown.z),
      ),
      0.0,
      1.0,
    ),
    motionNeighborsValid,
  );

  // motion.xy maps center-frame coordinates into the next decoded frame. The
  // future Lanczos footprint is therefore sampled around x + motion instead
  // of selecting only the rare position that lands exactly on an LR pixel.
  let futureCoordinate = sourceCoordinate + motion.xy;
  let future = futureKernel(futureCoordinate);
  let futureFiltered = future.premultiplied / max(future.weight, 0.0001);
  let futurePhase = futureCoordinate - round(futureCoordinate);
  let futurePhaseCoverage = exp(-12.0 * dot(futurePhase, futurePhase));
  let currentPhaseCoverage = exp(-12.0 * dot(sourcePhase, sourcePhase));
  // Temporal data is promoted only when motion maps this missing 2x lattice
  // site substantially closer to a real decoded sample than the center frame
  // can.  This is the actual sub-pixel super-resolution signal; equal-phase
  // frames contribute almost nothing and therefore cannot create a soft echo.
  let phaseAdvantage = max(0.0, futurePhaseCoverage - currentPhaseCoverage);
  let phaseEvidence = smoothstep(0.12, 0.78, phaseAdvantage)
    * smoothstep(0.18, 0.88, futurePhaseCoverage);
  let futureInBounds = all(futureCoordinate >= vec2<f32>(-0.5))
    && all(futureCoordinate <= uniforms.inputSize.xy - vec2<f32>(0.5));
  let motionConfidence = clamp(motion.z, 0.0, 1.0);
  let matchTrust = 1.0 - smoothstep(0.18, 0.72, motion.w);
  let photoError = abs(luma(futureFiltered) - luma(spatial));
  let photoTrust = 1.0 - smoothstep(0.055, 0.22, photoError);
  let colorError = length(futureFiltered - spatial) * 0.57735026919;
  let colorTrust = 1.0 - smoothstep(0.045, 0.18, colorError);
  let phaseGain = mix(0.035, 1.35, phaseEvidence);
  let localRange = max(0.0, luma(current.maximum) - luma(current.minimum));
  let motionMagnitude = length(motion.xy);
  // Temporal evidence is useful for broad, nearly static surfaces, but even a
  // good block match can smear text, faces, and object silhouettes. Give flat
  // regions a useful accumulation budget while collapsing it per pixel as
  // texture, hard edges, or displacement appear.
  let flatTrust = 1.0 - smoothstep(0.028, 0.115, localRange);
  let textureBand = smoothstep(0.014, 0.060, localRange)
    * (1.0 - smoothstep(0.105, 0.205, localRange));
  let edgeTrust = 1.0 - smoothstep(0.090, 0.205, localRange);
  let stillTrust = 1.0 - smoothstep(0.30, 1.80, motionMagnitude);
  let regionBudget = (0.060 + 0.560 * flatTrust + 0.240 * textureBand)
    * edgeTrust
    * mix(0.42, 1.0, stillTrust);
  let temporalValid = futureInBounds
    && finiteVec3(spatial)
    && finiteVec3(futureFiltered);
  let temporalScale = select(
    0.0,
    clamp(
      regionBudget
        * motionConfidence
        * mix(motionConfidence, neighborConfidence, 0.45)
        * motionCoherence
        * matchTrust
        * photoTrust
        * colorTrust
        * phaseGain,
      0.0,
      1.15,
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
