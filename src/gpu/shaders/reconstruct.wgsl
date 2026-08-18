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
@group(0) @binding(2) var historyPrevious: texture_2d<f32>;
@group(0) @binding(3) var historyCurrent: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var linearClamp: sampler;
@group(0) @binding(5) var<uniform> uniforms: FrameUniforms;
@group(0) @binding(6) var momentsPrevious: texture_2d<f32>;
@group(0) @binding(7) var momentsCurrent: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var reconstruction: texture_storage_2d<rgba16float, write>;

fn luma(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn finiteScalar(value: f32) -> bool {
  return value == value && abs(value) <= 65504.0;
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

// Step 1: construct the exact 2x lattice. Integer source positions remain
// exact input samples; the three new phases start as bounded Lanczos values.
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
  let hrCoordinate = vec2<f32>(id.xy) / scale;
  let sourceCenter = vec2<i32>(round(hrCoordinate));
  let phaseOffset = hrCoordinate - round(hrCoordinate);
  let sourceUv = (hrCoordinate + vec2<f32>(0.5)) * uniforms.inputSize.zw;
  let spatial = spatialTwoX(hrCoordinate);

  // On the exact 2x lattice the current LR frame directly constrains one of
  // the four HR phases. The other three remain unknown until differently
  // shifted frames provide observations for them.
  let exactTwoX = all(abs(scale - vec2<f32>(2.0)) < vec2<f32>(0.001));
  let alignedSample = all(abs(phaseOffset) < vec2<f32>(0.001));
  let hardCoverage = select(0.0, 1.0, alignedSample);
  let softCoverage = exp(-16.0 * dot(phaseOffset, phaseOffset));
  let currentWeight = select(softCoverage, hardCoverage, exactTwoX);
  let currentObservation = loadInput(sourceCenter);

  // Step 2: pull the previous observation field into the current coordinate
  // system. Subpixel motion moves known LR samples onto previously missing HR
  // phases; no synthetic spatial value is ever written into this history.
  let sampledMotion = textureSampleLevel(motionMetaCurrent, linearClamp, sourceUv, 0.0);
  let motion = select(
    vec4<f32>(0.0, 0.0, 0.0, 1.0),
    sampledMotion,
    finiteVec4(sampledMotion),
  );
  let previousHrCoordinate = hrCoordinate + motion.xy;
  let previousOutputCoordinate = previousHrCoordinate * scale;
  let previousUv = (
    previousOutputCoordinate + vec2<f32>(0.5)
  ) * uniforms.outputSize.zw;
  let inBounds = all(previousUv >= vec2<f32>(0.0))
    && all(previousUv <= vec2<f32>(1.0));
  let resetMask = select(1.0, 0.0, uniforms.flags.x > 0.5);

  let sampledHistory = textureSampleLevel(
    historyPrevious,
    linearClamp,
    previousUv,
    0.0,
  );
  let sampledMoments = textureSampleLevel(
    momentsPrevious,
    linearClamp,
    previousUv,
    0.0,
  );
  let oldHistory = select(vec4<f32>(0.0), sampledHistory, finiteVec4(sampledHistory));
  let oldMoments = select(vec4<f32>(0.0), sampledMoments, finiteVec4(sampledMoments));
  let oldCoverage = max(oldHistory.a, 0.0);
  let oldMean = oldHistory.rgb / max(oldCoverage, 0.0001);
  let photoError = abs(luma(spatial) - luma(oldMean));
  let photoTrust = 1.0 - smoothstep(0.035, 0.14, photoError);
  let motionTrust = mix(0.12, 1.0, clamp(motion.z, 0.0, 1.0));
  let matchTrust = 1.0 - smoothstep(0.18, 0.72, motion.w);
  let oldScaleCandidate = uniforms.thresholds.z
    * photoTrust
    * motionTrust
    * matchTrust
    * select(0.0, 1.0, inBounds)
    * resetMask;
  let oldScale = select(
    0.0,
    clamp(oldScaleCandidate, 0.0, 1.0),
    finiteScalar(oldScaleCandidate),
  );
  let oldWeight = oldCoverage * oldScale;

  let accumulatedPremul = currentObservation * currentWeight
    + oldMean * oldWeight;
  let accumulatedWeight = currentWeight + oldWeight;
  let oldSecondMoment = oldMoments.rgb / max(oldCoverage, 0.0001);
  let accumulatedSecondPremul = currentObservation * currentObservation
      * currentWeight
    + oldSecondMoment * oldWeight;
  let accumulatedSquaredWeight = currentWeight * currentWeight
    + max(oldMoments.a, 0.0) * oldScale * oldScale;

  // Keep the premultiplied observation field bounded while retaining its
  // effective sample count through sum(w^2).
  let storedWeight = min(accumulatedWeight, 1.0);
  let normalization = storedWeight / max(accumulatedWeight, 0.0001);
  let storedPremul = accumulatedPremul * normalization;
  let storedSecondPremul = accumulatedSecondPremul * normalization;
  let storedSquaredWeight = accumulatedSquaredWeight
    * normalization * normalization;
  textureStore(
    historyCurrent,
    vec2<i32>(id.xy),
    vec4<f32>(storedPremul, storedWeight),
  );
  textureStore(
    momentsCurrent,
    vec2<i32>(id.xy),
    vec4<f32>(storedSecondPremul, storedSquaredWeight),
  );

  let temporalMeanCandidate = storedPremul / max(storedWeight, 0.0001);
  let temporalMeanValid = finiteVec3(temporalMeanCandidate);
  let temporalMean = select(spatial, temporalMeanCandidate, temporalMeanValid);
  let secondMoment = storedSecondPremul / max(storedWeight, 0.0001);
  let variance = max(secondMoment - temporalMean * temporalMean, vec3<f32>(0.0));
  let varianceLuma = dot(variance, vec3<f32>(0.25, 0.5, 0.25));
  let effectiveSamples = storedWeight * storedWeight
    / max(storedSquaredWeight, 0.0001);
  let coverageTrust = smoothstep(0.08, 0.72, storedWeight);
  let sampleTrust = smoothstep(0.9, 2.4, effectiveSamples);
  let varianceTrust = exp(-varianceLuma * 72.0);
  let temporalTrustCandidate = coverageTrust * sampleTrust * varianceTrust;
  let temporalTrust = select(
    0.0,
    clamp(temporalTrustCandidate, 0.0, 1.0),
    finiteScalar(temporalTrustCandidate) && temporalMeanValid,
  );

  // Only real, aligned temporal observations may replace the spatial 2x
  // placeholder. No sharpening, contrast curve, deblock, or back-projection
  // is applied here.
  let resolved = mix(spatial, temporalMean, temporalTrust);
  textureStore(
    reconstruction,
    vec2<i32>(id.xy),
    vec4<f32>(clamp(resolved, vec3<f32>(0.0), vec3<f32>(1.0)), temporalTrust),
  );
}
