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
@group(0) @binding(6) var latentSeed: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var momentsPrevious: texture_2d<f32>;
@group(0) @binding(8) var momentsCurrent: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var latentPrevious: texture_2d<f32>;

struct NeighborhoodRange {
  minimum: vec3<f32>,
  maximum: vec3<f32>,
}

fn luma(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn rgbToYCoCg(color: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    color.r * 0.25 + color.g * 0.5 + color.b * 0.25,
    color.r * 0.5 - color.b * 0.5,
    -color.r * 0.25 + color.g * 0.5 - color.b * 0.25,
  );
}

fn yCoCgToRgb(color: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    color.x + color.y - color.z,
    color.x + color.z,
    color.x - color.y - color.z,
  );
}

fn loadInput(position: vec2<i32>) -> vec3<f32> {
  let maximum = vec2<i32>(uniforms.inputSize.xy) - vec2<i32>(1);
  return textureLoad(inputFrame, clamp(position, vec2<i32>(0), maximum), 0).rgb;
}

fn currentNeighborhoodRange(sourceCenter: vec2<i32>) -> NeighborhoodRange {
  var minimum = vec3<f32>(1000.0);
  var maximum = vec3<f32>(-1000.0);
  for (var y = -1; y <= 1; y += 1) {
    for (var x = -1; x <= 1; x += 1) {
      let sampleColor = rgbToYCoCg(loadInput(sourceCenter + vec2<i32>(x, y)));
      minimum = min(minimum, sampleColor);
      maximum = max(maximum, sampleColor);
    }
  }

  return NeighborhoodRange(minimum, maximum);
}

fn clampToCurrentNeighborhood(
  radiance: vec3<f32>,
  neighborhood: NeighborhoodRange,
) -> vec3<f32> {
  let expansion = vec3<f32>(0.025, 0.035, 0.035);
  let clamped = clamp(
    rgbToYCoCg(radiance),
    neighborhood.minimum - expansion,
    neighborhood.maximum + expansion,
  );
  return clamp(yCoCgToRgb(clamped), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let outputDimensions = vec2<u32>(uniforms.outputSize.xy);
  if (any(id.xy >= outputDimensions)) {
    return;
  }

  // Integer LR sample positions are aligned to integer HR texels. Exact 2x
  // therefore observes only the even/even phase in the current frame.
  let scale = uniforms.outputSize.xy * uniforms.inputSize.zw;
  let hrCoordinate = vec2<f32>(id.xy) / scale;
  let nearestSample = round(hrCoordinate);
  let phaseOffset = hrCoordinate - nearestSample;
  let sourceCenter = vec2<i32>(nearestSample);
  let sourceUv = (hrCoordinate + vec2<f32>(0.5)) * uniforms.inputSize.zw;
  let observedRadiance = loadInput(sourceCenter);
  let exactTwoX = all(abs(scale - vec2<f32>(2.0)) < vec2<f32>(0.001));
  let alignedSample = all(abs(phaseOffset) < vec2<f32>(0.001));
  let hardCoverage = select(0.0, 1.0, alignedSample);
  let softCoverage = exp(-16.0 * dot(phaseOffset, phaseOffset));
  let currentWeight = select(softCoverage, hardCoverage, exactTwoX);

  let motion = textureSampleLevel(motionMetaCurrent, linearClamp, sourceUv, 0.0);
  let previousHrCoordinate = hrCoordinate + motion.xy;
  let previousOutputCoordinate = previousHrCoordinate * scale;
  let previousUv = (previousOutputCoordinate + vec2<f32>(0.5)) * uniforms.outputSize.zw;
  let inBounds = all(previousUv >= vec2<f32>(0.0))
    && all(previousUv <= vec2<f32>(1.0));

  // History is a premultiplied observation accumulator:
  // RGB = observed radiance * weight, A = observation coverage.
  let sampledHistory = textureSampleLevel(historyPrevious, linearClamp, previousUv, 0.0);
  let sampledMoments = textureSampleLevel(momentsPrevious, linearClamp, previousUv, 0.0);
  let sampledPriorLatent = textureSampleLevel(latentPrevious, linearClamp, previousUv, 0.0);
  let sampledCoverage = max(sampledHistory.a, 0.0);
  let historyRadiance = sampledHistory.rgb / max(sampledCoverage, 0.0001);
  let neighborhood = currentNeighborhoodRange(sourceCenter);
  let clampedHistory = clampToCurrentNeighborhood(historyRadiance, neighborhood);
  let clampedPriorLatent = clampToCurrentNeighborhood(
    sampledPriorLatent.rgb,
    neighborhood,
  );

  let spatialProbe = textureSampleLevel(inputFrame, linearClamp, sourceUv, 0.0).rgb;
  let photoError = abs(luma(spatialProbe) - luma(clampedHistory));
  let photoTrust = 1.0 - smoothstep(0.035, 0.16, photoError);
  let priorPhotoError = abs(luma(spatialProbe) - luma(clampedPriorLatent));
  let priorPhotoTrust = 1.0 - smoothstep(0.045, 0.2, priorPhotoError);

  // Detect abrupt content changes against the unclamped history. The normal
  // neighborhood clamp can make an old subtitle edge look locally plausible,
  // so it must not be used for the reactive decision. These thresholds are
  // deliberately tolerant of 540p compression noise and film grain.
  let rawColorDelta = abs(
    rgbToYCoCg(spatialProbe) - rgbToYCoCg(historyRadiance)
  );
  let reactiveColorError = max(
    rawColorDelta.x,
    max(rawColorDelta.y, rawColorDelta.z) * 0.8,
  );
  let colorReactive = smoothstep(0.055, 0.18, reactiveColorError)
    * mix(0.2, 1.0, clamp(currentWeight, 0.0, 1.0));

  let motionTexel = uniforms.motionSize.zw;
  let leftMotion = textureSampleLevel(
    motionMetaCurrent, linearClamp, sourceUv - vec2<f32>(motionTexel.x, 0.0), 0.0
  ).xy;
  let rightMotion = textureSampleLevel(
    motionMetaCurrent, linearClamp, sourceUv + vec2<f32>(motionTexel.x, 0.0), 0.0
  ).xy;
  let upMotion = textureSampleLevel(
    motionMetaCurrent, linearClamp, sourceUv - vec2<f32>(0.0, motionTexel.y), 0.0
  ).xy;
  let downMotion = textureSampleLevel(
    motionMetaCurrent, linearClamp, sourceUv + vec2<f32>(0.0, motionTexel.y), 0.0
  ).xy;
  let motionDifference = 0.25 * (
    distance(motion.xy, leftMotion) + distance(motion.xy, rightMotion)
    + distance(motion.xy, upMotion) + distance(motion.xy, downMotion)
  );
  let coherence = exp(-motionDifference * 0.18);
  let matchingTrust = 1.0 - motion.w;
  let resetMask = select(1.0, 0.0, uniforms.flags.x > 0.5);
  let historyPresent = smoothstep(0.02, 0.15, sampledCoverage)
    * select(0.0, 1.0, inBounds)
    * resetMask;
  let motionReactive = max(
    smoothstep(0.35, 0.85, motion.w),
    smoothstep(1.5, 7.0, motionDifference) * 0.55,
  );
  let reactiveMask = clamp(
    max(colorReactive, motionReactive * 0.65) * historyPresent,
    0.0,
    1.0,
  );
  let reactiveTrust = 1.0 - reactiveMask;
  let historyReliability = motion.z * photoTrust * coherence * matchingTrust
    * select(0.0, 1.0, inBounds) * resetMask
    * reactiveTrust * reactiveTrust;
  let oldScale = historyReliability * uniforms.thresholds.z;
  let oldWeight = sampledCoverage * oldScale;

  // Preserve historical variance when neighborhood clamping moves the mean.
  // Moments RGB = sum(w*x^2), A = sum(w^2).
  let historySecondMoment = max(sampledMoments.rgb, vec3<f32>(0.0))
    / max(sampledCoverage, 0.0001);
  let historyVariance = max(
    historySecondMoment - historyRadiance * historyRadiance,
    vec3<f32>(0.0),
  );
  let clampedSecondMoment = historyVariance + clampedHistory * clampedHistory;
  let oldSecondPremul = clampedSecondMoment * oldWeight;
  let oldSquaredWeight = max(sampledMoments.a, 0.0) * oldScale * oldScale;

  let accumulatedPremul = observedRadiance * currentWeight
    + clampedHistory * oldWeight;
  let accumulatedWeight = currentWeight + oldWeight;
  let accumulatedSecondPremul = observedRadiance * observedRadiance * currentWeight
    + oldSecondPremul;
  let accumulatedSquaredWeight = currentWeight * currentWeight + oldSquaredWeight;
  let temporalRadiance = accumulatedPremul / max(accumulatedWeight, 0.0001);

  // Cap the accumulator while preserving its radiance. This keeps alpha an
  // interpretable coverage value instead of an unbounded frame counter.
  let storedWeight = min(accumulatedWeight, 1.0);
  let normalization = storedWeight / max(accumulatedWeight, 0.0001);
  let storedPremul = accumulatedPremul * normalization;
  let storedSecondPremul = accumulatedSecondPremul * normalization;
  let storedSquaredWeight = accumulatedSquaredWeight * normalization * normalization;
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

  // Keep inferred reconstruction separate from the observation accumulator.
  // RGB is a latent HR radiance seed. Alpha carries the transient reactive
  // mask through the IBP stages without allocating another full HR texture.
  let spatialSeed = textureSampleLevel(inputFrame, linearClamp, sourceUv, 0.0).rgb;
  let storedSecondMoment = storedSecondPremul / max(storedWeight, 0.0001);
  let observationVariance = max(
    storedSecondMoment - temporalRadiance * temporalRadiance,
    vec3<f32>(0.0),
  );
  let varianceLuma = dot(observationVariance, vec3<f32>(0.25, 0.5, 0.25));
  let effectiveSamples = storedWeight * storedWeight
    / max(storedSquaredWeight, 0.0001);
  let coverageTrust = smoothstep(0.02, 0.85, storedWeight);
  let varianceTrust = exp(-varianceLuma * 80.0);
  let sampleTrust = clamp((effectiveSamples - 1.0) / 3.0, 0.0, 1.0);
  let latentConfidence = coverageTrust * varianceTrust * mix(0.55, 1.0, sampleTrust);
  let observationSeed = mix(spatialSeed, temporalRadiance, latentConfidence);

  // Reproject the previous solved latent as a persistent optimization state.
  // Actual observations remain authoritative; the inferred prior is strongest
  // only on phases that are still unsupported by real temporal samples.
  let priorReactive = clamp(sampledPriorLatent.a, 0.0, 1.0);
  let priorTrust = motion.z
    * coherence
    * matchingTrust
    * priorPhotoTrust
    * select(0.0, 1.0, inBounds)
    * resetMask
    * reactiveTrust * reactiveTrust
    * mix(1.0, 0.1, priorReactive);
  let directObservation = clamp(currentWeight, 0.0, 1.0);
  let missingPhase = 1.0 - directObservation;
  let unresolvedPhase = 1.0 - latentConfidence;
  let persistentBlend = priorTrust
    * mix(0.12, 0.82, missingPhase)
    * mix(0.25, 1.0, unresolvedPhase);
  let persistentSeed = mix(observationSeed, clampedPriorLatent, persistentBlend);
  let observationConstraint = directObservation
    * varianceTrust
    * mix(0.75, 1.0, sampleTrust)
    * reactiveTrust;
  let seededRadiance = mix(
    persistentSeed,
    temporalRadiance,
    observationConstraint * 0.88,
  );
  textureStore(
    latentSeed,
    vec2<i32>(id.xy),
    vec4<f32>(seededRadiance, reactiveMask),
  );
}
