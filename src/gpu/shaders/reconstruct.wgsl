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

fn clampHistoryToCurrentNeighborhood(
  historyRadiance: vec3<f32>,
  sourceCenter: vec2<i32>,
) -> vec3<f32> {
  var minimum = vec3<f32>(1000.0);
  var maximum = vec3<f32>(-1000.0);
  for (var y = -1; y <= 1; y += 1) {
    for (var x = -1; x <= 1; x += 1) {
      let sampleColor = rgbToYCoCg(loadInput(sourceCenter + vec2<i32>(x, y)));
      minimum = min(minimum, sampleColor);
      maximum = max(maximum, sampleColor);
    }
  }

  let expansion = vec3<f32>(0.025, 0.035, 0.035);
  let clamped = clamp(rgbToYCoCg(historyRadiance), minimum - expansion, maximum + expansion);
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
  let sampledCoverage = max(sampledHistory.a, 0.0);
  let historyRadiance = sampledHistory.rgb / max(sampledCoverage, 0.0001);
  let clampedHistory = clampHistoryToCurrentNeighborhood(historyRadiance, sourceCenter);

  let spatialProbe = textureSampleLevel(inputFrame, linearClamp, sourceUv, 0.0).rgb;
  let photoError = abs(luma(spatialProbe) - luma(clampedHistory));
  let photoTrust = 1.0 - smoothstep(0.035, 0.16, photoError);

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
  let historyReliability = motion.z * photoTrust * coherence * matchingTrust
    * select(0.0, 1.0, inBounds) * resetMask;
  let oldWeight = sampledCoverage * historyReliability * uniforms.thresholds.z;

  let accumulatedPremul = observedRadiance * currentWeight
    + clampedHistory * oldWeight;
  let accumulatedWeight = currentWeight + oldWeight;
  let temporalRadiance = accumulatedPremul / max(accumulatedWeight, 0.0001);

  // Cap the accumulator while preserving its radiance. This keeps alpha an
  // interpretable coverage value instead of an unbounded frame counter.
  let storedWeight = min(accumulatedWeight, 1.0);
  let storedPremul = temporalRadiance * storedWeight;
  textureStore(
    historyCurrent,
    vec2<i32>(id.xy),
    vec4<f32>(storedPremul, storedWeight),
  );
}
