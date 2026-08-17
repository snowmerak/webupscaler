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
@group(0) @binding(1) var featureCurrent: texture_2d<f32>;
@group(0) @binding(2) var motionMetaCurrent: texture_2d<f32>;
@group(0) @binding(3) var historyPrevious: texture_2d<f32>;
@group(0) @binding(4) var historyCurrent: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var linearClamp: sampler;
@group(0) @binding(6) var<uniform> uniforms: FrameUniforms;

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
  return textureLoad(
    inputFrame,
    clamp(position, vec2<i32>(0), maximum),
    0,
  ).rgb;
}

fn clampHistoryToCurrentNeighborhood(
  historyColor: vec3<f32>,
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
  let clamped = clamp(rgbToYCoCg(historyColor), minimum - expansion, maximum + expansion);
  return clamp(yCoCgToRgb(clamped), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let outputDimensions = vec2<u32>(uniforms.outputSize.xy);
  if (any(id.xy >= outputDimensions)) {
    return;
  }

  // The radiance lattice is aligned so that integer LR sample positions land
  // directly on HR texels. At exact 2x, even HR texels are observed samples
  // and odd phases start as low-coverage spatial estimates.
  let scale = uniforms.outputSize.xy * uniforms.inputSize.zw;
  let hrCoordinate = vec2<f32>(id.xy) / scale;
  let nearestSample = round(hrCoordinate);
  let phaseOffset = hrCoordinate - nearestSample;
  let sourceCenter = vec2<i32>(nearestSample);
  let sourceUv = (hrCoordinate + vec2<f32>(0.5)) * uniforms.inputSize.zw;

  let spatialBase = textureSampleLevel(inputFrame, linearClamp, sourceUv, 0.0).rgb;
  let observedRadiance = loadInput(sourceCenter);
  let currentCoverage = exp(-8.0 * dot(phaseOffset, phaseOffset));

  let feature = textureSampleLevel(featureCurrent, linearClamp, sourceUv, 0.0);
  let gradient = feature.yz;
  let gradientLength = length(gradient);
  let tangent = vec2<f32>(-gradient.y, gradient.x) / max(gradientLength, 0.00001);
  let directionalRadius = uniforms.inputSize.zw * 0.45;
  let directional = 0.5 * (
    textureSampleLevel(inputFrame, linearClamp, sourceUv + tangent * directionalRadius, 0.0).rgb
    + textureSampleLevel(inputFrame, linearClamp, sourceUv - tangent * directionalRadius, 0.0).rgb
  );
  let edgeStrength = smoothstep(0.02, 0.14, gradientLength)
    * smoothstep(0.006, 0.09, feature.w);
  let spatial = mix(spatialBase, directional, edgeStrength * 0.12);

  let motion = textureSampleLevel(motionMetaCurrent, linearClamp, sourceUv, 0.0);
  let previousHrCoordinate = hrCoordinate + motion.xy;
  let previousOutputCoordinate = previousHrCoordinate * scale;
  let previousUv = (previousOutputCoordinate + vec2<f32>(0.5)) * uniforms.outputSize.zw;
  let inBounds = all(previousUv >= vec2<f32>(0.0))
    && all(previousUv <= vec2<f32>(1.0));
  let history = textureSampleLevel(historyPrevious, linearClamp, previousUv, 0.0);
  let clampedHistory = clampHistoryToCurrentNeighborhood(history.rgb, sourceCenter);
  let photoError = abs(luma(spatial) - luma(clampedHistory));
  let photoTrust = 1.0 - smoothstep(0.035, 0.16, photoError);

  let motionTexel = uniforms.motionSize.zw;
  let leftMotion = textureSampleLevel(
    motionMetaCurrent,
    linearClamp,
    sourceUv - vec2<f32>(motionTexel.x, 0.0),
    0.0,
  ).xy;
  let rightMotion = textureSampleLevel(
    motionMetaCurrent,
    linearClamp,
    sourceUv + vec2<f32>(motionTexel.x, 0.0),
    0.0,
  ).xy;
  let upMotion = textureSampleLevel(
    motionMetaCurrent,
    linearClamp,
    sourceUv - vec2<f32>(0.0, motionTexel.y),
    0.0,
  ).xy;
  let downMotion = textureSampleLevel(
    motionMetaCurrent,
    linearClamp,
    sourceUv + vec2<f32>(0.0, motionTexel.y),
    0.0,
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
  let warpedCoverage = history.a * historyReliability * uniforms.thresholds.z;

  let currentWeight = currentCoverage;
  let historyWeight = min(
    warpedCoverage * uniforms.timeScale.w,
    uniforms.thresholds.w,
  );
  let totalObservationWeight = currentWeight + historyWeight;
  let temporalRadiance = (
    observedRadiance * currentWeight + clampedHistory * historyWeight
  ) / max(totalObservationWeight, 0.0001);

  // Spatial interpolation remains the safe fallback for as-yet unobserved HR
  // phases. Reprojected high-coverage samples replace it as motion exposes
  // genuinely new subpixel observations.
  let resolveTrust = clamp(totalObservationWeight, 0.0, 1.0);
  let resolvedRadiance = mix(spatial, temporalRadiance, resolveTrust);
  let coverage = max(currentCoverage, warpedCoverage);

  textureStore(
    historyCurrent,
    vec2<i32>(id.xy),
    vec4<f32>(resolvedRadiance, clamp(coverage, 0.0, 1.0)),
  );
}
