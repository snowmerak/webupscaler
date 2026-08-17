struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  motionSize: vec4<f32>,
  timeScale: vec4<f32>,
  thresholds: vec4<f32>,
  flags: vec4<f32>,
}

@group(0) @binding(0) var featureCurrent: texture_2d<f32>;
@group(0) @binding(1) var featurePrevious: texture_2d<f32>;
@group(0) @binding(2) var motionStatePrevious: texture_2d<f32>;
@group(0) @binding(3) var motionMetaPrevious: texture_2d<f32>;
@group(0) @binding(4) var motionStateCurrent: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var motionMetaCurrent: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var linearClamp: sampler;
@group(0) @binding(7) var<uniform> uniforms: FrameUniforms;

const PATCH = array<vec2<f32>, 13>(
  vec2<f32>(0.0, 0.0),
  vec2<f32>(-1.0, 0.0), vec2<f32>(1.0, 0.0),
  vec2<f32>(0.0, -1.0), vec2<f32>(0.0, 1.0),
  vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0),
  vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
  vec2<f32>(-1.75, 0.0), vec2<f32>(1.75, 0.0),
  vec2<f32>(0.0, -1.75), vec2<f32>(0.0, 1.75),
);

fn candidateScore(centerUv: vec2<f32>, displacement: vec2<f32>) -> f32 {
  var score = 0.0;
  let displacementUv = displacement * uniforms.inputSize.zw;
  for (var index = 0u; index < 13u; index += 1u) {
    let patchUv = PATCH[index] * uniforms.analysisSize.zw;
    let currentLuma = textureSampleLevel(featureCurrent, linearClamp, centerUv + patchUv, 0.0).r;
    let previousLuma = textureSampleLevel(featurePrevious, linearClamp, centerUv + patchUv + displacementUv, 0.0).r;
    score += min(abs(currentLuma - previousLuma), 0.25);
  }
  return score / 13.0;
}

fn clampLength(value: vec2<f32>, maximum: f32) -> vec2<f32> {
  let magnitude = length(value);
  return select(value, value * (maximum / max(magnitude, 0.00001)), magnitude > maximum);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let motionDimensions = vec2<u32>(uniforms.motionSize.xy);
  if (any(id.xy >= motionDimensions)) {
    return;
  }

  if (uniforms.flags.x > 0.5) {
    textureStore(motionStateCurrent, vec2<i32>(id.xy), vec4<f32>(0.0));
    textureStore(motionMetaCurrent, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    return;
  }

  let motionUv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) * uniforms.motionSize.zw;
  let analysisCenter = vec2<f32>(id.xy) * 4.0 + vec2<f32>(2.0);
  let centerUv = analysisCenter * uniforms.analysisSize.zw;
  let previousState = textureSampleLevel(motionStatePrevious, linearClamp, motionUv, 0.0);
  let previousMeta = textureSampleLevel(motionMetaPrevious, linearClamp, motionUv, 0.0);
  let dt = max(uniforms.timeScale.x, 0.0001);
  let accelerationContribution = select(0.0, 0.5, previousMeta.z > 0.35);
  var predicted = previousState.xy * dt
    + previousState.zw * dt * dt * accelerationContribution;
  predicted = clamp(predicted, vec2<f32>(-32.0), vec2<f32>(32.0));

  var bestScore = 1000.0;
  var secondBest = 1000.0;
  var bestOffset = predicted;
  let searchRadius = select(3, 2, uniforms.flags.y > 1.5);

  for (var y = -3; y <= 3; y += 1) {
    for (var x = -3; x <= 3; x += 1) {
      if (abs(x) > searchRadius || abs(y) > searchRadius) {
        continue;
      }
      let candidate = predicted + vec2<f32>(f32(x), f32(y)) * 4.0;
      let score = candidateScore(centerUv, candidate);
      if (score < bestScore) {
        secondBest = bestScore;
        bestScore = score;
        bestOffset = candidate;
      } else if (score < secondBest) {
        secondBest = score;
      }
    }
  }

  let coarseBest = bestScore;
  let coarseSecond = secondBest;
  var refinementStep = 2.0;
  for (var level = 0u; level < 3u; level += 1u) {
    let refinementCenter = bestOffset;
    for (var y = -1; y <= 1; y += 1) {
      for (var x = -1; x <= 1; x += 1) {
        let candidate = refinementCenter + vec2<f32>(f32(x), f32(y)) * refinementStep;
        let score = candidateScore(centerUv, candidate);
        if (score < bestScore) {
          bestScore = score;
          bestOffset = candidate;
        }
      }
    }
    refinementStep *= 0.5;
  }

  let matchConfidence = exp(-bestScore * 14.0);
  let uniqueness = clamp(
    (coarseSecond - coarseBest) / max(coarseSecond, 0.0001),
    0.0,
    1.0,
  );
  let predictorAgreement = 1.0 - smoothstep(2.0, 14.0, distance(bestOffset, predicted));
  let predictionTrust = mix(0.45, 1.0, predictorAgreement);
  let previousTrust = mix(0.65, 1.0, previousMeta.z);
  let previousCenter = centerUv + bestOffset * uniforms.inputSize.zw;
  let margin = vec2<f32>(2.0) * uniforms.analysisSize.zw;
  let inBounds = all(previousCenter >= margin) && all(previousCenter <= vec2<f32>(1.0) - margin);
  let confidence = matchConfidence * uniqueness * predictionTrust * previousTrust
    * select(0.25, 1.0, inBounds);

  let observedVelocity = bestOffset / dt;
  let predictedVelocity = previousState.xy + previousState.zw * dt;
  let observedAcceleration = clampLength(
    (observedVelocity - previousState.xy) / dt,
    uniforms.thresholds.y,
  );
  let velocity = mix(predictedVelocity, observedVelocity, confidence);
  let acceleration = mix(previousState.zw, observedAcceleration, confidence * 0.35);
  let normalizedError = clamp(bestScore * 4.0, 0.0, 1.0);

  textureStore(motionStateCurrent, vec2<i32>(id.xy), vec4<f32>(velocity, acceleration));
  textureStore(motionMetaCurrent, vec2<i32>(id.xy), vec4<f32>(bestOffset, confidence, normalizedError));
}

