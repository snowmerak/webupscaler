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
@group(0) @binding(3) var motionMetaPrevious: texture_2d<f32>;
@group(0) @binding(4) var historyPrevious: texture_2d<f32>;
@group(0) @binding(5) var historyCurrent: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(6) var linearClamp: sampler;
@group(0) @binding(7) var<uniform> uniforms: FrameUniforms;

fn luma(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let outputDimensions = vec2<u32>(uniforms.outputSize.xy);
  if (any(id.xy >= outputDimensions)) {
    return;
  }

  let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) * uniforms.outputSize.zw;
  let base = textureSampleLevel(inputFrame, linearClamp, uv, 0.0).rgb;
  let feature = textureSampleLevel(featureCurrent, linearClamp, uv, 0.0);
  let gradient = feature.yz;
  let gradientLength = length(gradient);
  let tangent = vec2<f32>(-gradient.y, gradient.x) / max(gradientLength, 0.00001);
  let radius = uniforms.inputSize.zw * 0.55;
  let along = 0.5 * (
    textureSampleLevel(inputFrame, linearClamp, uv + tangent * radius, 0.0).rgb
    + textureSampleLevel(inputFrame, linearClamp, uv - tangent * radius, 0.0).rgb
  );
  let edgeStrength = smoothstep(0.015, 0.12, gradientLength)
    * smoothstep(0.005, 0.08, feature.w);
  let spatial = mix(base, along, edgeStrength * 0.22);

  let motion = textureSampleLevel(motionMetaCurrent, linearClamp, uv, 0.0);
  let previousMotion = textureSampleLevel(motionMetaPrevious, linearClamp, uv, 0.0);
  let previousUv = uv + motion.xy * uniforms.inputSize.zw;
  let inBounds = all(previousUv >= vec2<f32>(0.0)) && all(previousUv <= vec2<f32>(1.0));
  let history = textureSampleLevel(historyPrevious, linearClamp, previousUv, 0.0);
  let photoError = abs(luma(spatial) - luma(history.rgb));
  let photoTrust = 1.0 - smoothstep(0.025, 0.12, photoError);

  let motionTexel = uniforms.motionSize.zw;
  let leftMotion = textureSampleLevel(motionMetaCurrent, linearClamp, uv - vec2<f32>(motionTexel.x, 0.0), 0.0).xy;
  let rightMotion = textureSampleLevel(motionMetaCurrent, linearClamp, uv + vec2<f32>(motionTexel.x, 0.0), 0.0).xy;
  let upMotion = textureSampleLevel(motionMetaCurrent, linearClamp, uv - vec2<f32>(0.0, motionTexel.y), 0.0).xy;
  let downMotion = textureSampleLevel(motionMetaCurrent, linearClamp, uv + vec2<f32>(0.0, motionTexel.y), 0.0).xy;
  let motionDifference = 0.25 * (
    distance(motion.xy, leftMotion) + distance(motion.xy, rightMotion)
    + distance(motion.xy, upMotion) + distance(motion.xy, downMotion)
  );
  let coherence = exp(-motionDifference * 0.18);
  let matchingTrust = 1.0 - motion.w;
  let resetMask = select(1.0, 0.0, uniforms.flags.x > 0.5);
  let historyTrust = history.a * motion.z * photoTrust * coherence * matchingTrust
    * select(0.0, 1.0, inBounds) * resetMask;

  let sourceCoordinate = uv * uniforms.inputSize.xy - vec2<f32>(0.5);
  let sampleDelta = abs(sourceCoordinate - round(sourceCoordinate));
  let sampleProximity = exp(-4.0 * dot(sampleDelta, sampleDelta));
  let currentWeight = 0.35 + sampleProximity * 0.65;
  let historyWeight = min(
    historyTrust * uniforms.timeScale.w,
    uniforms.thresholds.w,
  );
  let color = (
    spatial * currentWeight + history.rgb * historyWeight
  ) / max(currentWeight + historyWeight, 0.0001);

  let phaseDelta = abs(fract(motion.xy) - fract(previousMotion.xy));
  let wrappedPhaseDelta = min(phaseDelta, vec2<f32>(1.0) - phaseDelta);
  let phaseNovelty = smoothstep(0.04, 0.3, length(wrappedPhaseDelta));
  let observation = mix(0.2, 1.0, sampleProximity) * mix(0.4, 1.0, phaseNovelty);
  let retained = historyTrust * uniforms.thresholds.z;
  let confidence = clamp(observation + retained * (1.0 - observation), 0.0, 1.0);

  textureStore(historyCurrent, vec2<i32>(id.xy), vec4<f32>(color, confidence));
}

