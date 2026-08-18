struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  motionSize: vec4<f32>,
  timeScale: vec4<f32>,
  thresholds: vec4<f32>,
  flags: vec4<f32>,
}

@group(0) @binding(0) var shockedFrame: texture_2d<f32>;
@group(0) @binding(1) var clarityFrame: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> uniforms: FrameUniforms;

const OFFSETS = array<vec2<i32>, 21>(
  vec2<i32>(0, 0),
  vec2<i32>(-1, 0), vec2<i32>(1, 0),
  vec2<i32>(0, -1), vec2<i32>(0, 1),
  vec2<i32>(-2, -2), vec2<i32>(2, -2),
  vec2<i32>(-2, 2), vec2<i32>(2, 2),
  vec2<i32>(-3, 0), vec2<i32>(3, 0),
  vec2<i32>(0, -3), vec2<i32>(0, 3),
  vec2<i32>(-2, -1), vec2<i32>(2, -1),
  vec2<i32>(-2, 1), vec2<i32>(2, 1),
  vec2<i32>(-1, -2), vec2<i32>(1, -2),
  vec2<i32>(-1, 2), vec2<i32>(1, 2),
);

const WEIGHTS = array<f32, 21>(
  5.0,
  3.0, 3.0, 3.0, 3.0,
  1.0, 1.0, 1.0, 1.0,
  1.0, 1.0, 1.0, 1.0,
  1.25, 1.25, 1.25, 1.25,
  1.25, 1.25, 1.25, 1.25,
);

fn luma(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn loadFrame(position: vec2<i32>) -> vec3<f32> {
  let maximum = vec2<i32>(uniforms.outputSize.xy) - vec2<i32>(1);
  return textureLoad(
    shockedFrame,
    clamp(position, vec2<i32>(0), maximum),
    0,
  ).rgb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dimensions = textureDimensions(clarityFrame);
  if (any(id.xy >= dimensions)) {
    return;
  }

  let position = vec2<i32>(id.xy);
  let center = loadFrame(position);
  let centerLuma = luma(center);
  var weightedSum = vec3<f32>(0.0);
  var weightSum = 0.0;
  var minimum = center;
  var maximum = center;
  var minimumLuma = centerLuma;
  var maximumLuma = centerLuma;

  for (var index = 0; index < 21; index += 1) {
    let sample = loadFrame(position + OFFSETS[index]);
    let sampleLuma = luma(sample);
    // Range weighting keeps the wide footprint from crossing a strong edge.
    let rangeWeight = exp(-abs(sampleLuma - centerLuma) * 18.0);
    let weight = WEIGHTS[index] * rangeWeight;
    weightedSum += sample * weight;
    weightSum += weight;
    minimum = min(minimum, sample);
    maximum = max(maximum, sample);
    minimumLuma = min(minimumLuma, sampleLuma);
    maximumLuma = max(maximumLuma, sampleLuma);
  }

  let wideBase = weightedSum / max(weightSum, 0.0001);
  let midDetail = centerLuma - luma(wideBase);
  let localRange = maximumLuma - minimumLuma;
  let left = luma(loadFrame(position + vec2<i32>(-1, 0)));
  let right = luma(loadFrame(position + vec2<i32>(1, 0)));
  let top = luma(loadFrame(position + vec2<i32>(0, -1)));
  let bottom = luma(loadFrame(position + vec2<i32>(0, 1)));
  let gradient = length(vec2<f32>((right - left) * 0.5, (bottom - top) * 0.5));

  // This pass fills the frequency gap between the 13-tap micro-detail pass
  // and the final 1-pixel CAS pass. It deliberately ignores perfectly flat
  // gradients and rolls off again on hard edges to avoid a second halo.
  let detailEvidence = smoothstep(0.0015, 0.022, abs(midDetail));
  let structureEvidence = smoothstep(0.010, 0.075, localRange);
  let hardEdgeProtection = 1.0 - smoothstep(0.135, 0.340, localRange);
  let gradientProtection = 1.0 - smoothstep(0.095, 0.230, gradient);
  let luminanceProtection = smoothstep(0.018, 0.110, centerLuma)
    * (1.0 - smoothstep(0.885, 0.992, centerLuma));
  let modeScale = select(
    1.28,
    select(0.96, 0.68, uniforms.flags.y > 1.5),
    uniforms.flags.y > 0.5,
  );
  let gain = detailEvidence
    * structureEvidence
    * hardEdgeProtection
    * gradientProtection
    * luminanceProtection
    * modeScale;
  let clarityDelta = clamp(midDetail * gain, -0.052, 0.052);
  let candidate = center + vec3<f32>(clarityDelta);
  let allowance = (maximum - minimum) * 0.10 + vec3<f32>(0.002);
  let resolved = clamp(candidate, minimum - allowance, maximum + allowance);

  textureStore(
    clarityFrame,
    position,
    vec4<f32>(clamp(resolved, vec3<f32>(0.0), vec3<f32>(1.0)), abs(clarityDelta)),
  );
}
