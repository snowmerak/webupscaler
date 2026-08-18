struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  motionSize: vec4<f32>,
  timeScale: vec4<f32>,
  thresholds: vec4<f32>,
  flags: vec4<f32>,
}

@group(0) @binding(0) var refinedFrame: texture_2d<f32>;
@group(0) @binding(1) var detailedFrame: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> uniforms: FrameUniforms;

const OFFSETS = array<vec2<i32>, 13>(
  vec2<i32>(0, 0),
  vec2<i32>(-1, 0), vec2<i32>(1, 0),
  vec2<i32>(0, -1), vec2<i32>(0, 1),
  vec2<i32>(-2, 0), vec2<i32>(2, 0),
  vec2<i32>(0, -2), vec2<i32>(0, 2),
  vec2<i32>(-1, -1), vec2<i32>(1, -1),
  vec2<i32>(-1, 1), vec2<i32>(1, 1),
);

const WEIGHTS = array<f32, 13>(
  4.0,
  2.0, 2.0, 2.0, 2.0,
  1.0, 1.0, 1.0, 1.0,
  1.5, 1.5, 1.5, 1.5,
);

fn luma(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn loadFrame(position: vec2<i32>) -> vec3<f32> {
  let maximum = vec2<i32>(uniforms.outputSize.xy) - vec2<i32>(1);
  return textureLoad(
    refinedFrame,
    clamp(position, vec2<i32>(0), maximum),
    0,
  ).rgb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dimensions = textureDimensions(detailedFrame);
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

  for (var index = 0; index < 13; index += 1) {
    let sample = loadFrame(position + OFFSETS[index]);
    let sampleLuma = luma(sample);
    weightedSum += sample * WEIGHTS[index];
    weightSum += WEIGHTS[index];
    minimum = min(minimum, sample);
    maximum = max(maximum, sample);
    minimumLuma = min(minimumLuma, sampleLuma);
    maximumLuma = max(maximumLuma, sampleLuma);
  }

  let localAverage = weightedSum / max(weightSum, 0.0001);
  let detail = centerLuma - luma(localAverage);
  let localRange = maximumLuma - minimumLuma;
  let leftLuma = luma(loadFrame(position + vec2<i32>(-1, 0)));
  let rightLuma = luma(loadFrame(position + vec2<i32>(1, 0)));
  let topLuma = luma(loadFrame(position + vec2<i32>(0, -1)));
  let bottomLuma = luma(loadFrame(position + vec2<i32>(0, 1)));
  let gradientMagnitude = length(vec2<f32>(
    (rightLuma - leftLuma) * 0.5,
    (bottomLuma - topLuma) * 0.5,
  ));

  // Amplify real, low-amplitude texture between edges. Completely flat areas
  // stay flat, while hard edges are reserved for the following shock pass.
  let detailEvidence = smoothstep(0.0012, 0.018, abs(detail));
  let textureEvidence = smoothstep(0.005, 0.055, localRange);
  let edgeProtection = 1.0 - smoothstep(0.110, 0.300, localRange);
  let gradientProtection = 1.0 - smoothstep(0.070, 0.180, gradientMagnitude);
  let luminanceProtection = smoothstep(0.025, 0.140, centerLuma)
    * (1.0 - smoothstep(0.860, 0.985, centerLuma));
  let modeScale = select(
    1.85,
    select(1.38, 0.92, uniforms.flags.y > 1.5),
    uniforms.flags.y > 0.5,
  );
  let gain = detailEvidence
    * textureEvidence
    * edgeProtection
    * gradientProtection
    * luminanceProtection
    * modeScale;
  let boostedDetail = clamp(detail * gain, -0.040, 0.040);
  let candidate = center + vec3<f32>(boostedDetail);
  let allowance = (maximum - minimum) * 0.20 + vec3<f32>(0.003);
  let resolved = clamp(candidate, minimum - allowance, maximum + allowance);

  textureStore(
    detailedFrame,
    position,
    vec4<f32>(clamp(resolved, vec3<f32>(0.0), vec3<f32>(1.0)), gain),
  );
}
