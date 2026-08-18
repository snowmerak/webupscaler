struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  motionSize: vec4<f32>,
  timeScale: vec4<f32>,
  thresholds: vec4<f32>,
  flags: vec4<f32>,
}

@group(0) @binding(0) var reconstruction: texture_2d<f32>;
@group(0) @binding(1) var refinedFrame: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> uniforms: FrameUniforms;

const OFFSETS = array<vec2<i32>, 4>(
  vec2<i32>(0, -1),
  vec2<i32>(-1, 0), vec2<i32>(1, 0),
  vec2<i32>(0, 1),
);

fn luma(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn loadFrame(position: vec2<i32>) -> vec3<f32> {
  let maximum = vec2<i32>(uniforms.outputSize.xy) - vec2<i32>(1);
  return textureLoad(
    reconstruction,
    clamp(position, vec2<i32>(0), maximum),
    0,
  ).rgb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dimensions = textureDimensions(refinedFrame);
  if (any(id.xy >= dimensions)) {
    return;
  }

  let position = vec2<i32>(id.xy);
  let center = loadFrame(position);
  let centerLuma = luma(center);
  let samples = array<vec3<f32>, 4>(
    loadFrame(position + OFFSETS[0]),
    loadFrame(position + OFFSETS[1]),
    loadFrame(position + OFFSETS[2]),
    loadFrame(position + OFFSETS[3]),
  );
  var filtered = center * 2.0;
  var weightSum = 2.0;
  var minimum = center;
  var maximum = center;
  var minimumLuma = centerLuma;
  var maximumLuma = centerLuma;

  for (var index = 0u; index < 4u; index += 1u) {
    let sample = samples[index];
    let sampleLuma = luma(sample);
    let rangeWeight = exp(-abs(sampleLuma - centerLuma) * 34.0);
    let weight = rangeWeight;
    filtered += sample * weight;
    weightSum += weight;
    minimum = min(minimum, sample);
    maximum = max(maximum, sample);
    minimumLuma = min(minimumLuma, sampleLuma);
    maximumLuma = max(maximumLuma, sampleLuma);
  }

  filtered /= max(weightSum, 0.0001);
  let localRange = maximumLuma - minimumLuma;
  let residual = abs(centerLuma - luma(filtered));
  let flatMask = 1.0 - smoothstep(0.022, 0.105, localRange);
  let horizontalCurvature = abs(
    luma(samples[1]) + luma(samples[2]) - 2.0 * centerLuma
  );
  let verticalCurvature = abs(
    luma(samples[0]) + luma(samples[3]) - 2.0 * centerLuma
  );
  let gradientMagnitude = max(
    abs(luma(samples[2]) - luma(samples[1])),
    abs(luma(samples[3]) - luma(samples[0])),
  ) * 0.5;
  let gradientContinuity = 1.0 - smoothstep(
    0.006,
    0.030,
    min(horizontalCurvature, verticalCurvature),
  );
  let gradientEvidence = smoothstep(0.0008, 0.012, gradientMagnitude);
  let noiseEvidence = smoothstep(0.0025, 0.020, residual);
  let ringEvidence = smoothstep(0.009, 0.050, residual)
    * (1.0 - smoothstep(0.085, 0.180, localRange));
  let modeScale = select(
    1.15,
    select(0.88, 0.62, uniforms.flags.y > 1.5),
    uniforms.flags.y > 0.5,
  );
  // A small continuous-tone contribution joins quantized steps in walls,
  // skin and other broad gradients. Curvature and range masks keep it away
  // from text and real object boundaries.
  let gradientSmoothing = flatMask * gradientContinuity
    * (0.14 + 0.22 * gradientEvidence);
  let smoothing = clamp(
    (
      gradientSmoothing
      + flatMask * (0.92 * noiseEvidence + 0.62 * ringEvidence)
    ) * modeScale,
    0.0,
    0.78,
  );
  let candidate = mix(center, filtered, smoothing);
  let allowance = (maximum - minimum) * 0.025 + vec3<f32>(0.0005);
  let resolved = clamp(candidate, minimum - allowance, maximum + allowance);

  textureStore(
    refinedFrame,
    position,
    vec4<f32>(clamp(resolved, vec3<f32>(0.0), vec3<f32>(1.0)), smoothing),
  );
}
