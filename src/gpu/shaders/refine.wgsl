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
  vec2<i32>(0, -2),
  vec2<i32>(-2, 0), vec2<i32>(2, 0),
  vec2<i32>(0, 2),
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
  let top = loadFrame(position + OFFSETS[0]);
  let left = loadFrame(position + OFFSETS[1]);
  let right = loadFrame(position + OFFSETS[2]);
  let bottom = loadFrame(position + OFFSETS[3]);
  let topLuma = luma(top);
  let leftLuma = luma(left);
  let rightLuma = luma(right);
  let bottomLuma = luma(bottom);
  let topWeight = exp(-abs(topLuma - centerLuma) * 34.0);
  let leftWeight = exp(-abs(leftLuma - centerLuma) * 34.0);
  let rightWeight = exp(-abs(rightLuma - centerLuma) * 34.0);
  let bottomWeight = exp(-abs(bottomLuma - centerLuma) * 34.0);
  let horizontalFiltered = (
    center * 2.0 + left * leftWeight + right * rightWeight
  ) / max(2.0 + leftWeight + rightWeight, 0.0001);
  let verticalFiltered = (
    center * 2.0 + top * topWeight + bottom * bottomWeight
  ) / max(2.0 + topWeight + bottomWeight, 0.0001);
  let horizontalCurvature = abs(leftLuma + rightLuma - 2.0 * centerLuma);
  let verticalCurvature = abs(topLuma + bottomLuma - 2.0 * centerLuma);
  let useHorizontal = horizontalCurvature <= verticalCurvature;
  let filtered = select(verticalFiltered, horizontalFiltered, useHorizontal);
  let minimum = min(center, min(min(top, bottom), min(left, right)));
  let maximum = max(center, max(max(top, bottom), max(left, right)));
  let minimumLuma = min(centerLuma, min(min(topLuma, bottomLuma), min(leftLuma, rightLuma)));
  let maximumLuma = max(centerLuma, max(max(topLuma, bottomLuma), max(leftLuma, rightLuma)));
  let localRange = maximumLuma - minimumLuma;
  let residual = abs(centerLuma - luma(filtered));
  let flatMask = 1.0 - smoothstep(0.022, 0.105, localRange);
  let gradientMagnitude = select(
    abs(bottomLuma - topLuma),
    abs(rightLuma - leftLuma),
    useHorizontal,
  ) * 0.5;
  let selectedCurvature = select(
    verticalCurvature,
    horizontalCurvature,
    useHorizontal,
  );
  let gradientContinuity = 1.0 - smoothstep(
    0.006,
    0.030,
    selectedCurvature,
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
  // The +/-2 footprint stays on the same phase of the 2x lattice. Only the
  // lower-curvature axis contributes, so synthesized subpixels are not mixed
  // across a real edge merely because the perpendicular direction is flat.
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
