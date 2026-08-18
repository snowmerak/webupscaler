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
@group(0) @binding(1) var enhancedFrame: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> uniforms: FrameUniforms;

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
  let dimensions = textureDimensions(enhancedFrame);
  if (any(id.xy >= dimensions)) {
    return;
  }

  let position = vec2<i32>(id.xy);
  let center = loadFrame(position);
  let tl = loadFrame(position + vec2<i32>(-1, -1));
  let top = loadFrame(position + vec2<i32>(0, -1));
  let tr = loadFrame(position + vec2<i32>(1, -1));
  let left = loadFrame(position + vec2<i32>(-1, 0));
  let right = loadFrame(position + vec2<i32>(1, 0));
  let bl = loadFrame(position + vec2<i32>(-1, 1));
  let bottom = loadFrame(position + vec2<i32>(0, 1));
  let br = loadFrame(position + vec2<i32>(1, 1));

  let lowPass = (
    tl + top * 2.0 + tr
    + left * 2.0 + center * 4.0 + right * 2.0
    + bl + bottom * 2.0 + br
  ) * 0.0625;
  let centerLuma = luma(center);
  let lowLuma = luma(lowPass);
  let detail = centerLuma - lowLuma;
  let gx = luma(tr + right * 2.0 + br - tl - left * 2.0 - bl) * 0.25;
  let gy = luma(bl + bottom * 2.0 + br - tl - top * 2.0 - tr) * 0.25;
  let edgeMagnitude = length(vec2<f32>(gx, gy));

  let minimum = min(center, min(min(min(tl, top), min(tr, left)), min(min(right, bl), min(bottom, br))));
  let maximum = max(center, max(max(max(tl, top), max(tr, left)), max(max(right, bl), max(bottom, br))));
  let minimumLuma = luma(minimum);
  let maximumLuma = luma(maximum);
  let localRange = maximumLuma - minimumLuma;
  let detailMask = smoothstep(0.0025, 0.022, abs(detail));
  let edgeMask = smoothstep(0.014, 0.125, edgeMagnitude);
  let extremity = smoothstep(0.16, 0.40, abs(centerLuma - 0.5));
  let textLike = smoothstep(0.075, 0.260, localRange) * edgeMask * extremity;
  let modeScale = select(
    1.15,
    select(0.88, 0.62, uniforms.flags.y > 1.5),
    uniforms.flags.y > 0.5,
  );

  // Deliberately aggressive starting point: visible enough to reveal ringing
  // and halo limits, then intended to be tuned down from real playback.
  let sharpenGain = (
    0.90 + 0.48 * edgeMask + 0.32 * textLike
  ) * detailMask * modeScale;
  let shoulder = max(0.0, 1.0 - 4.0 * (centerLuma - 0.5) * (centerLuma - 0.5));
  let globalContrast = (centerLuma - 0.5) * 0.12 * shoulder * modeScale;
  let localContrast = detail * 0.12 * (0.35 + 0.65 * edgeMask) * modeScale;
  let targetLuma = centerLuma + detail * sharpenGain + globalContrast + localContrast;
  let lumaShifted = center + vec3<f32>(targetLuma - centerLuma);

  let allowance = (maximum - minimum) * 0.12 + vec3<f32>(0.003);
  let bounded = clamp(lumaShifted, minimum - allowance, maximum + allowance);
  let valid = all(bounded == bounded)
    && all(abs(bounded) <= vec3<f32>(65504.0));
  let resolved = select(center, bounded, valid);

  textureStore(
    enhancedFrame,
    position,
    vec4<f32>(clamp(resolved, vec3<f32>(0.0), vec3<f32>(1.0)), sharpenGain),
  );
}
