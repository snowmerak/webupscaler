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
  let top = loadFrame(position + vec2<i32>(0, -1));
  let left = loadFrame(position + vec2<i32>(-1, 0));
  let right = loadFrame(position + vec2<i32>(1, 0));
  let bottom = loadFrame(position + vec2<i32>(0, 1));

  // Five-tap contrast-adaptive sharpening. The cross footprint is cheaper
  // than the old Sobel/3x3 pass and keeps the response centered on the pixel.
  let lowPass = center * 0.5 + (top + left + right + bottom) * 0.125;
  let centerLuma = luma(center);
  let lowLuma = luma(lowPass);
  let detail = centerLuma - lowLuma;
  let gx = (luma(right) - luma(left)) * 0.5;
  let gy = (luma(bottom) - luma(top)) * 0.5;
  let edgeMagnitude = length(vec2<f32>(gx, gy));

  let minimum = min(center, min(min(top, bottom), min(left, right)));
  let maximum = max(center, max(max(top, bottom), max(left, right)));
  let minimumLuma = luma(minimum);
  let maximumLuma = luma(maximum);
  let localRange = maximumLuma - minimumLuma;
  let detailMask = smoothstep(0.0018, 0.018, abs(detail));
  let edgeMask = smoothstep(0.010, 0.105, edgeMagnitude);
  let extremity = smoothstep(0.16, 0.40, abs(centerLuma - 0.5));
  let textLike = smoothstep(0.075, 0.260, localRange) * edgeMask * extremity;
  let structuredDetail = smoothstep(0.010, 0.050, localRange);
  let gradientSharpening = mix(0.30, 1.0, structuredDetail);
  let modeScale = select(
    1.15,
    select(0.88, 0.62, uniforms.flags.y > 1.5),
    uniforms.flags.y > 0.5,
  );

  // Deliberately aggressive starting point: visible enough to reveal ringing
  // and halo limits, then intended to be tuned down from real playback.
  let sharpenGain = (
    1.18 + 0.58 * edgeMask + 0.38 * textLike
  ) * detailMask * gradientSharpening * modeScale;
  let shoulder = max(0.0, 1.0 - 4.0 * (centerLuma - 0.5) * (centerLuma - 0.5));
  let globalContrast = (centerLuma - 0.5) * 0.12 * shoulder * modeScale;
  let localContrast = detail * 0.18 * (0.35 + 0.65 * edgeMask)
    * mix(0.45, 1.0, structuredDetail) * modeScale;
  let targetLuma = centerLuma + detail * sharpenGain + globalContrast + localContrast;
  let lumaShifted = center + vec3<f32>(targetLuma - centerLuma);

  let allowance = (maximum - minimum) * 0.16 + vec3<f32>(0.004);
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
