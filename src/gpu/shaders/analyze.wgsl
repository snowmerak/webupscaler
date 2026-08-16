struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  params: vec4<f32>,
}

@group(0) @binding(0) var inputFrame: texture_2d<f32>;
@group(0) @binding(1) var linearClamp: sampler;
@group(0) @binding(2) var featureOutput: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> uniforms: FrameUniforms;

fn luma(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let analysisDimensions = vec2<u32>(uniforms.analysisSize.xy);
  if (any(id.xy >= analysisDimensions)) {
    return;
  }

  let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) * uniforms.analysisSize.zw;
  let texel = uniforms.inputSize.zw * 2.0;
  let center = luma(textureSampleLevel(inputFrame, linearClamp, uv, 0.0).rgb);
  let left = luma(textureSampleLevel(inputFrame, linearClamp, uv - vec2<f32>(texel.x, 0.0), 0.0).rgb);
  let right = luma(textureSampleLevel(inputFrame, linearClamp, uv + vec2<f32>(texel.x, 0.0), 0.0).rgb);
  let up = luma(textureSampleLevel(inputFrame, linearClamp, uv - vec2<f32>(0.0, texel.y), 0.0).rgb);
  let down = luma(textureSampleLevel(inputFrame, linearClamp, uv + vec2<f32>(0.0, texel.y), 0.0).rgb);
  let gradient = vec2<f32>(right - left, down - up) * 0.5;
  let variance = 0.25 * (
    abs(left - center) + abs(right - center) + abs(up - center) + abs(down - center)
  );

  textureStore(featureOutput, vec2<i32>(id.xy), vec4<f32>(center, gradient, variance));
}

