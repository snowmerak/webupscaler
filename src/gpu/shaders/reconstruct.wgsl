struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  params: vec4<f32>,
}

@group(0) @binding(0) var inputFrame: texture_2d<f32>;
@group(0) @binding(1) var featureInput: texture_2d<f32>;
@group(0) @binding(2) var reconstructedOutput: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var linearClamp: sampler;
@group(0) @binding(4) var<uniform> uniforms: FrameUniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let outputDimensions = vec2<u32>(uniforms.outputSize.xy);
  if (any(id.xy >= outputDimensions)) {
    return;
  }

  let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) * uniforms.outputSize.zw;
  let base = textureSampleLevel(inputFrame, linearClamp, uv, 0.0).rgb;
  let feature = textureSampleLevel(featureInput, linearClamp, uv, 0.0);
  let gradient = feature.yz;
  let gradientLength = length(gradient);
  let safeLength = max(gradientLength, 0.00001);
  let tangent = vec2<f32>(-gradient.y, gradient.x) / safeLength;
  let radius = uniforms.inputSize.zw * 0.55;
  let along = 0.5 * (
    textureSampleLevel(inputFrame, linearClamp, uv + tangent * radius, 0.0).rgb
    + textureSampleLevel(inputFrame, linearClamp, uv - tangent * radius, 0.0).rgb
  );
  let edgeStrength = smoothstep(0.015, 0.12, gradientLength) * smoothstep(0.005, 0.08, feature.w);
  let spatial = mix(base, along, edgeStrength * 0.22);

  textureStore(reconstructedOutput, vec2<i32>(id.xy), vec4<f32>(spatial, 1.0));
}

