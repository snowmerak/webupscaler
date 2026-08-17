struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  motionSize: vec4<f32>,
  timeScale: vec4<f32>,
  thresholds: vec4<f32>,
  flags: vec4<f32>,
}

@group(0) @binding(0) var latentHr: texture_2d<f32>;
@group(0) @binding(1) var observedLr: texture_2d<f32>;
@group(0) @binding(2) var residualOutput: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> uniforms: FrameUniforms;

fn loadLatent(position: vec2<i32>) -> vec3<f32> {
  let maximum = vec2<i32>(uniforms.outputSize.xy) - vec2<i32>(1);
  return textureLoad(latentHr, clamp(position, vec2<i32>(0), maximum), 0).rgb;
}

fn forwardProject(lrPosition: vec2<i32>) -> vec3<f32> {
  let scale = uniforms.outputSize.xy * uniforms.inputSize.zw;
  let center = vec2<i32>(round(vec2<f32>(lrPosition) * scale));

  // A small pixel-response model: project the latent 2x lattice back into
  // the footprint of one source pixel rather than merely sampling its center.
  let centerSample = loadLatent(center) * 0.25;
  let axial = (
    loadLatent(center + vec2<i32>(-1, 0))
    + loadLatent(center + vec2<i32>(1, 0))
    + loadLatent(center + vec2<i32>(0, -1))
    + loadLatent(center + vec2<i32>(0, 1))
  ) * 0.125;
  let diagonal = (
    loadLatent(center + vec2<i32>(-1, -1))
    + loadLatent(center + vec2<i32>(1, -1))
    + loadLatent(center + vec2<i32>(-1, 1))
    + loadLatent(center + vec2<i32>(1, 1))
  ) * 0.0625;
  return centerSample + axial + diagonal;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let inputDimensions = vec2<u32>(uniforms.inputSize.xy);
  if (any(id.xy >= inputDimensions)) {
    return;
  }

  let position = vec2<i32>(id.xy);
  let observed = textureLoad(observedLr, position, 0).rgb;
  let predicted = forwardProject(position);
  let residual = observed - predicted;
  let sumSq = dot(residual, residual) / 3.0;
  textureStore(residualOutput, position, vec4<f32>(residual, sumSq));
}
