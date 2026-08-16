struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  params: vec4<f32>,
}

@group(0) @binding(0) var reconstructedInput: texture_2d<f32>;
@group(0) @binding(1) var linearClamp: sampler;
@group(0) @binding(2) var<uniform> uniforms: FrameUniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  var output: VertexOutput;
  output.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
  output.uv = positions[vertexIndex] * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let texel = uniforms.outputSize.zw;
  let center = textureSample(reconstructedInput, linearClamp, input.uv).rgb;
  let north = textureSample(reconstructedInput, linearClamp, input.uv - vec2<f32>(0.0, texel.y)).rgb;
  let south = textureSample(reconstructedInput, linearClamp, input.uv + vec2<f32>(0.0, texel.y)).rgb;
  let west = textureSample(reconstructedInput, linearClamp, input.uv - vec2<f32>(texel.x, 0.0)).rgb;
  let east = textureSample(reconstructedInput, linearClamp, input.uv + vec2<f32>(texel.x, 0.0)).rgb;
  let blur = center * 0.5 + (north + south + west + east) * 0.125;
  let detail = center - blur;
  let edgeMask = 1.0 - smoothstep(0.08, 0.3, max(max(abs(detail.r), abs(detail.g)), abs(detail.b)));
  let sharpened = center + detail * uniforms.params.y * edgeMask;
  return vec4<f32>(clamp(sharpened, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

