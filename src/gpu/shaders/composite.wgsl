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
@group(0) @binding(1) var<uniform> uniforms: FrameUniforms;

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

fn lanczos2Weight(distance: f32) -> f32 {
  let x = abs(distance);
  if (x < 0.0001) {
    return 1.0;
  }
  if (x >= 2.0) {
    return 0.0;
  }
  let pix = 3.14159265359 * x;
  return sin(pix) * sin(pix * 0.5) / (pix * pix * 0.5);
}

fn loadReconstruction(position: vec2<i32>) -> vec3<f32> {
  let maximum = vec2<i32>(uniforms.outputSize.xy) - vec2<i32>(1);
  return textureLoad(
    reconstruction,
    clamp(position, vec2<i32>(0), maximum),
    0,
  ).rgb;
}

// Resize the already-resolved 2x lattice to the presentation canvas exactly
// once. No spatial fallback or temporal mixing happens in this pass.
fn lanczosResolve(uv: vec2<f32>) -> vec3<f32> {
  let sourcePosition = uv * uniforms.outputSize.xy - vec2<f32>(0.5);
  let base = vec2<i32>(floor(sourcePosition));
  let phase = fract(sourcePosition);
  var result = vec3<f32>(0.0);
  var weightSum = 0.0;

  for (var y = -1; y <= 2; y += 1) {
    let weightY = lanczos2Weight(f32(y) - phase.y);
    for (var x = -1; x <= 2; x += 1) {
      let weight = lanczos2Weight(f32(x) - phase.x) * weightY;
      result += loadReconstruction(base + vec2<i32>(x, y)) * weight;
      weightSum += weight;
    }
  }

  let filtered = result / max(weightSum, 0.0001);
  let a = loadReconstruction(base);
  let b = loadReconstruction(base + vec2<i32>(1, 0));
  let c = loadReconstruction(base + vec2<i32>(0, 1));
  let d = loadReconstruction(base + vec2<i32>(1, 1));
  let minimum = min(min(a, b), min(c, d));
  let maximum = max(max(a, b), max(c, d));
  let allowance = (maximum - minimum) * 0.04 + vec3<f32>(0.001);
  return clamp(filtered, minimum - allowance, maximum + allowance);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let resolved = lanczosResolve(input.uv);
  let valid = all(resolved == resolved)
    && all(abs(resolved) <= vec3<f32>(65504.0));
  let safe = select(vec3<f32>(0.0), resolved, valid);
  return vec4<f32>(clamp(safe, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
