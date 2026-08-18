struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  motionSize: vec4<f32>,
  timeScale: vec4<f32>,
  thresholds: vec4<f32>,
  flags: vec4<f32>,
}

@group(0) @binding(0) var inputFrame: texture_2d<f32>;
@group(0) @binding(1) var filteredFrame: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> uniforms: FrameUniforms;

fn luma(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn loadInput(position: vec2<i32>) -> vec3<f32> {
  let maximum = vec2<i32>(uniforms.inputSize.xy) - vec2<i32>(1);
  return textureLoad(
    inputFrame,
    clamp(position, vec2<i32>(0), maximum),
    0,
  ).rgb;
}

fn axisOffset(horizontal: bool, amount: i32) -> vec2<i32> {
  return select(vec2<i32>(0, amount), vec2<i32>(amount, 0), horizontal);
}

// Estimate a codec transform boundary on the four-pixel grid. A real image
// edge normally has supporting gradients on both sides; a block seam has a
// large isolated step between two comparatively flat plateaus.
fn boundaryCorrection(position: vec2<i32>, horizontal: bool) -> vec3<f32> {
  let coordinate = select(position.y, position.x, horizontal);
  let boundary = ((coordinate + 2) / 4) * 4;
  let boundaryPosition = select(
    vec2<i32>(position.x, boundary),
    vec2<i32>(boundary, position.y),
    horizontal,
  );

  let a2 = loadInput(boundaryPosition + axisOffset(horizontal, -2));
  let a1 = loadInput(boundaryPosition + axisOffset(horizontal, -1));
  let b0 = loadInput(boundaryPosition);
  let b1 = loadInput(boundaryPosition + axisOffset(horizontal, 1));
  let seam = abs(luma(b0) - luma(a1));
  let shoulders = 0.5 * (
    abs(luma(a1) - luma(a2)) + abs(luma(b1) - luma(b0))
  );
  let isolatedStep = seam - shoulders * 1.20;
  let blockEvidence = smoothstep(0.003, 0.045, isolatedStep)
    * (1.0 - smoothstep(0.10, 0.24, seam))
    * (1.0 - smoothstep(0.14, 0.32, length(b0 - a1)));

  let center = loadInput(position);
  let n2 = loadInput(position + axisOffset(horizontal, -2));
  let n1 = loadInput(position + axisOffset(horizontal, -1));
  let p1 = loadInput(position + axisOffset(horizontal, 1));
  let p2 = loadInput(position + axisOffset(horizontal, 2));
  let lowPass = (n2 + n1 * 2.0 + center * 2.0 + p1 * 2.0 + p2) * 0.125;
  let boundaryCenter = f32(boundary) - 0.5;
  let distanceToBoundary = abs(f32(coordinate) - boundaryCenter);
  let boundaryInfluence = 1.0 - smoothstep(0.5, 2.5, distanceToBoundary);

  return (lowPass - center) * blockEvidence * boundaryInfluence;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dimensions = textureDimensions(filteredFrame);
  if (any(id.xy >= dimensions)) {
    return;
  }

  let position = vec2<i32>(id.xy);
  let center = loadInput(position);
  let horizontal = boundaryCorrection(position, true);
  let vertical = boundaryCorrection(position, false);
  let modeStrength = select(
    1.25,
    select(1.0, 0.72, uniforms.flags.y > 1.5),
    uniforms.flags.y > 0.5,
  );
  let candidate = center + (horizontal + vertical) * modeStrength;

  let left = loadInput(position + vec2<i32>(-1, 0));
  let right = loadInput(position + vec2<i32>(1, 0));
  let up = loadInput(position + vec2<i32>(0, -1));
  let down = loadInput(position + vec2<i32>(0, 1));
  let minimum = min(center, min(min(left, right), min(up, down)));
  let maximum = max(center, max(max(left, right), max(up, down)));
  let allowance = (maximum - minimum) * 0.02 + vec3<f32>(0.0005);
  let resolved = clamp(candidate, minimum - allowance, maximum + allowance);

  textureStore(
    filteredFrame,
    position,
    vec4<f32>(clamp(resolved, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0),
  );
}
