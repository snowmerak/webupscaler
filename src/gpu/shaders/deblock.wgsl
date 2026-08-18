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

// A block seam is an isolated moderate step between two locally flat
// plateaus. There is deliberately no fixed 4/8-pixel phase here: modern AV1
// transform partitions are variable, and a hard grid creates visible bands
// on clean gradients.
fn seamEvidence(
  leftOuter: vec3<f32>,
  left: vec3<f32>,
  right: vec3<f32>,
  rightOuter: vec3<f32>,
) -> f32 {
  let seam = abs(luma(right) - luma(left));
  let shoulder = max(
    abs(luma(left) - luma(leftOuter)),
    abs(luma(rightOuter) - luma(right)),
  );
  let isolatedStep = seam - shoulder * 1.75;
  let plateauTrust = 1.0 - smoothstep(0.025, 0.075, shoulder);
  let edgeProtection = 1.0 - smoothstep(0.095, 0.220, seam);
  let chromaProtection = 1.0 - smoothstep(0.10, 0.28, length(right - left));
  return smoothstep(0.012, 0.060, isolatedStep)
    * plateauTrust
    * edgeProtection
    * chromaProtection;
}

fn axisCorrection(position: vec2<i32>, horizontal: bool) -> vec3<f32> {
  let axis = axisOffset(horizontal, 1);
  let center = loadInput(position);
  let negativeOne = loadInput(position - axis);
  let negativeTwo = loadInput(position - axis * 2);
  let positiveOne = loadInput(position + axis);
  let positiveTwo = loadInput(position + axis * 2);

  let leftSeam = seamEvidence(negativeTwo, negativeOne, center, positiveOne);
  let rightSeam = seamEvidence(negativeOne, center, positiveOne, positiveTwo);
  let leftDelta = (center - negativeOne) * 0.5;
  let rightDelta = (positiveOne - center) * 0.5;
  return rightDelta * rightSeam - leftDelta * leftSeam;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dimensions = textureDimensions(filteredFrame);
  if (any(id.xy >= dimensions)) {
    return;
  }

  let position = vec2<i32>(id.xy);
  let center = loadInput(position);
  let horizontal = axisCorrection(position, true);
  let vertical = axisCorrection(position, false);
  let modeStrength = select(
    0.90,
    select(0.70, 0.50, uniforms.flags.y > 1.5),
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
