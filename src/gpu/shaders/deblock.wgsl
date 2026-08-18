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
  let horizontalCorrection = axisCorrection(position, true);
  let verticalCorrection = axisCorrection(position, false);
  let modeStrength = select(
    0.90,
    select(0.70, 0.50, uniforms.flags.y > 1.5),
    uniforms.flags.y > 0.5,
  );
  let deblocked = center
    + (horizontalCorrection + verticalCorrection) * modeStrength;

  let left = loadInput(position + vec2<i32>(-1, 0));
  let right = loadInput(position + vec2<i32>(1, 0));
  let up = loadInput(position + vec2<i32>(0, -1));
  let down = loadInput(position + vec2<i32>(0, 1));
  let centerLuma = luma(center);
  let leftLuma = luma(left);
  let rightLuma = luma(right);
  let upLuma = luma(up);
  let downLuma = luma(down);
  let leftWeight = exp(-abs(leftLuma - centerLuma) * 30.0);
  let rightWeight = exp(-abs(rightLuma - centerLuma) * 30.0);
  let upWeight = exp(-abs(upLuma - centerLuma) * 30.0);
  let downWeight = exp(-abs(downLuma - centerLuma) * 30.0);
  let horizontalFiltered = (
    center * 2.0 + left * leftWeight + right * rightWeight
  ) / max(2.0 + leftWeight + rightWeight, 0.0001);
  let verticalFiltered = (
    center * 2.0 + up * upWeight + down * downWeight
  ) / max(2.0 + upWeight + downWeight, 0.0001);
  let horizontalCurvature = abs(leftLuma + rightLuma - 2.0 * centerLuma);
  let verticalCurvature = abs(upLuma + downLuma - 2.0 * centerLuma);
  let useHorizontal = horizontalCurvature <= verticalCurvature;
  let directionalFiltered = select(
    verticalFiltered,
    horizontalFiltered,
    useHorizontal,
  );
  let minimumLuma = min(
    centerLuma,
    min(min(leftLuma, rightLuma), min(upLuma, downLuma)),
  );
  let maximumLuma = max(
    centerLuma,
    max(max(leftLuma, rightLuma), max(upLuma, downLuma)),
  );
  let localRange = maximumLuma - minimumLuma;
  let selectedCurvature = select(
    verticalCurvature,
    horizontalCurvature,
    useHorizontal,
  );
  let gradientMagnitude = select(
    abs(downLuma - upLuma),
    abs(rightLuma - leftLuma),
    useHorizontal,
  ) * 0.5;
  let residual = abs(centerLuma - luma(directionalFiltered));
  let flatMask = 1.0 - smoothstep(0.028, 0.120, localRange);
  let gradientContinuity = 1.0 - smoothstep(0.008, 0.042, selectedCurvature);
  let gradientEvidence = smoothstep(0.0015, 0.025, gradientMagnitude);
  let noiseEvidence = smoothstep(0.003, 0.026, residual);
  let flattenScale = select(
    1.0,
    select(0.76, 0.54, uniforms.flags.y > 1.5),
    uniforms.flags.y > 0.5,
  );
  // Flatten only the locally smooth, lower-curvature direction before motion
  // analysis and temporal reconstruction. This keeps true edges intact while
  // preventing faint quantization bands from being accumulated as detail.
  let flattening = clamp(
    flatMask * gradientContinuity
      * (0.16 + 0.30 * gradientEvidence + 0.16 * noiseEvidence)
      * flattenScale,
    0.0,
    0.52,
  );
  let candidate = deblocked
    + (directionalFiltered - center) * flattening;
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
