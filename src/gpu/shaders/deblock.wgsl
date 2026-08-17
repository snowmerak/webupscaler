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

fn loadInput(position: vec2<i32>) -> vec3<f32> {
  let maximum = vec2<i32>(uniforms.inputSize.xy) - vec2<i32>(1);
  return textureLoad(inputFrame, clamp(position, vec2<i32>(0), maximum), 0).rgb;
}

fn rgbToYCoCg(color: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    dot(color, vec3<f32>(0.25, 0.5, 0.25)),
    color.r * 0.5 - color.b * 0.5,
    color.g * 0.5 - color.r * 0.25 - color.b * 0.25,
  );
}

fn yCoCgToRgb(color: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    color.x + color.y - color.z,
    color.x + color.z,
    color.x - color.y - color.z,
  );
}

fn boundaryMask1d(coordinate: f32, period: f32) -> f32 {
  let phase = coordinate - floor(coordinate / period) * period;
  let distanceToBoundary = min(phase, period - phase);
  return 1.0 - smoothstep(0.5, 2.5, distanceToBoundary);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= u32(uniforms.inputSize.x) || id.y >= u32(uniforms.inputSize.y)) {
    return;
  }

  let position = vec2<i32>(id.xy);
  let centerRgb = loadInput(position);
  let center = rgbToYCoCg(centerRgb);

  let offsets = array<vec2<i32>, 16>(
    vec2<i32>(-1, 0), vec2<i32>(1, 0),
    vec2<i32>(0, -1), vec2<i32>(0, 1),
    vec2<i32>(-2, 0), vec2<i32>(2, 0),
    vec2<i32>(0, -2), vec2<i32>(0, 2),
    vec2<i32>(-4, 0), vec2<i32>(4, 0),
    vec2<i32>(0, -4), vec2<i32>(0, 4),
    vec2<i32>(-2, -2), vec2<i32>(2, -2),
    vec2<i32>(-2, 2), vec2<i32>(2, 2),
  );
  let spatialWeights = array<f32, 16>(
    1.0, 1.0, 1.0, 1.0,
    0.72, 0.72, 0.72, 0.72,
    0.38, 0.38, 0.38, 0.38,
    0.46, 0.46, 0.46, 0.46,
  );

  var lumaSum = center.x * 1.5;
  var lumaWeightSum = 1.5;
  var chromaSum = center.yz * 1.5;
  var chromaWeightSum = 1.5;
  for (var index = 0u; index < 16u; index += 1u) {
    let sampleColor = rgbToYCoCg(loadInput(position + offsets[index]));
    let lumaDelta = sampleColor.x - center.x;
    let chromaDelta = sampleColor.yz - center.yz;
    let spatialWeight = spatialWeights[index];
    let lumaWeight = spatialWeight * exp(-lumaDelta * lumaDelta * 360.0);
    let chromaWeight = spatialWeight * exp(
      -lumaDelta * lumaDelta * 420.0 - dot(chromaDelta, chromaDelta) * 24.0
    );
    lumaSum += sampleColor.x * lumaWeight;
    lumaWeightSum += lumaWeight;
    chromaSum += sampleColor.yz * chromaWeight;
    chromaWeightSum += chromaWeight;
  }

  let filtered = vec3<f32>(
    lumaSum / max(lumaWeightSum, 0.0001),
    chromaSum / max(chromaWeightSum, 0.0001),
  );
  let left = rgbToYCoCg(loadInput(position + vec2<i32>(-1, 0))).x;
  let right = rgbToYCoCg(loadInput(position + vec2<i32>(1, 0))).x;
  let up = rgbToYCoCg(loadInput(position + vec2<i32>(0, -1))).x;
  let down = rgbToYCoCg(loadInput(position + vec2<i32>(0, 1))).x;
  let localGradient = max(
    max(abs(center.x - left), abs(center.x - right)),
    max(abs(center.x - up), abs(center.x - down)),
  );

  let pixelCenter = vec2<f32>(id.xy) + vec2<f32>(0.5);
  let boundary4 = max(
    boundaryMask1d(pixelCenter.x, 4.0),
    boundaryMask1d(pixelCenter.y, 4.0),
  );
  let boundary8 = max(
    boundaryMask1d(pixelCenter.x, 8.0),
    boundaryMask1d(pixelCenter.y, 8.0),
  );
  let boundary16 = max(
    boundaryMask1d(pixelCenter.x, 16.0),
    boundaryMask1d(pixelCenter.y, 16.0),
  );
  let blockBoundary = max(max(boundary8, boundary16), boundary4 * 0.45);
  let structureProtection = 1.0 - smoothstep(0.035, 0.18, localGradient);
  let filterDifference = abs(filtered.x - center.x)
    + length(filtered.yz - center.yz) * 0.7;
  let artifactActivity = smoothstep(0.002, 0.04, filterDifference);
  let strength = clamp(uniforms.thresholds.w * 2.0, 0.0, 1.0);
  let lumaAmount = strength
    * structureProtection
    * mix(0.45, 1.0, blockBoundary)
    * mix(0.55, 1.0, artifactActivity);
  let chromaAmount = min(
    1.0,
    lumaAmount * 1.3 + strength * structureProtection * 0.08,
  );

  let result = vec3<f32>(
    mix(center.x, filtered.x, lumaAmount),
    mix(center.yz, filtered.yz, chromaAmount),
  );
  textureStore(
    filteredFrame,
    position,
    vec4<f32>(clamp(yCoCgToRgb(result), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0),
  );
}
