struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  motionSize: vec4<f32>,
  timeScale: vec4<f32>,
  thresholds: vec4<f32>,
  flags: vec4<f32>,
}

@group(0) @binding(0) var detailedFrame: texture_2d<f32>;
@group(0) @binding(1) var shockedFrame: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> uniforms: FrameUniforms;

fn luma(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn loadFrame(position: vec2<i32>) -> vec3<f32> {
  let maximum = vec2<i32>(uniforms.outputSize.xy) - vec2<i32>(1);
  return textureLoad(
    detailedFrame,
    clamp(position, vec2<i32>(0), maximum),
    0,
  ).rgb;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let dimensions = textureDimensions(shockedFrame);
  if (any(id.xy >= dimensions)) {
    return;
  }

  let position = vec2<i32>(id.xy);
  let center = loadFrame(position);
  let northWest = loadFrame(position + vec2<i32>(-1, -1));
  let north = loadFrame(position + vec2<i32>(0, -1));
  let northEast = loadFrame(position + vec2<i32>(1, -1));
  let west = loadFrame(position + vec2<i32>(-1, 0));
  let east = loadFrame(position + vec2<i32>(1, 0));
  let southWest = loadFrame(position + vec2<i32>(-1, 1));
  let south = loadFrame(position + vec2<i32>(0, 1));
  let southEast = loadFrame(position + vec2<i32>(1, 1));
  let centerLuma = luma(center);
  let nw = luma(northWest);
  let n = luma(north);
  let ne = luma(northEast);
  let w = luma(west);
  let e = luma(east);
  let sw = luma(southWest);
  let s = luma(south);
  let se = luma(southEast);

  let gx = ((ne + 2.0 * e + se) - (nw + 2.0 * w + sw)) * 0.125;
  let gy = ((sw + 2.0 * s + se) - (nw + 2.0 * n + ne)) * 0.125;
  let gradientMagnitude = length(vec2<f32>(gx, gy));
  let laplacian = n + s + w + e - 4.0 * centerLuma;
  let minimum = min(
    center,
    min(
      min(min(northWest, north), min(northEast, west)),
      min(min(east, southWest), min(south, southEast)),
    ),
  );
  let maximum = max(
    center,
    max(
      max(max(northWest, north), max(northEast, west)),
      max(max(east, southWest), max(south, southEast)),
    ),
  );
  let minimumLuma = min(
    centerLuma,
    min(min(min(nw, n), min(ne, w)), min(min(e, sw), min(s, se))),
  );
  let maximumLuma = max(
    centerLuma,
    max(max(max(nw, n), max(ne, w)), max(max(e, sw), max(s, se))),
  );
  let localRange = maximumLuma - minimumLuma;

  // PDE-style shock filtering compresses a multi-pixel ramp toward its local
  // extrema. The smooth sign avoids hard binary transitions, and both first-
  // and second-derivative gates keep broad gradients and flat areas untouched.
  let edgeEvidence = smoothstep(0.004, 0.095, gradientMagnitude);
  let curvatureEvidence = smoothstep(0.0015, 0.035, abs(laplacian));
  let textLike = smoothstep(0.070, 0.260, localRange)
    * smoothstep(0.018, 0.110, gradientMagnitude);
  let modeScale = select(
    1.08,
    select(0.82, 0.56, uniforms.flags.y > 1.5),
    uniforms.flags.y > 0.5,
  );
  let shockDelta = clamp(
    -tanh(laplacian * 28.0)
      * gradientMagnitude
      * edgeEvidence
      * curvatureEvidence
      * modeScale
      * (1.0 + 0.45 * textLike),
    -0.046,
    0.046,
  );
  let candidate = center + vec3<f32>(shockDelta);
  let allowance = (maximum - minimum) * 0.025 + vec3<f32>(0.001);
  let resolved = clamp(candidate, minimum - allowance, maximum + allowance);

  textureStore(
    shockedFrame,
    position,
    vec4<f32>(clamp(resolved, vec3<f32>(0.0), vec3<f32>(1.0)), abs(shockDelta)),
  );
}
