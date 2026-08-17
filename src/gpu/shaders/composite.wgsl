struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  motionSize: vec4<f32>,
  timeScale: vec4<f32>,
  thresholds: vec4<f32>,
  flags: vec4<f32>,
}

@group(0) @binding(0) var historyAccumulator: texture_2d<f32>;
@group(0) @binding(1) var inputFrame: texture_2d<f32>;
@group(0) @binding(2) var featureCurrent: texture_2d<f32>;
@group(0) @binding(3) var linearClamp: sampler;
@group(0) @binding(4) var<uniform> uniforms: FrameUniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

fn luma(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
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

fn sourceUvForOutput(outputPosition: vec2<f32>) -> vec2<f32> {
  let scale = uniforms.outputSize.xy * uniforms.inputSize.zw;
  let hrCoordinate = outputPosition / scale;
  return (hrCoordinate + vec2<f32>(0.5)) * uniforms.inputSize.zw;
}

fn spatialFallback(outputPosition: vec2<f32>) -> vec3<f32> {
  let sourceUv = sourceUvForOutput(outputPosition);
  let spatialBase = textureSampleLevel(inputFrame, linearClamp, sourceUv, 0.0).rgb;
  let feature = textureSampleLevel(featureCurrent, linearClamp, sourceUv, 0.0);
  let gradient = feature.yz;
  let gradientLength = length(gradient);
  let tangent = vec2<f32>(-gradient.y, gradient.x) / max(gradientLength, 0.00001);
  let directionalRadius = uniforms.inputSize.zw * 0.45;
  let directional = 0.5 * (
    textureSampleLevel(inputFrame, linearClamp, sourceUv + tangent * directionalRadius, 0.0).rgb
    + textureSampleLevel(inputFrame, linearClamp, sourceUv - tangent * directionalRadius, 0.0).rgb
  );
  let edgeStrength = smoothstep(0.02, 0.14, gradientLength)
    * smoothstep(0.006, 0.09, feature.w);
  return mix(spatialBase, directional, edgeStrength * 0.12);
}

fn coverageColor(coverage: f32) -> vec3<f32> {
  let cold = vec3<f32>(0.015, 0.025, 0.11);
  let observed = vec3<f32>(0.0, 0.86, 0.72);
  let full = vec3<f32>(1.0, 0.88, 0.18);
  let middle = mix(cold, observed, smoothstep(0.0, 0.65, coverage));
  return mix(middle, full, smoothstep(0.65, 1.0, coverage));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  // Map each presentation pixel center continuously onto the exact 2x HR
  // lattice. Avoid integer quantization when the canvas is smaller than HR.
  let outputPosition = input.uv * uniforms.outputSize.xy - vec2<f32>(0.5);
  let spatial = spatialFallback(outputPosition);
  let history = textureSample(historyAccumulator, linearClamp, input.uv);
  let coverage = clamp(history.a, 0.0, 1.0);

  if (uniforms.flags.w > 0.5) {
    return vec4<f32>(coverageColor(coverage), 1.0);
  }

  let temporalRadiance = history.rgb / max(coverage, 0.0001);
  let confidenceFromCoverage = smoothstep(0.02, 0.85, coverage);
  let resolved = mix(spatial, temporalRadiance, confidenceFromCoverage);

  // One LR-pixel cross neighborhood. A range-weighted average softens small
  // codec discontinuities while naturally rejecting samples across real
  // object edges. This is display-only; observations in history stay intact.
  let neighborRadius = 2.0;
  let north = spatialFallback(outputPosition - vec2<f32>(0.0, neighborRadius));
  let south = spatialFallback(outputPosition + vec2<f32>(0.0, neighborRadius));
  let west = spatialFallback(outputPosition - vec2<f32>(neighborRadius, 0.0));
  let east = spatialFallback(outputPosition + vec2<f32>(neighborRadius, 0.0));
  let northDelta = resolved - north;
  let southDelta = resolved - south;
  let westDelta = resolved - west;
  let eastDelta = resolved - east;
  let northWeight = exp(-dot(northDelta, northDelta) * 80.0);
  let southWeight = exp(-dot(southDelta, southDelta) * 80.0);
  let westWeight = exp(-dot(westDelta, westDelta) * 80.0);
  let eastWeight = exp(-dot(eastDelta, eastDelta) * 80.0);
  let bilateral = (
    resolved * 1.5
    + north * northWeight + south * southWeight
    + west * westWeight + east * eastWeight
  ) / max(1.5 + northWeight + southWeight + westWeight + eastWeight, 0.0001);

  let crossAverage = (north + south + west + east) * 0.25;
  let artifactMagnitude = abs(luma(resolved) - luma(crossAverage));
  let sourceUv = sourceUvForOutput(outputPosition);
  let sourceFeature = textureSampleLevel(featureCurrent, linearClamp, sourceUv, 0.0);
  let structureProtection = 1.0 - smoothstep(0.025, 0.12, length(sourceFeature.yz));
  let artifactSignal = smoothstep(0.004, 0.045, artifactMagnitude)
    * (1.0 - smoothstep(0.07, 0.18, artifactMagnitude));
  let deblockAmount = uniforms.thresholds.w * structureProtection * artifactSignal;
  let deblocked = mix(resolved, bilateral, deblockAmount);

  let blur = deblocked * 0.5 + (north + south + west + east) * 0.125;
  let detail = deblocked - blur;
  let edgeMagnitude = max(max(abs(detail.r), abs(detail.g)), abs(detail.b));
  let edgeMask = 1.0 - smoothstep(0.08, 0.3, edgeMagnitude);
  let sharpenStrength = uniforms.thresholds.x * edgeMask * (1.0 - artifactSignal * 0.8);
  let sharpened = deblocked + detail * sharpenStrength;
  return vec4<f32>(clamp(sharpened, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
