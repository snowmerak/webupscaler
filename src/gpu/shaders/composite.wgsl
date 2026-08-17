struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  motionSize: vec4<f32>,
  timeScale: vec4<f32>,
  thresholds: vec4<f32>,
  flags: vec4<f32>,
}

@group(0) @binding(0) var latentReconstruction: texture_2d<f32>;
@group(0) @binding(1) var inputFrame: texture_2d<f32>;
@group(0) @binding(2) var featureCurrent: texture_2d<f32>;
@group(0) @binding(3) var linearClamp: sampler;
@group(0) @binding(4) var<uniform> uniforms: FrameUniforms;
@group(0) @binding(5) var observationAccumulator: texture_2d<f32>;
@group(0) @binding(6) var observationMoments: texture_2d<f32>;
@group(0) @binding(7) var residualBefore: texture_2d<f32>;
@group(0) @binding(8) var residualAfter: texture_2d<f32>;
@group(0) @binding(9) var latentSeedTexture: texture_2d<f32>;
@group(0) @binding(10) var motionMetaCurrent: texture_2d<f32>;

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

fn sourceUvForOutput(outputPosition: vec2<f32>) -> vec2<f32> {
  let scale = uniforms.outputSize.xy * uniforms.inputSize.zw;
  let hrCoordinate = outputPosition / scale;
  return (hrCoordinate + vec2<f32>(0.5)) * uniforms.inputSize.zw;
}

fn loadInput(position: vec2<i32>) -> vec3<f32> {
  let maximum = vec2<i32>(uniforms.inputSize.xy) - vec2<i32>(1);
  return textureLoad(inputFrame, clamp(position, vec2<i32>(0), maximum), 0).rgb;
}

fn catmullRomWeights(phase: f32) -> vec4<f32> {
  let phase2 = phase * phase;
  let phase3 = phase2 * phase;
  return vec4<f32>(
    -0.5 * phase + phase2 - 0.5 * phase3,
    1.0 - 2.5 * phase2 + 1.5 * phase3,
    0.5 * phase + 2.0 * phase2 - 1.5 * phase3,
    -0.5 * phase2 + 0.5 * phase3,
  );
}

fn monotonicBicubic(sourceUv: vec2<f32>) -> vec3<f32> {
  let sourcePosition = sourceUv * uniforms.inputSize.xy - vec2<f32>(0.5);
  let base = vec2<i32>(floor(sourcePosition));
  let phase = fract(sourcePosition);
  let weightX = catmullRomWeights(phase.x);
  let weightY = catmullRomWeights(phase.y);
  var result = vec3<f32>(0.0);
  for (var y = 0; y < 4; y += 1) {
    for (var x = 0; x < 4; x += 1) {
      result += loadInput(base + vec2<i32>(x - 1, y - 1))
        * weightX[x] * weightY[y];
    }
  }

  // Catmull-Rom's negative lobes are useful for detail but can ring. Project
  // the result into the actual local 2x2 color envelope for monotonic output.
  let a = loadInput(base);
  let b = loadInput(base + vec2<i32>(1, 0));
  let c = loadInput(base + vec2<i32>(0, 1));
  let d = loadInput(base + vec2<i32>(1, 1));
  let minimum = min(min(a, b), min(c, d));
  let maximum = max(max(a, b), max(c, d));
  return clamp(result, minimum, maximum);
}

fn spatialFallback(outputPosition: vec2<f32>) -> vec3<f32> {
  let sourceUv = sourceUvForOutput(outputPosition);
  let spatialBase = monotonicBicubic(sourceUv);
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
  let directed = mix(spatialBase, directional, edgeStrength * 0.16);
  let sourcePosition = sourceUv * uniforms.inputSize.xy - vec2<f32>(0.5);
  let base = vec2<i32>(floor(sourcePosition));
  let a = loadInput(base);
  let b = loadInput(base + vec2<i32>(1, 0));
  let c = loadInput(base + vec2<i32>(0, 1));
  let d = loadInput(base + vec2<i32>(1, 1));
  return clamp(directed, min(min(a, b), min(c, d)), max(max(a, b), max(c, d)));
}

fn coverageColor(coverage: f32) -> vec3<f32> {
  let cold = vec3<f32>(0.015, 0.025, 0.11);
  let observed = vec3<f32>(0.0, 0.86, 0.72);
  let full = vec3<f32>(1.0, 0.88, 0.18);
  let middle = mix(cold, observed, smoothstep(0.0, 0.65, coverage));
  return mix(middle, full, smoothstep(0.65, 1.0, coverage));
}

fn scalarHeat(value: f32) -> vec3<f32> {
  let cold = vec3<f32>(0.015, 0.02, 0.09);
  let blue = vec3<f32>(0.02, 0.25, 0.95);
  let cyan = vec3<f32>(0.0, 0.9, 0.78);
  let hot = vec3<f32>(1.0, 0.82, 0.08);
  let first = mix(cold, blue, smoothstep(0.0, 0.33, value));
  let second = mix(first, cyan, smoothstep(0.28, 0.68, value));
  return mix(second, hot, smoothstep(0.62, 1.0, value));
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

fn loadLatent(position: vec2<i32>) -> vec4<f32> {
  let maximum = vec2<i32>(uniforms.outputSize.xy) - vec2<i32>(1);
  return textureLoad(
    latentReconstruction,
    clamp(position, vec2<i32>(0), maximum),
    0,
  );
}

fn lanczos2Resolve(uv: vec2<f32>) -> vec3<f32> {
  let sourcePosition = uv * uniforms.outputSize.xy - vec2<f32>(0.5);
  let base = vec2<i32>(floor(sourcePosition));
  let phase = fract(sourcePosition);
  var result = vec3<f32>(0.0);
  var weightSum = 0.0;
  for (var y = -1; y <= 2; y += 1) {
    let weightY = lanczos2Weight(f32(y) - phase.y);
    for (var x = -1; x <= 2; x += 1) {
      let weight = lanczos2Weight(f32(x) - phase.x) * weightY;
      result += loadLatent(base + vec2<i32>(x, y)).rgb * weight;
      weightSum += weight;
    }
  }
  let filtered = result / max(weightSum, 0.0001);

  // Negative Lanczos lobes can overshoot compressed video edges. Keep the
  // sharper resolve inside a relaxed local 2x2 radiance envelope.
  let a = loadLatent(base).rgb;
  let b = loadLatent(base + vec2<i32>(1, 0)).rgb;
  let c = loadLatent(base + vec2<i32>(0, 1)).rgb;
  let d = loadLatent(base + vec2<i32>(1, 1)).rgb;
  let minimum = min(min(a, b), min(c, d));
  let maximum = max(max(a, b), max(c, d));
  let expansion = (maximum - minimum) * 0.08 + vec3<f32>(0.004);
  return clamp(filtered, minimum - expansion, maximum + expansion);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  // Map each presentation pixel center continuously onto the exact 2x HR
  // lattice. Avoid integer quantization when the canvas is smaller than HR.
  let outputPosition = input.uv * uniforms.outputSize.xy - vec2<f32>(0.5);
  let spatial = spatialFallback(outputPosition);
  let presentationSize = vec2<f32>(1.0) / max(
    fwidth(input.uv),
    vec2<f32>(0.000001),
  );
  var latent = textureSample(latentReconstruction, linearClamp, input.uv);
  let reactiveMask = clamp(latent.a, 0.0, 1.0);
  if (all(uniforms.outputSize.xy > presentationSize * 1.02)) {
    let lanczos = lanczos2Resolve(input.uv);
    // Changed overlays and disocclusions use the non-ringing linear resolve.
    // Bilinear mask sampling also gives a small, free guard band at edges.
    latent = vec4<f32>(
      mix(lanczos, latent.rgb, smoothstep(0.08, 0.72, reactiveMask)),
      reactiveMask,
    );
  }
  let accumulator = textureSampleLevel(observationAccumulator, linearClamp, input.uv, 0.0);
  let moments = textureSampleLevel(observationMoments, linearClamp, input.uv, 0.0);
  let coverage = clamp(accumulator.a, 0.0, 1.0);
  let observationMean = accumulator.rgb / max(coverage, 0.0001);
  let variance = max(
    moments.rgb / max(coverage, 0.0001) - observationMean * observationMean,
    vec3<f32>(0.0),
  );
  let varianceLuma = dot(variance, vec3<f32>(0.25, 0.5, 0.25));
  let effectiveSamples = coverage * coverage / max(moments.a, 0.0001);
  let coverageTrust = smoothstep(0.02, 0.85, coverage);
  let varianceTrust = exp(-varianceLuma * 72.0);
  let sampleTrust = clamp((effectiveSamples - 1.0) / 3.0, 0.0, 1.0);
  let reconstructionConfidence = coverageTrust
    * varianceTrust
    * mix(0.55, 1.0, sampleTrust);
  let stableReconstructionConfidence = reconstructionConfidence
    * mix(1.0, 0.15, reactiveMask);

  let diagnosticView = i32(round(uniforms.flags.w));
  if (diagnosticView == 1) {
    return vec4<f32>(coverageColor(coverage), 1.0);
  }
  if (diagnosticView == 2) {
    return vec4<f32>(scalarHeat(clamp(sqrt(varianceLuma) * 7.0, 0.0, 1.0)), 1.0);
  }
  if (diagnosticView == 3) {
    return vec4<f32>(scalarHeat(clamp((effectiveSamples - 1.0) / 5.0, 0.0, 1.0)), 1.0);
  }
  if (diagnosticView == 4) {
    let before = textureSampleLevel(residualBefore, linearClamp, input.uv, 0.0);
    let after = textureSampleLevel(residualAfter, linearClamp, input.uv, 0.0);
    let beforeRms = sqrt(max(before.a, 0.0));
    let afterRms = sqrt(max(after.a, 0.0));
    let improvement = max(beforeRms - afterRms, 0.0);
    return vec4<f32>(
      clamp(vec3<f32>(beforeRms * 7.0, afterRms * 7.0, improvement * 12.0), vec3<f32>(0.0), vec3<f32>(1.0)),
      1.0,
    );
  }
  if (diagnosticView == 5) {
    let seed = textureSampleLevel(latentSeedTexture, linearClamp, input.uv, 0.0).rgb;
    let correction = length(latent.rgb - seed) * 10.0;
    return vec4<f32>(scalarHeat(clamp(correction, 0.0, 1.0)), 1.0);
  }
  if (diagnosticView == 6) {
    let motion = textureSampleLevel(motionMetaCurrent, linearClamp, input.uv, 0.0);
    let direction = clamp(motion.xy / 16.0 * 0.5 + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    return vec4<f32>(
      vec3<f32>(direction.x, direction.y, motion.z) * (1.0 - motion.w * 0.65),
      1.0,
    );
  }
  if (diagnosticView == 7) {
    return vec4<f32>(scalarHeat(reactiveMask), 1.0);
  }

  // RGB is the observation-seeded latent HR estimate after two residual
  // back-projection iterations. Alpha carries the transient reactive mask.
  let temporalRadiance = latent.rgb;
  let resolved = mix(spatial, temporalRadiance, stableReconstructionConfidence);

  // Deblocking now happens on the source before motion analysis and temporal
  // accumulation. Composite only applies a small final-detail recovery.
  let presentationTexel = vec2<f32>(1.0) / presentationSize;
  let north = textureSampleLevel(latentReconstruction, linearClamp, input.uv - vec2<f32>(0.0, presentationTexel.y), 0.0).rgb;
  let south = textureSampleLevel(latentReconstruction, linearClamp, input.uv + vec2<f32>(0.0, presentationTexel.y), 0.0).rgb;
  let west = textureSampleLevel(latentReconstruction, linearClamp, input.uv - vec2<f32>(presentationTexel.x, 0.0), 0.0).rgb;
  let east = textureSampleLevel(latentReconstruction, linearClamp, input.uv + vec2<f32>(presentationTexel.x, 0.0), 0.0).rgb;
  let blur = resolved * 0.5 + (north + south + west + east) * 0.125;
  let detail = resolved - blur;
  let edgeMagnitude = max(max(abs(detail.r), abs(detail.g)), abs(detail.b));
  let edgeMask = 1.0 - smoothstep(0.08, 0.3, edgeMagnitude);
  let sharpened = resolved + detail * uniforms.thresholds.x * edgeMask
    * stableReconstructionConfidence;
  return vec4<f32>(clamp(sharpened, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
