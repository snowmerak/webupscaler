struct FrameUniforms {
  inputSize: vec4<f32>,
  outputSize: vec4<f32>,
  analysisSize: vec4<f32>,
  motionSize: vec4<f32>,
  timeScale: vec4<f32>,
  thresholds: vec4<f32>,
  flags: vec4<f32>,
}

override correctionGain: f32 = 0.24;

@group(0) @binding(0) var latentInput: texture_2d<f32>;
@group(0) @binding(1) var lrResidual: texture_2d<f32>;
@group(0) @binding(2) var latentOutput: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var linearClamp: sampler;
@group(0) @binding(4) var<uniform> uniforms: FrameUniforms;

fn rgbToYCoCg(color: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    color.r * 0.25 + color.g * 0.5 + color.b * 0.25,
    color.r * 0.5 - color.b * 0.5,
    -color.r * 0.25 + color.g * 0.5 - color.b * 0.25,
  );
}

fn yCoCgToRgb(color: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    color.x + color.y - color.z,
    color.x + color.z,
    color.x - color.y - color.z,
  );
}

fn loadLatent(position: vec2<i32>) -> vec4<f32> {
  let maximum = vec2<i32>(uniforms.outputSize.xy) - vec2<i32>(1);
  return textureLoad(latentInput, clamp(position, vec2<i32>(0), maximum), 0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let outputDimensions = vec2<u32>(uniforms.outputSize.xy);
  if (any(id.xy >= outputDimensions)) {
    return;
  }

  let position = vec2<i32>(id.xy);
  let latent = loadLatent(position);
  let scale = uniforms.outputSize.xy * uniforms.inputSize.zw;
  let sourceCoordinate = vec2<f32>(id.xy) / scale;
  let sourceUv = (sourceCoordinate + vec2<f32>(0.5)) * uniforms.inputSize.zw;
  let residualTexel = uniforms.inputSize.zw;

  // Approximate the transpose of the LR pixel-response filter. The residual
  // is distributed over the HR footprint with a compact, normalized kernel.
  let centerResidual = textureSampleLevel(lrResidual, linearClamp, sourceUv, 0.0);
  let gatheredResidual = centerResidual * 0.625
    + textureSampleLevel(
      lrResidual, linearClamp, sourceUv - vec2<f32>(residualTexel.x, 0.0), 0.0
    ) * 0.09375
    + textureSampleLevel(
      lrResidual, linearClamp, sourceUv + vec2<f32>(residualTexel.x, 0.0), 0.0
    ) * 0.09375
    + textureSampleLevel(
      lrResidual, linearClamp, sourceUv - vec2<f32>(0.0, residualTexel.y), 0.0
    ) * 0.09375
    + textureSampleLevel(
      lrResidual, linearClamp, sourceUv + vec2<f32>(0.0, residualTexel.y), 0.0
    ) * 0.09375;

  let residualRms = sqrt(max(gatheredResidual.a, 0.0));
  let outlierProtection = 1.0 - smoothstep(0.12, 0.32, residualRms);
  let observationSupport = mix(0.42, 1.0, clamp(latent.a, 0.0, 1.0));
  let correctionRgb = clamp(
    gatheredResidual.rgb,
    vec3<f32>(-0.12),
    vec3<f32>(0.12),
  ) * correctionGain * outlierProtection * observationSupport;
  let corrected = rgbToYCoCg(latent.rgb + correctionRgb);

  // Projection onto a relaxed local color range prevents iterative ringing
  // while still allowing the residual to restore subpixel contrast.
  var minimum = vec3<f32>(1000.0);
  var maximum = vec3<f32>(-1000.0);
  for (var y = -1; y <= 1; y += 1) {
    for (var x = -1; x <= 1; x += 1) {
      let neighbor = rgbToYCoCg(loadLatent(position + vec2<i32>(x, y)).rgb);
      minimum = min(minimum, neighbor);
      maximum = max(maximum, neighbor);
    }
  }
  let expansion = vec3<f32>(0.035, 0.045, 0.045);
  let regularized = clamp(corrected, minimum - expansion, maximum + expansion);
  textureStore(
    latentOutput,
    position,
    vec4<f32>(
      clamp(yCoCgToRgb(regularized), vec3<f32>(0.0), vec3<f32>(1.0)),
      latent.a,
    ),
  );
}
