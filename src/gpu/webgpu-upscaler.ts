import type { OutputTarget } from '../runtime/output-policy'
import type { BaseUpscalerSettings } from '../shared/settings'
import analyzeShader from './shaders/analyze.wgsl?raw'
import clarityShader from './shaders/clarity.wgsl?raw'
import compositeShader from './shaders/composite.wgsl?raw'
import deblockShader from './shaders/deblock.wgsl?raw'
import enhanceShader from './shaders/enhance.wgsl?raw'
import microContrastShader from './shaders/micro-contrast.wgsl?raw'
import motionShader from './shaders/motion.wgsl?raw'
import reconstructShader from './shaders/reconstruct.wgsl?raw'
import refineShader from './shaders/refine.wgsl?raw'
import shockShader from './shaders/shock.wgsl?raw'

type Pair<T> = [T, T]

export interface FrameTiming {
  mediaTime: number
  skippedFrames: number
}

export interface ProcessResult {
  gpuQueueMs: number | null
  historyReset: boolean
  pendingSubmissions: number
}

interface GpuResources {
  input: Pair<GPUTexture>
  preprocessed: Pair<GPUTexture>
  features: Pair<GPUTexture>
  motionStates: Pair<GPUTexture>
  motionMeta: Pair<GPUTexture>
  reconstruction: Pair<GPUTexture>
  refined: Pair<GPUTexture>
  microDetailed: Pair<GPUTexture>
  shocked: Pair<GPUTexture>
  clarified: Pair<GPUTexture>
  enhanced: Pair<GPUTexture>
  deblockBindGroups: Pair<GPUBindGroup>
  analyzeBindGroups: Pair<GPUBindGroup>
  motionBindGroups: Pair<GPUBindGroup>
  reconstructBindGroups: Pair<GPUBindGroup>
  refineBindGroups: Pair<GPUBindGroup>
  microContrastBindGroups: Pair<GPUBindGroup>
  shockBindGroups: Pair<GPUBindGroup>
  clarityBindGroups: Pair<GPUBindGroup>
  enhanceBindGroups: Pair<GPUBindGroup>
  compositeBindGroups: Pair<GPUBindGroup>
  inputWidth: number
  inputHeight: number
  outputWidth: number
  outputHeight: number
  analysisWidth: number
  analysisHeight: number
  motionWidth: number
  motionHeight: number
}

export interface WebGpuUpscalerOptions {
  allowFallbackAdapter?: boolean
}

export class WebGpuUnavailableError extends Error {
  constructor(message = '이 환경에서는 WebGPU를 사용할 수 없습니다.') {
    super(message)
    this.name = 'WebGpuUnavailableError'
  }
}

export class ShaderCompilationError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ShaderCompilationError'
  }
}

export class WebGpuUpscaler {
  static readonly MAX_PENDING_SUBMISSIONS = 8
  private static readonly GPU_SAMPLE_BATCH_INTERVAL = 8
  private adapter: GPUAdapter | null = null
  private device: GPUDevice | null = null
  private context: GPUCanvasContext | null = null
  private canvasFormat: GPUTextureFormat | null = null
  private sampler: GPUSampler | null = null
  private uniformBuffer: GPUBuffer | null = null
  private deblockPipeline: GPUComputePipeline | null = null
  private analyzePipeline: GPUComputePipeline | null = null
  private motionPipeline: GPUComputePipeline | null = null
  private reconstructPipeline: GPUComputePipeline | null = null
  private refinePipeline: GPUComputePipeline | null = null
  private microContrastPipeline: GPUComputePipeline | null = null
  private shockPipeline: GPUComputePipeline | null = null
  private clarityPipeline: GPUComputePipeline | null = null
  private enhancePipeline: GPUComputePipeline | null = null
  private compositePipeline: GPURenderPipeline | null = null
  private resources: GpuResources | null = null
  private disposed = false
  private initialization: Promise<void> | null = null
  private readonly uniformData = new Float32Array(32)
  private frameIndex = 0
  private readonly preprocessedReady: Pair<boolean> = [false, false]
  private bufferedFrameTiming: FrameTiming | null = null
  private lastMediaTime: number | null = null
  private previousDt = 1 / 60
  private resetNextFrame = true
  private pendingSubmissions = 0
  private submissionBatchIndex = 0
  private batchFence: Promise<void> | null = null
  private completedGpuSampleMs: number | null = null

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly onDeviceLost: (message: string) => void,
    private readonly options: WebGpuUpscalerOptions = {},
  ) {}

  async initialize() {
    if (this.device) return
    if (this.initialization) return this.initialization
    this.initialization = this.initializeInternal()
    return this.initialization
  }

  private async initializeInternal() {
    if (!navigator.gpu) throw new WebGpuUnavailableError()

    this.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
    if (!this.adapter && this.options.allowFallbackAdapter) {
      this.adapter = await navigator.gpu.requestAdapter({ forceFallbackAdapter: true })
    }
    if (!this.adapter) throw new WebGpuUnavailableError('WebGPU 어댑터를 찾지 못했습니다.')

    this.device = await this.adapter.requestDevice()
    this.device.lost.then((info) => {
      if (!this.disposed && info.reason !== 'destroyed') {
        this.onDeviceLost(info.message || 'GPU 장치 연결이 끊어졌습니다.')
      }
    }).catch(() => undefined)

    this.context = this.canvas.getContext('webgpu')
    if (!this.context) throw new WebGpuUnavailableError('WebGPU canvas를 만들지 못했습니다.')

    this.canvasFormat = navigator.gpu.getPreferredCanvasFormat()
    this.context.configure({
      device: this.device,
      format: this.canvasFormat,
      alphaMode: 'opaque',
    })
    this.sampler = this.device.createSampler({
      label: 'Web Upscaler linear clamp sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    })
    this.uniformBuffer = this.device.createBuffer({
      label: 'Web Upscaler frame uniforms',
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    const deblockModule = this.device.createShaderModule({ label: 'Adaptive deblock shader', code: deblockShader })
    const analyzeModule = this.device.createShaderModule({ label: 'Analyze shader', code: analyzeShader })
    const motionModule = this.device.createShaderModule({ label: 'Motion shader', code: motionShader })
    const reconstructModule = this.device.createShaderModule({ label: 'Temporal reconstruct shader', code: reconstructShader })
    const refineModule = this.device.createShaderModule({ label: 'Flat-region refinement shader', code: refineShader })
    const microContrastModule = this.device.createShaderModule({ label: 'Micro-contrast shader', code: microContrastShader })
    const shockModule = this.device.createShaderModule({ label: 'Shock edge-compression shader', code: shockShader })
    const clarityModule = this.device.createShaderModule({ label: 'Wide-radius clarity shader', code: clarityShader })
    const enhanceModule = this.device.createShaderModule({ label: 'Contrast and sharpening shader', code: enhanceShader })
    const compositeModule = this.device.createShaderModule({ label: 'Composite shader', code: compositeShader })
    await Promise.all([
      this.assertShader(deblockModule, 'Adaptive deblock'),
      this.assertShader(analyzeModule, 'Analyze'),
      this.assertShader(motionModule, 'Motion'),
      this.assertShader(reconstructModule, 'Temporal reconstruct'),
      this.assertShader(refineModule, 'Flat-region refinement'),
      this.assertShader(microContrastModule, 'Micro-contrast'),
      this.assertShader(shockModule, 'Shock edge compression'),
      this.assertShader(clarityModule, 'Wide-radius clarity'),
      this.assertShader(enhanceModule, 'Contrast and sharpening'),
      this.assertShader(compositeModule, 'Composite'),
    ])

    this.deblockPipeline = await this.device.createComputePipelineAsync({
      label: 'Adaptive deblock preprocessing pipeline',
      layout: 'auto',
      compute: { module: deblockModule, entryPoint: 'main' },
    })
    this.analyzePipeline = await this.device.createComputePipelineAsync({
      label: 'Analyze pipeline',
      layout: 'auto',
      compute: { module: analyzeModule, entryPoint: 'main' },
    })
    this.motionPipeline = await this.device.createComputePipelineAsync({
      label: 'Acceleration-predicted motion pipeline',
      layout: 'auto',
      compute: { module: motionModule, entryPoint: 'main' },
    })
    this.reconstructPipeline = await this.device.createComputePipelineAsync({
      label: '2x temporal reconstruction pipeline',
      layout: 'auto',
      compute: { module: reconstructModule, entryPoint: 'main' },
    })
    this.refinePipeline = await this.device.createComputePipelineAsync({
      label: 'Flat-region refinement pipeline',
      layout: 'auto',
      compute: { module: refineModule, entryPoint: 'main' },
    })
    this.microContrastPipeline = await this.device.createComputePipelineAsync({
      label: 'Aggressive micro-contrast pipeline',
      layout: 'auto',
      compute: { module: microContrastModule, entryPoint: 'main' },
    })
    this.shockPipeline = await this.device.createComputePipelineAsync({
      label: 'Shock edge-compression pipeline',
      layout: 'auto',
      compute: { module: shockModule, entryPoint: 'main' },
    })
    this.clarityPipeline = await this.device.createComputePipelineAsync({
      label: 'Wide-radius clarity pipeline',
      layout: 'auto',
      compute: { module: clarityModule, entryPoint: 'main' },
    })
    this.enhancePipeline = await this.device.createComputePipelineAsync({
      label: 'Local contrast and sharpening pipeline',
      layout: 'auto',
      compute: { module: enhanceModule, entryPoint: 'main' },
    })
    this.compositePipeline = await this.device.createRenderPipelineAsync({
      label: 'Composite pipeline',
      layout: 'auto',
      vertex: { module: compositeModule, entryPoint: 'vertexMain' },
      fragment: {
        module: compositeModule,
        entryPoint: 'fragmentMain',
        targets: [{ format: this.canvasFormat }],
      },
      primitive: { topology: 'triangle-list' },
    })
  }

  private async assertShader(module: GPUShaderModule, label: string) {
    const info = await module.getCompilationInfo()
    const errors = info.messages.filter((message) => message.type === 'error')
    if (errors.length > 0) {
      throw new ShaderCompilationError(
        `${label}: ${errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join('\n')}`,
      )
    }
  }

  private pair<T>(factory: (index: number) => T): Pair<T> {
    return [factory(0), factory(1)]
  }

  private ensureResources(video: HTMLVideoElement, target: OutputTarget) {
    const device = this.requireDevice()
    const analysisWidth = Math.ceil(video.videoWidth / 4)
    const analysisHeight = Math.ceil(video.videoHeight / 4)
    const motionWidth = Math.ceil(video.videoWidth / 16)
    const motionHeight = Math.ceil(video.videoHeight / 16)
    const existing = this.resources

    if (Math.max(
      target.width,
      target.height,
      target.presentationWidth,
      target.presentationHeight,
    ) > device.limits.maxTextureDimension2D) {
      throw new WebGpuUnavailableError('요청한 출력 크기가 GPU texture 제한을 넘었습니다.')
    }

    // The temporal lattice stays at exact 2x. The swap-chain follows smaller
    // players but is capped at that lattice for oversized/HiDPI presentation;
    // CSS compositing performs the final display stretch without extra shader
    // invocations. Resizing only the canvas preserves accumulated HR history.
    if (this.canvas.width !== target.presentationWidth) {
      this.canvas.width = target.presentationWidth
    }
    if (this.canvas.height !== target.presentationHeight) {
      this.canvas.height = target.presentationHeight
    }

    if (existing
      && existing.inputWidth === video.videoWidth
      && existing.inputHeight === video.videoHeight
      && existing.outputWidth === target.width
      && existing.outputHeight === target.height) {
      return existing
    }

    this.destroyResources()
    this.resetTemporalState()
    const input = this.pair((index) => device.createTexture({
      label: `Video frame ${index}`,
      size: [video.videoWidth, video.videoHeight],
      format: 'rgba8unorm',
      // copyExternalImageToTexture requires both COPY_DST and
      // RENDER_ATTACHMENT on Chromium/Dawn destinations.
      usage: GPUTextureUsage.COPY_DST
        | GPUTextureUsage.RENDER_ATTACHMENT
        | GPUTextureUsage.TEXTURE_BINDING,
    }))
    const features = this.pair((index) => device.createTexture({
      label: `Quarter-resolution features ${index}`,
      size: [analysisWidth, analysisHeight],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }))
    const motionStates = this.pair((index) => device.createTexture({
      label: `Motion velocity and acceleration ${index}`,
      size: [motionWidth, motionHeight],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }))
    const motionMeta = this.pair((index) => device.createTexture({
      label: `Motion displacement and confidence ${index}`,
      size: [motionWidth, motionHeight],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }))
    const reconstruction = this.pair((index) => device.createTexture({
      label: `Resolved 2x observation lattice ${index}`,
      size: [target.width, target.height],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }))
    const refined = this.pair((index) => device.createTexture({
      label: `Flat-region refined lattice ${index}`,
      size: [target.width, target.height],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }))
    const microDetailed = this.pair((index) => device.createTexture({
      label: `Micro-contrast detailed lattice ${index}`,
      size: [target.width, target.height],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }))
    const shocked = this.pair((index) => device.createTexture({
      label: `Shock-compressed edge lattice ${index}`,
      size: [target.width, target.height],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }))
    const clarified = this.pair((index) => device.createTexture({
      label: `Wide-radius clarified lattice ${index}`,
      size: [target.width, target.height],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }))
    const enhanced = this.pair((index) => device.createTexture({
      label: `Contrast and sharpened lattice ${index}`,
      size: [target.width, target.height],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }))
    const preprocessed = this.pair((index) => device.createTexture({
      label: `Deblocked video frame ${index}`,
      size: [video.videoWidth, video.videoHeight],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }))
    const sampler = this.requireSampler()
    const uniformBuffer = this.requireUniformBuffer()
    const deblockPipeline = this.requireDeblockPipeline()
    const analyzePipeline = this.requireAnalyzePipeline()
    const motionPipeline = this.requireMotionPipeline()
    const reconstructPipeline = this.requireReconstructPipeline()
    const refinePipeline = this.requireRefinePipeline()
    const microContrastPipeline = this.requireMicroContrastPipeline()
    const shockPipeline = this.requireShockPipeline()
    const clarityPipeline = this.requireClarityPipeline()
    const enhancePipeline = this.requireEnhancePipeline()
    const compositePipeline = this.requireCompositePipeline()
    const deblockBindGroups = this.pair((current) => device.createBindGroup({
      label: `Adaptive deblock bind group ${current}`,
      layout: deblockPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: input[current].createView() },
        { binding: 1, resource: preprocessed[current].createView() },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    }))
    const analyzeBindGroups = this.pair((current) => device.createBindGroup({
      label: `Analyze bind group ${current}`,
      layout: analyzePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: preprocessed[current].createView() },
        { binding: 1, resource: sampler },
        { binding: 2, resource: features[current].createView() },
        { binding: 3, resource: { buffer: uniformBuffer } },
      ],
    }))
    const motionBindGroups = this.pair((current) => {
      const previous = 1 - current
      return device.createBindGroup({
        label: `Motion bind group ${current}`,
        layout: motionPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: features[current].createView() },
          { binding: 1, resource: features[previous].createView() },
          { binding: 2, resource: preprocessed[current].createView() },
          { binding: 3, resource: preprocessed[previous].createView() },
          { binding: 4, resource: motionStates[previous].createView() },
          { binding: 5, resource: motionMeta[previous].createView() },
          { binding: 6, resource: motionStates[current].createView() },
          { binding: 7, resource: motionMeta[current].createView() },
          { binding: 8, resource: sampler },
          { binding: 9, resource: { buffer: uniformBuffer } },
        ],
      })
    })
    const reconstructBindGroups = this.pair((current) => {
      const future = 1 - current
      return device.createBindGroup({
        label: `Temporal reconstruct bind group ${current}`,
        layout: reconstructPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: preprocessed[current].createView() },
          { binding: 1, resource: motionMeta[current].createView() },
          { binding: 2, resource: sampler },
          { binding: 3, resource: { buffer: uniformBuffer } },
          { binding: 4, resource: reconstruction[current].createView() },
          { binding: 5, resource: preprocessed[future].createView() },
        ],
      })
    })
    const refineBindGroups = this.pair((current) => device.createBindGroup({
      label: `Flat-region refinement bind group ${current}`,
      layout: refinePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: reconstruction[current].createView() },
        { binding: 1, resource: refined[current].createView() },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    }))
    const microContrastBindGroups = this.pair((current) => device.createBindGroup({
      label: `Micro-contrast bind group ${current}`,
      layout: microContrastPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: refined[current].createView() },
        { binding: 1, resource: microDetailed[current].createView() },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    }))
    const shockBindGroups = this.pair((current) => device.createBindGroup({
      label: `Shock edge-compression bind group ${current}`,
      layout: shockPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: microDetailed[current].createView() },
        { binding: 1, resource: shocked[current].createView() },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    }))
    const clarityBindGroups = this.pair((current) => device.createBindGroup({
      label: `Wide-radius clarity bind group ${current}`,
      layout: clarityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: shocked[current].createView() },
        { binding: 1, resource: clarified[current].createView() },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    }))
    const enhanceBindGroups = this.pair((current) => device.createBindGroup({
      label: `Contrast and sharpening bind group ${current}`,
      layout: enhancePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: clarified[current].createView() },
        { binding: 1, resource: enhanced[current].createView() },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    }))
    const compositeBindGroups = this.pair((current) => device.createBindGroup({
      label: `Composite bind group ${current}`,
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: enhanced[current].createView() },
        { binding: 1, resource: { buffer: uniformBuffer } },
      ],
    }))

    this.resources = {
      input,
      preprocessed,
      features,
      motionStates,
      motionMeta,
      reconstruction,
      refined,
      microDetailed,
      shocked,
      clarified,
      enhanced,
      deblockBindGroups,
      analyzeBindGroups,
      motionBindGroups,
      reconstructBindGroups,
      refineBindGroups,
      microContrastBindGroups,
      shockBindGroups,
      clarityBindGroups,
      enhanceBindGroups,
      compositeBindGroups,
      inputWidth: video.videoWidth,
      inputHeight: video.videoHeight,
      outputWidth: target.width,
      outputHeight: target.height,
      analysisWidth,
      analysisHeight,
      motionWidth,
      motionHeight,
    }
    return this.resources
  }

  private getTiming(mediaTime: number) {
    const difference = this.lastMediaTime === null ? this.previousDt : mediaTime - this.lastMediaTime
    const discontinuity = difference <= 0 || difference > Math.max(0.25, this.previousDt * 4)
    const reset = this.resetNextFrame || this.lastMediaTime === null || discontinuity
    const dt = reset ? this.previousDt : Math.min(0.2, Math.max(1 / 240, difference))
    return { dt, reset }
  }

  private writeUniforms(
    video: HTMLVideoElement,
    target: OutputTarget,
    settings: BaseUpscalerSettings,
    resources: GpuResources,
    timing: { dt: number; reset: boolean },
    skippedFrames: number,
  ) {
    const data = this.uniformData
    data.fill(0)
    data[0] = video.videoWidth
    data[1] = video.videoHeight
    data[2] = 1 / video.videoWidth
    data[3] = 1 / video.videoHeight
    data[4] = target.width
    data[5] = target.height
    data[6] = 1 / target.width
    data[7] = 1 / target.height
    data[8] = resources.analysisWidth
    data[9] = resources.analysisHeight
    data[10] = 1 / resources.analysisWidth
    data[11] = 1 / resources.analysisHeight
    data[12] = resources.motionWidth
    data[13] = resources.motionHeight
    data[14] = 1 / resources.motionWidth
    data[15] = 1 / resources.motionHeight
    data[16] = timing.dt
    data[17] = this.previousDt
    data[18] = target.scale
    data[19] = settings.mode === 'eco' ? 0.6 : 0.75
    data[20] = 0
    data[21] = 2400
    const historyRetention = settings.mode === 'eco' ? 0.92 : settings.mode === 'balanced' ? 0.955 : 0.965
    data[22] = Math.pow(historyRetention, skippedFrames + 1)
    data[23] = 0
    data[24] = timing.reset ? 1 : 0
    data[25] = settings.mode === 'auto' ? 0 : settings.mode === 'balanced' ? 1 : 2
    data[26] = this.frameIndex
    data[27] = 0
    this.requireDevice().queue.writeBuffer(this.requireUniformBuffer(), 0, data)
  }

  private closeSubmissionBatch(device: GPUDevice) {
    const shouldSample = this.submissionBatchIndex
      % WebGpuUpscaler.GPU_SAMPLE_BATCH_INTERVAL === 0
    const submittedAt = shouldSample ? performance.now() : 0
    this.submissionBatchIndex += 1
    this.batchFence = device.queue.onSubmittedWorkDone()
      .then(() => {
        if (shouldSample && !this.disposed) {
          this.completedGpuSampleMs = performance.now() - submittedAt
        }
      })
      .catch(() => undefined)
      .finally(() => {
        this.pendingSubmissions = 0
        this.batchFence = null
      })
  }

  private async drainPendingSubmissions(device: GPUDevice) {
    if (this.pendingSubmissions === 0) return
    if (this.batchFence) {
      await this.batchFence
      return
    }
    await device.queue.onSubmittedWorkDone()
    this.pendingSubmissions = 0
  }

  async process(
    video: HTMLVideoElement,
    target: OutputTarget,
    settings: BaseUpscalerSettings,
    frameTiming: FrameTiming,
  ): Promise<ProcessResult | null | undefined> {
    await this.initialize()
    if (this.pendingSubmissions >= WebGpuUpscaler.MAX_PENDING_SUBMISSIONS) {
      // Never pair an old center with a non-adjacent future frame after GPU
      // backpressure. The next accepted callback starts a fresh adjacent pair.
      this.bufferedFrameTiming = null
      this.resetNextFrame = true
      return null
    }

    const device = this.requireDevice()
    const context = this.requireContext()
    const existing = this.resources
    const resourcesNeedRebuild = existing !== null && (
      existing.inputWidth !== video.videoWidth
      || existing.inputHeight !== video.videoHeight
      || existing.outputWidth !== target.width
      || existing.outputHeight !== target.height
    )
    const presentationNeedsResize = this.canvas.width !== target.presentationWidth
      || this.canvas.height !== target.presentationHeight
    if ((resourcesNeedRebuild || presentationNeedsResize) && this.pendingSubmissions > 0) {
      await this.drainPendingSubmissions(device)
    }

    const resources = this.ensureResources(video, target)
    if (this.bufferedFrameTiming !== null && frameTiming.skippedFrames > 0) {
      // requestVideoFrameCallback can advance while a submission is in flight.
      // Drop the stale center so "future" always means the next accepted frame.
      this.bufferedFrameTiming = null
      this.resetNextFrame = true
    }
    const center = this.frameIndex % 2
    const future = 1 - center
    const destination = this.bufferedFrameTiming === null ? center : future

    try {
      device.queue.copyExternalImageToTexture(
        { source: video },
        { texture: resources.input[destination], colorSpace: 'srgb' },
        { width: video.videoWidth, height: video.videoHeight },
      )
      this.preprocessedReady[destination] = false
    } catch (error) {
      if (error instanceof DOMException && error.name === 'SecurityError') {
        throw new DOMException('이 영상은 브라우저 보안 정책상 GPU로 복사할 수 없습니다.', 'SecurityError')
      }
      throw error
    }

    // Keep exactly one decoded frame of lookahead. The first callback only
    // seeds the center slot; callback N+1 processes and presents frame N.
    if (this.bufferedFrameTiming === null) {
      this.bufferedFrameTiming = { ...frameTiming }
      return undefined
    }

    const centerFrameTiming = this.bufferedFrameTiming
    const timing = this.getTiming(centerFrameTiming.mediaTime)
    this.writeUniforms(
      video,
      target,
      settings,
      resources,
      timing,
      centerFrameTiming.skippedFrames,
    )

    const encoder = device.createCommandEncoder({ label: 'Web Upscaler preprocessing and reconstruction encoder' })
    const deblockCenter = !this.preprocessedReady[center]
    const deblockFuture = !this.preprocessedReady[future]
    if (deblockCenter || deblockFuture) {
      const deblockPass = encoder.beginComputePass({ label: 'New-frame adaptive deblock pass' })
      deblockPass.setPipeline(this.requireDeblockPipeline())
      if (deblockCenter) {
        deblockPass.setBindGroup(0, resources.deblockBindGroups[center])
        deblockPass.dispatchWorkgroups(Math.ceil(resources.inputWidth / 8), Math.ceil(resources.inputHeight / 8))
      }
      if (deblockFuture) {
        deblockPass.setBindGroup(0, resources.deblockBindGroups[future])
        deblockPass.dispatchWorkgroups(Math.ceil(resources.inputWidth / 8), Math.ceil(resources.inputHeight / 8))
      }
      deblockPass.end()
    }

    const analyzePass = encoder.beginComputePass({ label: 'Center and next-frame analyze pass' })
    analyzePass.setPipeline(this.requireAnalyzePipeline())
    analyzePass.setBindGroup(0, resources.analyzeBindGroups[center])
    analyzePass.dispatchWorkgroups(Math.ceil(resources.analysisWidth / 8), Math.ceil(resources.analysisHeight / 8))
    analyzePass.setBindGroup(0, resources.analyzeBindGroups[future])
    analyzePass.dispatchWorkgroups(Math.ceil(resources.analysisWidth / 8), Math.ceil(resources.analysisHeight / 8))
    analyzePass.end()

    const motionPass = encoder.beginComputePass({ label: 'Center-to-next-frame motion pass' })
    motionPass.setPipeline(this.requireMotionPipeline())
    motionPass.setBindGroup(0, resources.motionBindGroups[center])
    motionPass.dispatchWorkgroups(Math.ceil(resources.motionWidth / 8), Math.ceil(resources.motionHeight / 8))
    motionPass.end()

    const reconstructPass = encoder.beginComputePass({ label: '2x and temporal reconstruction pass' })
    reconstructPass.setPipeline(this.requireReconstructPipeline())
    reconstructPass.setBindGroup(0, resources.reconstructBindGroups[center])
    reconstructPass.dispatchWorkgroups(Math.ceil(target.width / 8), Math.ceil(target.height / 8))
    reconstructPass.end()

    const refinePass = encoder.beginComputePass({ label: 'Flat-region cleanup and dering pass' })
    refinePass.setPipeline(this.requireRefinePipeline())
    refinePass.setBindGroup(0, resources.refineBindGroups[center])
    refinePass.dispatchWorkgroups(Math.ceil(target.width / 8), Math.ceil(target.height / 8))
    refinePass.end()

    const microContrastPass = encoder.beginComputePass({ label: 'Aggressive micro-contrast pass' })
    microContrastPass.setPipeline(this.requireMicroContrastPipeline())
    microContrastPass.setBindGroup(0, resources.microContrastBindGroups[center])
    microContrastPass.dispatchWorkgroups(Math.ceil(target.width / 8), Math.ceil(target.height / 8))
    microContrastPass.end()

    const shockPass = encoder.beginComputePass({ label: 'Shock edge-compression pass' })
    shockPass.setPipeline(this.requireShockPipeline())
    shockPass.setBindGroup(0, resources.shockBindGroups[center])
    shockPass.dispatchWorkgroups(Math.ceil(target.width / 8), Math.ceil(target.height / 8))
    shockPass.end()

    const clarityPass = encoder.beginComputePass({ label: 'Wide-radius clarity pass' })
    clarityPass.setPipeline(this.requireClarityPipeline())
    clarityPass.setBindGroup(0, resources.clarityBindGroups[center])
    clarityPass.dispatchWorkgroups(Math.ceil(target.width / 8), Math.ceil(target.height / 8))
    clarityPass.end()

    const enhancePass = encoder.beginComputePass({ label: 'Local contrast, text clarity, and sharpening pass' })
    enhancePass.setPipeline(this.requireEnhancePipeline())
    enhancePass.setBindGroup(0, resources.enhanceBindGroups[center])
    enhancePass.dispatchWorkgroups(Math.ceil(target.width / 8), Math.ceil(target.height / 8))
    enhancePass.end()

    const compositePass = encoder.beginRenderPass({
      label: 'Lanczos presentation resize pass',
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    })
    compositePass.setPipeline(this.requireCompositePipeline())
    compositePass.setBindGroup(0, resources.compositeBindGroups[center])
    compositePass.draw(3)
    compositePass.end()

    device.queue.submit([encoder.finish()])
    if (deblockCenter) this.preprocessedReady[center] = true
    if (deblockFuture) this.preprocessedReady[future] = true
    this.pendingSubmissions += 1
    if (this.pendingSubmissions === WebGpuUpscaler.MAX_PENDING_SUBMISSIONS) {
      this.closeSubmissionBatch(device)
    }

    this.lastMediaTime = centerFrameTiming.mediaTime
    this.previousDt = timing.dt
    this.resetNextFrame = false
    this.bufferedFrameTiming = { ...frameTiming }
    this.frameIndex += 1
    const gpuQueueMs = this.completedGpuSampleMs
    this.completedGpuSampleMs = null
    return {
      gpuQueueMs,
      historyReset: timing.reset,
      pendingSubmissions: this.pendingSubmissions,
    }
  }

  resetHistory() {
    this.resetNextFrame = true
    this.bufferedFrameTiming = null
    this.preprocessedReady[0] = false
    this.preprocessedReady[1] = false
  }

  private resetTemporalState() {
    this.frameIndex = 0
    this.preprocessedReady[0] = false
    this.preprocessedReady[1] = false
    this.bufferedFrameTiming = null
    this.lastMediaTime = null
    this.previousDt = 1 / 60
    this.resetNextFrame = true
    this.completedGpuSampleMs = null
    this.submissionBatchIndex = 0
  }

  private requireDevice() {
    if (!this.device) throw new WebGpuUnavailableError('WebGPU device가 초기화되지 않았습니다.')
    return this.device
  }

  private requireContext() {
    if (!this.context) throw new WebGpuUnavailableError('WebGPU canvas가 초기화되지 않았습니다.')
    return this.context
  }

  private requireSampler() {
    if (!this.sampler) throw new WebGpuUnavailableError('GPU sampler가 초기화되지 않았습니다.')
    return this.sampler
  }

  private requireUniformBuffer() {
    if (!this.uniformBuffer) throw new WebGpuUnavailableError('GPU uniform buffer가 초기화되지 않았습니다.')
    return this.uniformBuffer
  }

  private requireAnalyzePipeline() {
    if (!this.analyzePipeline) throw new WebGpuUnavailableError('Analyze pipeline이 초기화되지 않았습니다.')
    return this.analyzePipeline
  }

  private requireDeblockPipeline() {
    if (!this.deblockPipeline) throw new WebGpuUnavailableError('Deblock pipeline이 초기화되지 않았습니다.')
    return this.deblockPipeline
  }

  private requireMotionPipeline() {
    if (!this.motionPipeline) throw new WebGpuUnavailableError('Motion pipeline이 초기화되지 않았습니다.')
    return this.motionPipeline
  }

  private requireReconstructPipeline() {
    if (!this.reconstructPipeline) throw new WebGpuUnavailableError('Temporal reconstruction pipeline이 초기화되지 않았습니다.')
    return this.reconstructPipeline
  }

  private requireRefinePipeline() {
    if (!this.refinePipeline) throw new WebGpuUnavailableError('Refinement pipeline이 초기화되지 않았습니다.')
    return this.refinePipeline
  }

  private requireMicroContrastPipeline() {
    if (!this.microContrastPipeline) throw new WebGpuUnavailableError('Micro-contrast pipeline is not initialized.')
    return this.microContrastPipeline
  }

  private requireShockPipeline() {
    if (!this.shockPipeline) throw new WebGpuUnavailableError('Shock pipeline is not initialized.')
    return this.shockPipeline
  }

  private requireClarityPipeline() {
    if (!this.clarityPipeline) throw new WebGpuUnavailableError('Wide-radius clarity pipeline is not initialized.')
    return this.clarityPipeline
  }

  private requireEnhancePipeline() {
    if (!this.enhancePipeline) throw new WebGpuUnavailableError('Enhancement pipeline이 초기화되지 않았습니다.')
    return this.enhancePipeline
  }

  private requireCompositePipeline() {
    if (!this.compositePipeline) throw new WebGpuUnavailableError('Composite pipeline이 초기화되지 않았습니다.')
    return this.compositePipeline
  }

  private destroyResources() {
    for (const texture of this.resources?.input ?? []) texture.destroy()
    for (const texture of this.resources?.preprocessed ?? []) texture.destroy()
    for (const texture of this.resources?.features ?? []) texture.destroy()
    for (const texture of this.resources?.motionStates ?? []) texture.destroy()
    for (const texture of this.resources?.motionMeta ?? []) texture.destroy()
    for (const texture of this.resources?.reconstruction ?? []) texture.destroy()
    for (const texture of this.resources?.refined ?? []) texture.destroy()
    for (const texture of this.resources?.microDetailed ?? []) texture.destroy()
    for (const texture of this.resources?.shocked ?? []) texture.destroy()
    for (const texture of this.resources?.clarified ?? []) texture.destroy()
    for (const texture of this.resources?.enhanced ?? []) texture.destroy()
    this.resources = null
  }

  destroy() {
    this.disposed = true
    this.destroyResources()
    this.uniformBuffer?.destroy()
    this.device?.destroy()
    this.device = null
    this.pendingSubmissions = 0
    this.batchFence = null
    this.completedGpuSampleMs = null
    this.context?.unconfigure()
    this.context = null
  }
}
