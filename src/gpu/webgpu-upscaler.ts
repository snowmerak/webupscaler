import type { OutputTarget } from '../runtime/output-policy'
import type { BaseUpscalerSettings } from '../shared/settings'
import analyzeShader from './shaders/analyze.wgsl?raw'
import compositeShader from './shaders/composite.wgsl?raw'
import motionShader from './shaders/motion.wgsl?raw'
import reconstructShader from './shaders/reconstruct.wgsl?raw'

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
  features: Pair<GPUTexture>
  motionStates: Pair<GPUTexture>
  motionMeta: Pair<GPUTexture>
  history: Pair<GPUTexture>
  analyzeBindGroups: Pair<GPUBindGroup>
  motionBindGroups: Pair<GPUBindGroup>
  reconstructBindGroups: Pair<GPUBindGroup>
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
  private static readonly MAX_PENDING_SUBMISSIONS = 2
  private static readonly GPU_SAMPLE_BATCH_INTERVAL = 8
  private adapter: GPUAdapter | null = null
  private device: GPUDevice | null = null
  private context: GPUCanvasContext | null = null
  private canvasFormat: GPUTextureFormat | null = null
  private sampler: GPUSampler | null = null
  private uniformBuffer: GPUBuffer | null = null
  private analyzePipeline: GPUComputePipeline | null = null
  private motionPipeline: GPUComputePipeline | null = null
  private reconstructPipeline: GPUComputePipeline | null = null
  private compositePipeline: GPURenderPipeline | null = null
  private resources: GpuResources | null = null
  private disposed = false
  private initialization: Promise<void> | null = null
  private readonly uniformData = new Float32Array(32)
  private frameIndex = 0
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

    const analyzeModule = this.device.createShaderModule({ label: 'Analyze shader', code: analyzeShader })
    const motionModule = this.device.createShaderModule({ label: 'Motion shader', code: motionShader })
    const reconstructModule = this.device.createShaderModule({ label: 'Temporal reconstruct shader', code: reconstructShader })
    const compositeModule = this.device.createShaderModule({ label: 'Composite shader', code: compositeShader })
    await Promise.all([
      this.assertShader(analyzeModule, 'Analyze'),
      this.assertShader(motionModule, 'Motion'),
      this.assertShader(reconstructModule, 'Temporal reconstruct'),
      this.assertShader(compositeModule, 'Composite'),
    ])

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
      label: 'Temporal reconstruction pipeline',
      layout: 'auto',
      compute: { module: reconstructModule, entryPoint: 'main' },
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

    if (existing
      && existing.inputWidth === video.videoWidth
      && existing.inputHeight === video.videoHeight
      && existing.outputWidth === target.width
      && existing.outputHeight === target.height) {
      return existing
    }

    if (Math.max(target.width, target.height) > device.limits.maxTextureDimension2D) {
      throw new WebGpuUnavailableError('요청한 출력 크기가 GPU texture 제한을 넘었습니다.')
    }

    this.destroyResources()
    this.resetTemporalState()
    this.canvas.width = target.width
    this.canvas.height = target.height

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
    const history = this.pair((index) => device.createTexture({
      label: `HR radiance and observation coverage ${index}`,
      size: [target.width, target.height],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    }))
    const sampler = this.requireSampler()
    const uniformBuffer = this.requireUniformBuffer()
    const analyzePipeline = this.requireAnalyzePipeline()
    const motionPipeline = this.requireMotionPipeline()
    const reconstructPipeline = this.requireReconstructPipeline()
    const compositePipeline = this.requireCompositePipeline()
    const analyzeBindGroups = this.pair((current) => device.createBindGroup({
      label: `Analyze bind group ${current}`,
      layout: analyzePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: input[current].createView() },
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
          { binding: 2, resource: input[current].createView() },
          { binding: 3, resource: input[previous].createView() },
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
      const previous = 1 - current
      return device.createBindGroup({
        label: `Temporal reconstruct bind group ${current}`,
        layout: reconstructPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: input[current].createView() },
          { binding: 1, resource: motionMeta[current].createView() },
          { binding: 2, resource: history[previous].createView() },
          { binding: 3, resource: history[current].createView() },
          { binding: 4, resource: sampler },
          { binding: 5, resource: { buffer: uniformBuffer } },
        ],
      })
    })
    const compositeBindGroups = this.pair((current) => device.createBindGroup({
      label: `Composite bind group ${current}`,
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: history[current].createView() },
        { binding: 1, resource: input[current].createView() },
        { binding: 2, resource: features[current].createView() },
        { binding: 3, resource: sampler },
        { binding: 4, resource: { buffer: uniformBuffer } },
      ],
    }))

    this.resources = {
      input,
      features,
      motionStates,
      motionMeta,
      history,
      analyzeBindGroups,
      motionBindGroups,
      reconstructBindGroups,
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
    data[20] = settings.mode === 'eco' ? 0 : settings.sharpness
    data[21] = 2400
    data[22] = Math.pow(0.94, skippedFrames + 1)
    data[23] = settings.mode === 'eco' ? 0.7 : 0.85
    data[24] = timing.reset ? 1 : 0
    data[25] = settings.mode === 'auto' ? 0 : settings.mode === 'balanced' ? 1 : 2
    data[26] = this.frameIndex
    data[27] = settings.coverageOverlay ? 1 : 0
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
  ): Promise<ProcessResult | null> {
    await this.initialize()
    if (this.pendingSubmissions >= WebGpuUpscaler.MAX_PENDING_SUBMISSIONS) return null

    const device = this.requireDevice()
    const context = this.requireContext()
    const existing = this.resources
    const resourcesNeedRebuild = existing !== null && (
      existing.inputWidth !== video.videoWidth
      || existing.inputHeight !== video.videoHeight
      || existing.outputWidth !== target.width
      || existing.outputHeight !== target.height
    )
    if (resourcesNeedRebuild && this.pendingSubmissions > 0) {
      await this.drainPendingSubmissions(device)
    }

    const resources = this.ensureResources(video, target)
    const timing = this.getTiming(frameTiming.mediaTime)
    const current = this.frameIndex % 2
    this.writeUniforms(video, target, settings, resources, timing, frameTiming.skippedFrames)

    try {
      device.queue.copyExternalImageToTexture(
        { source: video },
        { texture: resources.input[current], colorSpace: 'srgb' },
        { width: video.videoWidth, height: video.videoHeight },
      )
    } catch (error) {
      if (error instanceof DOMException && error.name === 'SecurityError') {
        throw new DOMException('이 영상은 브라우저 보안 정책상 GPU로 복사할 수 없습니다.', 'SecurityError')
      }
      throw error
    }

    const encoder = device.createCommandEncoder({ label: 'Web Upscaler temporal frame encoder' })
    const analyzePass = encoder.beginComputePass({ label: 'Analyze pass' })
    analyzePass.setPipeline(this.requireAnalyzePipeline())
    analyzePass.setBindGroup(0, resources.analyzeBindGroups[current])
    analyzePass.dispatchWorkgroups(Math.ceil(resources.analysisWidth / 8), Math.ceil(resources.analysisHeight / 8))
    analyzePass.end()

    const motionPass = encoder.beginComputePass({ label: 'Acceleration-predicted motion pass' })
    motionPass.setPipeline(this.requireMotionPipeline())
    motionPass.setBindGroup(0, resources.motionBindGroups[current])
    motionPass.dispatchWorkgroups(Math.ceil(resources.motionWidth / 8), Math.ceil(resources.motionHeight / 8))
    motionPass.end()

    const reconstructPass = encoder.beginComputePass({ label: 'Temporal reconstruction pass' })
    reconstructPass.setPipeline(this.requireReconstructPipeline())
    reconstructPass.setBindGroup(0, resources.reconstructBindGroups[current])
    reconstructPass.dispatchWorkgroups(Math.ceil(target.width / 8), Math.ceil(target.height / 8))
    reconstructPass.end()

    const compositePass = encoder.beginRenderPass({
      label: 'Composite pass',
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    })
    compositePass.setPipeline(this.requireCompositePipeline())
    compositePass.setBindGroup(0, resources.compositeBindGroups[current])
    compositePass.draw(3)
    compositePass.end()

    device.queue.submit([encoder.finish()])
    this.pendingSubmissions += 1
    if (this.pendingSubmissions === WebGpuUpscaler.MAX_PENDING_SUBMISSIONS) {
      this.closeSubmissionBatch(device)
    }

    this.lastMediaTime = frameTiming.mediaTime
    this.previousDt = timing.dt
    this.resetNextFrame = false
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
  }

  private resetTemporalState() {
    this.frameIndex = 0
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

  private requireMotionPipeline() {
    if (!this.motionPipeline) throw new WebGpuUnavailableError('Motion pipeline이 초기화되지 않았습니다.')
    return this.motionPipeline
  }

  private requireReconstructPipeline() {
    if (!this.reconstructPipeline) throw new WebGpuUnavailableError('Temporal reconstruction pipeline이 초기화되지 않았습니다.')
    return this.reconstructPipeline
  }

  private requireCompositePipeline() {
    if (!this.compositePipeline) throw new WebGpuUnavailableError('Composite pipeline이 초기화되지 않았습니다.')
    return this.compositePipeline
  }

  private destroyResources() {
    for (const texture of this.resources?.input ?? []) texture.destroy()
    for (const texture of this.resources?.features ?? []) texture.destroy()
    for (const texture of this.resources?.motionStates ?? []) texture.destroy()
    for (const texture of this.resources?.motionMeta ?? []) texture.destroy()
    for (const texture of this.resources?.history ?? []) texture.destroy()
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
