import type { BaseUpscalerSettings } from '../shared/settings'
import type { OutputTarget } from '../runtime/output-policy'
import analyzeShader from './shaders/analyze.wgsl?raw'
import compositeShader from './shaders/composite.wgsl?raw'
import reconstructShader from './shaders/reconstruct.wgsl?raw'

interface GpuResources {
  input: GPUTexture
  feature: GPUTexture
  output: GPUTexture
  analyzeBindGroup: GPUBindGroup
  reconstructBindGroup: GPUBindGroup
  compositeBindGroup: GPUBindGroup
  inputWidth: number
  inputHeight: number
  outputWidth: number
  outputHeight: number
  analysisWidth: number
  analysisHeight: number
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
  private adapter: GPUAdapter | null = null
  private device: GPUDevice | null = null
  private context: GPUCanvasContext | null = null
  private canvasFormat: GPUTextureFormat | null = null
  private sampler: GPUSampler | null = null
  private uniformBuffer: GPUBuffer | null = null
  private analyzePipeline: GPUComputePipeline | null = null
  private reconstructPipeline: GPUComputePipeline | null = null
  private compositePipeline: GPURenderPipeline | null = null
  private resources: GpuResources | null = null
  private disposed = false
  private initialization: Promise<void> | null = null

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly onDeviceLost: (message: string) => void,
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
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    const analyzeModule = this.device.createShaderModule({ label: 'Analyze shader', code: analyzeShader })
    const reconstructModule = this.device.createShaderModule({ label: 'Reconstruct shader', code: reconstructShader })
    const compositeModule = this.device.createShaderModule({ label: 'Composite shader', code: compositeShader })
    await Promise.all([
      this.assertShader(analyzeModule, 'Analyze'),
      this.assertShader(reconstructModule, 'Reconstruct'),
      this.assertShader(compositeModule, 'Composite'),
    ])

    this.analyzePipeline = await this.device.createComputePipelineAsync({
      label: 'Analyze pipeline',
      layout: 'auto',
      compute: { module: analyzeModule, entryPoint: 'main' },
    })
    this.reconstructPipeline = await this.device.createComputePipelineAsync({
      label: 'Reconstruct pipeline',
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

  private ensureResources(video: HTMLVideoElement, target: OutputTarget) {
    const device = this.requireDevice()
    const analysisWidth = Math.ceil(video.videoWidth / 4)
    const analysisHeight = Math.ceil(video.videoHeight / 4)
    const existing = this.resources

    if (existing
      && existing.inputWidth === video.videoWidth
      && existing.inputHeight === video.videoHeight
      && existing.outputWidth === target.width
      && existing.outputHeight === target.height) {
      return existing
    }

    this.destroyResources()
    this.canvas.width = target.width
    this.canvas.height = target.height

    const input = device.createTexture({
      label: 'Current video frame',
      size: [video.videoWidth, video.videoHeight],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    })
    const feature = device.createTexture({
      label: 'Quarter-resolution features',
      size: [analysisWidth, analysisHeight],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })
    const output = device.createTexture({
      label: 'Spatial reconstruction',
      size: [target.width, target.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    })
    const sampler = this.requireSampler()
    const uniformBuffer = this.requireUniformBuffer()
    const analyzePipeline = this.requireAnalyzePipeline()
    const reconstructPipeline = this.requireReconstructPipeline()
    const compositePipeline = this.requireCompositePipeline()

    this.resources = {
      input,
      feature,
      output,
      inputWidth: video.videoWidth,
      inputHeight: video.videoHeight,
      outputWidth: target.width,
      outputHeight: target.height,
      analysisWidth,
      analysisHeight,
      analyzeBindGroup: device.createBindGroup({
        label: 'Analyze bind group',
        layout: analyzePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: input.createView() },
          { binding: 1, resource: sampler },
          { binding: 2, resource: feature.createView() },
          { binding: 3, resource: { buffer: uniformBuffer } },
        ],
      }),
      reconstructBindGroup: device.createBindGroup({
        label: 'Reconstruct bind group',
        layout: reconstructPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: input.createView() },
          { binding: 1, resource: feature.createView() },
          { binding: 2, resource: output.createView() },
          { binding: 3, resource: sampler },
          { binding: 4, resource: { buffer: uniformBuffer } },
        ],
      }),
      compositeBindGroup: device.createBindGroup({
        label: 'Composite bind group',
        layout: compositePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: output.createView() },
          { binding: 1, resource: sampler },
          { binding: 2, resource: { buffer: uniformBuffer } },
        ],
      }),
    }

    return this.resources
  }

  async process(video: HTMLVideoElement, target: OutputTarget, settings: BaseUpscalerSettings) {
    await this.initialize()
    const device = this.requireDevice()
    const context = this.requireContext()
    const resources = this.ensureResources(video, target)
    const uniforms = new Float32Array([
      video.videoWidth, video.videoHeight, 1 / video.videoWidth, 1 / video.videoHeight,
      target.width, target.height, 1 / target.width, 1 / target.height,
      resources.analysisWidth, resources.analysisHeight, 1 / resources.analysisWidth, 1 / resources.analysisHeight,
      target.scale, settings.mode === 'eco' ? 0 : settings.sharpness, 0, 0,
    ])
    device.queue.writeBuffer(this.requireUniformBuffer(), 0, uniforms)

    try {
      device.queue.copyExternalImageToTexture(
        { source: video },
        { texture: resources.input, colorSpace: 'srgb' },
        { width: video.videoWidth, height: video.videoHeight },
      )
    } catch (error) {
      if (error instanceof DOMException && error.name === 'SecurityError') {
        throw new DOMException('이 영상은 브라우저 보안 정책상 GPU로 복사할 수 없습니다.', 'SecurityError')
      }
      throw error
    }

    const encoder = device.createCommandEncoder({ label: 'Web Upscaler frame encoder' })
    const analyzePass = encoder.beginComputePass({ label: 'Analyze pass' })
    analyzePass.setPipeline(this.requireAnalyzePipeline())
    analyzePass.setBindGroup(0, resources.analyzeBindGroup)
    analyzePass.dispatchWorkgroups(Math.ceil(resources.analysisWidth / 8), Math.ceil(resources.analysisHeight / 8))
    analyzePass.end()

    const reconstructPass = encoder.beginComputePass({ label: 'Reconstruct pass' })
    reconstructPass.setPipeline(this.requireReconstructPipeline())
    reconstructPass.setBindGroup(0, resources.reconstructBindGroup)
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
    compositePass.setBindGroup(0, resources.compositeBindGroup)
    compositePass.draw(3)
    compositePass.end()

    const startedAt = performance.now()
    device.queue.submit([encoder.finish()])
    await device.queue.onSubmittedWorkDone()
    return performance.now() - startedAt
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

  private requireReconstructPipeline() {
    if (!this.reconstructPipeline) throw new WebGpuUnavailableError('Reconstruct pipeline이 초기화되지 않았습니다.')
    return this.reconstructPipeline
  }

  private requireCompositePipeline() {
    if (!this.compositePipeline) throw new WebGpuUnavailableError('Composite pipeline이 초기화되지 않았습니다.')
    return this.compositePipeline
  }

  private destroyResources() {
    this.resources?.input.destroy()
    this.resources?.feature.destroy()
    this.resources?.output.destroy()
    this.resources = null
  }

  destroy() {
    this.disposed = true
    this.destroyResources()
    this.uniformBuffer?.destroy()
    this.device?.destroy()
    this.device = null
    this.context?.unconfigure()
    this.context = null
  }
}

