import { ShaderCompilationError, WebGpuUnavailableError, WebGpuUpscaler } from '../gpu/webgpu-upscaler'
import type { SiteAdapter } from '../sites/adapter'
import type { BaseUpscalerSettings } from '../shared/settings'
import type { RuntimeStatus } from '../shared/status'
import { FrameScheduler } from './frame-scheduler'
import { MetricsTracker } from './metrics'
import { OverlayController } from './overlay-controller'
import { calculateOutputTarget } from './output-policy'

export interface UpscalerControllerOptions {
  adapter: SiteAdapter
  settings: BaseUpscalerSettings
  onStatus(status: RuntimeStatus): void
}

export class UpscalerController {
  private settings: BaseUpscalerSettings
  private status: RuntimeStatus
  private video: HTMLVideoElement | null = null
  private overlay: OverlayController | null = null
  private scheduler: FrameScheduler | null = null
  private processor: WebGpuUpscaler | null = null
  private metrics = new MetricsTracker()
  private observer: MutationObserver | null = null
  private urlTimer: number | null = null
  private lastUrl = location.href
  private lastStatusSentAt = 0
  private recoveryAttempts = 0
  private skippedSinceLastFrame = 0

  constructor(private readonly options: UpscalerControllerOptions) {
    this.settings = options.settings
    this.status = {
      state: 'idle',
      enabled: this.settings.enabled,
      supportedSite: true,
      adapter: options.adapter.id,
      message: this.settings.enabled
        ? `${options.adapter.playerLabel}를 찾는 중입니다.`
        : '현재 사이트에서 꺼져 있습니다.',
    }
  }

  start() {
    this.observer = new MutationObserver(() => this.syncVideo())
    this.observer.observe(document.documentElement, { childList: true, subtree: true })
    for (const eventName of this.options.adapter.navigationEvents ?? []) {
      document.addEventListener(eventName, this.handleNavigation)
    }
    this.urlTimer = window.setInterval(() => {
      if (location.href !== this.lastUrl) {
        this.lastUrl = location.href
        this.syncVideo(true)
      }
    }, 1000)
    this.syncVideo()
  }

  getStatus() {
    return this.status
  }

  updateSettings(settings: BaseUpscalerSettings) {
    const wasEnabled = this.settings.enabled
    const deblockChanged = this.settings.deblockStrength !== settings.deblockStrength
    this.settings = settings

    if (!settings.enabled) {
      this.stopSession()
      this.publish({
        state: 'idle',
        enabled: false,
        supportedSite: true,
        adapter: this.options.adapter.id,
        message: '현재 사이트에서 꺼져 있습니다.',
      }, true)
      return
    }

    if (!wasEnabled) this.syncVideo(true)
    if (deblockChanged) this.processor?.resetHistory()
    this.updateDebugOverlay()
  }

  private syncVideo(force = false) {
    if (!this.settings.enabled) return

    const candidate = this.options.adapter.findVideos(document)[0] ?? null
    if (!force && candidate === this.video) return

    this.stopSession()
    if (!candidate) {
      this.publish({
        state: 'waiting-video',
        enabled: true,
        supportedSite: true,
        adapter: this.options.adapter.id,
        message: `${this.options.adapter.playerLabel}를 기다리는 중입니다.`,
      }, true)
      return
    }

    this.video = candidate
    if (candidate.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
      && candidate.videoWidth > 0
      && candidate.videoHeight > 0) {
      this.attachVideo(candidate)
      return
    }

    this.publish({
      state: 'waiting-video',
      enabled: true,
      supportedSite: true,
      adapter: this.options.adapter.id,
      message: `${this.options.adapter.playerLabel}의 첫 프레임을 기다리는 중입니다.`,
    }, true)
    candidate.addEventListener('loadeddata', this.handleVideoReady, { once: true })
    candidate.addEventListener('resize', this.handleVideoReset)
    candidate.addEventListener('emptied', this.handleVideoReset)
    candidate.addEventListener('seeking', this.handleHistoryReset)
  }

  private readonly handleVideoReady = () => {
    if (this.video) this.attachVideo(this.video)
  }

  private readonly handleNavigation = () => {
    this.lastUrl = location.href
    this.syncVideo(true)
  }

  private readonly handleVideoReset = () => {
    this.syncVideo(true)
  }

  private readonly handleHistoryReset = () => {
    this.processor?.resetHistory()
    this.overlay?.hide()
  }

  private attachVideo(video: HTMLVideoElement) {
    video.addEventListener('resize', this.handleVideoReset)
    video.addEventListener('emptied', this.handleVideoReset)
    video.addEventListener('seeking', this.handleHistoryReset)
    const host = this.options.adapter.getOverlayHost(video)
    this.overlay = new OverlayController(video, host)
    this.processor = this.createProcessor(this.overlay.canvas)
    this.metrics = new MetricsTracker()
    this.skippedSinceLastFrame = 0
    this.recoveryAttempts = 0
    this.scheduler = new FrameScheduler(video, {
      process: async (metadata) => this.processFrame(video, metadata),
      onSkipped: () => {
        this.metrics.recordSkipped()
        this.skippedSinceLastFrame += 1
      },
      onError: (error) => this.handleFrameError(error),
    })
    this.publish({
      state: 'initializing',
      enabled: true,
      supportedSite: true,
      adapter: this.options.adapter.id,
      message: 'WebGPU 업스케일러를 준비하고 있습니다.',
    }, true)
    this.scheduler.start()
  }

  private createProcessor(canvas: HTMLCanvasElement) {
    return new WebGpuUpscaler(canvas, (message) => this.handleDeviceLost(message))
  }

  private async processFrame(video: HTMLVideoElement, metadata: VideoFrameCallbackMetadata) {
    if (!this.settings.enabled || video !== this.video || video.videoWidth === 0) return

    const target = calculateOutputTarget(video, this.settings)
    if (target.kind === 'bypass') {
      this.overlay?.hide()
      this.publish({
        state: 'bypass',
        enabled: true,
        supportedSite: true,
        adapter: this.options.adapter.id,
        message: '정확한 2× 내부 복원이 GPU 픽셀 예산을 초과합니다.',
        bypassReason: target.reason,
      })
      return
    }

    const result = await this.processor?.process(video, target, this.settings, {
      mediaTime: metadata.mediaTime,
      skippedFrames: this.skippedSinceLastFrame,
    })
    if (result === undefined) return
    if (result === null) {
      this.metrics.recordSkipped()
      this.skippedSinceLastFrame += 1
      return
    }
    this.skippedSinceLastFrame = 0
    if (result.historyReset) this.metrics.recordHistoryReset()

    this.metrics.recordSubmitted(result.gpuQueueMs, result.pendingSubmissions)
    this.overlay?.show()
    this.overlay?.layout()
    const metrics = this.metrics.snapshot(
      video.videoWidth,
      video.videoHeight,
      target.width,
      target.height,
      target.presentationWidth,
      target.presentationHeight,
    )
    this.publish({
      state: 'running',
      enabled: true,
      supportedSite: true,
      adapter: this.options.adapter.id,
      message: `${video.videoWidth}×${video.videoHeight} → 2× ${target.width}×${target.height} → 표시 ${target.presentationWidth}×${target.presentationHeight}`,
      metrics,
    })
    this.updateDebugOverlay()
  }

  private updateDebugOverlay() {
    const metrics = this.status.metrics
    const text = metrics
      ? [
          'Web Upscaler v2',
          `Source ${metrics.sourceWidth}×${metrics.sourceHeight}`,
          `Internal ${metrics.outputWidth}×${metrics.outputHeight}`,
          `Display ${metrics.presentationWidth}×${metrics.presentationHeight}`,
          `Mode ${this.settings.mode}`,
          `GPU queue ${metrics.gpuQueueMs.toFixed(1)} ms / p90 ${metrics.gpuQueueP90Ms.toFixed(1)} ms`,
          `Pending ${metrics.pendingSubmissions}/2`,
          `Skipped ${metrics.skippedFrames}`,
          `History resets ${metrics.historyResets}`,
        ].join('\n')
      : `Web Upscaler v2\nState ${this.status.state}`
    this.overlay?.setDebug(text, this.settings.debugOverlay)
  }

  private handleDeviceLost(message: string) {
    this.overlay?.hide()
    this.recoveryAttempts += 1

    if (this.recoveryAttempts > 1 || !this.overlay) {
      this.handleFrameError(new Error(`GPU 복구 실패: ${message}`))
      return
    }

    this.publish({
      state: 'recovering',
      enabled: true,
      supportedSite: true,
      adapter: this.options.adapter.id,
      message: 'GPU 장치를 한 번 복구하고 있습니다.',
    }, true)
    this.processor?.destroy()
    this.processor = this.createProcessor(this.overlay.canvas)
  }

  private handleFrameError(error: unknown) {
    this.scheduler?.stop()
    this.overlay?.hide()

    let errorCode = 'UNKNOWN'
    let message = error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다.'
    if (error instanceof DOMException && error.name === 'SecurityError') {
      errorCode = 'VIDEO_SOURCE_SECURITY_ERROR'
      message = '이 영상은 브라우저 보안 정책상 처리할 수 없습니다.'
    } else if (error instanceof WebGpuUnavailableError) {
      errorCode = 'WEBGPU_UNAVAILABLE'
    } else if (error instanceof ShaderCompilationError) {
      errorCode = 'SHADER_COMPILE_FAILED'
    }

    this.publish({
      state: 'error',
      enabled: true,
      supportedSite: true,
      adapter: this.options.adapter.id,
      message,
      errorCode,
    }, true)
  }

  private publish(status: RuntimeStatus, force = false) {
    this.status = status
    const now = performance.now()
    if (!force && now - this.lastStatusSentAt < 1000) return
    this.lastStatusSentAt = now
    this.options.onStatus(status)
  }

  private stopSession() {
    if (this.video) {
      this.video.removeEventListener('loadeddata', this.handleVideoReady)
      this.video.removeEventListener('resize', this.handleVideoReset)
      this.video.removeEventListener('emptied', this.handleVideoReset)
      this.video.removeEventListener('seeking', this.handleHistoryReset)
    }
    this.scheduler?.stop()
    this.processor?.destroy()
    this.overlay?.destroy()
    this.scheduler = null
    this.processor = null
    this.overlay = null
    this.video = null
  }

  destroy() {
    this.observer?.disconnect()
    if (this.urlTimer !== null) clearInterval(this.urlTimer)
    for (const eventName of this.options.adapter.navigationEvents ?? []) {
      document.removeEventListener(eventName, this.handleNavigation)
    }
    this.stopSession()
  }
}
