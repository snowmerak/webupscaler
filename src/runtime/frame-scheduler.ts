export interface FrameSchedulerCallbacks {
  process(metadata: VideoFrameCallbackMetadata): Promise<void>
  onSkipped(): void
  onError(error: unknown): void
}

export class FrameScheduler {
  private callbackId: number | null = null
  private inFlight = false
  private running = false

  constructor(
    private readonly video: HTMLVideoElement,
    private readonly callbacks: FrameSchedulerCallbacks,
  ) {}

  start() {
    if (this.running) return
    this.running = true
    this.schedule()
  }

  private schedule() {
    if (!this.running) return
    this.callbackId = this.video.requestVideoFrameCallback((_now, metadata) => {
      this.callbackId = null
      this.schedule()

      if (document.hidden || this.video.paused || this.video.ended) return
      if (this.inFlight) {
        this.callbacks.onSkipped()
        return
      }

      this.inFlight = true
      void this.callbacks.process(metadata)
        .catch((error: unknown) => this.callbacks.onError(error))
        .finally(() => {
          this.inFlight = false
        })
    })
  }

  stop() {
    this.running = false
    if (this.callbackId !== null) {
      this.video.cancelVideoFrameCallback(this.callbackId)
      this.callbackId = null
    }
  }
}

