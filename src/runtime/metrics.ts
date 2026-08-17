import type { RuntimeMetrics } from '../shared/status'

export class MetricsTracker {
  private readonly samples: number[] = []
  private processedFrames = 0
  private skippedFrames = 0
  private historyResets = 0
  private pendingSubmissions = 0

  recordSubmitted(gpuQueueMs: number | null, pendingSubmissions: number) {
    this.processedFrames += 1
    this.pendingSubmissions = pendingSubmissions
    if (gpuQueueMs !== null) {
      this.samples.push(gpuQueueMs)
      if (this.samples.length > 120) this.samples.shift()
    }
  }

  recordSkipped() {
    this.skippedFrames += 1
  }

  recordHistoryReset() {
    this.historyResets += 1
  }

  snapshot(
    sourceWidth: number,
    sourceHeight: number,
    outputWidth: number,
    outputHeight: number,
    presentationWidth: number,
    presentationHeight: number,
  ): RuntimeMetrics {
    const sorted = [...this.samples].sort((a, b) => a - b)
    const p90Index = Math.max(0, Math.ceil(sorted.length * 0.9) - 1)
    const latest = this.samples.at(-1) ?? 0

    return {
      sourceWidth,
      sourceHeight,
      outputWidth,
      outputHeight,
      presentationWidth,
      presentationHeight,
      gpuQueueMs: latest,
      gpuQueueP90Ms: sorted[p90Index] ?? latest,
      processedFrames: this.processedFrames,
      skippedFrames: this.skippedFrames,
      historyResets: this.historyResets,
      pendingSubmissions: this.pendingSubmissions,
    }
  }
}
