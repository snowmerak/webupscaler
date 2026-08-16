import type { RuntimeMetrics } from '../shared/status'

export class MetricsTracker {
  private readonly samples: number[] = []
  private processedFrames = 0
  private skippedFrames = 0

  recordCompletion(value: number) {
    this.processedFrames += 1
    this.samples.push(value)
    if (this.samples.length > 120) this.samples.shift()
  }

  recordSkipped() {
    this.skippedFrames += 1
  }

  snapshot(sourceWidth: number, sourceHeight: number, outputWidth: number, outputHeight: number): RuntimeMetrics {
    const sorted = [...this.samples].sort((a, b) => a - b)
    const p90Index = Math.max(0, Math.ceil(sorted.length * 0.9) - 1)
    const latest = this.samples.at(-1) ?? 0

    return {
      sourceWidth,
      sourceHeight,
      outputWidth,
      outputHeight,
      completionMs: latest,
      completionP90Ms: sorted[p90Index] ?? latest,
      processedFrames: this.processedFrames,
      skippedFrames: this.skippedFrames,
    }
  }
}

