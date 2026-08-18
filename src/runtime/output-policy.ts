import type { BaseUpscalerSettings } from '../shared/settings'

export interface OutputTarget {
  kind: 'upscale'
  /** Exact 2x temporal reconstruction lattice. */
  width: number
  height: number
  scale: number
  /** Canvas backing size, capped at the exact 2x reconstruction lattice. */
  presentationWidth: number
  presentationHeight: number
}

export interface BypassTarget {
  kind: 'bypass'
  reason: 'pixel-budget-exceeded'
}

export type FrameTarget = OutputTarget | BypassTarget

function floorEven(value: number) {
  return Math.max(2, Math.floor(value / 2) * 2)
}

export function calculateOutputTarget(
  video: HTMLVideoElement,
  settings: BaseUpscalerSettings,
): FrameTarget {
  const inputWidth = video.videoWidth
  const inputHeight = video.videoHeight
  const rect = video.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1
  const displayWidth = Math.max(1, rect.width * dpr)
  const displayHeight = Math.max(1, rect.height * dpr)
  const pixelBudget = settings.mode === 'eco' ? 3_700_000 : 8_300_000
  const scale = settings.requestedScale
  const width = floorEven(inputWidth * scale)
  const height = floorEven(inputHeight * scale)

  if (!Number.isFinite(scale) || width * height > pixelBudget) {
    return { kind: 'bypass', reason: 'pixel-budget-exceeded' }
  }

  // Never render more fragment invocations than the reconstructed 2x lattice.
  // The browser compositor handles only the unavoidable final enlargement
  // when the physical player is larger than the recovered lattice.
  const canvasScale = Math.min(
    1,
    width / displayWidth,
    height / displayHeight,
  )

  return {
    kind: 'upscale',
    width,
    height,
    scale,
    presentationWidth: floorEven(displayWidth * canvasScale),
    presentationHeight: floorEven(displayHeight * canvasScale),
  }
}
