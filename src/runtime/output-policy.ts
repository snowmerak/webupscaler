import type { BaseUpscalerSettings } from '../shared/settings'

export interface OutputTarget {
  kind: 'upscale'
  width: number
  height: number
  scale: number
}

export interface BypassTarget {
  kind: 'bypass'
  reason: 'source-large-enough'
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
  const objectFit = getComputedStyle(video).objectFit
  const displayScale = objectFit === 'cover'
    ? Math.max(displayWidth / inputWidth, displayHeight / inputHeight)
    : Math.min(displayWidth / inputWidth, displayHeight / inputHeight)
  const pixelBudget = settings.mode === 'eco' ? 3_700_000 : 8_300_000
  const budgetScale = Math.sqrt(pixelBudget / (inputWidth * inputHeight))
  const scale = Math.min(settings.requestedScale, displayScale, budgetScale)

  if (!Number.isFinite(scale) || scale < 1.05) {
    return { kind: 'bypass', reason: 'source-large-enough' }
  }

  return {
    kind: 'upscale',
    width: floorEven(inputWidth * scale),
    height: floorEven(inputHeight * scale),
    scale,
  }
}

