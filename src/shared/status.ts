export type RuntimeState =
  | 'idle'
  | 'waiting-video'
  | 'initializing'
  | 'running'
  | 'bypass'
  | 'recovering'
  | 'error'

export type SupportedSiteAdapter = 'soop' | 'youtube'

export type BypassReason =
  | 'source-large-enough'
  | 'pixel-budget-exceeded'
  | 'video-paused'
  | 'document-hidden'
  | 'webgpu-unavailable'

export interface RuntimeMetrics {
  sourceWidth: number
  sourceHeight: number
  outputWidth: number
  outputHeight: number
  presentationWidth: number
  presentationHeight: number
  gpuQueueMs: number
  gpuQueueP90Ms: number
  processedFrames: number
  skippedFrames: number
  historyResets: number
  pendingSubmissions: number
}

export interface RuntimeStatus {
  state: RuntimeState
  enabled: boolean
  supportedSite: boolean
  adapter: SupportedSiteAdapter | 'unsupported'
  message: string
  bypassReason?: BypassReason
  errorCode?: string
  metrics?: RuntimeMetrics
}

export const INITIAL_STATUS: RuntimeStatus = {
  state: 'idle',
  enabled: false,
  supportedSite: false,
  adapter: 'unsupported',
  message: '지원 사이트가 아닙니다.',
}
