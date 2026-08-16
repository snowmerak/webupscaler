import type { UpscalerSettings } from './settings'
import type { RuntimeStatus } from './status'

export type ExtensionMessage =
  | { type: 'GET_STATUS' }
  | { type: 'SET_ENABLED'; enabled: boolean }
  | { type: 'UPDATE_SETTINGS'; patch: Partial<UpscalerSettings> }
  | { type: 'STATUS_UPDATE'; status: RuntimeStatus }

export interface ExtensionResponse {
  ok: boolean
  status?: RuntimeStatus
  settings?: UpscalerSettings
  error?: string
}

