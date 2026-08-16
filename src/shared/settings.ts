export const SETTINGS_KEY = 'upscalerSettings'

export type UpscalerMode = 'auto' | 'balanced' | 'eco'

export interface BaseUpscalerSettings {
  enabled: boolean
  mode: UpscalerMode
  requestedScale: 2
  sharpness: number
  debugOverlay: boolean
}

export interface UpscalerSettings extends BaseUpscalerSettings {
  perHost: Record<string, Partial<BaseUpscalerSettings>>
}

export const DEFAULT_SETTINGS: UpscalerSettings = {
  enabled: false,
  mode: 'auto',
  requestedScale: 2,
  sharpness: 0.12,
  debugOverlay: false,
  perHost: {},
}

function isMode(value: unknown): value is UpscalerMode {
  return value === 'auto' || value === 'balanced' || value === 'eco'
}

export function normalizeSettings(value: unknown): UpscalerSettings {
  const input = typeof value === 'object' && value !== null
    ? value as Partial<UpscalerSettings>
    : {}

  return {
    enabled: typeof input.enabled === 'boolean' ? input.enabled : DEFAULT_SETTINGS.enabled,
    mode: isMode(input.mode) ? input.mode : DEFAULT_SETTINGS.mode,
    requestedScale: 2,
    sharpness: typeof input.sharpness === 'number'
      ? Math.min(0.25, Math.max(0, input.sharpness))
      : DEFAULT_SETTINGS.sharpness,
    debugOverlay: typeof input.debugOverlay === 'boolean'
      ? input.debugOverlay
      : DEFAULT_SETTINGS.debugOverlay,
    perHost: typeof input.perHost === 'object' && input.perHost !== null
      ? input.perHost
      : {},
  }
}

export function settingsForHost(settings: UpscalerSettings, hostname: string): BaseUpscalerSettings {
  const override = settings.perHost[hostname] ?? {}
  const mode = isMode(override.mode) ? override.mode : settings.mode

  return {
    enabled: typeof override.enabled === 'boolean' ? override.enabled : settings.enabled,
    mode,
    requestedScale: 2,
    sharpness: typeof override.sharpness === 'number'
      ? Math.min(0.25, Math.max(0, override.sharpness))
      : settings.sharpness,
    debugOverlay: typeof override.debugOverlay === 'boolean'
      ? override.debugOverlay
      : settings.debugOverlay,
  }
}

