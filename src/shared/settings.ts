export const SETTINGS_KEY = 'upscalerSettings'

export type UpscalerMode = 'auto' | 'balanced' | 'eco'
export type DiagnosticView = 'off' | 'coverage' | 'variance' | 'samples' | 'residual' | 'correction' | 'motion'

export interface BaseUpscalerSettings {
  enabled: boolean
  mode: UpscalerMode
  requestedScale: 2
  sharpness: number
  deblockStrength: number
  debugOverlay: boolean
  diagnosticView: DiagnosticView
}

export interface UpscalerSettings extends BaseUpscalerSettings {
  perHost: Record<string, Partial<BaseUpscalerSettings>>
}

export const DEFAULT_SETTINGS: UpscalerSettings = {
  enabled: false,
  mode: 'auto',
  requestedScale: 2,
  sharpness: 0.12,
  deblockStrength: 0.3,
  debugOverlay: false,
  diagnosticView: 'off',
  perHost: {},
}

function isMode(value: unknown): value is UpscalerMode {
  return value === 'auto' || value === 'balanced' || value === 'eco'
}

function isDiagnosticView(value: unknown): value is DiagnosticView {
  return value === 'off'
    || value === 'coverage'
    || value === 'variance'
    || value === 'samples'
    || value === 'residual'
    || value === 'correction'
    || value === 'motion'
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
    deblockStrength: typeof input.deblockStrength === 'number'
      ? Math.min(0.5, Math.max(0, input.deblockStrength))
      : DEFAULT_SETTINGS.deblockStrength,
    debugOverlay: typeof input.debugOverlay === 'boolean'
      ? input.debugOverlay
      : DEFAULT_SETTINGS.debugOverlay,
    diagnosticView: isDiagnosticView(input.diagnosticView)
      ? input.diagnosticView
      : (input as { coverageOverlay?: unknown }).coverageOverlay === true
        ? 'coverage'
        : DEFAULT_SETTINGS.diagnosticView,
    perHost: typeof input.perHost === 'object' && input.perHost !== null
      ? input.perHost
      : {},
  }
}

export function settingsForHost(settings: UpscalerSettings, hostname: string): BaseUpscalerSettings {
  const override = settings.perHost[hostname] ?? {}
  const legacyOverride = override as Partial<BaseUpscalerSettings> & { coverageOverlay?: unknown }
  const mode = isMode(override.mode) ? override.mode : settings.mode

  return {
    enabled: typeof override.enabled === 'boolean' ? override.enabled : settings.enabled,
    mode,
    requestedScale: 2,
    sharpness: typeof override.sharpness === 'number'
      ? Math.min(0.25, Math.max(0, override.sharpness))
      : settings.sharpness,
    deblockStrength: typeof override.deblockStrength === 'number'
      ? Math.min(0.5, Math.max(0, override.deblockStrength))
      : settings.deblockStrength,
    debugOverlay: typeof override.debugOverlay === 'boolean'
      ? override.debugOverlay
      : settings.debugOverlay,
    diagnosticView: isDiagnosticView(override.diagnosticView)
      ? override.diagnosticView
      : legacyOverride.coverageOverlay === true
        ? 'coverage'
        : settings.diagnosticView,
  }
}
