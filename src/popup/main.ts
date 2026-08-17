import type { ExtensionMessage, ExtensionResponse } from '../shared/messages'
import {
  DEFAULT_SETTINGS,
  normalizeSettings,
  type UpscalerSettings,
} from '../shared/settings'
import { INITIAL_STATUS, type RuntimeStatus } from '../shared/status'
import './style.css'

function required<T extends Element>(selector: string) {
  const element = document.querySelector<T>(selector)
  if (!element) throw new Error(`Popup UI element is missing: ${selector}`)
  return element
}

const controls = {
  enabled: required<HTMLInputElement>('#enabled'),
  mode: required<HTMLSelectElement>('#mode'),
  sharpness: required<HTMLInputElement>('#sharpness'),
  sharpnessValue: required<HTMLOutputElement>('#sharpness-value'),
  deblockStrength: required<HTMLInputElement>('#deblock-strength'),
  deblockStrengthValue: required<HTMLOutputElement>('#deblock-strength-value'),
  debugOverlay: required<HTMLInputElement>('#debug-overlay'),
  diagnosticView: required<HTMLSelectElement>('#diagnostic-view'),
}
const view = {
  badge: required<HTMLSpanElement>('#state-badge'),
  dot: required<HTMLSpanElement>('#status-dot'),
  title: required<HTMLElement>('#status-title'),
  message: required<HTMLParagraphElement>('#status-message'),
  metrics: required<HTMLElement>('#metrics'),
  sourceSize: required<HTMLElement>('#source-size'),
  outputSize: required<HTMLElement>('#output-size'),
  frameTime: required<HTMLElement>('#frame-time'),
}

let settings: UpscalerSettings = DEFAULT_SETTINGS
let status: RuntimeStatus = INITIAL_STATUS
const extensionAvailable = typeof chrome !== 'undefined' && Boolean(chrome.tabs)

async function sendToActiveTab(message: ExtensionMessage): Promise<ExtensionResponse> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
  if (tab?.id === undefined) throw new Error('활성 탭을 찾지 못했습니다.')
  return chrome.tabs.sendMessage(tab.id, message) as Promise<ExtensionResponse>
}

function stateLabel(runtime: RuntimeStatus) {
  if (!runtime.enabled || runtime.state === 'idle') return ['꺼짐', '대기 중', 'off'] as const
  if (runtime.state === 'running') return ['실행 중', '업스케일링 중', 'running'] as const
  if (runtime.state === 'bypass') return ['원본', '원본 사용 중', 'bypass'] as const
  if (runtime.state === 'error') return ['오류', '처리할 수 없음', 'error'] as const
  if (runtime.state === 'recovering') return ['복구', 'GPU 복구 중', 'pending'] as const
  return ['준비', '영상 준비 중', 'pending'] as const
}

function render() {
  const [badge, title, tone] = stateLabel(status)
  view.badge.textContent = badge
  view.badge.dataset.tone = tone
  view.dot.dataset.tone = tone
  view.title.textContent = title
  view.message.textContent = status.message

  controls.enabled.checked = settings.enabled
  controls.mode.value = settings.mode
  controls.sharpness.value = String(Math.round(settings.sharpness * 100))
  controls.sharpnessValue.value = `${Math.round(settings.sharpness * 100)}%`
  controls.deblockStrength.value = String(Math.round(settings.deblockStrength * 100))
  controls.deblockStrengthValue.value = `${Math.round(settings.deblockStrength * 100)}%`
  controls.debugOverlay.checked = settings.debugOverlay
  controls.diagnosticView.value = settings.diagnosticView

  const disabled = !status.supportedSite
  for (const control of [controls.enabled, controls.mode, controls.sharpness, controls.deblockStrength, controls.debugOverlay, controls.diagnosticView]) {
    control.disabled = disabled
  }

  if (status.metrics) {
    view.metrics.hidden = false
    view.sourceSize.textContent = `${status.metrics.sourceWidth}×${status.metrics.sourceHeight}`
    view.outputSize.textContent = `${status.metrics.outputWidth}×${status.metrics.outputHeight} → ${status.metrics.presentationWidth}×${status.metrics.presentationHeight}`
    view.frameTime.textContent = `${status.metrics.gpuQueueP90Ms.toFixed(1)} ms`
  } else {
    view.metrics.hidden = true
  }
}

async function update(message: ExtensionMessage) {
  if (!extensionAvailable) {
    if (message.type === 'SET_ENABLED') settings = { ...settings, enabled: message.enabled }
    if (message.type === 'UPDATE_SETTINGS') settings = normalizeSettings({ ...settings, ...message.patch })
    status = {
      state: settings.enabled ? 'waiting-video' : 'idle',
      enabled: settings.enabled,
      supportedSite: true,
      adapter: 'soop',
      message: settings.enabled ? 'SOOP LivePlayer를 기다리는 중입니다.' : '현재 사이트에서 꺼져 있습니다.',
    }
    render()
    return
  }

  const response = await sendToActiveTab(message)
  if (!response.ok) throw new Error(response.error ?? '요청을 적용하지 못했습니다.')
  if (response.settings) settings = normalizeSettings(response.settings)
  if (response.status) status = response.status
  render()
}

controls.enabled.addEventListener('change', () => {
  void update({ type: 'SET_ENABLED', enabled: controls.enabled.checked }).catch(showError)
})
controls.mode.addEventListener('change', () => {
  void update({ type: 'UPDATE_SETTINGS', patch: { mode: controls.mode.value as UpscalerSettings['mode'] } }).catch(showError)
})
controls.sharpness.addEventListener('input', () => {
  controls.sharpnessValue.value = `${controls.sharpness.value}%`
})
controls.sharpness.addEventListener('change', () => {
  void update({ type: 'UPDATE_SETTINGS', patch: { sharpness: Number(controls.sharpness.value) / 100 } }).catch(showError)
})
controls.deblockStrength.addEventListener('input', () => {
  controls.deblockStrengthValue.value = `${controls.deblockStrength.value}%`
})
controls.deblockStrength.addEventListener('change', () => {
  void update({ type: 'UPDATE_SETTINGS', patch: { deblockStrength: Number(controls.deblockStrength.value) / 100 } }).catch(showError)
})
controls.debugOverlay.addEventListener('change', () => {
  void update({ type: 'UPDATE_SETTINGS', patch: { debugOverlay: controls.debugOverlay.checked } }).catch(showError)
})
controls.diagnosticView.addEventListener('change', () => {
  void update({
    type: 'UPDATE_SETTINGS',
    patch: { diagnosticView: controls.diagnosticView.value as UpscalerSettings['diagnosticView'] },
  }).catch(showError)
})

function showError(error: unknown) {
  status = {
    state: 'error',
    enabled: settings.enabled,
    supportedSite: true,
    adapter: 'soop',
    message: error instanceof Error ? error.message : '요청을 적용하지 못했습니다.',
  }
  render()
}

async function initialize() {
  if (!extensionAvailable) {
    settings = DEFAULT_SETTINGS
    status = {
      state: 'idle',
      enabled: false,
      supportedSite: true,
      adapter: 'soop',
      message: '미리보기 모드 — SOOP 탭에서 확장을 열면 실제 상태가 표시됩니다.',
    }
    render()
    return
  }

  try {
    const response = await sendToActiveTab({ type: 'GET_STATUS' })
    if (response.settings) settings = normalizeSettings(response.settings)
    if (response.status) status = response.status
  } catch {
    status = {
      ...INITIAL_STATUS,
      message: '연결되지 않았습니다. 확장을 새로고침한 뒤 SOOP 방송 탭도 새로고침해 주세요.',
    }
  }
  render()
}

void initialize()
