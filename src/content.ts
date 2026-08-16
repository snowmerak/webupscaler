import { UpscalerController } from './runtime/controller'
import { resolveSiteAdapter } from './sites'
import type { ExtensionMessage, ExtensionResponse } from './shared/messages'
import {
  normalizeSettings,
  SETTINGS_KEY,
  settingsForHost,
  type UpscalerSettings,
} from './shared/settings'
import { INITIAL_STATUS, type RuntimeStatus } from './shared/status'

let storedSettings: UpscalerSettings
let status: RuntimeStatus = INITIAL_STATUS
let controller: UpscalerController | null = null

async function loadSettings() {
  const stored = await chrome.storage.local.get(SETTINGS_KEY)
  return normalizeSettings(stored[SETTINGS_KEY])
}

async function saveSettings(next: UpscalerSettings) {
  storedSettings = normalizeSettings(next)
  await chrome.storage.local.set({ [SETTINGS_KEY]: storedSettings })
  controller?.updateSettings(settingsForHost(storedSettings, location.hostname))
}

function publishStatus(next: RuntimeStatus) {
  status = next
  void chrome.runtime.sendMessage({ type: 'STATUS_UPDATE', status } satisfies ExtensionMessage)
    .catch(() => undefined)
}

async function initialize() {
  storedSettings = await loadSettings()
  const adapter = resolveSiteAdapter()

  if (!adapter) {
    publishStatus(INITIAL_STATUS)
    return
  }

  controller = new UpscalerController({
    adapter,
    settings: settingsForHost(storedSettings, location.hostname),
    onStatus: publishStatus,
  })
  controller.start()
  status = controller.getStatus()
  publishStatus(status)
}

chrome.runtime.onMessage.addListener(
  (message: ExtensionMessage, _sender, sendResponse: (response: ExtensionResponse) => void) => {
    if (message.type === 'GET_STATUS') {
      sendResponse({ ok: true, status, settings: storedSettings })
      return
    }

    if (message.type === 'SET_ENABLED') {
      void saveSettings({ ...storedSettings, enabled: message.enabled })
        .then(() => sendResponse({ ok: true, status: controller?.getStatus() ?? status, settings: storedSettings }))
        .catch((error: unknown) => sendResponse({
          ok: false,
          error: error instanceof Error ? error.message : '설정을 저장하지 못했습니다.',
        }))
      return true
    }

    if (message.type === 'UPDATE_SETTINGS') {
      void saveSettings({ ...storedSettings, ...message.patch })
        .then(() => sendResponse({ ok: true, status: controller?.getStatus() ?? status, settings: storedSettings }))
        .catch((error: unknown) => sendResponse({
          ok: false,
          error: error instanceof Error ? error.message : '설정을 저장하지 못했습니다.',
        }))
      return true
    }
  },
)

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName !== 'local' || !changes[SETTINGS_KEY]?.newValue) return
  storedSettings = normalizeSettings(changes[SETTINGS_KEY].newValue)
  controller?.updateSettings(settingsForHost(storedSettings, location.hostname))
})

void initialize()

