import type { ExtensionMessage } from './shared/messages'
import { DEFAULT_SETTINGS, normalizeSettings, SETTINGS_KEY } from './shared/settings'
import type { RuntimeStatus } from './shared/status'

async function ensureDefaultSettings() {
  const stored = await chrome.storage.local.get(SETTINGS_KEY)
  if (stored[SETTINGS_KEY] === undefined) {
    await chrome.storage.local.set({ [SETTINGS_KEY]: DEFAULT_SETTINGS })
    return
  }

  await chrome.storage.local.set({
    [SETTINGS_KEY]: normalizeSettings(stored[SETTINGS_KEY]),
  })
}

function badgeFor(status: RuntimeStatus) {
  if (!status.enabled || status.state === 'idle') return { text: '', color: '#6b7280' }
  if (status.state === 'running') return { text: '2×', color: '#635bff' }
  if (status.state === 'error') return { text: '!', color: '#d94b5b' }
  if (status.state === 'bypass') return { text: '1×', color: '#6b7280' }
  return { text: '…', color: '#c58a21' }
}

async function updateBadge(tabId: number, status: RuntimeStatus) {
  const badge = badgeFor(status)
  await Promise.all([
    chrome.action.setBadgeText({ tabId, text: badge.text }),
    chrome.action.setBadgeBackgroundColor({ tabId, color: badge.color }),
    chrome.action.setTitle({ tabId, title: `Web Upscaler — ${status.message}` }),
  ])
}

chrome.runtime.onInstalled.addListener(() => {
  void ensureDefaultSettings()
})

chrome.runtime.onStartup.addListener(() => {
  void ensureDefaultSettings()
})

chrome.runtime.onMessage.addListener((message: ExtensionMessage, sender) => {
  if (message.type === 'STATUS_UPDATE' && sender.tab?.id !== undefined) {
    void updateBadge(sender.tab.id, message.status)
  }
})

