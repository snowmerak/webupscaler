import type { ExtensionMessage, ExtensionResponse } from './messages'

chrome.runtime.onInstalled.addListener(({ reason }) => {
  if (reason === 'install') {
    console.info('Web Upscaler installed and ready.')
  }
})

chrome.runtime.onMessage.addListener(
  (message: ExtensionMessage, _sender, sendResponse: (response: ExtensionResponse) => void) => {
    if (message.type === 'PING') {
      sendResponse({ ok: true, message: 'Background service worker is running.' })
    }
  },
)

