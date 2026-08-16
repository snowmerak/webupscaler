import type { ExtensionMessage, ExtensionResponse } from './messages'

const STYLE_ID = 'webupscaler-image-highlight-style'

function isEnabled() {
  return document.getElementById(STYLE_ID) !== null
}

function toggleImageHighlights() {
  const currentStyle = document.getElementById(STYLE_ID)

  if (currentStyle) {
    currentStyle.remove()
    return false
  }

  const style = document.createElement('style')
  style.id = STYLE_ID
  style.textContent = `
    img {
      outline: 3px solid #6d5dfc !important;
      outline-offset: 3px !important;
      border-radius: 2px;
    }
  `
  document.documentElement.append(style)
  return true
}

chrome.runtime.onMessage.addListener(
  (message: ExtensionMessage, _sender, sendResponse: (response: ExtensionResponse) => void) => {
    if (message.type === 'GET_HIGHLIGHT_STATE') {
      sendResponse({ ok: true, enabled: isEnabled() })
    }

    if (message.type === 'TOGGLE_IMAGE_HIGHLIGHTS') {
      sendResponse({ ok: true, enabled: toggleImageHighlights() })
    }
  },
)

