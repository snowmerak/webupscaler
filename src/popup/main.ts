import type { ExtensionMessage, ExtensionResponse } from '../messages'
import './style.css'

function getRequiredElement<T extends Element>(selector: string) {
  const element = document.querySelector<T>(selector)
  if (!element) throw new Error(`Popup UI element is missing: ${selector}`)
  return element
}

const toggleButton = getRequiredElement<HTMLButtonElement>('#toggle')
const status = getRequiredElement<HTMLParagraphElement>('#status')

async function getActiveTabId() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
  return tab?.id
}

async function sendToActiveTab(message: ExtensionMessage) {
  const tabId = await getActiveTabId()
  if (tabId === undefined) throw new Error('활성 탭을 찾지 못했습니다.')
  return chrome.tabs.sendMessage(tabId, message) as Promise<ExtensionResponse>
}

function render(enabled: boolean) {
  toggleButton.classList.toggle('active', enabled)
  toggleButton.querySelector('span')!.textContent = enabled
    ? '이미지 표시 해제'
    : '페이지 이미지 표시'
  status.textContent = enabled
    ? '페이지의 이미지가 강조 표시되고 있습니다.'
    : '현재 페이지의 이미지를 감지할 수 있습니다.'
}

async function initialize() {
  try {
    const response = await sendToActiveTab({ type: 'GET_HIGHLIGHT_STATE' })
    render(Boolean(response.enabled))
  } catch {
    status.textContent = '이 탭에서는 실행할 수 없습니다. 일반 웹 페이지를 열어 주세요.'
    toggleButton.disabled = true
  }
}

toggleButton.addEventListener('click', async () => {
  toggleButton.disabled = true

  try {
    const response = await sendToActiveTab({ type: 'TOGGLE_IMAGE_HIGHLIGHTS' })
    render(Boolean(response.enabled))
  } catch {
    status.textContent = '페이지를 새로고침한 뒤 다시 시도해 주세요.'
  } finally {
    toggleButton.disabled = false
  }
})

void initialize()
