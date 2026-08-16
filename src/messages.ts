export type ExtensionMessage =
  | { type: 'PING' }
  | { type: 'GET_HIGHLIGHT_STATE' }
  | { type: 'TOGGLE_IMAGE_HIGHLIGHTS' }

export type ExtensionResponse = {
  ok: boolean
  enabled?: boolean
  message?: string
}

