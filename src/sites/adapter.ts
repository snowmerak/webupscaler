export interface SiteAdapter {
  readonly id: 'soop'
  matches(url: URL): boolean
  findVideos(root: ParentNode): HTMLVideoElement[]
  getOverlayHost(video: HTMLVideoElement): HTMLElement
}

export function isUsableVideo(video: HTMLVideoElement): boolean {
  const style = getComputedStyle(video)
  const rect = video.getBoundingClientRect()

  return video.id !== 'pipMedia'
    && style.display !== 'none'
    && style.visibility !== 'hidden'
    && Number.parseFloat(style.opacity || '1') > 0
    && rect.width > 0
    && rect.height > 0
}

