import type { SiteAdapter } from './adapter'
import { isUsableVideo } from './adapter'

export class SoopAdapter implements SiteAdapter {
  readonly id = 'soop' as const
  readonly playerLabel = 'SOOP LivePlayer'

  matches(url: URL): boolean {
    return url.hostname === 'play.sooplive.com'
      || url.hostname === 'play.sooplive.co.kr'
  }

  findVideos(root: ParentNode): HTMLVideoElement[] {
    const primary = root.querySelector<HTMLVideoElement>('video#livePlayer')
    const candidates = Array.from(
      root.querySelectorAll<HTMLVideoElement>('#videoLayer video, #player video, video.af_video'),
    )
    const unique = new Set<HTMLVideoElement>()

    if (primary) unique.add(primary)
    for (const video of candidates) unique.add(video)

    return Array.from(unique)
      .filter(isUsableVideo)
      .sort((a, b) => {
        if (a.id === 'livePlayer') return -1
        if (b.id === 'livePlayer') return 1
        const aRect = a.getBoundingClientRect()
        const bRect = b.getBoundingClientRect()
        return bRect.width * bRect.height - aRect.width * aRect.height
      })
  }

  getOverlayHost(video: HTMLVideoElement): HTMLElement {
    return video.closest<HTMLElement>('#videoLayer')
      ?? video.parentElement
      ?? document.body
  }
}
