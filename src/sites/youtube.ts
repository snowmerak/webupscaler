import type { SiteAdapter } from './adapter'
import { isUsableVideo } from './adapter'

const YOUTUBE_HOSTS = new Set([
  'youtube.com',
  'www.youtube.com',
  'm.youtube.com',
  'www.youtube-nocookie.com',
])

export class YouTubeAdapter implements SiteAdapter {
  readonly id = 'youtube' as const
  readonly playerLabel = 'YouTube 플레이어'
  readonly navigationEvents = ['yt-navigate-finish'] as const

  matches(url: URL): boolean {
    return YOUTUBE_HOSTS.has(url.hostname)
  }

  findVideos(root: ParentNode): HTMLVideoElement[] {
    const candidates = Array.from(root.querySelectorAll<HTMLVideoElement>([
      '#movie_player video',
      '#shorts-player video',
      'video.html5-main-video',
      'ytd-player video',
    ].join(', ')))

    return Array.from(new Set(candidates))
      .filter(isUsableVideo)
      .sort((a, b) => this.videoScore(b) - this.videoScore(a))
  }

  getOverlayHost(video: HTMLVideoElement): HTMLElement {
    return video.closest<HTMLElement>('.html5-video-container')
      ?? video.closest<HTMLElement>('#movie_player, #shorts-player, .html5-video-player')
      ?? video.parentElement
      ?? document.body
  }

  private videoScore(video: HTMLVideoElement): number {
    const rect = video.getBoundingClientRect()
    const mainPlayerBonus = video.closest('#movie_player, #shorts-player') ? 1_000_000_000 : 0
    const mainVideoBonus = video.classList.contains('html5-main-video') ? 100_000_000 : 0
    return mainPlayerBonus + mainVideoBonus + rect.width * rect.height
  }
}
