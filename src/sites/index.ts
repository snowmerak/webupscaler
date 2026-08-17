import type { SiteAdapter } from './adapter'
import { SoopAdapter } from './soop'
import { YouTubeAdapter } from './youtube'

const adapters: SiteAdapter[] = [new SoopAdapter(), new YouTubeAdapter()]

export function resolveSiteAdapter(url = new URL(location.href)): SiteAdapter | null {
  return adapters.find((adapter) => adapter.matches(url)) ?? null
}
