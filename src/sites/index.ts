import type { SiteAdapter } from './adapter'
import { SoopAdapter } from './soop'

const adapters: SiteAdapter[] = [new SoopAdapter()]

export function resolveSiteAdapter(url = new URL(location.href)): SiteAdapter | null {
  return adapters.find((adapter) => adapter.matches(url)) ?? null
}

