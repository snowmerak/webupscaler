import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { WgslReflect } from 'wgsl_reflect/wgsl_reflect.module.js'

const shaders = [
  { file: 'analyze.wgsl', compute: ['main'], bindings: 4 },
  { file: 'motion.wgsl', compute: ['main'], bindings: 8 },
  { file: 'reconstruct.wgsl', compute: ['main'], bindings: 8 },
  { file: 'composite.wgsl', vertex: ['vertexMain'], fragment: ['fragmentMain'], bindings: 3 },
]

for (const expectation of shaders) {
  const path = resolve('src/gpu/shaders', expectation.file)
  const source = await readFile(path, 'utf8')
  const reflection = new WgslReflect(source)
  const entries = {
    compute: reflection.entry.compute.map((entry) => entry.name),
    vertex: reflection.entry.vertex.map((entry) => entry.name),
    fragment: reflection.entry.fragment.map((entry) => entry.name),
  }

  for (const stage of ['compute', 'vertex', 'fragment']) {
    for (const name of expectation[stage] ?? []) {
      if (!entries[stage].includes(name)) {
        throw new Error(`${expectation.file}: missing ${stage} entry point ${name}`)
      }
    }
  }

  const bindingCount = reflection.getBindGroups()
    .flat()
    .filter(Boolean)
    .length
  if (bindingCount !== expectation.bindings) {
    throw new Error(
      `${expectation.file}: expected ${expectation.bindings} bindings, found ${bindingCount}`,
    )
  }

  console.log(`✓ ${expectation.file}: ${bindingCount} bindings`)
}

