# Vercel AI SDK case study / proof asset

Updated: **July 1, 2026**

Use this file when you want one concrete `ai-knot` integration story that
starts from a mainstream TypeScript app stack rather than from Python. The
Vercel AI SDK is a strong follow-up surface because its `system` / `messages`
shape is already familiar to app builders, and `ai-knot` can slot into that
surface without taking over the model/runtime layer.

Official references:

- Vercel AI repo: https://github.com/vercel/ai
- Vercel AI SDK docs: https://ai-sdk.dev

As of **July 1, 2026**, the official `vercel/ai` repo shows **25,279 GitHub
stars**. That makes it a real app-builder acquisition channel, not just a niche
TypeScript side path.

---

## The angle

Do not pitch this as "a new framework." Pitch it as:

> **Keep your AI SDK app flow, add deterministic long-term memory.**

That means:

- your existing `generateText()` / `streamText()` code stays in place,
- ai-knot fills the exact `system` or `messages` surface the app already uses,
- model choice, streaming, and UI routing stay inside your own AI SDK code.

The hook is not "another SDK." The hook is that the app developer can keep the
same TypeScript ergonomics and gain persistence underneath.

---

## Fastest proof paths

### Repo-native surface proof

Inspect the repo-native example in:

```text
npm/examples/vercel-ai-sdk-surface.ts
```

What it proves:

- `AiKnotAISDKMemory` builds the exact `system` surface now,
- recall is deterministic and local,
- the integration can be inspected before wiring a real model call.

### Real Vercel AI SDK wiring

```bash
npm install ai-knot ai @ai-sdk/openai
OPENAI_API_KEY=... node npm/examples/vercel-ai-sdk.ts
```

What it proves:

- the named adapter plugs into a real AI SDK app flow,
- ai-knot handles recall while the SDK keeps the runtime/model layer,
- the TypeScript path is explicit, not generic.

---

## What to emphasize

### Problem

TypeScript app builders want persistent memory without moving to a Python-first
stack or replaying too much chat history into every `system` prompt.

### What ai-knot adds

- deterministic recalled facts for the current user input,
- self-hosted storage behind one npm package path,
- `system` / `messages` helpers that fit AI SDK apps directly,
- no LLM on the retrieval path,
- a local-first proof before any real model call.

### What not to claim

- Do not say it replaces the Vercel AI SDK.
- Do not overfocus on MCP or Python when talking to TS builders.
- Do not make this about benchmarks first; lead with the app surface.

---

## Copy blocks

### GitHub discussion / follow-up comment

> A concrete TypeScript app surface that is ready today: Vercel AI SDK.
>
> `ai-knot` now has a named `AiKnotAISDKMemory` adapter that fills the same
> `system` / `messages` surface AI SDK apps already use, so you can keep your
> app runtime and add deterministic long-term memory underneath.
>
> Fastest proof:
> - `npm/examples/vercel-ai-sdk-surface.ts` for the repo-native surface proof
> - `npm/examples/vercel-ai-sdk.ts` for the real model-wiring path

### X / LinkedIn

> Vercel AI SDK path for `ai-knot` is ready.
>
> `AiKnotAISDKMemory` builds the exact `system` / `messages` surface AI SDK apps
> already use, with deterministic recalled facts and no LLM on the read path.
>
> Repo-native surface proof: `npm/examples/vercel-ai-sdk-surface.ts`
> Full wiring: `npm/examples/vercel-ai-sdk.ts`
>
> https://github.com/alsoleg89/ai-knot

### Reddit comment or reply

> If you want the TypeScript app path instead of a Python adapter, start with
> the Vercel AI SDK surface. The named adapter now builds the `system` /
> `messages` inputs directly, and `npm/examples/vercel-ai-sdk-surface.ts`
> lets you inspect the local proof before any model call.

---

## Recommended CTA

Lead with one of these, in order:

1. `npm/examples/vercel-ai-sdk-surface.ts`
2. `npm/examples/vercel-ai-sdk.ts`
3. [../npm/README.md](../npm/README.md)

Do not send TypeScript app builders to Python docs first. Send them to the app-shaped proof.
