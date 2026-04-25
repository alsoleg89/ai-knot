# Cognitive Memory Model — ai-knot as Psychological Memory System

## Date: 2026-04-10

---

## The Full Model

ai-knot уже реализует части когнитивной модели памяти. Strand добавляет недостающий слой.
Дальше — rescripting.

### Уровень 1: ЗАПОМИНАНИЕ (existing)
- `kb.add()` / `kb.learn()` — encoding, как мозг формирует новые воспоминания
- Facts = episodic traces (конкретные события с контекстом)
- Extraction = consolidation (переход из рабочей памяти в долговременную)
- Entity/attribute/slot = schema formation (структурирование опыта)

### Уровень 2: ЗАБЫВАНИЕ (existing)
- Ebbinghaus decay curves — retention ослабевает со временем
- `valid_from` / `valid_until` — temporal validity windows
- `access_count` / `access_intervals` — spacing effect (чаще обращаешься → крепче помнишь)
- Episodic TTL — эпизодические воспоминания угасают быстрее семантических

This maps to cognitive psychology's forgetting curve + spacing effect.

### Уровень 3: ВОСПОМИНАНИЕ / ASSOCIATION (NEW — Strand)
- Strand = associative memory network (семантическая сеть связей)
- Co-occurrence = Hebbian learning ("neurons that fire together wire together")
- Query expansion from Strand = spreading activation (активация одного концепта активирует связанные)
- Intent-dependent expansion depth = attentional control (фокус внимания определяет глубину активации)
- Temporal decay on Strand = synaptic pruning (неиспользуемые связи ослабевают)

This is the layer that enables RECALL — not just storage, but the ability to
reconstruct knowledge from partial cues. When you hear "Melanie" and remember
"pottery, camping, swimming" — that's spreading activation through your
associative network. Strand does exactly this.

### Уровень 4: РЕСКРИПТИНГ / REWRITING (FUTURE)
Rescripting (schema therapy, EMDR) — процесс изменения ИНТЕРПРЕТАЦИИ
существующих воспоминаний без изменения самих фактов.

В контексте ai-knot:

**Что хранится**: факт "User lost their job at Acme"
**Что меняется при рескриптинге**: ассоциации и контекст вокруг этого факта

Примеры:

1. **Reframing** — изменение контекста:
   - Before: "lost job" associates with {failure, problem, crisis}
   - After rescripting: "lost job" associates with {new opportunity, career change, growth}
   - Implementation: modify Strand weights without changing facts
   - The FACT is the same. The MEANING (associations) changes.

2. **Consolidation rescripting** — пересборка связей:
   - User learns new facts that change the significance of old facts
   - "User lost job at Acme" + later "User started successful startup"
   - Strand naturally updates: "acme" now co-occurs with "startup", "success"
   - Old negative associations get overwritten by new co-occurrences

3. **Schema update** — когда новая информация меняет структуру знания:
   - Old schema: User is a banker → {finance, trading, wall_street}
   - New info: User is now a dance instructor → {dance, studio, teaching}
   - Strand reflects the shift: co-occurrence weights shift over time
   - Ebbinghaus decay weakens old associations, new ones strengthen

4. **Selective strengthening** — терапевтический приём усиления позитивных связей:
   - Repeated recall of certain facts strengthens their Strand associations
   - `access_count` increases → co-occurrence weights for those tokens increase
   - This is literally how EMDR works: repeated processing of a memory
     changes its emotional associations

### Уровень 5: МЕТАКОГНИЦИЯ (FUTURE FUTURE)
- The system monitors its own memory quality
- Detects when Strand associations are inconsistent with stored facts
- Triggers re-consolidation (rebuild Strand from current facts)
- Self-correction of false associations

---

## Mapping to Cognitive Psychology

| Psychological concept | ai-knot implementation | Status |
|----------------------|----------------------|--------|
| Encoding | `kb.learn()` / `kb.add()` | Existing |
| Consolidation | Extraction → Fact storage | Existing |
| Forgetting curve | Ebbinghaus decay | Existing |
| Spacing effect | `access_intervals` | Existing |
| Episodic memory | Individual facts with timestamps | Existing |
| Semantic memory | **Strand** (co-occurrence patterns) | **NEW** |
| Spreading activation | Strand query expansion | **NEW** |
| Hebbian learning | Co-occurrence counting | **NEW** |
| Synaptic pruning | Strand temporal decay | **NEW** |
| Attentional control | Intent-dependent expansion depth | **NEW** |
| Rescripting / reframing | Strand weight modification | **FUTURE** |
| Schema therapy | Strand rebuild on schema change | **FUTURE** |
| EMDR | Repeated recall → association strengthening | **FUTURE** |
| Metacognition | Self-monitoring of memory quality | **FUTURE** |

---

## Why This Matters for Positioning

### Current narrative: "ai-knot has forgetting curves"
- True but narrow
- Forgetting is one aspect of memory
- Competitors can copy Ebbinghaus decay in a weekend

### New narrative: "ai-knot is a cognitive memory system"
- Encoding → Consolidation → Forgetting → Association → Rescripting
- Full cognitive cycle, not just storage + retrieval
- Strand is the KEY missing piece that makes it a SYSTEM, not a database

### Competitive moat
- Mem0: storage + retrieval (no forgetting, no association, no rescripting)
- Letta: conversation management (not memory)
- Zep: knowledge graph (static, no forgetting, no association dynamics)
- Hindsight: spreading activation on curated graph (close but no forgetting, no rescripting)
- memvid: video storage + search (no memory model at all)
- **ai-knot + Strand**: encoding + forgetting + association + (future) rescripting

Nobody has the full cognitive stack.

---

## Strand as the Bridge

Without Strand, ai-knot has:
- Good encoding (learn)
- Good forgetting (Ebbinghaus)
- Poor recall (flat BM25 search)

With Strand, ai-knot has:
- Good encoding
- Good forgetting
- **Good recall** (associative expansion → diverse retrieval)

Strand is the missing piece between "memory storage" and "memory system".
It's what turns a database into something that resembles how memory actually works.

And it opens the door to rescripting — because once you have an explicit
associative structure, you can MODIFY the associations without touching the facts.
That's exactly what therapy does: change the meaning, not the memory.

---

## Implementation Priority

1. **Strand core** — binary structure, build/query/update
2. **Integration** — plug into recall pipeline as query expansion
3. **Benchmark** — measure LOCOMO improvement
4. **Decay integration** — link Strand decay to Ebbinghaus curves
5. **Rescripting API** — `kb.rescript(topic, new_associations)`
6. **Marketing** — "ai-knot: the first cognitive memory system for LLM agents"
