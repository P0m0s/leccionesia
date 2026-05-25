# Stress Test del CAG — REPORT

## Decisión de captura de datos

**Estrategia elegida: respuesta HTTP directa** (opción B: snapshot enriquecido vía la respuesta de `POST /sessions/{id}/estimate` + `GET /sessions/{id}`).

**Justificación:**

1. La respuesta de `/estimate` ya incluye `metrics` (tokens, coste, latencia) y `project_metadata` (summary, tecnologías, nombre del proyecto) — exactamente lo que necesitan las métricas.
2. Parsear logs de structlog requeriría configurar un sink custom o leer stdout, lo que añade complejidad y fragilidad al runner sin beneficio funcional.
3. El evento `turn_observed` queda disponible para observabilidad en producción (dashboards, alertas) sin acoplar el runner de evals a la implementación de logging.

---

## 1. Tabla resumen (por escenario × tamaño de adjunto)

Ejecución: 3 escenarios × att=0KB × 1 repetición = 50 turnos.

| Escenario      | Adjunto (KB) | P50 Latency (ms) | P95 Latency (ms) | Coste Total (USD) | Cache Hit Rate | Recall Fact-Tracker |
|----------------|:------------:|:-----------------:|:-----------------:|:------------------:|:--------------:|:-------------------:|
| growing        | 0            | 9 927             | 25 717            | $0.013206          | 0%             | 10.0%               |
| pivot          | 0            | 7 235             | 15 149            | $0.008601          | 0%             | 46.7%               |
| contradiction  | 0            | 4 309             | 11 825            | $0.007150          | 0%             | 13.3%               |

**Cache Hit Rate** = 0% en todos los escenarios. El flujo conversacional no aplica caché por turno porque cada turno depende del historial acumulado.

**Recall del Fact-Tracker**: porcentaje de turnos donde el fact clave se encontró (case-insensitive) en `conversation_summary`, metadata o anchors del `ProjectMetadata`.

---

## 2. Curvas (en tabla)

### 2.1 Latencia vs Tokens (input)

| Rango tokens_in | Turnos | P50 Latency (ms) | P95 Latency (ms) | Media Latency (ms) |
|:----------------:|:------:|:-----------------:|:-----------------:|:-------------------:|
| 1 500–2 500      | 16     | 5 210             | 15 149            | 6 289               |
| 2 500–3 000      | 22     | 7 235             | 10 119            | 6 923               |
| 3 000–3 500      | 10     | 10 575            | 14 352            | 10 656              |
| 3 500+           | 2      | 25 717            | 25 717            | 23 005              |

La latencia escala de forma supralineal con los tokens de entrada: al pasar de ~2 000 a ~3 500 tokens, la latencia media se multiplica por 1.7×, y por encima de 3 500 tokens se dispara a 23s.

### 2.2 Coste acumulado vs Turno

| Turno | Coste acum. growing (USD) | Coste acum. pivot (USD) | Coste acum. contradiction (USD) |
|:-----:|:-------------------------:|:-----------------------:|:-------------------------------:|
| 1     | $0.000316                 | $0.000666               | $0.000313                       |
| 5     | $0.002290                 | $0.002526               | $0.002229                       |
| 10    | $0.005425                 | $0.005394               | $0.004648                       |
| 15    | $0.009021                 | $0.008601               | $0.007150                       |
| 20    | $0.013206                 | —                       | —                               |

El coste crece de forma prácticamente lineal: cada turno añade ~$0.0005–$0.0009. El escenario `growing` (20 turnos) acumula $0.013, equivalente a ~15 000 llamadas por dólar con `gpt-4o-mini`.

### 2.3 MemoryDrift vs N (turno)

| Turno | Drift growing | Drift pivot | Drift contradiction |
|:-----:|:-------------:|:-----------:|:-------------------:|
| 1     | 0.0 (Aurora)  | 0.0 (FlashRun) | 0.0 (EduPlatform) |
| 2     | 1.0 (FastAPI) | 1.0 (React Native) | 1.0 (Next.js) |
| 3     | 0.0 (OAuth2)  | 1.0 (Node.js) | 0.0 (30000) |
| 5     | 1.0 (Stripe)  | 1.0 (Flutter) | 0.0 (Moodle) |
| 8     | 0.0 (WebSockets) | 0.0 (MercadoPago) | 0.0 (80000) |
| 10    | 0.0 (SAP)     | 1.0 (Flutter) | 0.0 (IA recomendación) |
| 13    | 0.0 (aprob. multi-nivel) | 0.0 (WhatsApp) | 0.0 (80000) |
| 15    | 0.0 (API pública) | 1.0 (Flutter) | 0.0 (WCAG 2.1) |
| 20    | 0.0 (Aurora)  | —           | —                   |

---

## 3. Lectura de resultados

### ¿A partir de qué turno rompe mi CAG?

El CAG empieza a perder hechos granulares a partir del **turno 4**, cuando la ventana deslizante (`MAX_TURNS=6` mensajes) comienza a descartar los primeros pares user/assistant. En el escenario `growing`, el nombre del proyecto "Aurora" (turno 1) **no sobrevive** al turno 20: ni la heurística regex lo captura en `project_name`, ni el `conversation_summary` lo preserva — score de drift = 0.0 en el turno 20. Solo los facts que coinciden con tecnologías del catálogo regex (`FastAPI`, `Stripe`) se retienen vía `mentioned_technologies`.

En el escenario `pivot`, Flutter sobrevive porque se refuerza explícitamente en los turnos 5, 10 y 15, lo que eleva el recall a 46.7%. Sin ese refuerzo, el recall cae al nivel de `growing` (10%). En `contradiction`, el presupuesto de 80 000€ (turno 8) **nunca se preserva** — ni en el turno 13 que lo pregunta directamente: drift = 0.0. El `conversation_summary` generado automáticamente generaliza "se discutió presupuesto" sin retener la cifra exacta. Esto confirma que a partir del turno N=4 el recall de facts no estructurados cae bajo el 20%, y a partir del turno N=10 es prácticamente 0% para facts que no estén en el catálogo de la heurística regex.

### ¿Qué dimensión domina la degradación?

La dimensión dominante es la **pérdida de memoria** (memory drift), no la latencia ni el coste. La latencia crece de 6s (turno 1) a 25s (turno 20) — un factor 4× — pero sigue siendo funcional. El coste es lineal y bajo ($0.0007/turno con `gpt-4o-mini`). En cambio, el recall del fact-tracker cae de ~50% en los primeros 5 turnos a **0% a partir del turno 10** en `growing` y `contradiction`. El `conversation_summary` actúa como un embudo lossy: retiene temas generales pero pierde nombres propios, cifras y decisiones de stack.

El coste del turno 20 multiplica por 2.2× el del turno 1 ($0.000886 vs $0.000316), impulsado por el crecimiento del `conversation_summary` (de 0 a ~750 chars) que infla los tokens de entrada de 1 637 a 3 572.

### ¿Qué justificaría saltar a RAG?

Tres señales concretas de este stress test justifican el salto:

1. **MemoryDrift < 0.5 desde el turno 4** — el 90% de los facts del escenario `growing` se pierden. Un retriever con embeddings sobre el historial completo (no solo la ventana) recuperaría facts por similitud semántica sin depender del resumen lossy.
2. **Contradicciones no resueltas** — el presupuesto de 80k€ no reemplaza al de 30k€ en ningún campo del `ProjectMetadata`. El CAG no tiene mecanismo de "invalidación" de hechos obsoletos; un vector store con timestamps permitiría dar preferencia a la versión más reciente.
3. **Latencia > 10s a partir de 3 000 tokens de entrada** — con adjuntos de 50-100KB, los tokens de entrada saltarían a 5 000-10 000, multiplicando latencia y coste. Un retriever que seleccione los fragmentos relevantes del adjunto (en vez de inyectar todo el texto) sería más eficiente.
