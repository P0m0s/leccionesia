"""
Escenarios sintéticos multi-turno para stress test del CAG.

Cada perfil expone:
- `scenario_name`: identificador del escenario.
- `turns`: lista de tuplas ``(turn_index, transcript, fact_to_remember)``.
- `fact_tracker`: dict que mapea turn_index → fact clave que debería sobrevivir.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StressScenario:
    scenario_name: str
    turns: list[tuple[int, str, str]]
    fact_tracker: dict[int, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Proyecto que crece — requisitos se acumulan turno a turno
# ---------------------------------------------------------------------------

_GROWING_TURNS: list[tuple[int, str, str]] = [
    (1, "Quiero construir un proyecto llamado Aurora. Es una plataforma SaaS de gestión de gastos.", "Aurora"),
    (2, "Usaremos Python con FastAPI para el backend y PostgreSQL como base de datos.", "FastAPI"),
    (3, "Necesitamos autenticación con OAuth2 y roles (admin, manager, viewer).", "OAuth2"),
    (4, "Añade un módulo de reportes con gráficos exportables a PDF.", "reportes PDF"),
    (5, "Integración con Stripe para pagos recurrentes mensuales.", "Stripe"),
    (6, "El equipo es de 4 personas: 2 backend, 1 frontend, 1 QA.", "4 personas"),
    (7, "Añade notificaciones push vía Firebase Cloud Messaging.", "Firebase"),
    (8, "Necesitamos un dashboard en tiempo real con WebSockets.", "WebSockets"),
    (9, "Añade soporte multi-idioma (i18n) con al menos español e inglés.", "i18n"),
    (10, "Integración con SAP para sincronizar facturas.", "SAP"),
    (11, "Módulo de auditoría que registre cada acción del usuario.", "auditoría"),
    (12, "Añade un chatbot interno con IA para consultas de gastos.", "chatbot IA"),
    (13, "Necesitamos un workflow de aprobación multi-nivel configurable.", "aprobación multi-nivel"),
    (14, "Soporte para adjuntos (facturas escaneadas) con OCR automático.", "OCR"),
    (15, "API pública documentada con OpenAPI para integraciones de terceros.", "API pública"),
    (16, "Módulo de presupuestos con alertas al 80% y 100% del límite.", "presupuestos alertas"),
    (17, "Añade tests E2E con Playwright y cobertura mínima del 80%.", "Playwright"),
    (18, "Despliegue en AWS con Terraform y CI/CD en GitHub Actions.", "Terraform AWS"),
    (19, "Necesitamos SSO con Azure AD para clientes enterprise.", "SSO Azure AD"),
    (20, "El nombre del proyecto sigue siendo Aurora. Confirma el alcance total.", "Aurora"),
]

_GROWING_TRACKER = {1: "Aurora", 5: "Stripe", 10: "SAP", 15: "API pública", 20: "Aurora"}

GROWING = StressScenario(
    scenario_name="growing",
    turns=_GROWING_TURNS,
    fact_tracker=_GROWING_TRACKER,
)

# ---------------------------------------------------------------------------
# 2. Proyecto que pivota — cambio de stack en turno 5
# ---------------------------------------------------------------------------

_PIVOT_TURNS: list[tuple[int, str, str]] = [
    (1, "Vamos a crear una app móvil de delivery llamada FlashRun.", "FlashRun"),
    (2, "Empezaremos con React Native para iOS y Android.", "React Native"),
    (3, "Backend en Node.js con Express y MongoDB.", "Node.js"),
    (4, "Integración con Google Maps para tracking en tiempo real.", "Google Maps"),
    (5, "CAMBIO DE DIRECCIÓN: Abandonamos React Native. Pasamos a Flutter con Dart.", "Flutter"),
    (6, "El backend ahora será en Python con Django REST Framework.", "Django"),
    (7, "Cambiamos MongoDB por PostgreSQL con PostGIS para geolocalización.", "PostgreSQL PostGIS"),
    (8, "Añade sistema de pagos con MercadoPago en lugar de Stripe.", "MercadoPago"),
    (9, "Módulo de gestión de repartidores con asignación automática.", "asignación automática"),
    (10, "Recuerda: ya NO usamos React Native ni Node.js. La tecnología activa es Flutter + Django.", "Flutter"),
    (11, "Añade un panel admin web en Angular para operadores.", "Angular"),
    (12, "Sistema de calificaciones y reseñas para repartidores y restaurantes.", "calificaciones"),
    (13, "Integración con WhatsApp Business API para notificaciones.", "WhatsApp"),
    (14, "Módulo de cupones y programa de lealtad.", "cupones lealtad"),
    (15, "Confirma: ¿cuál es la tecnología activa del frontend móvil?", "Flutter"),
]

_PIVOT_TRACKER = {2: "React Native", 5: "Flutter", 10: "Flutter", 15: "Flutter"}

PIVOT = StressScenario(
    scenario_name="pivot",
    turns=_PIVOT_TURNS,
    fact_tracker=_PIVOT_TRACKER,
)

# ---------------------------------------------------------------------------
# 3. Proyecto que se contradice — presupuesto cambia
# ---------------------------------------------------------------------------

_CONTRADICTION_TURNS: list[tuple[int, str, str]] = [
    (1, "Proyecto EduPlatform: plataforma e-learning para universidades.", "EduPlatform"),
    (2, "Stack: Next.js frontend, Python FastAPI backend, PostgreSQL.", "Next.js"),
    (3, "El presupuesto máximo es de 30.000 euros. No podemos pasarnos.", "30000"),
    (4, "Módulos: cursos en vídeo, exámenes online, foro de discusión.", "cursos vídeo"),
    (5, "Integración con Moodle para importar cursos existentes.", "Moodle"),
    (6, "Necesitamos videoconferencia integrada con Jitsi.", "Jitsi"),
    (7, "Sistema de certificados digitales con verificación blockchain.", "blockchain"),
    (8, "ACTUALIZACIÓN: El presupuesto ha subido a 80.000 euros tras aprobación del comité.", "80000"),
    (9, "Con el nuevo presupuesto, añade app móvil nativa en Swift y Kotlin.", "Swift Kotlin"),
    (10, "Añade IA para recomendación personalizada de cursos.", "IA recomendación"),
    (11, "Gamificación: badges, leaderboards, racha de estudio.", "gamificación"),
    (12, "Integración con pasarelas de pago para cursos premium.", "pasarela pago"),
    (13, "¿Cuál es el presupuesto vigente del proyecto?", "80000"),
    (14, "Analytics avanzado para profesores: engagement, tasas de aprobación.", "analytics"),
    (15, "Accesibilidad WCAG 2.1 nivel AA en toda la plataforma.", "WCAG 2.1"),
]

_CONTRADICTION_TRACKER = {3: "30000", 8: "80000", 13: "80000"}

CONTRADICTION = StressScenario(
    scenario_name="contradiction",
    turns=_CONTRADICTION_TURNS,
    fact_tracker=_CONTRADICTION_TRACKER,
)

# ---------------------------------------------------------------------------
# Registro global
# ---------------------------------------------------------------------------

ALL_SCENARIOS: dict[str, StressScenario] = {
    "growing": GROWING,
    "pivot": PIVOT,
    "contradiction": CONTRADICTION,
}
