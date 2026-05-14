ESTIMATION_EXAMPLES: list[dict[str, str]] = [
    {
        "meeting_summary": (
            "El cliente necesita una plataforma web de gestión de inventario "
            "con roles, alertas de stock bajo y exportación a Excel."
        ),
        "estimation": """
## Estimación: Plataforma de Gestión de Inventario

### Desglose de tareas:
1. Diseño UI/UX: 40 horas
2. Backend API (CRUD inventario): 60 horas
3. Autenticación y roles: 20 horas
4. Dashboard con métricas: 30 horas
5. Testing y QA: 25 horas

**Total estimado: 175 horas**
**Equipo recomendado: 2 full-stack + 1 UX (part-time)**
**Duración estimada: 6–8 semanas**
""".strip(),
    },
    {
        "meeting_summary": (
            "Startup quiere un MVP de app móvil para reservas de citas con "
            "notificaciones push, pago con Stripe y panel admin básico."
        ),
        "estimation": """
## Estimación: MVP Reservas + Pagos

### Desglose de tareas:
1. Diseño flujos y pantallas clave: 24 horas
2. Backend (usuarios, citas, disponibilidad): 45 horas
3. Integración Stripe (checkout web / deep link): 16 horas
4. App móvil (2 plataformas, flujo principal): 70 horas
5. Panel admin mínimo: 20 horas
6. Push notifications + hardening: 15 horas

**Total estimado: 190 horas**
**Equipo recomendado: 1 mobile + 1 backend + 0.5 diseño**
**Duración estimada: 8–10 semanas (MVP)**
""".strip(),
    },
]
