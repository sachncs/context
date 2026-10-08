---
title: "Errors"
description: "The exception hierarchy, which errors are retried, and what to catch at your service boundary."
---

Every deliberate error derives from `FoveateError`; catch that at the boundary of your service.

| Error | When | Retried? |
|---|---|---|
| `ConfigError` | A bad setting, unknown method name or missing optional dependency | No |
| `ValidationError` | Bad input or a malformed model reply (not JSON, no choices, no text after recovery) | No |
| `BudgetExceededError` | A context cannot be compressed under its budget (with `Overflow.RAISE`) | No |
| `CompressionError` | A compression step failed; carries the step name and cause | No |
| `BackendError` | Base for the errors below | |
| `TransientBackendError` | Timeouts, connection errors, HTTP 408, 409 and 5xx | Yes, with backoff and jitter |
| `RateLimitError` | HTTP 429; honours `Retry-After` | Yes |
| `BackendTimeoutError` | A call exceeded its timeout | Yes |
| `CircuitOpenError` | The circuit breaker is open after repeated failures | Not until the cool-down ends |
| `PermanentBackendError` | Authentication, bad request, any other 4xx | Never |

```python
from foveate import errors

try:
    answer = foveator.ask(question, [report])
except errors.RateLimitError:
    ...        # back off further, or queue
except errors.FoveateError as exc:
    ...        # log and show a generic failure
```

Compression and answering degrade where they safely can: a failed model call in a `a|b` method falls
back to `b`; a failed query expansion falls back to the original question; a history compressor falls
back to an offline method rather than aborting the agent run.
