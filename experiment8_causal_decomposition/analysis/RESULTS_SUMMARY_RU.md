# Experiment 8 — итог реального прогона

Дата прогона: 2026-09-16.

## Целостность данных

- Модели: Qwen3-4B, Phi-3-mini-4k-instruct, Mistral-7B-Instruct-v0.3.
- Собрано 5 400 траекторий: по 1 800 на модель.
- В каждой модели: 100 задач, 50 семейств, 6 условий и 3 варианта.
- Невалидных траекторий: 0.
- Все 19 unit-тестов проходят.
- Все frozen-input SHA-256 совпадают с конфигурациями прогонов.
- Предварительная валидация: 300/300 canonical/opaque пар прошли; 945/945 deterministic execution cases прошли; tokenizer semantic-step coverage gate пройден во всех IR-derived условиях.

## Preregistered contrasts

Значения — средняя разность primary `mean_entropy_top20` и 99% task-bootstrap CI.

| Contrast | Qwen3-4B | Phi-3 mini | Mistral 7B | Межмодельный итог |
|---|---:|---:|---:|---|
| C1: procedural prose − controlled NL | 0.099901 [0.079718, 0.120241] | 0.042507 [0.022026, 0.062341] | -0.005783 [-0.031967, 0.019920] | Не реплицирован во всех моделях |
| C2: controlled NL − pseudocode | 0.071814 [0.054219, 0.089369] | 0.041733 [0.026986, 0.055823] | 0.096837 [0.079799, 0.113927] | ROBUST во всех моделях |
| C3: pseudocode − canonical Python | 0.183493 [0.161553, 0.206682] | 0.275816 [0.257954, 0.293805] | 0.373046 [0.348157, 0.398851] | ROBUST во всех моделях |
| C4: opaque Python − canonical Python | 0.089878 [0.074440, 0.105922] | 0.209187 [0.185184, 0.234131] | 0.164407 [0.142963, 0.186529] | ROBUST nonzero effect во всех моделях |

Для всех поддержанных contrasts Holm-adjusted Monte Carlo p = 0.0000199999. Для Mistral C1 Holm-adjusted p = 0.716371.

## C4 equivalence

| Модель | Frozen delta | 99% CI(C4) | Решение |
|---|---:|---:|---|
| Qwen3-4B | 0.036046 | [0.074440, 0.105922] | Не equivalent; весь CI выше +delta |
| Phi-3 mini | 0.065922 | [0.185184, 0.234131] | Не equivalent; весь CI выше +delta |
| Mistral 7B | 0.063529 | [0.142963, 0.186529] | Не equivalent; весь CI выше +delta |

Identifier-opacity intervention повышает predictive uncertainty во всех трёх моделях. Это не evidence for practical equivalence и не просто nonsignificant result.

## Средние по условиям

| Условие | Qwen3-4B | Phi-3 mini | Mistral 7B |
|---|---:|---:|---:|
| Experiment 7 NL anchor | 0.539876 | 1.000773 | 1.056317 |
| Procedural prose | 0.534622 | 0.701636 | 0.885267 |
| Controlled NL | 0.434722 | 0.659129 | 0.891050 |
| Pseudocode | 0.362907 | 0.617396 | 0.794213 |
| Canonical Python | 0.179415 | 0.341580 | 0.421167 |
| Opaque Python | 0.269292 | 0.550767 | 0.585574 |

## Вывод

Наиболее сильный и наиболее устойчивый последовательный переход — C3, от pseudocode к canonical Python. C2 также устойчиво уменьшает uncertainty во всех моделях. C1 не является универсальным: он поддержан для Qwen3 и Phi-3, но полностью не поддержан для Mistral (49/100 положительных задач, 25/50 положительных семейств, CI пересекает ноль).

C4 показывает, что замена scope-bound идентификаторов на непрозрачные имена существенно повышает uncertainty. Эффект реплицируется во всех моделях, сохраняет направление в valid-vocabulary и semantic-step-balanced метриках и превышает заранее замороженные equivalence margins.

C1–C3 следует интерпретировать только как sequential representational intervention effects. Они не изолируют одну поверхностную причину, потому что каждый переход меняет пакет свойств representation.
