---
area: universidad
estado: activo
actualizado: 2026-10-13
---

# Decisiones abiertas de la entrega 2

Los tres bloqueantes que `ROADMAP.md` §4 declara vencidos antes de la entrega 2.

| # | Asunto | Estado (02-sep-2026) |
|---|---|---|
| 1 | Ensemble | ⬜ **Abierto.** Hay recomendación y medición; David lo decide más adelante |
| 2 | Peso de arista | ✅ **Cerrado.** Exposición cruda, limitación declarada |
| 3 | Tesis de Wightman | ✅ PDF completo disponible (288 páginas); ⏳ verificar la adaptación de A3 contra el texto |

---

## 1. El ensemble — ⬜ ABIERTO. Recomendación: **no construirlo**

> **Estado:** David decidió posponerlo («después lo hacemos», 02-sep-2026). La recomendación y la
> medición de abajo quedan listas para cuando se retome. **No asumir que está cerrado al escribir
> la entrega 2**: si se llega a la sección de metodología sin esta decisión, la opción por defecto
> es la recomendada, y hay que decirlo explícitamente en el texto.

**El problema, como lo dejó el `HITO0` §0.3.** Koç et al. varían la topología a través de un
*ensemble* de grids sintéticos derivados del IEEE 118 y correlacionan cada métrica espectral
contra el umbral crítico de carga *entre miembros del ensemble*. Acá hay **un solo grafo
observado en el tiempo**. El `HITO0` marcó esto como «la decisión de diseño abierta más
importante».

**Por qué la reparametrización lo disuelve en buena parte.** El ensemble de Koç es necesario para
correlacionar un escalar contra un umbral crítico *a través de topologías*. Después de que el
`HITO0` §0.8c movió el centro del paper al **experimento contrafactual de remoción**, esa
correlación deja de ser el resultado principal: la comparación de los seis criterios ocurre
**dentro de cada instantánea**, sobre el mismo grafo y el mismo presupuesto $k$, y lo que se
reporta son curvas de daño. Para eso no hace falta ningún ensemble. Los 174 snapshots posteriores
a 2021 aportan la variación necesaria para mostrar que el orden entre criterios es estable.

**Cuánta variación hay realmente.** Medido sobre `output/serie_espectral.csv`, residualizando cada
invariante contra $\log N$ y $\log E$ y corrigiendo por autocorrelación
($n_{\text{ef}} = n\,(1-\rho_1)/(1+\rho_1)$):

| serie | R² contra tamaño | ρ₁ (residuo) | n observado | **n efectivo** |
|---|---|---|---|---|
| λ₁ | 0,987 | 0,135 | 174 | **132,5** |
| λ₂ | 0,455 | 0,336 | 174 | **86,5** |
| R_G | 1,000 | 0,637 | 174 | **38,6** |
| `ncomp/N` | — | 0,915 | 174 | **7,7** |

Dos lecturas, y las dos importan:

- **λ₂ tiene ~87 observaciones efectivas.** Es una muestra razonable para el claim 2 reducido
  (λ₂ contra SIS). No es un ensemble sintético, pero tampoco lo necesita.
- **`ncomp/N` tiene ~8.** Prácticamente ocho observaciones independientes en tres años. Esto
  explica por qué el t de Welch ingenuo del `HITO0` §0.8b daba t = 6,76 y el test de permutación
  circular lo mataba: no había 21 observaciones pre-evento, había una o dos. **Cualquier
  afirmación que descanse sobre la serie de fragmentación hay que reportarla con este número al
  lado.**

**Recomendación.** No construir ensemble sintético ni rewiring ad hoc. Usar los 174 snapshots
como la variación, reportando $n_{\text{ef}}$ junto a cada resultado. Si hace falta un nulo, el
correcto es el de **máxima entropía de Macchiati** (`DELTA_VENUE.md` §1) —calibrado sobre la red
real y con expectativa analítica— y no un rewiring inventado para este paper.

**Consecuencia si se acepta:** el bloqueante deja de serlo para la entrega 2 y pasa a ser una
subsección de metodología. La única deuda que queda es reportar $n_{\text{ef}}$.

---

## 2. Exposición vs. apalancamiento — ✅ CERRADO: **exposición, con la limitación declarada**

> **Decisión de David, 02-sep-2026.** Confirma lo que `entrega1.tex` §II-G ya declaraba, así que
> no hay que reescribir nada de la entrega 1.

**El problema, como lo dejó el `HITO0` §0.4 y §0.6.** Macchiati et al. señalan que la conexión
entre el radio espectral y el tamaño de pérdida pasa por la **matriz de apalancamiento**
(exposición normalizada por capital propio), no por la de exposición cruda. Y es un resultado
establecido que las pérdidas totales de capital crecen monótonamente con el radio espectral de
esa matriz.

**Por qué no es ejecutable acá.** En DeFi **el capital propio no se observa**, y TVL no lo es:
Luo et al. demostraron que está inflado por doble conteo, y es exactamente la métrica que el
paper se comprometió a no usar como peso. Los propios Macchiati escriben en su Discusión que *no
se ha diseñado un algoritmo* para estimar esa magnitud a partir de la información parcial
habitual.

**Recomendación.** Peso de arista = la **exposición del panel de DeXposure**, como ya declara
`entrega1.tex` §II-G. La matriz de apalancamiento entra al paper como **limitación explícita y
como trabajo futuro**, citando la frase de Macchiati: es la puerta que ellos dejan abierta, y
declararla es más fuerte que ignorarla o que fabricar un proxy de capital propio.

---

## 3. Tesis de Wightman — PDF completo disponible; lectura metodológica pendiente

`digitalcommons.usf.edu` está detrás de un *managed challenge* de Cloudflare: `curl` recibe 403
aunque se le pase User-Agent de navegador, cookies y referer. **La página del registro (`/etd/1807/`)
sí responde 200; lo que está protegido es `/cgi/viewcontent.cgi`.** El archivo de 0 bytes en
`~/proyectos/topologia-colateral/data/wightman2010.pdf` es el residuo de ese intento fallido del
23-ago.

La vía que funciona es un navegador real, que resuelve el desafío. Documento verificado:
*Topology Control in Wireless Sensor Networks*, Pedro Mario Wightman Rojas, USF, 2010-02-12,
**288 páginas**.

URL del PDF: `https://digitalcommons.usf.edu/cgi/viewcontent.cgi?article=2806&context=etd`

**Hecho el 13-oct-2026:** David descargó el PDF desde el navegador. Copia verificada en

```
estado-del-arte/pdfs/2010_Wightman_topology-control-wireless-sensor-networks.pdf
```

El archivo de 0 bytes en `~/proyectos/topologia-colateral/data/wightman2010.pdf`, si todavía
existe, es el residuo del intento fallido del 23-ago. Falta leer los capítulos de A3 y CDS
antes de afirmar que la adaptación al problema de remoción está respaldada por la tesis.
