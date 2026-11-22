# critset — conjuntos críticos en el grafo de exposición de DeFi

Herramienta del proyecto *Estructura frente a tamaño: selección de conjuntos críticos en el
grafo de exposición de DeFi* (Teoría de Grafos 2026-II, Universidad del Rosario).
Autores: David Sánchez Lombana, Juan Sebastián Cuesta Montenegro.

Recibe una instantánea semanal del panel de exposición inter-protocolo de DeXposure
(`time,source,target,size`), construye la componente gigante de su proyección no dirigida y
devuelve, para cada regla de selección y presupuesto `k`, los protocolos seleccionados, las
curvas de pérdida estructural y ponderada, la descomposición `L_pond = L_directa + L_frag` y el
tiempo de ejecución.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) y Python ≥ 3.11.

```bash
uv sync
uv run pytest -q          # 35 pruebas contra networkx y fuerza bruta
```

## Datos

Ya incluido en `datos/DeXposure/network_data/` (licencia MIT, ver `datos/DeXposure/LICENSE`).
Para obtenerlo de nuevo desde el repositorio de DeXposure (Wu et al., 2025):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/dthinkr/DeXposure
# panel: DeXposure/data/network_data/filtered_graph_data.csv (314 semanas, 2018-04 a 2024-04)
```

## Uso

```bash
# Una instantánea: conjuntos, curvas y descomposición (JSON + CSV)
uv run critset run --panel filtered_graph_data.csv --week 2022-05-02 --out salida/

# Con el control voraz sin restricción (más lento)
uv run critset run --panel filtered_graph_data.csv --week 2022-05-02 --greedy --out salida/

# Protocolo experimental completo (13 semanas, k = 1, 2, 5, 10, 20)
uv run critset experiment --panel filtered_graph_data.csv \
    --nodes filtered_nodes_terra.csv filtered_nodes_ftx.csv --out resultados --jobs 8
uv run critset case --panel filtered_graph_data.csv --results resultados
uv run critset characterize --panel filtered_graph_data.csv --out resultados --jobs 8
uv run critset plots --results resultados --out figuras

# Control exploratorio por TVL y SIS (el TVL publicado no cubre toda C_t: faltantes = 0)
uv run critset tvl --panel filtered_graph_data.csv \
    --nodes filtered_nodes_terra.csv filtered_nodes_ftx.csv --results resultados

# Sensibilidad: repetir sin el nodo agregado "Unknown" y sin la regla espectral (lenta)
uv run critset experiment --panel filtered_graph_data.csv --out resultados/sin_unknown \
    --exclude Unknown --rules fuerza intermediacion pagerank articulacion cds voraz
```

`experiment` guarda cada semana terminada en `resultados/por_semana/` y, si se interrumpe,
retoma desde ahí. La regla espectral exacta es la más costosa (del orden de una hora por semana
con `k = 20` cuando corren 13 semanas en paralelo).

## Reglas

| Regla | Clave | Tipo |
|---|---|---|
| Fuerza de exposición (suma de pesos entrantes y salientes) | `fuerza` | ranking estático |
| Intermediación sin pesos en la proyección | `intermediacion` | ranking estático |
| PageRank dirigido y ponderado (α = 0,85, tolerancia ℓ₁ 1e-10) | `pagerank` | ranking estático |
| Articulación adaptativa (menor componente residual; si no hay, intermediación) | `articulacion` | adaptativa |
| CDS de Guha–Khuller (algoritmo I modificado) + voraz ponderado dentro del CDS | `cds` | adaptativa |
| Inmunización espectral voraz exacta sobre `log(1 + w)` | `espectral` | adaptativa |
| Controles: voraz sin restricción, TVL, SIS, aleatorio (20 permutaciones) | `voraz`, `tvl`, `sis`, `aleatorio` | — |

Los empates se resuelven por orden lexicográfico de identificador.

## Archivos de salida (`experiment`)

- `curvas.csv`: semana, regla, k, R_est, R_pond, L_est, L_pond, L_directa, L_frag, segundos, MB pico.
- `semanas.csv`: dimensiones de cada instantánea, tamaño del CDS y cobertura de TVL.
- `conjuntos.json`: secuencia seleccionada por cada regla y semana (identificadores).
- `jaccard.csv`: solapamiento entre reglas por semana y k.
- `estudio_caso.csv`, `afectados_*.txt`: cotejo con el indicador de afectación (caída ≥ 50 % de la
  fuerza en 4 semanas).
- `caracterizacion.csv`, `autocorrelacion.csv`: invariantes de las 314 semanas y n efectivo AR(1).
- `tvl_exploratorio.csv`: TVL y SIS con faltantes en cero, junto a las demás reglas, en las
  cuatro semanas con datos de tamaño.
- `sin_unknown/`: los mismos archivos para el análisis de sensibilidad sin `Unknown`.
