# Controlabilidad y asignación estratégica de actuadores en sistemas BBM

**[PRESENTACIÓN ORAL — Guion de 20 minutos](PRESENTACION.md)**

Explicación para público no especializado, con
variables matemáticas definidas, gráficas y preguntas frecuentes.

**Proyecto final - Maestría en Ciencia de Datos**
**Universidad Yachay Tech - Octubre de 2026**

**Autor:** Héctor Alexander Román Potosí

> **Proyecto académico:** controlabilidad espectral,
> juegos finitos de asignación de actuadores y
> experimentos computacionales sobre redes BBM.

**Resultados destacados:**

- Equilibrios puros, equilibrio mixto exacto y precio
  de la anarquía en un juego finito de actuadores.
- Validación matemática reproducible mediante Python
  y C++17: cuatro pruebas ejecutadas correctamente.
- Comparación de métricas de sincronización en
  900 observaciones emparejadas.
- Evaluación de Machine Learning bajo validación
  agrupada y LOICFO.
- Estudio independiente de sensibilidad numérica.

---

## 1. Descripción

Este proyecto presenta resultados matemáticos y computacionales
de una investigacion sobre sistemas Benjamin-Bona-Mahony (BBM),
controlabilidad, teoría de grafos y teoría de juegos.

Se estudian cuatro componentes:

1. Controlabilidad de aproximaciones espectrales BBM.
2. Asignación estratégica de actuadores y equilibrio de Nash.
3. Experimentos numericos sobre redes BBM.
4. Analisis de datos y aprendizaje automático.

Los estudios de juegos finitos y redes BBM no lineales son
experimentos distintos y se presentan por separado.

## 2. Resultados matemáticos: juego de actuadores

### 2.1. Modelo reducido

Se considera el sistema lineal Fourier-Galerkin:

```math
\dot{x}_N(t)=A_Nx_N(t)+B_N(s)v(t).
```

El ejemplo utiliza tres modos espectrales,
dos jugadores y un horizonte temporal T=1.5.

Los canales de control son:

```math
c_1=(1,0,1)^\top,\qquad c_2=(1,1,1)^\top.
```

La similitud modal es:

```math
\rho=\frac{2}{3}.
```

Las funciones de beneficio incluyen una medida basada
en el Gramiano, costos individuales y redundancia modal.

### 2.2. Equilibrios de Nash

Sin penalización por redundancia, el único equilibrio
puro del ejemplo es (1,1).

Con redundancia, los equilibrios puros son:

```math
\mathrm{NE}_{pure}=\{(1,0),(0,1)\}.
```

El equilibrio mixto exacto es:

```math
(p_1^*,p_2^*)=
\left(\frac{33}{38},\frac{6}{17}\right).
```

### 2.3. Controlabilidad de la truncación

Los rangos de las matrices de controlabilidad son:

| Perfil | Rango | Controlable |
|---|---:|---|
| (0,0) | 0/3 | No |
| (0,1) | 3/3 | Sí |
| (1,0) | 2/3 | No |
| (1,1) | 3/3 | Sí |

Un equilibrio de Nash no garantiza por si mismo
controlabilidad completa del sistema truncado.

### 2.4. Bienestar y precio de la anarquía

El óptimo social es (1,1), con:

```math
W^{opt}=\frac{8}{15}.
```

El peor equilibrio es (1,0), con:

```math
W(1,0)=\frac{107}{323}.
```

Por tanto:

```math
\mathrm{PoA}_{pure}=\mathrm{PoA}_{all}
=\frac{2584}{1605}\approx1.609969.
```

### 2.5. Alineación de incentivos

Se estudian transferencias presupuestariamente equilibradas:

```math
t_1^\tau(s)=\tau(s_1-s_2),
\qquad t_2^\tau(s)=-t_1^\tau(s).
```

El umbral critico es:

```math
\tau_1=\frac{22}{255}.
```

Para valores estrictamente superiores a este umbral,
el único equilibrio es (1,1) y el precio de la
anarquía es igual a uno.

### Alcance

Estos resultados corresponden a un juego finito
construido a partir de una truncación lineal BBM.
No prueban controlabilidad de la ecuacion BBM
no lineal en dimension infinita.

### 2.6. Validación computacional

Los resultados analíticos se complementan con pruebas
deterministas implementadas en Python y C++17.

| Programa | Verificacion | Estado |
|---|---|---|
| `parameter_regions_exact_test.py` | Equilibrios, bienestar, PoA e incentivos con fracciones exactas | PASS |
| `redundancy_game_test.cpp` | Redundancia modal y mejores respuestas | PASS |
| `parameter_regions_test.cpp` | 15 casos de regiones paramétricas | PASS |
| `finite_gramian_game_test.cpp` | Juego finito adicional basado en Gramianos | PASS |

El ultimo programa examina un juego diferente del
ejemplo reducido de redundancia. Sus equilibrios
no deben confundirse con los de ese experimento.

#### Requisitos

- Python 3.
- Compilador compatible con C++17.
- PowerShell para ejecutar el script incluido.

La validación racional utiliza la biblioteca estandar
`fractions` de Python.

#### Ejecución

Desde la raiz del repositorio:

```powershell
.\scripts\run_validations.ps1
```

El script compila los programas C++ y ejecuta las
cuatro verificaciones.

La ejecución satisfactoria termina con:

```text
RESULTADO FINAL: 4/4 PRUEBAS PASS
```

Los ejecutables generados se guardan en `.build/`,
directorio excluido del control de versiones.

Estas pruebas verifican los casos implementados;
no sustituyen las demostraciones matemáticas
ni establecen controlabilidad de la BBM no lineal
en dimension infinita.

---
## 3. Experimentos de redes BBM

Esta etapa estudia el comportamiento de sistemas BBM
acoplados mediante grafos.

Los experimentos de redes no constituyen una validación
empirica del juego finito de Nash presentado anteriormente.

### 3.1. Comparación de métricas de sincronizacion

Se comparan dos formas de evaluar la respuesta de la red:

- Una métrica global de sincronizacion.
- Una métrica local basada en las aristas del grafo.

La comparación V2 incluye 900 observaciones emparejadas.

| Indicador | Resultado |
|---|---:|
| Comparaciones | 900 |
| Concordancias de clase | 833 |
| Discordancias de clase | 67 |
| Concordancia porcentual | 92.56 % |
| Correlación de Pearson | 0.8071 |
| Correlación de Spearman | 0.8518 |

![Comparacion entre metricas de sincronizacion](figures/eda_v2/04_edge_global_agreement_v2.png)

**Figura 1.** Comparación de las métricas local y global
en los experimentos BBM V2.

### 3.2. Efecto del acoplamiento

La concordancia también se examina por régimen
del parametro de acoplamiento.

| Régimen | Comparaciones | Concordancia |
|---|---:|---:|
| gamma = 0 | 225 | 91.11 % |
| gamma > 0 | 675 | 93.04 % |
| Total | 900 | 92.56 % |

Los resultados indican una elevada concordancia entre
las clasificaciones obtenidas con las dos métricas.

Sin embargo, las discrepancias observadas muestran
que no deben considerarse medidas intercambiables.

La interpretacion de las clases depende de las reglas
de clasificacion utilizadas en el experimento.

Los resultados originales estan disponibles en:

`data/derived/v2/bbm_network_metric_agreement_v2.csv`

**Alcance de la comparación entre métricas**

Estos resultados describen el conjunto de simulaciones
estudiado. No constituyen una prueba de controlabilidad
exacta ni de superioridad universal de una topología.

---
### 3.3. Sensibilidad numérica

Una campaña independiente de sensibilidad V2
contiene 1944 integraciones y 1458 pares
controlado-baseline.

Se evaluaron tres topologías, varios tamanos de
red y condiciones iniciales, junto con:

- Resoluciones espectrales K = 4, 6 y 8.
- Pasos temporales h = 0.0025 y 0.00125.
- Horizonte final T = 1.

El refinamiento espacial compara K = 4 frente
a K = 8 manteniendo h = 0.00125.

El refinamiento temporal compara ambos pasos
manteniendo K = 8.

| Comparación | Magnitud | Diferencia mediana |
|---|---|---:|
| Espacial | Error final edge | 1.70e-8 |
| Espacial | Error final global | 2.29e-8 |
| Temporal | Error final edge | 5.30e-15 |
| Temporal | Error final global | 5.00e-15 |

#### Refinamiento espacial

![Refinamiento espacial](figures/sensitivity/bbm_network_v2/01_spatial_refinement_v2.png)

#### Refinamiento temporal

![Refinamiento temporal](figures/sensitivity/bbm_network_v2/02_temporal_refinement_v2.png)

Las diferencias reportadas son pequeñas dentro
de las comparaciones numéricas realizadas.

Esta evidencia permite evaluar sensibilidad
frente a cambios de resolución, pero no
constituye una demostracion general de
convergencia o controlabilidad infinita.

---
## 4. Ciencia de Datos y Machine Learning

Esta etapa estudia la posibilidad de predecir los
resultados de las simulaciones BBM mediante modelos
supervisados de aprendizaje automatico.

### 4.1. Problema de prediccion

Las configuraciones experimentales contienen
informacion sobre la topologia, el tamano de la red,
el acoplamiento, las condiciones iniciales y las
estrategias de actuacion.

El objetivo es estudiar las relaciones entre estos
factores y las metricas de sincronizacion.

### 4.2. Generalizacion a condiciones iniciales nuevas

Se examina la validacion
Leave-One-Initial-Condition-Family-Out (LOICFO).

En este protocolo se excluye una familia completa
de condiciones iniciales durante el entrenamiento
y se utiliza posteriormente para evaluar el modelo.

Esto permite estudiar un problema distinto de la
evaluacion convencional dentro de un conjunto de
configuraciones conocidas.

![Generalizacion LOICFO](figures/modeling_v2/02_loicfo_metrics_v2.png)

**Figura 2.** Evaluacion de modelos predictivos
sobre familias de condiciones iniciales excluidas
del entrenamiento, segun el experimento V2.

### 4.3. Interpretacion y limitaciones

Un modelo puede presentar buen rendimiento en
particiones de un diseno experimental conocido
y resultados diferentes cuando debe predecir
sobre condiciones iniciales no observadas.

Por ello, los resultados deben interpretarse
junto con su protocolo de validacion.

### 4.4. Resultados cuantitativos LOICFO

La evaluacion LOICFO se realiza dejando fuera una
familia completa de condiciones iniciales por fold.

Los valores reportados son medias de cinco folds.

| Objetivo | Modelo | MAE | RMSE | R2 |
|---|---|---:|---:|---:|
| Edge | Dummy | 0.036191 | 0.046981 | -0.195797 |
| Edge | Ridge | 0.037181 | 0.047270 | -0.294806 |
| Edge | Random Forest | 0.034470 | 0.046420 | -0.413929 |
| Edge | HistGradientBoosting | 0.035088 | 0.046533 | -0.444177 |
| Global | Dummy | 0.036297 | 0.043542 | -0.135153 |
| Global | Ridge | 0.037451 | 0.043291 | -0.257992 |
| Global | Random Forest | 0.035007 | 0.044182 | -0.448291 |
| Global | HistGradientBoosting | 0.036449 | 0.046170 | -0.598605 |

### 4.5. Comparación de protocolos

En la validacion agrupada principal V2,
Random Forest obtuvo para el objetivo Edge:

```math
R^2=0.931225.
```

En cambio, bajo LOICFO, su R2 medio
desciende a -0.413929 para Edge y
-0.448291 para Global.

Los resultados no implican que el modelo falle
bajo todas las metricas. En LOICFO,
Random Forest presenta un MAE medio ligeramente
inferior al baseline Dummy.

Sin embargo, los R2 medios negativos y la elevada
variabilidad entre familias indican limitaciones
importantes de generalizacion.

Por ejemplo, al excluir la familia inicial 2,
Random Forest obtuvo aproximadamente:

- Edge: R2 = -2.9220.
- Global: R2 = -2.4492.

Estos valores corresponden a protocolos y
objetivos concretos; no deben interpretarse como
propiedades universales de Random Forest.

### 4.6. Evidencia experimental

Los resultados se conservan en:

- `data/modeling_v2/bbm_network_ml_metrics_summary_v2.csv`
- `data/modeling_v2/bbm_network_ml_metrics_by_fold_v2.csv`

Estos archivos documentan las evaluaciones realizadas.
La reproduccion completa del entrenamiento de ML
requiere integrar y comprobar el codigo correspondiente.

---

## 5. Reproducibilidad

### 5.1. Requisitos

Para ejecutar los validadores matemáticos se necesita:

- Python 3, con la biblioteca estándar `fractions`.
- Un compilador C++ compatible con C++17.
- PowerShell para utilizar el script de automatización.

El entorno utilizado durante la validación fue
Python 3.12.13 y g++ 13.2.0 en Windows.

### 5.2. Ejecutar las pruebas

Desde la raíz del repositorio:

```powershell
.\scripts\run_validations.ps1
```

El script ejecuta un validador racional en Python
y compila y ejecuta tres pruebas C++.

La ejecución verificada produjo:

```text
RESULTADO FINAL: 4/4 PRUEBAS PASS
```

Los ejecutables se generan dentro de `.build/`,
carpeta excluida de Git.

### 5.3. Datos y figuras

Los directorios `data/` y `figures/` contienen
resultados seleccionados de los experimentos.

Los archivos CSV permiten consultar los valores
reportados en las tablas y las figuras documentan
resultados de simulación y evaluación de modelos.

**Alcance de la reproducibilidad:** las cuatro
pruebas matemáticas incluidas son ejecutables
desde este repositorio. La reproducción integral
de la campaña de redes BBM y del entrenamiento
de Machine Learning no está incluida todavía.

## 6. Limitaciones científicas

- Los resultados de controlabilidad del juego
  corresponden a una aproximación lineal finita.
- Los experimentos no demuestran controlabilidad
  exacta de la ecuación BBM no lineal completa.
- La asignación mediante Nash y la campaña de
  redes BBM son experimentos diferenciados.
- Los resultados predictivos dependen del protocolo
  de validación y de las familias consideradas.
- Los estudios de sensibilidad no constituyen
  una demostración general de convergencia.

El objetivo de este repositorio es presentar
resultados seleccionados y verificables del trabajo,
sin sustituir la exposición matemática completa
desarrollada en la investigación.
