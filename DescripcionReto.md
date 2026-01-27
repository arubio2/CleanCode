# Descripción Reto 2026 

El reto consistirá en desarrollar un generador automático de informes a partir de una fuente de datos.  

El programa deberá leer la fuente de datos (uno o varios ficheros Excel, csv o similares) y una descripción de estos en un fichero de texto. A partir de los datos y su descripción, y ayudado por una IA, el programa deberá generar de forma automática un informe sobre esos datos.  

---
**Requisitos mínimos** del generador automático de informes: 

1. El informe generado deberá representar los datos con los gráficos y tablas adecuadas.  

2. Los gráficos y tablas deben estar correctamente etiquetados, con sus correspondientes pies de figura o de tabla. Las etiquetas dentro de los gráficos deben ser legibles.  

3. El formato de salida del informe debe poder seleccionarse entre pdf, Word o PowerPoint.  

4. El informe contiene al menos un motivo relacionado con los propios datos que han sido seleccionados: foto o ilustración. 

**Funcionalidades valoradas positivamente**: 

1. El formato de salida del informe incluye HTML, jupyter notebook o similares. 

2. En informe se puede generar tanto en español como en inglés. 

3. El informe contiene varios motivos relacionados con los propios datos que han sido seleccionados: fotos o ilustraciones. 

4. Interfaz de usuario interactiva para la generación de los informes.
5. Incorporación de nuevos agentes.
6. Calidad del informe.

7. Gestión de recursos

 ---

La entrada será por tanto el .csv de los datos y un .txt con la descripción de cada columna o especificando el tipo de análisis a realizar. 

En el repositorio de github arubio2/CleanData tenéis un código que cumple los requisitos mínimos planteados en este reto y las instrucciones para usarlo.  

Vuestro trabajo consistirá en mejorar el código y aumentar la calidad de los informes generados con las bases de datos facilitadas. La evaluación se realizará con una base de datos diferente que se os proporcionará 1 hora antes de la presentación final. 

---
Para ayudar con el proceso se han creado **3 Metas Volantes** que habrá que ir cumpliendo estos dos días: 

## Primera Meta Volante: Martes 27 a las 19:00 

- Se ha realizado un repartición del trabajo entre los integrantes del equipo. 

- Haber conseguido ejecutar el código y generar informes en pdf, word y powerpoint de al menos 2 bases de datos distintas. 

 

## Segunda Meta Volante: Miércoles 28 a las 11:30 

- El informe incluye imágenes relacionadas con el tema de los datos proporcionados. 

- El código sigue las instrucciones del .txt para realizar el informe 

 

## Tercera Meta Volante: Miércoles a las 16:00 

- Se ve una mejoría en la calidad de los informes generados 

- Descripciones de los gráficos/ilustraciones/tablas 

- Se sacan conclusiones con sentido y bien razonadas 

- Se hace referencia a datos concretos en los análisis 

 ---

El Miércoles 28 a las 18:00 se os entregará la base de datos nueva para la realización de los informes a presentar a las 19:30 al jurado invitado. 

Los criterios de evaluación serán los siguientes: 

 

# Rúbrica de Evaluación del Generador Automático de Informes 

## 1. Funcionalidad Técnica (40%) 

### 1.1 Lectura de Fuentes de Datos (5%) 

0 pts: El programa no puede leer fuentes de datos. 

2 pts: Puede leer solo un tipo de fuente de datos (p.ej., solo CSV). 

4 pts: Puede leer múltiples formatos de datos (p.ej., Excel, CSV, etc.). 

5 pts: Además, interpreta correctamente la descripción del fichero de texto asociada. 

 

### 1.2 Generación Automática del Informe (10%) 

0 pts: No existe una arquitectura basada en agentes o esta no es funcional. 

4 pts: Los agentes producen texto genérico, sin análisis estructurado. 

8 pts: El sistema utiliza múltiples agentes con roles diferenciados pero no todos los elementos están correctamente justificados o conectados entre sí. 

10 pts: Arquitectura de agentes bien definida y eficiente, donde cada agente tiene un rol claro y existe coordinación efectiva.


### 1.4 Formatos de Salida (5%) 

0 pts: Solo genera un formato de salida. 

2 pts: Genera informes en dos formatos seleccionables por el usuario. 

4 pts: Genera informes en tres formatos seleccionables por el usuario (PDF, Word, PowerPoint). 

5 pts: Incluye además otros formatos de salida con gráficos adaptados. 

 

### 1.5 Idiomas del Informe (5%) 

0 pts: Solo soporta un idioma. 

2 pts: Soporta español o inglés. 

4 pts: Soporta ambos idiomas con traducción parcial. 

5 pts: Traducción completa y precisa en ambos idiomas. 

 

### 1.6 Representación Visual y Etiquetado (10%) 

0 pts: Gráficos ilegibles o mal etiquetados. 

4 pts: Gráficos y tablas básicos con etiquetas claras pero poco detalladas. 

8 pts: Representaciones legibles y etiquetadas, con pies de figura/tabla. 

10 pts: Gráficos/tablas legibles, etiquetados, con pies de figura detallados y descripciones adecuadas. 

 

### 1.7 Interactividad (5%) 

0 pts: Sin elementos interactivos. 

2 pts: Interactividad básica en gráficos. 

4 pts: Interactividad avanzada (filtros, hover) en gráficos de HTML. 

5 pts: Interactividad coherente y adaptada a los datos presentados. 

 

## 2. Análisis de Datos y Generación de Resultados (30%) 

### 2.1 Profundidad del Análisis (15%) 

0 pts: Sin análisis o extracción superficial de datos. 

5 pts: Análisis básico pero incompleto o redundante. 

10 pts: Análisis lógico y adecuado, pero sin insights novedosos. 

15 pts: Análisis profundo, con resultados no obvios y bien fundamentados. 

 

### 2.2 Sentido y Coherencia en la Presentación de Resultados (10%) 

0 pts: Resultados mal estructurados o sin relación con los datos. 

4 pts: Resultados coherentes pero poco claros o confusos. 

8 pts: Presentación clara y lógica, pero con limitaciones menores en la interpretación. 

10 pts: Resultados claros, coherentes y relevantes, con interpretaciones significativas. 


### 2.3 Creatividad en los Resultados (5%) 

0 pts: Sin elementos creativos o innovadores. 

2 pts: Uso básico de creatividad (ej., inclusión de gráficos estándar). 

4 pts: Incorporación de elementos creativos relevantes al contexto de los datos. 

5 pts: Innovación sobresaliente en gráficos, imágenes o presentaciones ilustrativas. 

 

## 3. Optimización de Recursos y Gasto en Tokens (15%) 

### 3.1 Uso Eficiente de Tokens (5%) (https://openai.com/api/pricing/) 

0 pts: Uso ineficiente o excesivo de tokens. 

2 pts: Uso aceptable, pero con costos elevados o redundancias. 

4 pts: Optimización adecuada, con generación eficiente de informes. 

5 pts: Uso altamente eficiente, manteniendo calidad y minimizando costos. 

 

### 3.2 Velocidad de Generación (5%) 

0 pts: Tiempo de generación inaceptablemente largo. 

2 pts: Tiempo aceptable pero con demoras perceptibles. 

4 pts: Generación rápida en la mayoría de los casos. 

5 pts: Generación casi inmediata, incluso con grandes volúmenes de datos. 

 

### 3.3 Gestión de Recursos (5%) 

0 pts: Recursos mal gestionados, errores frecuentes. 

2 pts: Recursos gestionados adecuadamente pero con espacio para mejoras. 

4 pts: Gestión eficiente de recursos en la mayoría de los casos. 

5 pts: Gestión óptima, asegurando estabilidad y calidad. 

 

## 4. Experiencia de Usuario (15%) 

### 4.1 Interfaz de Usuario (5%) 

0 pts: Interfaz inexistente o poco usable. 

2 pts: Interfaz básica, funcional pero poco intuitiva. 

4 pts: Interfaz intuitiva y fácil de usar. 

5 pts: Interfaz excepcionalmente amigable y eficiente. 

 

### 4.2 Personalización del Informe (5%) 

0 pts: Sin opciones de personalización. 

2 pts: Opciones limitadas (p.ej., solo idioma o formato). 

4 pts: Amplias opciones de personalización (idioma, formato, estilos visuales). 

5 pts: Personalización total, incluyendo temas visuales relacionados con los datos. 

 

### 4.3 Satisfacción General del Usuario (5%) 

0 pts: Opinión negativa del usuario final. 

2 pts: Opinión neutral, con puntos negativos claros. 

4 pts: Opinión mayoritariamente positiva, con algunos aspectos a mejorar. 

5 pts: Opinión altamente positiva, superando expectativas. 

## 5. Criterios adicionales (10%) 

Se aplicará un 10% extra de puntuación a aquellos que cumplan los siguientes criterios en relación al uso de Agentes:

**a) Diseño y número de agentes**

Adecuación del número de agentes al problema.

Justificación implícita o explícita de la división de tareas.

**b) Coherencia inter-agente**

Consistencia entre análisis, visualizaciones y conclusiones generadas por distintos agentes.

Ausencia de contradicciones o repeticiones innecesarias.

**c) Eficiencia y reutilización de información**

Uso de resultados intermedios generados por agentes previos.

Minimización de llamadas redundantes a modelos de IA.

**d) Robustez del sistema**

Capacidad del sistema para generar informes coherentes ante distintos tipos de datasets y objetivos.

# Puntuación Total: 

110 puntos posibles. 

90-110 pts: Excelencia. Cumple con creces los objetivos propuestos. 

70-89 pts: Bueno. Cumple con la mayoría de los objetivos, con margen de mejora. 

40-69 pts: Aceptable. Resultados básicos, pero insuficientes en profundidad o funcionalidad. 

<40 pts: Insuficiente. No cumple con los requisitos mínimos. 
