# AI-Powered Shopping Assistant

En este proyecto, crearemos e implementaremos un asistente de compras conversacional basado en IA utilizando LangGraph y Large Language Models. El sistema gestionará las consultas de los clientes, las búsquedas de productos, la gestión del carrito de compras y escalará los problemas complejos al soporte técnico cuando sea necesario. No es necesario que comprenda completamente el funcionamiento interno de los LLM, ya que lo explicaremos en detalle más adelante. Por ahora, puede centrarse en implementar las funciones necesarias para que el sistema funcione.

La estructura del proyecto ya está definida y verá que los módulos ya incluyen código y comentarios para ayudarle a empezar.

A continuación se muestra la estructura completa del proyecto:

```
├── src
│   ├── assistants.py
│   ├── tools.py
│   ├── graph.py
│   ├── state.py
│   ├── prompts.py
│   ├── conversation_runner.py
├── tests
│   ├── test_cart_and_schema.py
│   ├── test_end_to_end.py
│   ├── test_graph.py
│   ├── test_sales_assistant.py
│   ├── test_structured_search.py
│   ├── test_tool_node.py
│   └── test_vector_search.py
├── dataset
│   ├── products.csv
│   ├── orders.csv
│   ├── aisles.csv
│   └── departments.csv
├── app.py
├── download_dataset.py
├── README.md
└── requirements.txt
```

Veamos brevemente cada módulo:

- src: Contiene la implementación principal del sistema de asistente de compras con IA.
- `src/tools.py`: Implementa las herramientas que pueden usar los agentes de IA, incluyendo la búsqueda de productos, la gestión del carrito de compras y las funciones de escalamiento. Debe implementar las siguientes funciones:
- `structured_search_tool()`: Proporciona un filtrado similar a SQL en el catálogo de productos con compatibilidad con el historial de compras.
- `search_products()`: Realiza una búsqueda vectorial semántica mediante incrustaciones.
- `search_tool()`: Envoltorio de herramientas LangChain para la búsqueda vectorial.
- `RouteToCustomerSupport`: Esquema de Pydantic para escalar a soporte.
- `create_tool_node_with_fallback()`: Gestión de errores para la ejecución de la herramienta.
- `src/assistants.py`: Contiene las implementaciones de los agentes de IA para ventas y atención al cliente.
- `src/graph.py`: Implementa el flujo de conversación mediante LangGraph.
- `src/state.py`: Define la estructura del estado de la conversación.
- `src/prompts.py`: Contiene las indicaciones para los agentes de IA.
- `src/conversation_runner.py`: Utilidades para probar conversaciones.
- tests: Este módulo contiene pruebas unitarias y de integración para validar el comportamiento del sistema.
- dataset: Contiene el catálogo de productos y el historial de pedidos.
- app.py: Interfaz web de Streamlit para probar el asistente de compras.

Su tarea consistirá en completar el código correspondiente en las partes marcadas con `#TODO` en todos los módulos. Puede validar su correcto funcionamiento utilizando las pruebas proporcionadas. Le recomendamos leer atentamente la documentación, ya que contiene una guía detallada de implementación.

**Importante**: Antes de comenzar, debe crear la base de datos vectorial, ya que el sistema no funcionará sin ella.

## Creación de la base de datos vectorial

La base de datos vectorial es esencial para la búsqueda semántica de productos. Tienes dos opciones:

### Opción A: Google Colab (Recomendado)

Crear incrustaciones para miles de productos requiere un uso intensivo de recursos computacionales. Google Colab ofrece acceso gratuito a las GPU NVIDIA T4, que pueden procesar las incrustaciones entre 10 y 20 veces más rápido que una CPU típica.

1. Abre Google Colab y crea un nuevo notebook.
2. Habilita la GPU: Ve a Tiempo de ejecución → Cambiar tipo de tiempo de ejecución → Acelerador de hardware → GPU T4.
3. Clona tu repositorio:

```python
!git clone YOUR_REPO_URL
%cd your-repo-name
```

4. Instala las dependencias:

```python
!pip install -r requirements.txt
```

5. Crea la base de datos de vectores:

```python
!python src/build_vector_db.py
```

Esto debería tardar entre 5 y 10 minutos con la GPU, frente a 1 o 2 horas con la CPU.

6. Descarga la carpeta `vector_db` y colócala en el directorio raíz de tu proyecto local.

### Opción B: Máquina local

Si prefiere ejecutar localmente:

```bash
python src/build_vector_db.py
```

### Descarga del conjunto de datos (obligatorio)

Antes de compilar la base de datos vectorial o ejecutar cualquier prueba, debe descargar y extraer el conjunto de datos.

Ejecute el siguiente script una vez desde la raíz del proyecto:

```bash
python download_dataset.py
```

Esto permitirá:

- Descargar el archivo ZIP del conjunto de datos desde Google Drive
- Descomprimirlo directamente en la carpeta `dataset/`

Después de ejecutarlo, debería ver archivos como `products.csv`, `orders.csv`, etc., dentro del directorio `dataset/`.

### Método recomendado para trabajar con todos esos archivos

Nuestra recomendación sobre el orden en que debe completar estos archivos es la siguiente:

## 1. `src/tools.py`

Dentro de este módulo, complete las funciones en este orden:

1. Clase `RouteToCustomerSupport`. Este es un modelo base de Pydantic simple que define el esquema para escalar a soporte al cliente.

2. Función `search_products()`. Esta función realiza una búsqueda vectorial semántica utilizando la base de datos vectorial. Lea atentamente la documentación, ya que explica las funciones exactas que debe llamar y el formato de retorno esperado.

3. Función `search_tool()`. Este es un contenedor de herramientas LangChain que formatea los resultados de `search_products()` para el agente de IA.

4. Función `structured_search_tool()`. Proporciona un filtrado similar a SQL en el catálogo de productos. La documentación contiene ejemplos detallados y sugerencias de implementación. Preste atención al parámetro `history_only` y a la gestión de errores.

5. Función `create_tool_node_with_fallback()`. Implementa la gestión de errores para la ejecución de la herramienta mediante el mecanismo de reserva de LangChain.

Puede comprobar su progreso ejecutando:

```console
$ pytest tests/test_structured_search.py ​​-v
```

## 2. `src/assistants.py`

Dentro de este módulo, complete:

1. Función `sales_assistant()`. Esta función configura el hilo y el contexto del usuario, y luego invoca el flujo de trabajo del agente de ventas. La documentación explica exactamente lo que debe hacer; puede consultar la función `support_assistant()` a continuación para ver el patrón.

Ahora vuelva a ejecutar las pruebas para comprobar si funcionan correctamente.

## 3. Pruebas y validación

Ejecute el conjunto de pruebas completo para validar su implementación:

```console
$ pytest tests/ -v
```

También puede probar funciones individuales manualmente:

```python
from src.conversation_runner import run_single_turn
result = run_single_turn("Hola, necesito plátanos", "test-thread-123")
print(result)
```

## 4. Pruebe la interfaz web

Una vez que sus funciones funcionen y las pruebas sean correctas:

```bash
streamlit run app.py
```

Pruebe estos escenarios:

- Búsqueda de productos: "Necesito refrigerios saludables"
- Gestión del carrito: "Agregar plátanos a mi carrito"
- Escalada: "Quiero un reembolso" (activa la aprobación humana)

## 5. Implemente la búsqueda web a través de Brave Search MCP: https://github.com/brave/brave-search-mcp-server

Para el ejercicio de búsqueda web, necesitará tener instalado Node.js para ejecutar los comandos `npx`. El servidor MCP se ejecuta mediante `npx -y @brave/brave-search-mcp-server` con los argumentos de transporte y clave API adecuados. También deberá obtener una clave API de Brave Search de https://brave.com/search/api y agregarla a su archivo `.env` como `BRAVE_API_KEY=your_key_here`. La implementación implica el uso de adaptadores `langchain-mcp` para conectarse al servidor MCP e integrar la herramienta resultante en el conjunto de herramientas de su asistente de ventas.

## Notas clave de implementación

Al implementar las funciones, tenga en cuenta lo siguiente:

- **Lea atentamente la documentación**: Cada función cuenta con documentación detallada que explica las entradas, las salidas y los requisitos de implementación. - **Usar llamadas de herramientas adecuadas**: Para las herramientas LangChain, utilice siempre el formato `.invoke({"param": value})`, no llamadas directas a funciones.
- **Gestionar errores con precisión**: Devolver los mensajes de error adecuados en el formato esperado en lugar de generar excepciones.
- **Seguir el flujo de datos**: Los productos tienen identificadores que se vinculan a los nombres, las operaciones del carrito necesitan identificadores de subprocesos y la búsqueda vectorial devuelve datos diferentes a los de la búsqueda estructurada.
- **Pruebas incrementales**: Ejecutar pruebas después de completar cada función para detectar problemas de forma temprana.

¡Mucha suerte!
