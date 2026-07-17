# Plano de Implementação: Motor de Inferência Otimizado (Benchmark)

Com base no nosso alinhamento, adicionamos os conceitos mais avançados de **2026** para garantir que seu portfólio seja estado da arte. Vamos utilizar o hardware da sua **AMD RX 9060XT (16GB VRAM)** para demonstrar não apenas o KV-Cache/PagedAttention, mas também o **Speculative Decoding** e **Chunked Prefill**.

## Estrutura de Pastas (Final 2026 Edition)
**Diretório Base**: `E:\PERSONAL SOFTWARE PROJECTS\llm-inference-benchmark`

```text
llm-inference-benchmark/
├── README.md               
├── requirements.txt        
├── .gitignore              # Ignorar arquivos temporários, venv e cache
├── .env                    # Variáveis de ambiente (ex: HF_TOKEN)
├── engines/                # APIs de Inferência
│   ├── __init__.py
│   ├── config.py             # Leitura centralizada do .env via pydantic-settings
│   ├── base_engine.py        # Classe abstrata para os motores (OOP)
│   ├── hf_engine.py          # Motor 1: HuggingFace (Baseline)
│   ├── vllm_engine.py        # Motor 2: vLLM (PagedAttention + Prefix Caching)
│   ├── awq_engine.py         # Motor 3: vLLM Quantizado (AWQ 4-bits)
│   └── speculative_engine.py # Motor 4: vLLM com Speculative Decoding
├── load_tester/            # Ferramenta de Teste de Carga
│   ├── __init__.py
│   └── tester.py           # Script asyncio/httpx (Testará o Chunked Prefill)
├── telemetry/              # Monitoramento de Hardware (AMD)
│   ├── __init__.py
│   └── monitor.py          # Script usando amdsmi / rocm-smi
├── dashboard/              # Visualização
│   └── app.py              # Interface Streamlit
└── data/                   
```

## Regras e Estrutura Geral do Código

Para garantir que a base de código seja lida como a de um Engenheiro de Software Sênior em produção, seguiremos diretrizes estritas em todo o repositório:

1. **Idioma Padrão**: Todas as variáveis, nomes de funções e classes serão escritas estritamente em **inglês**.
2. **Comentários**: Comentários curtos e diretos em **inglês**, focados exclusivamente nas partes mais importantes e complexas do código (explicando o *porquê*, não o *que*).
3. **Modularidade e OOP**: Construção dos arquivos separando cada implementação lógica. Para os motores, utilizaremos Orientação a Objetos: criaremos uma interface `BaseEngine`, e faremos todos os motores herdarem dela, garantindo que tenham a mesma estrutura de métodos e facilitando o plug-and-play.
4. **Type Hinting e Pydantic**: Uso rigoroso de tipagem forte em todas as funções (ex: `def run_inference(prompt: str) -> str:`) e validação estrita de dados, o que é obrigatório em frameworks modernos como o Litestar.
5. **Logging Profissional**: É terminantemente proibido o uso de `print()`. Utilizaremos uma biblioteca profissional (como `logging` ou `loguru`) para registrar os eventos do servidor com timestamps e níveis de severidade adequados (INFO, WARNING, ERROR).
6. **Gestão de Configuração (Settings)**: Teremos um arquivo `config.py` dedicado a ler as variáveis do `.env` de forma segura. Nenhuma variável de ambiente deve ser chamada via `os.getenv()` no meio das lógicas de negócio.

## Dependências (`requirements.txt`)

```text
# Servidor de API e Runtime
litestar              # Framework web assíncrono muito mais rápido e tipado que o FastAPI antigo
granian               # Servidor web Rust-based para Python (substitui o Uvicorn com o dobro de performance)
pydantic-settings     # Para gestão do arquivo .env e tipagem de configurações

# Modelos e Inferência (Precisam ser instalados com suporte a ROCm/AMD)
torch
transformers
vllm
autoawq

# Teste de Carga (Assíncrono)
httpx                 # Cliente HTTP moderno (substitui o aiohttp)
asyncio

# Telemetria de GPU (Específico para AMD)
amdsmi                

# Dashboard e Dados (Mantidos a pedido)
streamlit
pandas
matplotlib
```

## Plano de Ação Detalhado (5-6 Dias)

- **Dia 1: Motores 1 e 2 (Baseline HF e vLLM Padrão)** 
  - Foco: Setup inicial, API Litestar, implementação da classe Base, e os dois primeiros motores.
- **Dia 2: Motores 3 e 4 (AWQ e Speculative Decoding)**
  - Foco: Configurar a quantização e hospedar um modelo "draft" (rascunho) minúsculo com vLLM para multiplicar o TPOT.
- **Dia 3: Load Tester Assíncrono com Chunked Prefill**
  - Foco: Programar picos de tráfego e injetar um prompt colossal (100k tokens) para provar o Chunked Prefill.
- **Dia 4: Sistema de Telemetria AMD (Consumo de Hardware)**
  - Foco: Desenvolvimento do script `monitor.py` utilizando a biblioteca `amdsmi` (ou lendo a saída do `rocm-smi`). Este script será executado em background paralelamente ao teste de carga para coletar, em tempo real, os picos de consumo de VRAM e a porcentagem de uso do processador da sua RX 9060XT. Os dados brutos serão gravados estruturalmente em arquivos `.csv` dentro da pasta `data/`.
- **Dia 5: Dashboard Streamlit e Análise dos Dados**
  - Foco: Criação da interface interativa visual em `app.py`. O Streamlit vai ler os logs de resposta do load tester e os `.csv` da telemetria para renderizar gráficos comparativos explícitos. Os gráficos provarão a relação "Latência vs Concorrência" e o "Consumo de VRAM ao longo do tempo", ilustrando de forma inquestionável o ponto exato de falha (Out-Of-Memory) do Hugging Face contra a estabilidade dos motores em vLLM.
- **Dia 6: Documentação Final e Preparação do Portfólio (GitHub/LinkedIn)**
  - Foco: Criação da apresentação técnica nos dois principais canais de recrutamento.
  - **GitHub (`README.md`)**: Escrito inteiramente em **Inglês**. A estrutura será altamente técnica e seccionada por tópicos lógicos (Arquitetura, Como Rodar, Resultados). Incluirá placeholders/espaços visuais pré-definidos para a inserção das screenshots geradas no Dia 5.
  - **LinkedIn (Artigo/Post)**: A comunicação será o meio-termo ideal entre formal e informal, servindo como um texto claro de contextualização e referência para o leitor entender a engenharia aplicada. A publicação deve, obrigatoriamente, incluir uma explicação didática de fácil compreensão (usando analogias acessíveis) para que profissionais de RH e recrutadores não-técnicos entendam perfeitamente o impacto de negócio do projeto. Embora se inspire em uma didática lógica (Problema, Solução, Resultados), não deve ser tratada como um *paper* acadêmico rígido. **Regras Estritas de Redação**: Ausência absoluta de travessões, zero utilização de emojis, e eliminação de todo o vocabulário clichê comumente detectado como gerado por IA.

---

## ⚠️ User Review Required

Com a inclusão das regras de MLOps (Logging, OOP, Type Hinting) e o detalhamento extremo dos Dias 4, 5 e 6 com as regras de redação do LinkedIn, o plano arquitetural está finalizado.

Podemos marcar este plano como **Aprovado** e iniciar imediatamente a programação do **Dia 1**?
