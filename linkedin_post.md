# LinkedIn Post Draft

## Problema

Executar modelos de linguagem localmente esbarra em um gargalo comum: a GPU não consegue atender múltiplas requisições simultâneas sem travar ou ficar ociosa. Para quem quer colocar um modelo em produção, entender esse limite é essencial.

## Solução

Construí um benchmark comparativo de motores de inferência rodando na minha AMD RX 9060XT com 16GB de VRAM. O projeto testa quatro configurações diferentes usando o modelo Qwen2.5-3B-Instruct:

- HuggingFace Transformers como baseline
- vLLM com PagedAttention e Chunked Prefill
- vLLM com modelo quantizado em 4 bits via AWQ
- vLLM com Decodificação Especulativa usando um modelo draft de 0.5B

A arquitetura separa responsabilidades de forma clara: uma API Litestar é a dona da GPU, um dashboard Streamlit serve apenas como interface, e um load tester assíncrono simula múltiplos usuários com httpx e asyncio. Telemetria de VRAM e uso de GPU é coletada via amdsmi e exportada para CSV.

## Resultados

O HuggingFace puro atinge o limite de VRAM rapidamente sob concorrência, gerando falhas de OOM. O vLLM com PagedAttention mantém estabilidade mesmo com dezenas de requisições simultâneas, aproveitando a memória de forma muito mais eficiente. A quantização AWQ reduz o footprint do modelo pela metade e libera espaço para batch maior. A decodificação especulativa entrega ganho real de throughput ao usar um modelo menor para antecipar tokens.

O projeto completo está no GitHub com instruções para rodar em ambientes ROCm. O código mostra na prática como escolhas de arquitetura de inferência impactam latência, throughput e estabilidade sob carga.

Link para o repositório nos comentários.
