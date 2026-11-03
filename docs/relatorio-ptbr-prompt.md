# Prompt para o relatório comparativo

Use o scribe para redigir, em português brasileiro, um relatório técnico após receber `comparison.json`, `comparison.csv`, `comparison.png` e os diretórios completos das sessões dos dois PCs. Ainda não há resultados de benchmark. Não invente números, fontes, execução, comportamento do Bitdefender nem medições de SDR.

Antes de escrever, confira a identidade dos casos, os hashes dos dados, as condições registradas, `environment.json`, `results.jsonl`, telemetria e as exclusões. Cite cada afirmação quantitativa com o arquivo, ID do caso, métrica, unidade e número de observações que a sustenta. Separe observação de interpretação. Não inclua ensaios falhos, contaminados, com aviso de telemetria ou incompatíveis nas estimativas de efeito. Registre-os em uma tabela de cobertura e interferências.

Estruture o relatório em: objetivo e escopo, método e condições de execução, equivalência e diferenças residuais dos PCs, resultados de treinamento de ML, carregamento e checkpoints, operações de arquivos, uso de CPU, memória e GPU, interpretação, limitações e implicações qualitativas para SDR/telecom. Mostre valores absolutos por condição, diferença relativa, dispersão, tamanho amostral e intervalos quando houver unidades independentes suficientes. Explique que intervalos baseados em repetições de uma única sessão não cobrem variação entre sessões.

Para SDR/telecom, discuta apenas mecanismos plausíveis como competição por CPU e atrasos de leitura ou escrita. Não atribua números medidos de ML ou arquivos a latência de rádio, perda de amostras, throughput de rede ou desempenho de uma aplicação SDR. Como as condições vêm de dois PCs, trate diferenças observadas como evidência comparativa sujeita a diferenças residuais entre máquinas. Não atribua causalidade exclusiva ao Bitdefender.

Faça uma revisão de consistência de unidades, percentuais, sinais e referências aos artefatos antes de entregar o texto.

## Coleta e comparação

No PC com Bitdefender, execute `./run-with-bitdefender.sh`. No PC sem Bitdefender, execute `./run-without-bitdefender.sh`. Cada script prepara o perfil `standard` com trabalho fixo, três repetições, RTX 4070 e dados em `benchmark-data/`. Defina `BDBENCH_STORAGE_ROOT` antes de executar se precisar usar outro disco. A preparação pode baixar CIFAR-10 e WikiText-2, e a campanha pode ser longa.

Copie os diretórios inteiros `sessions/<nome>` para um PC e execute:

```bash
./bench.sh compare --baseline /caminho/sem-bitdefender --protected /caminho/com-bitdefender --output reports/comparison
```

Guarde `prepared/<nome>.json` de cada PC como registro da configuração local. Consulte `comparison.json` para diferenças de ambiente, exclusões, falhas e interferências. O gráfico e a tabela ficam em `comparison.png` e `comparison.csv`.
