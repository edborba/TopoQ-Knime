# TopoQ — extensão KNIME em Python puro

Extensão KNIME com três nós que cobrem o fluxo completo de otimização de
geometria semiempírica: execução em lote, reconstrução da estrutura preservando
a topologia original e validação geométrica do resultado.

Todos os nós aparecem em **Community Nodes → TopoQ**.

## Nós

### Batch Geometry Optimizer

Recebe uma tabela com coluna de ID e coluna de molécula (SDF ou MOL). Converte
cada molécula com o OpenBabel, executa o MOPAC em paralelo e faz o parsing dos
arquivos `.out`. Devolve uma linha por molécula de entrada, na ordem de entrada.

Arquivos `.mop` e `.out` existentes são reaproveitados — reexecutar o nó calcula
apenas o que falta. As palavras-chave do MOPAC são aplicadas somente quando um
`.mop` é criado; arquivos reaproveitados mantêm as palavras-chave com que foram
gerados. Marque "Delete saved files before running" para recalcular tudo.

A falha de uma molécula é registrada no log e a linha correspondente fica com
valores ausentes — o lote não é interrompido. O cancelamento encerra os
processos em execução e remove arquivos parciais.

> A coluna `Molecule (MOPAC)` contém a conversão bruta do `.out` pelo OpenBabel:
> as coordenadas estão otimizadas, mas o grafo (ligações e **cargas formais**) é
> inferido a partir de distâncias e pode estar errado. Ligue esta saída ao nó de
> transferência de coordenadas.

### Structure Correction

Duas entradas: os resultados otimizados (1ª) e as moléculas originais de
referência (2ª). Pareia as linhas por ID e reconstrói cada estrutura como **o
grafo original com as coordenadas otimizadas**, preservando ligações,
estereoquímica e cargas.

Resolve a perda de carga formal em espécies como amônio quaternário, N-óxidos e
zwitteriônicos, que o XYZ/`.out` não consegue representar. Linhas que não podem
ser corrigidas (sem correspondente, ID ambíguo, contagem de átomos divergente)
ficam com a célula de molécula vazia e o motivo é registrado no log.

### Bond Distance Checker

Compara a distância de cada ligação com a soma dos raios covalentes multiplicada
por um fator de tolerância, sinalizando ligações longas demais ou átomos
colapsados. Como o nó anterior preserva o grafo original, ele mascararia uma
otimização que falhou de fato — este nó é o que restaura a capacidade de
detectar esse caso, e por isso é parte obrigatória do fluxo.

Adiciona as colunas `Bond Warning`, `Max Bond Distance (Å)` e `Long Bonds`. O
`Bond Warning` é tri-estado: verdadeiro, falso ou **ausente** (não foi possível
verificar) — ausência não significa aprovação.

## Fluxo recomendado

```
Moléculas → [Batch Geometry Optimizer] → [Structure Correction] → [Bond Distance Checker]
                 │                              ▲
                 └──────────────────────────────┘
                        (originais na 2ª entrada)
```

## Estrutura do projeto

```
TopoQ - Knime/
├── config.yml          registro da extensão para desenvolvimento
├── topoq/              a extensão (código e ícones) — é o que vai no pacote
├── tests/              testes; fora de topoq/ para não entrar no pacote
├── topoq_build_5.4/    update site compilado para KNIME 5.4.x
└── documentos/         documentação dos nós, guias e material do artigo
```

Documentação detalhada, em `documentos/`:

| Arquivo | Conteúdo |
|---|---|
| `Documentação - Batch Geometry Optimizer.txt` | nó de cálculo: parâmetros, saída, cache, internals |
| `Documentação - Structure Correction.txt` | nó de correção: entradas, casos de falha, notas de projeto |
| `Documentação - Bond Distance Checker.txt` | nó de validação: onde conectar, colunas, tolerâncias |
| `Documentação - Compilação para KNIME 5.4.txt` | como compilar e instalar em outra máquina |
| `Roteiro - Artigo Cientifico.md` | fundamentação, plano do artigo e nomenclatura |

## Desenvolvimento

Registre a extensão adicionando ao `knime.ini`:

```text
-Dknime.python.extension.config=C:/Users/eborba/Documents/TopoQ - Knime/config.yml
```

Testes (rodam sem KNIME, um script autônomo por nó):

```text
C:\Users\eborba\.conda\envs\topoq-env\python.exe tests\test_batch_geometry_optimizer.py
```

## Compilação e distribuição

Ver `documentos/Documentação - Compilação para KNIME 5.4.txt`. Em resumo: a
versão do pacote `knime-extension-bundling` **é** a versão de KNIME de destino.
O update site já compilado para KNIME 5.4.x está em `topoq_build_5.4/`.

## Requisitos

- KNIME Analytics Platform (o pacote em `topoq_build_5.4` exige 5.4.x)
- Extensão KNIME Chemistry Types (`org.knime.features.chem.types`)
- MOPAC no PATH ou em `C:\mopac\bin\mopac.exe`
- OpenBabel (`obabel.exe`) no PATH ou configurado no diálogo do nó

MOPAC e OpenBabel são projetos independentes, não distribuídos com esta
extensão. Esta extensão não é afiliada a eles nem endossada por eles.
