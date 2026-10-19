# ts-hrm-finetuning
Sim, existem ótimos datasets de TypeScript (TS) prontos no Hugging Face focados exatamente em Supervised Fine-Tuning (SFT) e que servem perfeitamente para essa finalidade.
Como o objetivo do HRM-Text-1B é desenvolver raciocínio lógico profundo, você deve priorizar datasets no formato Instrução-Resposta em vez de código bruto puro. Abaixo estão os melhores datasets prontos para TS:

1. Os Datasets Recomendados

• grenishrai/typescript-dataset: Este é provavelmente o melhor dataset para começar o treino do HRM. Ele foca em Advanced Reasoning (Raciocínio Avançado) em TypeScript. Ele contém milhares de desafios complexos sobre inferência de tipos, condicionais, genéricos e análise estática. É ideal porque força o "módulo lento/supervisor" do HRM a planejar os tipos lógicos complexos antes de gerar a sintaxe. [1] (https://huggingface.co/datasets/grenishrai/typescript-dataset)
• mhhmm/typescript-instruct-20k: Um dataset bem maior contendo 20 mil pares de instruções e explicações/gerações de código puramente em TypeScript. Excelente para dar "corpo" e vocabulário de código ao modelo. [1] (https://huggingface.co/datasets/mhhmm/typescript-instruct-20k/viewer/default/train)
• bleugreen/typescript-instruct: Outra opção pronta em formato de instrução voltada para a construção de funções específicas em TypeScript de nível de produção. [1] (https://huggingface.co/datasets/bleugreen/typescript-instruct/viewer)

2. Como preparar o Dataset de TS para o HRM no Colab

Os datasets acima vêm no formato padrão do Hugging Face (com colunas como instruction e output). Lembra que o HRM-Text exige os Special Condition Tokens e a marcação no token_type_ids? [1] (https://www.youtube.com/watch?v=oCx22mwtw2Y&t=53)
Você precisará usar uma função de mapeamento (map) simples no Colab para estruturar o dataset de TypeScript antes do treino. Veja o exemplo de código prático:
python
from datasets import load_dataset

# 1. Carrega o dataset de raciocínio em TypeScript do Hugging Face
dataset = load_dataset("grenishrai/typescript-dataset", split="train")

# 2. Função para injetar os prefixos exigidos pelo HRM e tokenizar
def prepare_hrm_data(example):
    # Formata o texto com as tags que o cérebro hierárquico usa para "pensar"
    prompt = f"<|quad_end|><|object_ref_end|> Instrução: {example['instruction']}\nResposta: "
    target = example['output']
    
    # Junta os dois em um texto contínuo de treino
    full_text = prompt + target
    
    # Tokeniza
    tokenized = tokenizer(full_text, truncation=True, max_length=512)
    
    # Tokeniza apenas o prompt para sabermos o tamanho dele
    prompt_len = len(tokenizer(prompt)["input_ids"])
    
    # CRUCIAL PARA O HRM: Criar a máscara onde o Prompt recebe '1' (Atenção Bidirecional)
    # e a Resposta recebe '0' (Atenção Causal)
    token_type_ids = [1] * prompt_len + [0] * (len(tokenized["input_ids"]) - prompt_len)
    
    # Adiciona os vetores necessários para o cálculo da Loss do modelo
    tokenized["token_type_ids"] = token_type_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

# 3. Aplica a formatação em lote
dataset_pronto = dataset.map(prepare_hrm_data, remove_columns=dataset.column_names)
Use o código com cuidado.

O que vai acontecer na prática?

Ao alimentar o modelo com o dataset do grenishrai/typescript-dataset usando essa formatação, você ensinará o HRM-Text-1B a não apenas cuspir código decorado, mas a usar o loop de recorrência interna (módulos H e L trabalhando juntos) para mapear a complexa árvore de tipos do TypeScript. Como o dataset tem pouco mais de 3.000 linhas densas de alto nível, o treinamento por QLoRA na GPU T4 gratuita do Colab vai terminar em pouquíssimos minutos. [1] (https://huggingface.co/sapientinc/HRM-Text-1B), [2] (https://huggingface.co/datasets/grenishrai/typescript-dataset)
