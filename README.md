# ARC Episode Viewer - starter kit

Este projeto é independente do código dos agentes ARC. A aplicação não importa,
executa ou altera o simulador. Limita-se a ler ficheiros JSON produzidos por um
exportador privado e a apresentar os episódios de forma visual.

O pacote inclui dois episódios de desenvolvimento exportados com o mesmo
cenário e seed: uma política de referência e um agente v4. Estes ficheiros
permitem começar imediatamente a navegação temporal e a comparação, sem acesso
ao repositório, ao modelo ou ao checkpoint.

## Arranque rápido no Windows

Requer Python 3.10 ou superior.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Também é possível arrastar novos ficheiros JSON para a aplicação. Cada ficheiro
é validado antes de ser apresentado.

## Estrutura

- `app.py`: ponto de partida da interface;
- `arc_student/bundles.py`: leitura e validação dos ficheiros;
- `schema/arc_episode_bundle_v1.schema.json`: contrato público do formato;
- `sample_bundles/`: dois outputs aprovados para desenvolvimento;
- `TODO_STUDENT.md`: funcionalidades a desenvolver;
- `INSTRUCTIONS_TO_STUDENT.md`: instruções iniciais;
- `SECURITY_AND_SCOPE.md`: separação entre a interface e o ARC.

## Modelo de trabalho

1. O ARC executa um cenário dentro do repositório privado.
2. O exportador escreve um `arc-episode-bundle-v1` em JSON.
3. Apenas o JSON aprovado é enviado para este projeto.
4. A interface lê o JSON e permite reproduzir, analisar e comparar episódios.

Não é necessário conhecer aprendizagem por reforço para começar. Para a
interface, uma política é apenas a versão do agente indicada nos metadados do
ficheiro.

## Testes

```powershell
python -m unittest discover -s tests -v
```

Os outputs fornecidos são sintéticos e exploratórios. Não são dados operacionais
nem demonstram superioridade de um agente.

