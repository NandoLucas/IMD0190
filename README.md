# IMD0190
Repositório destinado ao desenvolvimento do projeto final da disciplina.

## Descrição do projeto
Nesse trabalho, desenvolvemos três projetos diferentes com a temática deep learning. 

O primeiro projeto visa desenvolver um agente capaz de classificar as avaliações dos cursos oferecidos pela Escola Judiciária Eleitoral do Rio Grande do Norte (EJE/RN). A EJE tem como objetivo promover a formação, atualização e especialização de servidores, magistrados e estagiários, oferecendo uma variedade de cursos de capacitação. Esses cursos são avaliados por meio de feedbacks fornecidos pelos participantes. O projeto propõe realizar uma análise de sentimentos, classificando as avaliações coletadas pelo Google Forms em categorias de feedback positivo, negativo e neutro. Para proporcionar uma experiência interativa e amigável para os usuários foi utilizada a biblioteca Streamlit como interface do chatbot.

O segundo projeto tem como objetivo desenvolver um agente inteligente capaz de responder às dúvidas dos usuários sobre os dados disponíveis no Portal de Dados Abertos do TRE-RN. Essa plataforma disponibiliza conjuntos de dados públicos relacionados ao Tribunal Regional Eleitoral do Rio Grande do Norte, promovendo o acesso à informação e fortalecendo a cultura da transparência. O agente é capaz de responder a diversas perguntas sobre os arquivos CSV disponíveis, como, por exemplo, informar a quantidade de servidores do TRE-RN.

O terceiro projeto aborda o mesmo tema do segundo, oferecendo uma solução para responder às dúvidas dos usuários do Portal de Dados Abertos. No entanto, neste caso, será utilizada a técnica de Retrieval-Augmented Generation (RAG), que combina a busca de informações com a geração de respostas mais contextuais e precisas.

## Como executar o projeto

Inicialmente, é necessário possuir uma chave da API do Gemini. Foi utilizada uma chave de API gratuita.
ELa pode ser obtida em [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

Adicione a sua chave da API em um arquivo .env na variável **GOOGLE_API_KEY**

Para a execução do projeto localmente, listamos os comandos necessários:

```bash
# Clone este repositório
git clone [https://github.com/NandoLucas/IMD0190.git](https://github.com/NandoLucas/IMD0190.git)


# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual

source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt

# Execute o notebook IMD0190.ipynb para ver os resultados obtidos nos projetos

# Para visualizar o chatbot para análise de sentimentos, execute o comando abaixo
streamlit run .\app-sentiment-analysis.py
```

