<body>
<h1 align="center"> --- </h1>

<div align="center">
<p>Antes de tudo; estatística nunca poderá ser resumida em uma única publicação. 
  Ela abrange muitos conceitos matemáticos e é um campo próprio dentro de cálculo. 
  Esse como as outras publicações para aprendizado de <a href="https://www.tabnews.com.br/BorboletaVermelha/o-que-e-ser-um-cientista-de-dados"> 
  Data scientist<a> são apenas as abordagens iniciais dos conteúdos e existe muito mais para ser estudado e analisado. 
  <br><br>E quero agradecer os comentários da última publicação sobre 
  <a href="https://www.tabnews.com.br/BorboletaVermelha/visualizando-dados-em-python"> visualizando dados em python<a>, 
  pois tive a oportunidade de conhecer novas ferramentas, 
  como a <a href="https://plotly.com/python/getting-started/"> PlotLy <a> e até mesmo sites como 
  <a href="https://python-graph-gallery.com/"> The Python Graph Gallery<a>
  Está se tornando cada vez mais prazeroso poder compartilhar esse 
  pequeno guia e ter a colaboração dos membros da comunidade da TabNews. 
<p>
</div>

---

## O QUE É ESTATÍSTICA? 
<div align="justify">
<p>
  A estatística é uma disciplina da matemática que se concentra na coleta, análise, interpretação e visualização de dados. 
  Então ela permite a gente a identificar padrões, fazer previsões e tomar decisões. A estatística é muito velha, ela foi 
  utilizada até mesmo no antigo Egito para registro das colheitas. Há mais de dois mil anos, a China já se preocupava com o 
  crescimento populacional por meio de censos. Já no século XIV, o início do Renascimento na Europa também proporcionou novos rumos à ESTATÍSTICA, necessária especialmente para aprimorar 
  a administração de governos. E esses são otimos <a href="http://www.juventudect.fiocruz.br/estatistica#:~:text=A%20ESTAT%C3%8DSTICA%20surgiu%20quando%20governos,registros%20estat%C3%ADsticos%20de%20suas%20colheitas"> exemplos<a>.
  <br><br> A estatística é amplamente aplicada em diversas áreas, não são apenas os cientistas de dados que utilizam estatística, há utilização em 
    ciências da natureza, ciências sociais, negócios, economia, engenharia e muitas outras áreas. É uma ferramenta poderosa para principalmente 
    compreender e lidar com a incerteza, fornecendo métodos para quantificar e avaliar a variabilidade nos dados. Em resumo, a estatística desempenha 
    um papel crucial na compreensão do mundo ao nosso redor por meio da análise e interpretação de dados.
    <br><br>E compreender essa área pelo menos na sua forma básica é uma parte fundamental da análise de dados, irei fornecer os conceitos e ferramentas necessários para entender e interpretar conjuntos de dados. Um dos primeiros passos na análise de dados é a descrição das características de nossas informações, o que inclui *medidas de tendência central* e *dispersão*.


---

## **Medidas de Tendência Central:**

<div align="justify">
<p>As medidas de tendência central descrevem onde os dados tendem a se concentrar. 
Qual é a maior quantidade em que eles existem por exemplo. Aqui estão três das principais 
  medidas de tendência central. <strong>A média, mediana e moda</strong>. Mas antes de tudo, iremos 
  primeiro usar bibliotecas para a utilização das fórmulas adequadas. 
</div>

```python
import numpy as np
from scipy import stats
```
<div align="justify">

Quer que eu seja sincero? Essas importações realmente nem são tão importantes assim para se utilizar. 
As fórmulas de tendência central são básicas o suficiente para serem criadas sem problemas com <code>python</code> 
sem adição de outras bibliotecas, e até mesmo com outras linguagens, como <code>C++</code> ou <code>Lua</code> existe uma certa 
facilidade para criar esses cálculos. Mas eu utilizarei o <code>Numpy</code> e o <code>SciPy</code> porque já são bibliotecas que 
teremos que constantemente utilizar durante as próximas publicações. Então é melhor usar agora do que depois, certo?
<br><br>
Eu fiz aleatoriamente uma série de valores para ser nossa base de dados, pois o intuito no momento é apenas apresentar as fórmulas, mas sempre costume a testar com base de dados reais.
</div>

```python
Base_De_Dados = [29, 34, 48, 47, 41, 38, 33, 42, 43, 22, 35, 43, 25, 34, 25, 28, 34, 32, 47,
 46, 27, 44, 22, 35, 20, 49, 43, 34, 43, 42, 30, 26, 43, 33, 24, 27, 45, 49, 41, 28]
```

<div align="justify">

Caso seja difícil ver uma série de valores sem sentido nenhum e começar a analisar, 
criarei um contexto; pense que é o registro de 50 pessoas numa sala, e esses valores
representam a idade de cada uma delas e queremos descobrir como está a distribuição desses dados 
<br><br>Vamos começar com a análise através da média, isso é um conteúdo que se estuda durante o 
ensino médio, mas vou descrever como a <em>soma de todos os valores dividida pela quantidade de produtos </em>. 
Ela é utilizada para representarmos o valor típico central de nossos dados. Se você possui dois pontos, provavelmente 
a média vai ser o meio do caminho entre eles. E quanto mais pontos você adiciona para alguma das direções, mais ele cresce ou diminui. 
Caso visualize os dados e perceba que há valores muito extremos que alterem a média. Sempre será necessário uma limpeza na informação.  
<br>
E a fórmula de forma prática é assim: 
</div>

---


$$
X=\sum\frac{Xi}{N}
$$

<div align="center">
<strong>X</strong> = ᴍᴇᴅɪᴀ <br>
<strong>Xi</strong> = ꜱᴏᴍᴀ ᴅᴏꜱ ᴠᴀʟᴏʀᴇꜱ <br>
<strong>N</strong>  = Qᴜᴀɴᴛɪᴅᴀᴅᴇ ᴅᴇ ᴠᴀʟᴏʀᴇꜱ
</div>

---

<div align="justify">
  <p>
  Olhando assim pode parecer bem complicado para alguns, afinal de conta, existem esses símbolos como Σ, que apenas representa 
  <strong>somatória</strong>. E isso também é algo simples, para os que já sabem como média funciona, devem olhar para isso e 
    até mesmo questionar o motivo de eu ter deixado dessa forma, e direi que as fórmulas são ferramentas importantes e fazem
    parte de trabalhos científicos, é importante treinar sua leitura. Principalmente caso queira entrar na parte mais técnica, 
    e ler artigos mais difíceis. Mas eu irei trazer um exemplo para deixar mais claro. 
  </p>
<br>
</div>

  ---

$$
X = \frac {10+12+9+6+13} {5}
$$

$$
X = \frac{50}{5}
$$

$$
X = 10
$$

---
<div align="justify">
  <p>

Esse é um exemplo bem simples de uma das fórmulas de tendencia central, viu? Não é tão difícil assim, 
foi soma e divisão. Apenas coisas que se utilizava no fundamental II. E é claro, caso sinta a dúvida em alguma das 
fórmulas que aparecer posteriormente, procure exemplos na internet e vídeos no youtube. Matemática pode ser um assunto 
delicado para alguns e para outros pode ser facilmente explicada. E está tudo bem!!! Nunca se julgue caso não entenda esses conceitos, 
busque com calma e faça sempre no seu jeito e no seu tempo. 
<br><br>
Mas bem, agora que temos um exemplo e base de dados, vamos executar em código, o que é muito mais fácil do que pegar um lápis e começar 
a montar toda a fórmula para encontrar o resultado, e o código fica assim:
  <p>
<br>
</div>

```python
media = np.mean(Base_De_Dados)
print("Média:", media)
```

<img src="https://github.com/Borboleta-Vermelha/BLOGS/blob/main/imagens/M%C3%89DIA.png?raw=true" align="center">

---

<div align="justify">
  <p> Fácil, não?
    <br><br>
    Agora vamos falar da mediana. A mediana é o valor que divide o conjunto de dados ao meio quando ordenado. 
    Ele é literalmente o valor do meio de nossa base de dados. Mas se houver um número ímpar de valores, a mediana é simplesmente o valor do meio. 
    Se houver um número par de observações, a mediana é a média dos dois valores do meio.
    <br><br>
    Um exemplo simples de mediana:
  <p>
<br>
</div>

---

$$
X = [4, 6, 8, 10, 12]
$$

$$
X = [8]
$$

---

<div align="justify">
  <p> O valor do meio. Outro exemplo simples. 
    Entretanto; a base de dados que normalmente utilizaremos vão ser em grande escala,
    então porque não fazemos isso em código? 
  <p>
</div>

```python
mediana = np.median(Base_De_Dados)
print("Mediana:", mediana)
```

<img src="https://github.com/Borboleta-Vermelha/BLOGS/blob/main/imagens/MEDIANA.png?raw=true" align="center">

---

<br>
<div align="justify">
  <p> Um número bem próximo da média. Com a diferença de 1,2 para os resultados.
    Mas repare por exemplo que diferente da média, a mediana não depende de cada 
    valor nos seus dados. Por exemplo, se você aumentar o maior ponto (ou diminuir o menor ponto), 
    os pontos do meio permanecem intactos, logo, a mediana também. Ela só iria se alterar caso os dois 
    valores do meio tivessem alteração. Então ela pode ser suspeita se apenas for usada sozinha.
    <br><br>
    Mas por último entre as medidas de tendencia central é a <strong>Moda</strong>, que descreve o 
    valor mais frequente no conjunto de dados. O que mais aparece. Isso tem como função principal uma certa limpeza nós dados.
  <p>
</div>

```python
moda = stats.mode(Base_De_Dados)
print("Moda:", moda)
```

<img src="https://github.com/Borboleta-Vermelha/BLOGS/blob/main/imagens/MODA.png?raw=true" align="center">

---


<br>
<div align="justify">
  <p>Além do resultado da moda, que para nossa base de dados é 43, ele também nos fornece a quantidade de vezes que o valor aparece, sendo no caso 5 vezes.
  <br><br>
  As medidas de tendencia central são o básico da estatística, normalmente nem se utiliza tanto mediana ou moda. O que mais usaremos provavelmente será 
  a média. Mas também é necessário apresentar as medidas de dispersão desses dados.
</p> </div>

---

## **Medidas de Dispersão:**
<div align="justify">
  <p>A dispersão se refere à medida de como os nossos dados estão espalhados. 
    Se eles têm valores muito distantes ou estão muito próximos, caso o resultado 
    dos valores é perto de zero significam não estão espalhados de forma alguma e para
    valores maiores significa que estão muito espalhados. Por exemplo, uma simples medida é a <strong>amplitude</strong>, 
    que é a diferença entre o maior e o menor elemento:</p></div>

$$
\text {Amplitude = Valor Máximo−Valor Mínimo}
$$

```python
amplitude = np.max(Base_De_Dados) - np.min(Base_De_Dados)
print("Amplitude:", amplitude)
```

<img src="https://github.com/Borboleta-Vermelha/BLOGS/blob/main/imagens/AMPLITUDE.png?raw=true" align="center">

---

<div align="justify">
  <p> Amplitude alta significa que os dados estão bem espalhados, caso o resultado fosse 0, 
    obrigatoriamente todos os dados teriam o mesmo valor. Por outro lado, se a amplitude é ampla, 
    então o max é bem maior do que o min e os dados estão mais espalhados. E isso não necessariamente 
    significa algo bom, apenas depende da base de dados e a distribuição dos valores e quantidade de valores. 
    É preciso fazer <strong>análise</strong> e isso é algo consideravelmente difícil de se ensinar, é algo que precisamos fazer com calma e compreender o contexto para cada caso.
    <br><br> Mas agora vamos partir para a próxima medida de dispersão, a variância, essa é uma medida da dispersão dos dados em relação à média. Valores de dados que estão 
    mais distantes da média contribuem mais para a variância, enquanto valores mais próximos têm uma contribuição menor. Deixe-me apresentar a fórmula:
  </p></div>

---

$$
Variância=\sum \frac{(xi−x)^{2}}{n}
$$

<div align="center">
<strong>X</strong> = ᴍᴇᴅɪᴀ <br>
<strong>Xi</strong> = ꜱᴏᴍᴀ ᴅᴏꜱ ᴠᴀʟᴏʀᴇꜱ <br>
<strong>N</strong>  = Qᴜᴀɴᴛɪᴅᴀᴅᴇ ᴅᴇ ᴠᴀʟᴏʀᴇꜱ
</div>

---
<div align="justify">
  <p> Esse pode parecer um pouco mais difícil de entender, mas como a última a da média, 
    recomendo que busque tutoriais ou mais exemplos caso queira compreender afundo. Porém, 
    vamos direto ao código desta vez:
  </p></div>

```python
variancia = np.var(Base_De_Dados)
print("Variância:", variancia)
```

<img src="https://github.com/Borboleta-Vermelha/BLOGS/blob/main/imagens/VARI%C3%82NCIA.png?raw=true" align="center">

---

<div align="justify">
  <p> A variância é um método mais complexo para a análise de dispersão, pois ela utiliza a média como base e ela torna o resultado mais 
    “preciso”. A variância apenas nos diz se os nossos valores estão distantes ou não da nossa média. Nesse nosso exemplo, como sabemos 
    a média é 35, e temos números como 20 e 49, existe uma grande taxa de variância em relação à média.
    <br><br> Vamos agora partir para algo mais intuitivo, uma medida onde a dispersão pode ser baseada na média,
    mas ela é considerada mais visual para nós. E essa é o <strong>Desvio Padrão(σ)</strong>. Que de forma prática, é simplesmente a raiz da variância.

$$
σ= \sqrt {Variância}
$$

<div align="center">
<strong>σ</strong> = ᴅᴇꜱᴠɪᴏ ᴘᴀᴅʀᴀᴏ
</div>

---

```python
desvio_padrao = np.std(Base_De_Dados)
print("Desvio Padrão:", desvio_padrao)
```

<img src="https://github.com/Borboleta-Vermelha/BLOGS/blob/main/imagens/DESVIO_PADR%C3%83O.png?raw=true" align="center">

---

<div align="justify">
  <p>Normalmente a distribuição dos dados, tem uma diferença de 8 com a média. Então existe um desvio..padrão. 
    Acho que o nome já diz bastante coisa. É um desvio padrão dá média. E muitos preferem usar ele diretamente 
    do que a variância, pois o resultado é muito mais compreensível. 
  </p></div>
  
---

## E agora?

<div align="center">
  <p> Primeiramente quero reforçar a ideia de que a estatística é uma ferramenta poderosa para compreender e interpretar conjuntos de dados, 
    identificar padrões e tomar decisões informadas. No entanto, este é apenas o básico, e eu apenas quero apresentar alguns conceitos que utilizaremos.
  <br><br> Caso queira alguma recomendação para começar do zero, eu direi para ver a plataforma do <a href="https://pt.khanacademy.org/math/em-mat-estatistica/x5d13d3b4b5b8c419:introducao-a-estatistica"> Khan Academy<a>
   , que é um lugar gratuito e bom para reforçar os conteúdos do ensino fundamental e superior. 
    Se tiver mais experiencia e um inglês consideravelmente avançado, sempre existem recursos como opções de livros, como o 
    <a href="https://openstax.org/details/books/introductory-statistics"> introdução para estatistica</a>.
    <br><br>
      E para os programadores mais experientes, compartilhem aqui lugares que estudaram sobre estatística e até mesmo conceitos que acham importante. Logo mais quero falar de probabilidade, que é um ramo mais complexo e muito mais diversos em estatística.
<br><br>
E logo mais terá mais postagens desse mundo de Data Science. 🦋 </p></div>
