import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from gensim.models import Word2Vec
import pandas as pd
import jieba


# 读取语料文件并分词
def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 使用jieba进行分词
            words = jieba.cut(line.strip())
            yield list(words)

# 训练 Word2Vec 模型
def train_word2vec(corpus_file):
    sentences = list(read_corpus(corpus_file))
    model = Word2Vec(sentences, vector_size=20, window=5, min_count=1, workers=4)
    return model

# 加载 Word2Vec 模型
loaded_model = train_word2vec('/Users/chenxin/近期工作/Embedding算法/三国演义.txt')

# 创建 Dash 应用
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Word2Vec Similar Words"),
    dcc.Input(id='input-word', type='text', value='Enter word'),
    html.Button('Find Similar Words', id='submit-val', n_clicks=0),
    html.Div(id='output-container-button')
])

@app.callback(
    Output('output-container-button', 'children'),
    [Input('submit-val', 'n_clicks')],
    [Input('input-word', 'value')]
)
def update_output(n_clicks, input_word):
    if n_clicks > 0:
        similar_words = loaded_model.wv.most_similar(input_word, topn=10)
        similar_words_df = pd.DataFrame(similar_words, columns=['Word', 'Similarity'])
        return html.Table([
            html.Thead(html.Tr([html.Th(col) for col in similar_words_df.columns])),
            html.Tbody([
                html.Tr([
                    html.Td(similar_words_df.iloc[i][col]) for col in similar_words_df.columns
                ]) for i in range(len(similar_words_df))
            ])
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
