import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from gensim.models import Word2Vec
import pandas as pd

# 读取语料文件并分词
def read_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield line.split()

# 训练 Word2Vec 模型
def train_word2vec(corpus_file):
    sentences = list(read_corpus(corpus_file))
    model = Word2Vec(sentences, vector_size=300, window=20, min_count=5, workers=5)
    return model

# 加载 Word2Vec 模型
loaded_model = train_word2vec('/Users/chenxin/近期工作/Embedding算法/text8.txt')

# 创建 Dash 应用
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Word2Vec Similar Words"),
    dcc.Input(id='input-word', type='text', value='Enter word'),
    html.Button('Find Similar Words', id='submit-val', n_clicks=0),
    html.Button('Show All Word Vectors', id='show-all-val', n_clicks=0),
    html.Button('Export Word Vectors to Excel', id='export-val', n_clicks=0),
    html.Div(id='output-container-button'),
    html.Div(id='output-all-word-vectors'),
    html.Div(id='output-save-message')
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

@app.callback(
    Output('output-all-word-vectors', 'children'),
    [Input('show-all-val', 'n_clicks')]
)
def show_all_word_vectors(n_clicks):
    if n_clicks > 0:
        all_words = loaded_model.wv.index_to_key
        all_vectors = [loaded_model.wv[word] for word in all_words]
        all_vectors_df = pd.DataFrame(all_vectors, index=all_words)
        return html.Table([
            html.Thead(html.Tr([html.Th(col) for col in all_vectors_df.columns])),
            html.Tbody([
                html.Tr([
                    html.Td(all_vectors_df.iloc[i][col]) for col in all_vectors_df.columns
                ]) for i in range(len(all_vectors_df))
            ])
        ])

@app.callback(
    Output('output-save-message', 'children'),
    [Input('export-val', 'n_clicks')]
)
def save_word_vectors(n_clicks):
    if n_clicks > 0:
        all_words = loaded_model.wv.index_to_key
        all_vectors = [loaded_model.wv[word] for word in all_words]
        all_vectors_df = pd.DataFrame(all_vectors, index=all_words)
        file_path = 'word_vectors.xlsx'
        all_vectors_df.to_excel(file_path)
        return html.Div(f'Word vectors exported to {file_path}')

if __name__ == '__main__':
    app.run_server(debug=True)
