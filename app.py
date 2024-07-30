from flask import Flask, render_template, request
import yt_data_api

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        df, summary = yt_data_api.main(url)
        table_html = df.to_html(index=False, classes=['dataframe'])  # Remove index and add class for styling
        
        # Generate visualizations
        yt_data_api.wordcloud(df)
        yt_data_api.barchart(df)
        yt_data_api.piechart(df)
        
        return render_template('results.html', 
                               summary=summary, 
                               table_data=table_html,
                               wordcloud='wordcloud.png',
                               barchart='barchart.png',
                               piechart='piechart.png')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)