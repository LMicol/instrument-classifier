from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # for testing:
    # app.run(debug=True)

    # for serving: 
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)