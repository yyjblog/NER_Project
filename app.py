import threading
import webbrowser

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from Config import Config
from process.Predictor import Predictor

app = Flask(__name__)
CORS(app)
# 允许的模型和数据集
ALLOWED_DATASETS = ['CNER', 'CLUENER', 'CMEEE']
ALLOWED_MODELS = ['bert_crf', 'bilstm_crf', 'roberta_crf']

# 全局变量
global_config = Config()
global_predictor = None

def reload_predictor():
    """根据当前 config 重新加载 predictor 实例"""
    global global_predictor
    global_predictor = Predictor(global_config, None)

# 第一次加载
reload_predictor()

@app.route("/exchange", methods=["POST"])
def exchange():
    data = request.get_json()
    dataset = data.get("dataset", "").strip()
    model_name = data.get("model_name", "").strip()

    if not dataset:
        return jsonify({"error": "数据集不能为空"}), 400
    if not model_name:
        return jsonify({"error": "模型不能为空"}), 400
    if dataset not in ALLOWED_DATASETS:
        return jsonify({"error": f"不支持的数据集：{dataset}，可选：{ALLOWED_DATASETS}"}), 400
    if model_name not in ALLOWED_MODELS:
        return jsonify({"error": f"不支持的模型：{model_name}，可选：{ALLOWED_MODELS}"}), 400

    global_config.dataset = dataset
    global_config.model_name = model_name
    global_config.update_paths()

    # 打印日志确认切换
    print(f"[模型切换] Dataset: {dataset}, Model: {model_name}")

    reload_predictor()

    return jsonify({
        "message": "模型与数据集切换成功",
        "dataset": dataset,
        "model_name": model_name
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "文本不能为空"}), 400

    try:
        print(f"[预测] 输入文本: {text}")
        result = global_predictor.predict_text(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 运行直接打开网页版本
# @app.route('/')
# def home():
#     return render_template('index.html')
#
# def open_browser():
#     webbrowser.open_new("http://127.0.0.1:8000/")
#
# if __name__ == "__main__":
#     threading.Timer(1.25, open_browser).start()  # 延迟打开浏览器
#     app.run(host='127.0.0.1', port=8000, debug=True)


# 前后端分离版本
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)