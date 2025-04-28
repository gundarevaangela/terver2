from flask import Flask, render_template, request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from scipy.stats import kstest, norm

app = Flask(__name__)

def parse_input(data):
    return [float(x.strip()) for x in data.split(",") if x.strip()]

def generate_histogram(values, title):
    fig, ax = plt.subplots()
    count, bins, ignored = ax.hist(values, bins=8, color='blue', alpha=0.7, edgecolor='black', density=False)
    
    mu, std = norm.fit(values)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    
    p = p * max(count) / max(p)
    
    ax.plot(x, p, 'r', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel('Отклонения')
    ax.set_ylabel('Количество измерений')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return graph_url

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            deviations_a = parse_input(request.form['deviations_a'])
            counts_a = parse_input(request.form['counts_a'])
            deviations_b = parse_input(request.form['deviations_b'])
            counts_b = parse_input(request.form['counts_b'])

            if len(deviations_a) != len(counts_a) or len(deviations_b) != len(counts_b):
                raise ValueError("Количество отклонений и измерений должно совпадать!")

            # Завод A
            expanded_a = []
            for value, count in zip(deviations_a, counts_a):
                expanded_a.extend([value] * int(count))

            mean_a = np.mean(expanded_a)
            std_a = np.std(expanded_a, ddof=1)
            min_a = np.min(expanded_a)
            max_a = np.max(expanded_a)

            # Завод B
            expanded_b = []
            for value, count in zip(deviations_b, counts_b):
                expanded_b.extend([value] * int(count))

            mean_b = np.mean(expanded_b)
            std_b = np.std(expanded_b, ddof=1)
            min_b = np.min(expanded_b)
            max_b = np.max(expanded_b)

            # Проверка нормальности (Kolmogorov-Smirnov test)
            statistic_a, p_value_a = kstest(expanded_a, 'norm', args=(mean_a, std_a))
            normal_a = p_value_a > 0.05

            histogram_a = generate_histogram(expanded_a, "Распределение отклонений (Завод A)")
            histogram_b = generate_histogram(expanded_b, "Распределение отклонений (Завод B)")

            result = {
                'a': {
                    'n': len(expanded_a),
                    'mean': mean_a,
                    'std': std_a,
                    'min': min_a,
                    'max': max_a,
                    'statistic': statistic_a,
                    'p_value': p_value_a,
                    'normal': normal_a,
                    'histogram': histogram_a
                },
                'b': {
                    'n': len(expanded_b),
                    'mean': mean_b,
                    'std': std_b,
                    'min': min_b,
                    'max': max_b,
                    'histogram': histogram_b
                }
            }

        except Exception as e:
            result = {'error': str(e)}

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
