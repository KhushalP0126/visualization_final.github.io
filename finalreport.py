# === Final Full Working Report ===

import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import base64
from plotly.offline import plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# === Paths ===
data_path = '/Users/khushalpandala/Downloads/final_project'
project_dir = '/'

final_report_html = os.path.join(project_dir, 'final_AI_energy_co2_report.html')
stats_summary_csv = os.path.join(project_dir, 'ieee_stats_summary.csv')
correlation_heatmap_png = os.path.join(project_dir, 'correlation_heatmap.png')
pred_vs_actual_png = os.path.join(project_dir, 'predicted_vs_actual.png')

# === Story Sections ===
story_section = """
<p>Ever since ChatGPT entered the public limelight, I was curious to know more about AI and its functioning. 
As a current computer engineering student with a research interest in AI, I have been exploring how AI model training affects power usage and how to make it more efficient. 
I am learning how hardware architecture can be maximized through simulation software that replicates the impact of design choices on energy conservation.</p>

<p>My interest deepened after watching a Veritasium video on AI solving the protein folding problem. 
Around the same time, my sister, a USC student, shared her experience with protein structure analysis using Python, which inspired me to bridge software innovation with circuit design and mathematical modeling. 
I realized that to truly solve these global challenges, both hardware and algorithms must evolve sustainably.</p>

<p>Through this project, I highlight the growing carbon footprint of modern AI infrastructures. 
Massive server farms, crypto mining, and large-scale AI model training all consume enormous energy—contributing to carbon emissions. 
By visualizing these impacts, I hope to advocate for a future where AI development is powerful yet environmentally sustainable.</p>
"""

story_server_power = """<p><i>Exploring server energy dynamics revealed how computational spikes during AI training dramatically increase energy needs. This realization strengthened my interest in energy-efficient AI training methods. Understanding these behaviors is critical for building the next generation of sustainable cloud architectures.</i></p>"""
story_cumulative_energy = """<p><i>Over time, even small server power spikes accumulate into substantial energy consumption. This cumulative effect became clear to me during my project, emphasizing the importance of efficient server management. Long-running AI experiments significantly contribute to total energy demand and emissions.</i></p>"""
story_gpu_summary = """<p><i>GPU architectures fascinated me due to their critical role in AI workloads. Observing average GPU power metrics showed how some hardware designs are inherently more efficient than others. Selecting energy-efficient GPUs could dramatically reduce AI's carbon footprint in the future.</i></p>"""
story_gpu_heatmap = """<p><i>Learning about TDP (Thermal Design Power) deepened my understanding of energy profiles for GPUs. Higher TDPs directly relate to higher emissions when GPUs are used extensively for AI model training. Future AI sustainability will hinge on better balancing performance with lower TDP hardware.</i></p>"""
story_co2_emissions = """<p><i>Looking at global CO₂ emissions trends from 2010–2021 contextualized AI's role in a larger environmental picture. It's easy to forget that even virtual models running on servers contribute to very real atmospheric changes. AI sustainability isn't just a technical challenge—it's a global imperative.</i></p>"""
story_stats_summary = """<p><i>Working with server statistics helped me understand baseline behaviors and outliers in AI-driven workloads. Statistical analysis uncovered not just typical loads, but rare surges that could be optimized away. These insights guide smarter, greener system designs going forward.</i></p>"""
story_correlation_heatmap = """<p><i>Strong correlations between GPU loads, CPU power, and energy usage taught me that these systems are tightly coupled. Optimizing one without considering the others misses opportunities for holistic improvement. Understanding these correlations is crucial for designing truly sustainable AI infrastructures.</i></p>"""
story_ml_prediction = """<p><i>Applying machine learning to predict energy consumption was both empowering and humbling. It showed me that models could help manage energy smarter, but also reminded me that model training itself has an energy cost. Machine learning is thus both a contributor to, and potential solution for, the CO₂ challenge.</i></p>"""

# === Load and clean data ===
ieee_file = os.path.join(data_path, "1mayo - agosto 2021.csv")
ieee_cols = ['fecha_servidor', 'potencia', 'energia', 'WORKSTATION_CPU_POWER', 'WORKSTATION_GPU_POWER']
ieee_df = pd.read_csv(ieee_file, usecols=ieee_cols)
ieee_df.rename(columns={'fecha_servidor': 'timestamp', 'potencia': 'power', 'energia': 'energy'}, inplace=True)
ieee_df['timestamp'] = pd.to_datetime(ieee_df['timestamp'], errors='coerce')
ieee_df.dropna(subset=['timestamp', 'power', 'energy'], inplace=True)

owid_file = os.path.join(data_path, "owid-co2-data.csv")
owid_cols = ['country', 'year', 'co2', 'co2_per_capita', 'primary_energy_consumption']
owid_df = pd.read_csv(owid_file, usecols=owid_cols)
owid_df.dropna(subset=['year', 'co2'], inplace=True)

gpu_file = os.path.join(data_path, "gpus.csv")
gpu_df = pd.read_csv(gpu_file)
gpu_df['name'] = gpu_df['name'].astype(str)

# === Statistical summaries ===
gpu_summary = gpu_df.describe().transpose().reset_index()[['index', 'mean']]
gpu_summary.columns = ['GPU_Model', 'Average_Metric']
stats_summary = ieee_df.describe()
stats_summary.to_csv(stats_summary_csv)

# === Correlation heatmap ===
plt.figure(figsize=(10, 8))
sns.heatmap(ieee_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap of IEEE Metrics")
plt.tight_layout()
plt.savefig(correlation_heatmap_png)
plt.close()

with open(correlation_heatmap_png, "rb") as f:
    encoded_heatmap = base64.b64encode(f.read()).decode('utf-8')
heatmap_img_tag = f'<img src="data:image/png;base64,{encoded_heatmap}" width="700">'

# === Machine Learning Section ===
ml_section = ""
try:
    ml_df = ieee_df[['WORKSTATION_CPU_POWER', 'WORKSTATION_GPU_POWER', 'power']].dropna()
    X = ml_df[['WORKSTATION_CPU_POWER', 'WORKSTATION_GPU_POWER']]
    y = ml_df['power']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Power")
    plt.ylabel("Predicted Power")
    plt.title("Predicted vs Actual Server Power")
    plt.tight_layout()
    plt.savefig(pred_vs_actual_png)
    plt.close()

    with open(pred_vs_actual_png, "rb") as f:
        encoded_pred_plot = base64.b64encode(f.read()).decode('utf-8')
    pred_img_tag = f'<img src="data:image/png;base64,{encoded_pred_plot}" width="500">'

    ml_section = f"""
    <p><b>Model:</b> Linear Regression</p>
    <p><b>R² Score:</b> {r2:.3f}</p>
    <p><b>Mean Squared Error:</b> {mse:.2f}</p>
    {pred_img_tag}
    """
except Exception as e:
    ml_section = f"<p>Error during model training: {str(e)}</p>"

# === Generate Plotly Visualizations ===
embed_line_power = plot(px.line(ieee_df.head(1000), x="timestamp", y="power", title="⚡ Server Power Over Time"), include_plotlyjs='cdn', output_type='div')
embed_area_energy = plot(px.area(ieee_df.head(1000), x="timestamp", y="energy", title="🔋 Cumulative Server Energy Consumption"), include_plotlyjs='cdn', output_type='div')
embed_gpu_bar = plot(px.bar(gpu_summary, x="GPU_Model", y="Average_Metric", title="🎮 Average GPU Metrics"), include_plotlyjs='cdn', output_type='div')
embed_gpu_heatmap = plot(px.density_heatmap(gpu_df, x='name', y='tdp_watts', title='🔥 GPU TDP Heatmap', color_continuous_scale='Turbo').update_layout(xaxis_tickangle=45), include_plotlyjs='cdn', output_type='div')

co2_by_year = owid_df[(owid_df['year'] >= 2010) & (owid_df['year'] <= 2021)].groupby('year')['co2'].sum().reset_index()
embed_co2_line = plot(px.line(co2_by_year, x='year', y='co2', title="🌡️ Global CO₂ Emissions (2010–2021)"), include_plotlyjs='cdn', output_type='div')

# === Build HTML ===
html_report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='UTF-8'>
<title>AI Energy & CO₂ Report</title>

<style>
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f9f9f9;
        margin: 0;
        padding: 20px;
        text-align: center;
    }}
    .viz-section {{
        background: #ffffff;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        padding: 30px;
        margin: 50px auto;
        width: 90%;
        max-width: 1000px;
        border-radius: 12px;
        text-align: center;
    }}
    .story-section {{
        background-color: #fff8dc;
        padding: 30px;
        margin: 50px auto;
        width: 90%;
        max-width: 1000px;
        border-radius: 12px;
        text-align: left;
        font-size: 18px;
        color: #333;
    }}
    h1 {{
        color: #2c3e50;
        font-size: 36px;
    }}
    h2 {{
        color: #3498db;
        font-size: 28px;
        margin-top: 30px;
    }}
</style>
</head>
<body>

<h1>📊 AI Energy Consumption and Carbon Emissions Dashboard</h1>

<div class="story-section">
<h2>🚀 My Journey: From AI Curiosity to Environmental Awareness</h2>
{story_section}
</div>

<div class="viz-section">
<h2>⚡ Server Power Over Time</h2>
{story_server_power}
{embed_line_power}
</div>

<div class="viz-section">
<h2>🔋 Cumulative Server Energy Consumption</h2>
{story_cumulative_energy}
{embed_area_energy}
</div>

<div class="viz-section">
<h2>🎮 GPU Metrics Summary</h2>
{story_gpu_summary}
{embed_gpu_bar}
</div>

<div class="viz-section">
<h2>🔥 GPU TDP Heatmap</h2>
{story_gpu_heatmap}
{embed_gpu_heatmap}
</div>

<div class="viz-section">
<h2>🌡️ Global CO₂ Emissions (2010–2021)</h2>
{story_co2_emissions}
{embed_co2_line}
</div>

<div class="viz-section">
<h2>📈 Statistical Summary</h2>
{story_stats_summary}
<p>Saved as CSV at: <code>{stats_summary_csv}</code></p>
</div>

<div class="viz-section">
<h2>📊 Correlation Heatmap</h2>
{story_correlation_heatmap}
{heatmap_img_tag}
</div>

<div class="viz-section">
<h2>🤖 ML Prediction Results</h2>
{story_ml_prediction}
{ml_section}
</div>

</body>
</html>"""

# === Save the final HTML ===
with open(final_report_html, "w") as f:
    f.write(html_report)

print("✅ Final report generated and saved!")
print(f"📂 Location: {final_report_html}")
