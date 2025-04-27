# 📊 AI Energy Consumption and Carbon Emissions Dashboard

This project explores the hidden environmental cost of artificial intelligence workloads, focusing on server power usage, GPU characteristics, and global CO₂ emissions. It combines real server energy data with worldwide emissions trends to showcase AI’s growing environmental footprint.

Originally proposed to compare AI model energy consumption directly (e.g., GPT-3 vs BERT), the project evolved into a real-time server power analysis with machine learning prediction, using publicly available server logs and emissions datasets.

---

## 📚 Project Overview

- **Server Power Usage:** Analyzed real-time server energy (power and cumulative energy consumption).
- **GPU Metrics:** Studied GPU characteristics like TDP (thermal design power) to highlight inefficiencies.
- **CO₂ Emissions:** Visualized global CO₂ emission trends (2010–2021) to contextualize AI's indirect impact.
- **Correlation Analysis:** Explored relationships between CPU load, GPU load, and total server power usage.
- **Machine Learning Prediction:** Built a regression model to estimate server power draw from CPU/GPU activity.

While the original plan to directly compare AI model training footprints (GPT-3, BERT, ResNet, etc.) was limited by dataset availability, this project successfully visualizes infrastructure-level trends and highlights key challenges for AI sustainability.

---

## 📂 Datasets Used

- **Server Energy Dataset (IEEE Dataport)**  
  `1mayo - agosto 2021.csv`  
  Real-time measurements of server power (watts), cumulative energy (kWh), and CPU/GPU activity.

- **Global CO₂ Emissions Dataset (Our World in Data)**  
  `owid-co2-data.csv`  
  Annual CO₂ emissions per country, from 2010 to 2021.

- **GPU Specifications Dataset**  
  `gpus.csv`  
  Details of GPU models including thermal design power (TDP) and hardware characteristics.

---

## 📈 Visualizations Created

- **Server Power Over Time:** Line plot of real-time server power measurements.
- **Cumulative Energy Usage:** Area plot showing growth of total energy consumed.
- **GPU Metrics Analysis:** Bar chart showing average GPU metrics (TDP and others).
- **GPU TDP Heatmap:** Heatmap showing GPU model vs thermal design power.
- **Global CO₂ Emissions Trend:** Line chart visualizing total global CO₂ emissions from 2010–2021.
- **Correlation Heatmap:** Heatmap exploring correlations between server power, CPU usage, and GPU usage.
- **Machine Learning Prediction Plot:** Scatter plot comparing predicted vs actual server power consumption.

---

## 🔄 Changes from Original Proposal

- Instead of comparing deep model energy usage (GPT-3, BERT, ResNet), focused on infrastructure (server/GPU) level trends.
- Switched from a planned **world map** of CO₂ emissions to a **time-series line plot** (because of dataset structure).
- Deep CO₂ calculators (like DeepEn2023) were not used due to access limitations.
- Machine learning model simplified from complex deep networks to a **linear regression** approach for interpretability and simplicity.

---

## 🔥 Highlights

- **Interactive visualizations** using Plotly (line charts, area plots, heatmaps, bar graphs).
- **Linear regression model** predicting server energy consumption with strong R² accuracy.
- **Fully styled HTML dashboard** automatically generated.
- **GitHub Pages** live hosting for easy access.

---

## 🛠 Setup Instructions

```bash
pip install pandas matplotlib seaborn scikit-learn plotly
