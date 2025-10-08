# 📊 Proactive AI Forecasting System for Digital Resource Management  
**Saudi Food and Drug Authority (SFDA)**  
تطبيق ويب يعتمد على الذكاء الاصطناعي للتنبؤ بالاستخدام المستقبلي لسعة التخزين في الخوادم.

---

## 🚀 Overview
This Flask web application predicts future disk usage (in MB/GB) for one or multiple servers using **NeuralProphet** — a time-series forecasting model based on PyTorch.  
The system supports:
- **Single Server Mode:** Train, forecast, and visualize one server.
- **Multi Server Mode:** Upload an Excel file with multiple servers, forecast them all, and summarize results.

---

## 🧩 Project Structure
📁 DiskForecastUI/
│
├── app.py # Flask web application
├── model.py # Model training and forecasting logic (NeuralProphet)
├── preprocess_of_data.py # Data cleaning and monthly aggregation
├── requirements.txt # Required Python packages
│
├── templates/
│ ├── base.html # Base layout (header, footer)
│ ├── index.html # Single server workflow
│ ├── multi.html # Multi-server workflow
│ ├── results.html # Results and charts page
│
├── static/
│ ├── style.css # Main CSS style
│ ├── sfda_header_full.png # Header logo
│ └── Use_guide.pdf # User manual (optional)
│
├── runs/ # Generated model and result folders
└── README.md


---

## 🧠 How It Works
1. **Upload Excel File**  
   Upload a file containing columns:  
   `ServerName`, `DateRecorded`, `FileType` (`data` or `log`), `SizeInMB`.

2. **Data Preprocessing**  
   The file is filtered per server and converted into monthly end-of-month data.

3. **Model Training**  
   NeuralProphet models are trained separately for `DATA` and `LOG` types.  
   Each model is saved as `.np` (pickled).

4. **Forecast Generation**  
   Forecasts are generated for the selected duration (3–12 months) and saved as CSVs:
future_DATA_6M.csv
future_LOG_6M.csv

5. **Visualization**
- Gray line → historical growth  
- Green line → forecasted growth  
- Interactive table of results in MB and GB

---


