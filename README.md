# UIDAI Dataset Analysis Tool

A comprehensive CLI-based Python program for analyzing Aadhaar datasets including Enrolment, Biometric Update, and Demographic Update data. This tool automatically detects dataset types, performs detailed analysis, and generates professional reports with visualizations.

## 🎯 Features

- **Automatic Dataset Detection**: Identifies dataset type based on column names
- **Robust Data Cleaning**: Handles missing values, invalid dates, and incomplete records
- **Comprehensive Analysis**: 
  - Overall statistics and trends
  - Regional demand patterns (state, district, pincode)
  - Age-group distribution analysis
  - Temporal trend analysis
  - Anomaly detection
- **Multiple Output Formats**:
  - Detailed CLI output
  - High-quality visualizations (PNG graphs)
  - Professional PDF report

## 📋 Requirements

- Python 3.7 or higher
- Required libraries (see `requirements.txt`)

## 🚀 Installation

### Step 1: Clone or Download

Download the following files to your project directory:
- `uidai_analysis.py`
- `requirements.txt`

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas matplotlib reportlab
```

## 📊 Supported Dataset Types

### 1. Enrolment Dataset
**Required Columns:**
- `date`, `state`, `district`, `pincode`
- `age_0_5`, `age_5_17`, `age_18_greater`

### 2. Biometric Update Dataset
**Required Columns:**
- `date`, `state`, `district`, `pincode`
- `bio_age_5_17`, `bio_age_17_`

### 3. Demographic Update Dataset
**Required Columns:**
- `date`, `state`, `district`, `pincode`
- `demo_age_5_17`, `demo_age_17_`

## 💻 Usage

### Basic Usage

```bash
python uidai_analysis.py <path_to_csv_file>
```

### Example

```bash
python uidai_analysis.py enrolment_data.csv
```

### Expected CSV Format

```csv
date,state,district,pincode,age_0_5,age_5_17,age_18_greater
01-01-2024,Maharashtra,Mumbai,400001,150,300,850
02-01-2024,Delhi,New Delhi,110001,200,400,1000
...
```

**Important Notes:**
- Date format must be **DD-MM-YYYY** (day-first format)
- Column names must match exactly (case-sensitive)
- Missing values in numeric columns will be replaced with 0
- Missing location data will be marked as "Unknown State/District/Pincode"

## 📂 Output Files

After running the analysis, the following files will be generated:

```
{filename}_analysis_{timestamp}.pdf              # Complete PDF report
{filename}_analysis_{timestamp}_trend.png        # Time series line graph
{filename}_analysis_{timestamp}_states.png       # Top states bar chart
{filename}_analysis_{timestamp}_age_groups.png   # Age distribution bar chart
```

### Example Output:
```
enrolment_data_analysis_20240120_143052.pdf
enrolment_data_analysis_20240120_143052_trend.png
enrolment_data_analysis_20240120_143052_states.png
enrolment_data_analysis_20240120_143052_age_groups.png
```

## 📈 Analysis Components

### CLI Output Includes:
- Dataset type and validation summary
- Records processed and date range
- Total activity count
- Highest activity location details
- Lowest activity location details
- Age-group distribution with percentages
- Top 5 contributing states
- Detected anomalies

### PDF Report Contains:
- **Dataset Summary**: Type, records, date range, totals
- **Key Findings**: Highest/lowest activity, anomalies
- **Tables**: Top 10 states, age-group breakdown
- **Visualizations**: All generated graphs embedded
- **Operational Insights**: Data-driven recommendations

### Visualizations:
1. **Trend Analysis**: Line graph showing activity over time
2. **Regional Distribution**: Bar chart of top 10 states
3. **Age Demographics**: Bar chart showing age-group contributions

## 🔍 Sample Output

```
============================================================
LOADING AND VALIDATING DATASET
============================================================
✓ CSV file loaded: 1000 rows
✓ Dataset type detected: ENROLMENT
✓ Activity type: Enrolments
⚠ Skipped 5 rows with invalid dates
✓ Records processed: 995
✓ Data cleaning completed

============================================================
PERFORMING ANALYSIS
============================================================
✓ Analysis completed

============================================================
ANALYSIS REPORT: Enrolments
============================================================

Dataset Type: ENROLMENT
Records Processed: 995
Date Range: 01-01-2024 to 31-12-2024
Total Enrolments: 1,250,000

--- Highest Activity Location ---
Date: 15-06-2024
State: Maharashtra
District: Mumbai
Pincode: 400001
Count: 5,000

...
```

## ⚠️ Error Handling

The tool provides clear error messages for:
- Missing or invalid CSV files
- Unrecognized dataset types
- Column name mismatches
- Data format issues

### Common Errors:

**1. Dataset Type Not Recognized**
```
✗ ERROR: CSV does not match any supported dataset type

Expected column sets:
ENROLMENT:
  Required columns: date, state, district, pincode, age_0_5, age_5_17, age_18_greater
...
```

**Solution**: Verify your CSV has the exact column names listed above.

**2. File Not Found**
```
Error: File 'data.csv' not found
```

**Solution**: Check the file path and ensure the file exists.

## 🛠️ Troubleshooting

### Issue: "Module not found" error
**Solution**: Install required dependencies using `pip install -r requirements.txt`

### Issue: Date parsing errors
**Solution**: Ensure dates are in DD-MM-YYYY format (e.g., 15-01-2024, not 01-15-2024)

### Issue: Column name errors
**Solution**: Column names are case-sensitive. Use exact names as specified in the Supported Datasets section

### Issue: PDF generation fails
**Solution**: Ensure you have write permissions in the current directory

## 🎓 Use Cases

- **Government Reporting**: Generate official analysis reports for UIDAI operations
- **Trend Analysis**: Identify temporal patterns in enrolment/update activities
- **Resource Planning**: Determine high-demand regions for infrastructure allocation
- **Performance Monitoring**: Track operational metrics across states and districts
- **Anomaly Detection**: Identify unusual activity patterns for investigation

## 📝 Data Privacy & Security

- **No Data Storage**: Tool processes data locally, no external connections
- **No Personal Information**: Analyzes aggregated statistics only
- **Temporary Processing**: Data exists only in memory during execution
- **Local Output**: All reports saved locally on your machine

## 🏆 UIDAI Hackathon Ready

This tool is specifically designed for UIDAI hackathon submissions with:
- ✅ Single-file Python implementation
- ✅ CLI-based execution
- ✅ Professional reporting
- ✅ Operational insights focus
- ✅ Government-appropriate language
- ✅ Beginner-friendly code
- ✅ Comprehensive documentation

## 📄 License

This tool is created for educational and government hackathon purposes.

## 👥 Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify your CSV format matches requirements
3. Ensure all dependencies are installed correctly

## 🔄 Version

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Python Compatibility**: 3.7+

---

**Note**: This tool analyzes aggregated operational data and does not process or store any personally identifiable information (PII). All analysis focuses on statistical patterns and operational insights for administrative purposes.
