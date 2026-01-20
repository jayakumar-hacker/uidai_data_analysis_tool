#!/usr/bin/env python3
"""
UIDAI Dataset Analysis Tool
A CLI-based program for analyzing Aadhaar datasets (Enrolment, Biometric Update, Demographic Update)
Generates CLI output, visualizations, and PDF reports
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.units import inch
import os

# Dataset configurations
DATASET_CONFIGS = {
    'ENROLMENT': {
        'required_columns': ['date', 'state', 'district', 'pincode', 'age_0_5', 'age_5_17', 'age_18_greater'],
        'age_columns': ['age_0_5', 'age_5_17', 'age_18_greater'],
        'total_column': 'total_enrolments',
        'activity_name': 'Enrolments'
    },
    'BIOMETRIC_UPDATE': {
        'required_columns': ['date', 'state', 'district', 'pincode', 'bio_age_5_17', 'bio_age_17_'],
        'age_columns': ['bio_age_5_17', 'bio_age_17_'],
        'total_column': 'total_biometric_updates',
        'activity_name': 'Biometric Updates'
    },
    'DEMOGRAPHIC_UPDATE': {
        'required_columns': ['date', 'state', 'district', 'pincode', 'demo_age_5_17', 'demo_age_17_'],
        'age_columns': ['demo_age_5_17', 'demo_age_17_'],
        'total_column': 'total_demographic_updates',
        'activity_name': 'Demographic Updates'
    }
}


def detect_dataset_type(columns):
    """
    Auto-detect dataset type based on exact column names
    
    Args:
        columns: List of column names from CSV
        
    Returns:
        Tuple of (dataset_type, config) or (None, None) if no match
    """
    columns_set = set(columns)
    
    for dataset_type, config in DATASET_CONFIGS.items():
        required_set = set(config['required_columns'])
        if required_set.issubset(columns_set):
            return dataset_type, config
    
    return None, None


def load_and_clean_data(filepath):
    """
    Load CSV and perform data cleaning
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Tuple of (dataframe, dataset_type, config)
    """
    print(f"\n{'='*60}")
    print("LOADING AND VALIDATING DATASET")
    print(f"{'='*60}")
    
    # Read CSV
    try:
        df = pd.read_csv(filepath)
        print(f"✓ CSV file loaded: {len(df)} rows")
    except Exception as e:
        print(f"✗ Error reading CSV file: {e}")
        sys.exit(1)
    
    # Detect dataset type
    dataset_type, config = detect_dataset_type(df.columns.tolist())
    
    if dataset_type is None:
        print("\n✗ ERROR: CSV does not match any supported dataset type")
        print("\nExpected column sets:")
        for dt, cfg in DATASET_CONFIGS.items():
            print(f"\n{dt}:")
            print(f"  Required columns: {', '.join(cfg['required_columns'])}")
        print(f"\nFound columns: {', '.join(df.columns.tolist())}")
        sys.exit(1)
    
    print(f"✓ Dataset type detected: {dataset_type}")
    print(f"✓ Activity type: {config['activity_name']}")
    
    # Convert date column with day-first format
    original_count = len(df)
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce', dayfirst=True)
    
    # Track and remove rows with invalid dates
    invalid_dates = df['date'].isna().sum()
    if invalid_dates > 0:
        print(f"⚠ Skipped {invalid_dates} rows with invalid dates")
        df = df.dropna(subset=['date'])
    
    # Handle missing values in numeric columns
    for col in config['age_columns']:
        if df[col].isna().sum() > 0:
            print(f"⚠ Replaced {df[col].isna().sum()} missing values in '{col}' with 0")
        df[col] = df[col].fillna(0)
    
    # Handle missing values in location columns
    location_cols = {'state': 'Unknown State', 'district': 'Unknown District', 'pincode': 'Unknown Pincode'}
    for col, default in location_cols.items():
        if df[col].isna().sum() > 0:
            print(f"⚠ Replaced {df[col].isna().sum()} missing values in '{col}' with '{default}'")
        df[col] = df[col].fillna(default)
    
    # Calculate total activity column
    df[config['total_column']] = df[config['age_columns']].sum(axis=1)
    
    print(f"✓ Records processed: {len(df)}")
    print(f"✓ Data cleaning completed")
    
    return df, dataset_type, config


def perform_analysis(df, config):
    """
    Perform comprehensive analysis on the dataset
    
    Args:
        df: Cleaned dataframe
        config: Dataset configuration
        
    Returns:
        Dictionary containing all analysis results
    """
    print(f"\n{'='*60}")
    print("PERFORMING ANALYSIS")
    print(f"{'='*60}")
    
    results = {}
    total_col = config['total_column']
    
    # Overall statistics
    results['total_count'] = df[total_col].sum()
    results['date_range'] = (df['date'].min(), df['date'].max())
    results['records_count'] = len(df)
    
    # Highest activity
    max_idx = df[total_col].idxmax()
    results['highest'] = {
        'date': df.loc[max_idx, 'date'],
        'state': df.loc[max_idx, 'state'],
        'district': df.loc[max_idx, 'district'],
        'pincode': df.loc[max_idx, 'pincode'],
        'value': df.loc[max_idx, total_col]
    }
    
    # Lowest activity (non-zero)
    df_nonzero = df[df[total_col] > 0]
    if len(df_nonzero) > 0:
        min_idx = df_nonzero[total_col].idxmin()
        results['lowest'] = {
            'date': df.loc[min_idx, 'date'],
            'state': df.loc[min_idx, 'state'],
            'district': df.loc[min_idx, 'district'],
            'pincode': df.loc[min_idx, 'pincode'],
            'value': df.loc[min_idx, total_col]
        }
    else:
        results['lowest'] = None
    
    # Age-group analysis
    age_totals = {}
    for col in config['age_columns']:
        age_totals[col] = df[col].sum()
    
    total = sum(age_totals.values())
    age_percentages = {col: (val/total*100 if total > 0 else 0) for col, val in age_totals.items()}
    
    results['age_groups'] = {
        'totals': age_totals,
        'percentages': age_percentages
    }
    
    # State-wise aggregation
    state_agg = df.groupby('state')[total_col].sum().sort_values(ascending=False)
    results['state_wise'] = state_agg
    
    # District-wise aggregation (top 10)
    district_agg = df.groupby('district')[total_col].sum().sort_values(ascending=False).head(10)
    results['district_wise'] = district_agg
    
    # Date-wise trend
    date_trend = df.groupby('date')[total_col].sum().sort_index()
    results['date_trend'] = date_trend
    
    # Anomaly detection (simple threshold-based)
    mean_activity = date_trend.mean()
    std_activity = date_trend.std()
    threshold = 2 * std_activity  # 2 standard deviations
    
    anomalies = date_trend[(date_trend > mean_activity + threshold) | (date_trend < mean_activity - threshold)]
    results['anomalies'] = anomalies
    results['mean_activity'] = mean_activity
    results['std_activity'] = std_activity
    
    print(f"✓ Analysis completed")
    
    return results


def print_cli_output(dataset_type, config, results):
    """
    Display analysis results in CLI
    
    Args:
        dataset_type: Type of dataset
        config: Dataset configuration
        results: Analysis results dictionary
    """
    print(f"\n{'='*60}")
    print(f"ANALYSIS REPORT: {config['activity_name']}")
    print(f"{'='*60}")
    
    # Dataset summary
    print(f"\nDataset Type: {dataset_type}")
    print(f"Records Processed: {results['records_count']:,}")
    print(f"Date Range: {results['date_range'][0].strftime('%d-%m-%Y')} to {results['date_range'][1].strftime('%d-%m-%Y')}")
    print(f"Total {config['activity_name']}: {results['total_count']:,.0f}")
    
    # Highest activity
    print(f"\n--- Highest Activity Location ---")
    h = results['highest']
    print(f"Date: {h['date'].strftime('%d-%m-%Y')}")
    print(f"State: {h['state']}")
    print(f"District: {h['district']}")
    print(f"Pincode: {h['pincode']}")
    print(f"Count: {h['value']:,.0f}")
    
    # Lowest activity
    if results['lowest']:
        print(f"\n--- Lowest Activity Location (Non-Zero) ---")
        l = results['lowest']
        print(f"Date: {l['date'].strftime('%d-%m-%Y')}")
        print(f"State: {l['state']}")
        print(f"District: {l['district']}")
        print(f"Pincode: {l['pincode']}")
        print(f"Count: {l['value']:,.0f}")
    
    # Age-group distribution
    print(f"\n--- Age-Group Distribution ---")
    for col, total in results['age_groups']['totals'].items():
        pct = results['age_groups']['percentages'][col]
        print(f"{col}: {total:,.0f} ({pct:.2f}%)")
    
    # Top states
    print(f"\n--- Top 5 Contributing States ---")
    for i, (state, count) in enumerate(results['state_wise'].head(5).items(), 1):
        pct = (count / results['total_count'] * 100)
        print(f"{i}. {state}: {count:,.0f} ({pct:.2f}%)")
    
    # Anomalies
    print(f"\n--- Detected Anomalies ---")
    if len(results['anomalies']) > 0:
        print(f"Found {len(results['anomalies'])} dates with unusual activity:")
        for date, value in results['anomalies'].items():
            deviation = ((value - results['mean_activity']) / results['std_activity'])
            print(f"  {date.strftime('%d-%m-%Y')}: {value:,.0f} ({deviation:+.2f} std dev)")
    else:
        print("No significant anomalies detected")
    
    print(f"\n{'='*60}\n")


def generate_graphs(results, config, output_prefix):
    """
    Generate and save visualization graphs
    
    Args:
        results: Analysis results dictionary
        config: Dataset configuration
        output_prefix: Prefix for output filenames
        
    Returns:
        Dictionary of graph filenames
    """
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    graph_files = {}
    
    # 1. Line graph: Activity vs Date
    plt.figure(figsize=(12, 6))
    plt.plot(results['date_trend'].index, results['date_trend'].values, linewidth=2, color='#2E86AB')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'Total {config["activity_name"]}', fontsize=12)
    plt.title(f'{config["activity_name"]} Trend Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    graph_files['trend'] = f'{output_prefix}_trend.png'
    plt.savefig(graph_files['trend'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {graph_files['trend']}")
    
    # 2. Bar chart: Activity by State (Top 10)
    plt.figure(figsize=(12, 6))
    top_states = results['state_wise'].head(10)
    plt.barh(range(len(top_states)), top_states.values, color='#A23B72')
    plt.yticks(range(len(top_states)), top_states.index)
    plt.xlabel(f'Total {config["activity_name"]}', fontsize=12)
    plt.ylabel('State', fontsize=12)
    plt.title(f'Top 10 States by {config["activity_name"]}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    graph_files['states'] = f'{output_prefix}_states.png'
    plt.savefig(graph_files['states'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {graph_files['states']}")
    
    # 3. Bar chart: Age-group contribution
    plt.figure(figsize=(10, 6))
    age_data = results['age_groups']['totals']
    colors_age = ['#F18F01', '#C73E1D', '#6A994E']
    plt.bar(range(len(age_data)), list(age_data.values()), color=colors_age[:len(age_data)])
    plt.xticks(range(len(age_data)), list(age_data.keys()), rotation=45, ha='right')
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel(f'Total {config["activity_name"]}', fontsize=12)
    plt.title(f'{config["activity_name"]} by Age Group', fontsize=14, fontweight='bold')
    plt.tight_layout()
    graph_files['age_groups'] = f'{output_prefix}_age_groups.png'
    plt.savefig(graph_files['age_groups'], dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated: {graph_files['age_groups']}")
    
    print(f"✓ All visualizations saved")
    
    return graph_files


def generate_pdf_report(dataset_type, config, results, graph_files, output_filename):
    """
    Generate comprehensive PDF report
    
    Args:
        dataset_type: Type of dataset
        config: Dataset configuration
        results: Analysis results dictionary
        graph_files: Dictionary of graph filenames
        output_filename: Output PDF filename
    """
    print(f"\n{'='*60}")
    print("GENERATING PDF REPORT")
    print(f"{'='*60}")
    
    doc = SimpleDocTemplate(output_filename, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=30,
        alignment=1  # Center
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#A23B72'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph(f"UIDAI {config['activity_name']} Analysis Report", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Dataset Summary
    story.append(Paragraph("Dataset Summary", heading_style))
    summary_data = [
        ['Dataset Type', dataset_type],
        ['Activity Type', config['activity_name']],
        ['Records Processed', f"{results['records_count']:,}"],
        ['Date Range', f"{results['date_range'][0].strftime('%d-%m-%Y')} to {results['date_range'][1].strftime('%d-%m-%Y')}"],
        [f'Total {config["activity_name"]}', f"{results['total_count']:,.0f}"]
    ]
    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F0F0')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Key Findings
    story.append(Paragraph("Key Findings", heading_style))
    
    # Highest activity
    h = results['highest']
    highest_text = f"<b>Highest Activity Location:</b> {h['state']}, {h['district']} (Pincode: {h['pincode']}) on {h['date'].strftime('%d-%m-%Y')} with {h['value']:,.0f} {config['activity_name'].lower()}."
    story.append(Paragraph(highest_text, styles['BodyText']))
    story.append(Spacer(1, 0.1*inch))
    
    # Lowest activity
    if results['lowest']:
        l = results['lowest']
        lowest_text = f"<b>Lowest Non-Zero Activity:</b> {l['state']}, {l['district']} (Pincode: {l['pincode']}) on {l['date'].strftime('%d-%m-%Y')} with {l['value']:,.0f} {config['activity_name'].lower()}."
        story.append(Paragraph(lowest_text, styles['BodyText']))
        story.append(Spacer(1, 0.1*inch))
    
    # Anomalies
    if len(results['anomalies']) > 0:
        anomaly_text = f"<b>Anomalies Detected:</b> {len(results['anomalies'])} dates showed unusual activity patterns (beyond 2 standard deviations from mean)."
        story.append(Paragraph(anomaly_text, styles['BodyText']))
    else:
        story.append(Paragraph("<b>Anomalies:</b> No significant anomalies detected.", styles['BodyText']))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Top States Table
    story.append(Paragraph("Top 10 Contributing States", heading_style))
    state_data = [['Rank', 'State', 'Total', 'Percentage']]
    for i, (state, count) in enumerate(results['state_wise'].head(10).items(), 1):
        pct = (count / results['total_count'] * 100)
        state_data.append([str(i), state, f"{count:,.0f}", f"{pct:.2f}%"])
    
    state_table = Table(state_data, colWidths=[0.7*inch, 2.5*inch, 1.5*inch, 1.3*inch])
    state_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(state_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Age Group Table
    story.append(Paragraph("Age-Group Distribution", heading_style))
    age_data = [['Age Group', 'Total', 'Percentage']]
    for col, total in results['age_groups']['totals'].items():
        pct = results['age_groups']['percentages'][col]
        age_data.append([col, f"{total:,.0f}", f"{pct:.2f}%"])
    
    age_table = Table(age_data, colWidths=[2.5*inch, 2*inch, 1.5*inch])
    age_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#A23B72')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(age_table)
    story.append(PageBreak())
    
    # Visualizations
    story.append(Paragraph("Data Visualizations", heading_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Add graphs
    for graph_name, graph_file in graph_files.items():
        if os.path.exists(graph_file):
            img = Image(graph_file, width=6*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
    
    story.append(PageBreak())
    
    # Conclusion
    story.append(Paragraph("Operational Insights & Conclusion", heading_style))
    
    # Generate insights based on data
    top_state = results['state_wise'].index[0]
    top_state_pct = (results['state_wise'].iloc[0] / results['total_count'] * 100)
    
    conclusion_points = [
        f"The analysis covers {results['records_count']:,} records spanning from {results['date_range'][0].strftime('%d-%m-%Y')} to {results['date_range'][1].strftime('%d-%m-%Y')}, representing {results['total_count']:,.0f} total {config['activity_name'].lower()}.",
        f"<b>Regional Concentration:</b> {top_state} leads with {top_state_pct:.1f}% of total activity, indicating significant regional demand patterns.",
        f"<b>Age-Group Patterns:</b> The distribution across age groups reveals operational priorities and demographic trends in service utilization.",
    ]
    
    if len(results['anomalies']) > 0:
        conclusion_points.append(f"<b>Temporal Patterns:</b> {len(results['anomalies'])} dates showed unusual activity levels, suggesting specific events or operational changes that warrant further investigation.")
    else:
        conclusion_points.append("<b>Temporal Patterns:</b> Activity levels remained relatively consistent throughout the period with no significant anomalies.")
    
    conclusion_points.append("These insights can guide resource allocation, infrastructure planning, and operational strategy for improved service delivery.")
    
    for point in conclusion_points:
        story.append(Paragraph(point, styles['BodyText']))
        story.append(Spacer(1, 0.15*inch))
    
    # Build PDF
    doc.build(story)
    print(f"✓ PDF report generated: {output_filename}")


def main():
    """
    Main program execution
    """
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python uidai_analysis.py <csv_file_path>")
        print("\nExample: python uidai_analysis.py enrolment_data.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Validate file exists
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    
    # Extract base filename for output files
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_prefix = f"{base_name}_analysis_{timestamp}"
    
    # Step 1: Load and clean data
    df, dataset_type, config = load_and_clean_data(csv_file)
    
    # Step 2: Perform analysis
    results = perform_analysis(df, config)
    
    # Step 3: Display CLI output
    print_cli_output(dataset_type, config, results)
    
    # Step 4: Generate graphs
    graph_files = generate_graphs(results, config, output_prefix)
    
    # Step 5: Generate PDF report
    pdf_filename = f"{output_prefix}.pdf"
    generate_pdf_report(dataset_type, config, results, graph_files, pdf_filename)
    
    # Final summary
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"\nGenerated files:")
    print(f"  • PDF Report: {pdf_filename}")
    for graph_name, graph_file in graph_files.items():
        print(f"  • {graph_name.title()} Graph: {graph_file}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
