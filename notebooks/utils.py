import json
import pandas as pd
import re
from bs4 import BeautifulSoup

def clean_text(text):
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(separator=" ")
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def json_to_dataframe(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    rows = []
    for case_id, case_data in data.items():
        case_identifier = case_data['case_identifier']
        findings = clean_text(case_data['report']['Findings'])
        conclusions = clean_text(case_data['report']['Conclusions and Recommendations'])
        
        rows.append({
            'case_identifier': case_identifier,
            'findings': findings,
            'conclusions_and_recommendations': conclusions
        })
    
    return pd.DataFrame(rows)

def json_to_string_list(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    string_list = []
    for case_id, case_data in data.items():
        findings = clean_text(case_data['report']['Findings'])
        conclusions = clean_text(case_data['report']['Conclusions and Recommendations'])
        
        combined_text = f"Findings: {findings} Conclusions and recommendations: {conclusions}"
        string_list.append(combined_text)
    
    return string_list