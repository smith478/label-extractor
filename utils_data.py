import pandas as pd
from sqlalchemy import create_engine

# Connect to the MIMIC-III database
engine = create_engine('postgresql://username:password@localhost/mimic')

# Query the radiology reports
query = """
SELECT report_text, diagnosis
FROM mimiciii.noteevents
WHERE category = 'Radiology'
"""
radiology_reports = pd.read_sql(query, engine)
