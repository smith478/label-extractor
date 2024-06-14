import openai

#TODO Get this working locally with llama 3 8B

# Load your OpenAI API key
openai.api_key = 'your-api-key'

# Define the list of abnormalities
abnormalities = ["pulmonary edema", "consolidation", "pleural effusion", "pneumothorax", "cardiomegaly"]

def classify_abnormalities(report):
    # Initialize results
    results = {abnormality: 0 for abnormality in abnormalities}
    
    # Prepare the prompt for the GPT-4 model
    prompt = f"Read the following radiology report and identify the presence or absence of the following abnormalities: {', '.join(abnormalities)}.\n\nReport:\n{report}\n\nOutput the results as a list of abnormalities with 0 for absence and 1 for presence."
    
    # Get the classification results from GPT-4
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0
    )
    
    # Process the response
    output = response.choices[0].text.strip().split(",")
    
    # Map the results back to the dictionary
    for item in output:
        abnormality, presence = item.strip().split(":")
        abnormality = abnormality.strip()
        presence = int(presence.strip())
        if abnormality in results:
            results[abnormality] = presence
    
    return results

def generate_xml(results, report_id):
    # Create XML structure
    xml_output = f'<RadiologyReport id="{report_id}">\n'
    for abnormality, presence in results.items():
        xml_output += f'    <{abnormality.replace(" ", "_")}>{presence}</{abnormality.replace(" ", "_")}>\n'
    xml_output += '</RadiologyReport>'
    
    return xml_output

# Example usage
report = "The patient shows signs of pulmonary edema and pleural effusion. No evidence of consolidation or pneumothorax."
report_id = "12345"

# Classify abnormalities
results = classify_abnormalities(report)

# Generate XML
xml_output = generate_xml(results, report_id)
print(xml_output)
