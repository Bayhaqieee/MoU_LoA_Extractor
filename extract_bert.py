import re
from transformers import BertTokenizer, BertForTokenClassification, pipeline
import pymupdf

class AgreementExtractor:
    def __init__(self):
        self.model_name = "dslim/bert-base-NER"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForTokenClassification.from_pretrained(self.model_name)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
    
    def extract_text_from_pdf(self, file_path):
        doc = pymupdf.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def extract_entities_with_bert(self, text, entity_type):
        entities = self.ner_pipeline(text)
        extracted_entities = []
        current_entity = []
        for entity in entities:
            if entity['entity'] == f'B-{entity_type}':
                if current_entity:
                    extracted_entities.append(' '.join(current_entity))
                current_entity = [entity['word']]
            elif entity['entity'] == f'I-{entity_type}':
                current_entity.append(entity['word'].replace('##', ''))
            else:
                if current_entity:
                    extracted_entities.append(' '.join(current_entity))
                    current_entity = []

        if current_entity:
            extracted_entities.append(' '.join(current_entity))

        return extracted_entities
    
    def extract_date_of_agreement(self, text):
        date_pattern_first_page = r"On this date ([\d\w\s]+), we the undersigned below:"
        date_pattern_chapter_v_english = r"The term of this Memorandum of Understanding is valid for a period of [\w\s(),]+\s*from\s([\d\w\s]+)\sto"
        date_pattern_chapter_v_indonesian = r"Jangka waktu Nota Kesepahaman ini\s*berlaku untuk jangka waktu\s*[\w\s(),]+\s*sejak\s([\d\s\w]+)\s*sampai"
        
        dates = self.extract_entities_with_bert(text, 'DATE')
        
        if not dates:
            match_first_page = re.search(date_pattern_first_page, text)
            if match_first_page:
                dates.append(match_first_page.group(1))
            
            match_chapter_v_english = re.search(date_pattern_chapter_v_english, text)
            if match_chapter_v_english:
                dates.append(match_chapter_v_english.group(1))
            
            match_chapter_v_indonesian = re.search(date_pattern_chapter_v_indonesian, text)
            if match_chapter_v_indonesian:
                dates.append(match_chapter_v_indonesian.group(1))
        
        return dates
    
    def extract_letter_number(self, text):
        letter_number_pattern = r"Number:\s*(\S+)\s*|Nomor:\s*(\S+)\s*"
        
        letter_numbers = self.extract_entities_with_bert(text, 'MISC')
        
        if not letter_numbers:
            match = re.search(letter_number_pattern, text)
            if match:
                letter_numbers.append(match.group(1) or match.group(2))
        
        return letter_numbers
    
    def extract_party_names(self, text):
        # Preprocess text to remove HTML tags if present
        text = re.sub(r'<b>(.*?)</b>', r'\1', text)
        
        # Updated regex patterns to extract company names
        first_party_pattern_english = r"On this date [\d\w\s,]+, we the\s*undersigned below:\s*1\.\s*(?P<company_name>[\w\s,]+), a"
        first_party_pattern_indonesian = r"Pada hari ini tanggal [\d\s\w]+, pihak-pihak\s*yang bertanda tangan di bawah ini:\s*1\.\s*(?P<company_name>[\w\s,]+), sebuah"
        
        second_party_pattern_company_english = r"2\.\s*(?P<company_name>[\w\s,]+), a"
        second_party_pattern_company_indonesian = r"2\.\s*(?P<company_name>[\w\s,]+), suatu"
        
        second_party_pattern_speaker_english = r"2\.\s*(?P<speaker_name>[\w\s,]+), who is located"
        second_party_pattern_speaker_indonesian = r"2\.\s*(?P<speaker_name>[\w\s,]+), yang berkedudukan"
        
        first_party = []
        second_party = []
        
        match_first_party_english = re.search(first_party_pattern_english, text)
        match_first_party_indonesian = re.search(first_party_pattern_indonesian, text)
        
        if match_first_party_english:
            first_party.append(match_first_party_english.group('company_name'))
        if match_first_party_indonesian:
            first_party.append(match_first_party_indonesian.group('company_name'))
        
        match_second_party_company_english = re.search(second_party_pattern_company_english, text)
        match_second_party_company_indonesian = re.search(second_party_pattern_company_indonesian, text)
        match_second_party_speaker_english = re.search(second_party_pattern_speaker_english, text)
        match_second_party_speaker_indonesian = re.search(second_party_pattern_speaker_indonesian, text)
        
        if match_second_party_company_english:
            second_party.append(match_second_party_company_english.group('company_name'))
        if match_second_party_company_indonesian:
            second_party.append(match_second_party_company_indonesian.group('company_name'))
        if match_second_party_speaker_english:
            second_party.append(match_second_party_speaker_english.group('speaker_name'))
        if match_second_party_speaker_indonesian:
            second_party.append(match_second_party_speaker_indonesian.group('speaker_name'))
        
        return first_party, second_party
    
    def extract_pic_data(self, text):
        # Preprocess text to remove HTML tags if present
        text = re.sub(r'<b>(.*?)</b>', r'\1', text)
        
        
        # Extract entities with BERT
        first_party_data = []
        second_party_data = []
        entities = self.nlp(text)
        for entity in entities:
            if entity['entity'] == 'PERSON':
                if 'PIHAK PERTAMA' in entity['word']:
                    first_party_data.append({'name': entity['word']})
                elif 'PIHAK KEDUA' in entity['word']:
                    second_party_data.append({'name': entity['word']})
        
        # If no PIC data is found using BERT, use regex patterns
        if not first_party_data and not second_party_data:
            pic_pattern_indonesian = r"Nama\s*:\s*(?P<name>[\w\s]+)\s*Jabatan\s*:\s*(?P<position>[\w\s]+)\s*Telp/fax\s*:\s*(?P<telephone>[\w\s]+)\s*Email\s*:\s*(?P<email>[\w\s]+)\s*Alamat\s*:\s*(?P<address>[\w\s]+)"
            pic_pattern_english = r"Name\s*:\s*(?P<name>[\w\s]+)\s*Position\s*:\s*(?P<position>[\w\s]+)\s*Telp/fax\s*:\s*(?P<telephone>[\w\s]+)\s*Email\s*:\s*(?P<email>[\w\s]+)\s*Address\s*:\s*(?P<address>[\w\s]+)"
            
            first_party_match_indonesian = re.search(r'PIHAK PERTAMA.*?' + pic_pattern_indonesian, text, re.DOTALL)
            first_party_match_english = re.search(r'FIRST PARTY.*?' + pic_pattern_english, text, re.DOTALL)
            
            second_party_match_indonesian = re.search(r'PIHAK KEDUA.*?' + pic_pattern_indonesian, text, re.DOTALL)
            second_party_match_english = re.search(r'SECOND PARTY.*?' + pic_pattern_english, text, re.DOTALL)
            
            if first_party_match_indonesian:
                first_party_data.append(first_party_match_indonesian.groupdict())
            if first_party_match_english:
                first_party_data.append(first_party_match_english.groupdict())
            
            if second_party_match_indonesian:
                second_party_data.append(second_party_match_indonesian.groupdict())
            if second_party_match_english:
                second_party_data.append(second_party_match_english.groupdict())
        
        return first_party_data, second_party_data
    
    def extract_supply_data(self, text):
        supply_patterns = [
            r"FIRST PARTY responsibilities to place a logo placement of SECOND PARTY in official poster event FIRST PARTY, and LPJ internal FIRST PARTY.",
            r"FIRST PARTY responsibilities to inform all things needed related the partnership with SECOND PARTY.",
            r"FIRST PARTY responsibilities to keep track of cooperation in order running well and according to the agreement",
            r"FIRST PARTY responsibilities to obey entirely regulation which has been agreed.",
            r"FIRST PARTY responsibilities to give a certificate and newsletter report to SECOND PARTY.",
            r"FIRST PARTY responsibilities to include SECOND PARTY in pre-event article",
            r"FIRST PARTY responsibilities to conduct selling space of SECOND PARTY product which will be held for [\w\s()]+ minutes.",
            r"FIRST PARTY responsibilities to fulfill SECOND PARTY Research Survey which in total of [\w\s()]+ Participant.",
            r"FIRST PARTY responsibilities to conduct Ad-Libs of SECOND PARTY when the event is ongoing.",
            r"FIRST PARTY responsibilities to play Company Video Promotion of SECOND PARTY when the event is ongoing.",
            r"FIRST PARTY responsibilities to post 1 \(one\) Story for SECOND PARTY with 20,000\+ Account Follower on Instagram."
        ]
        
        supply_data = []
        for pattern in supply_patterns:
            match = re.search(pattern, text)
            if match:
                supply_data.append(match.group())
        
        return supply_data
    
    def extract_demand_data(self, text):
        # Define demand patterns similar to supply patterns
        demand_patterns = [
            # Add regex patterns for demand data here
        ]
        
        demand_data = []
        for pattern in demand_patterns:
            match = re.search(pattern, text)
            if match:
                demand_data.append(match.group())
        
        return demand_data
    
    def extract_duration(self, text):
        duration_pattern_indonesian = r"Jangka waktu Nota Kesepahaman ini\s*berlaku untuk jangka waktu\s*[\w\s(),]+\s*sejak\s([\d\s\w]+)\s*sampai"
        duration_pattern_english = r"The term of this Memorandum of\s*Understanding is valid for a period of\s*[\w\s(),]+\s*from\s([\d\w\s]+)\sto"
        
        duration = self.extract_entities_with_bert(text, 'DURATION')
        
        if not duration:
            match_indonesian = re.search(duration_pattern_indonesian, text)
            if match_indonesian:
                duration.append(match_indonesian.group(1))
            
            match_english = re.search(duration_pattern_english, text)
            if match_english:
                duration.append(match_english.group(1))
        
        return duration
    
    def extract_roi(self, supply_data, demand_data):
        # Implement the RoI calculation logic based on supply and demand data
        roi = calculate_roi(supply_data, demand_data)
        return roi
    
def calculate_roi(supply_data, demand_data):
    # Define the RoI calculation logic here
    pass

# Example usage
file_path = "STAMP_MoUSawulan (1).pdf"
extractor = AgreementExtractor()
text = extractor.extract_text_from_pdf(file_path)

# Extract different entities using BERT and regex
dates = extractor.extract_date_of_agreement(text)
letter_numbers = extractor.extract_letter_number(text)
first_party, second_party = extractor.extract_party_names(text)
first_party_pic = extractor.extract_pic_data(text)
second_party_pic = extractor.extract_pic_data(text)
supply_data = extractor.extract_supply_data(text)
demand_data = extractor.extract_demand_data(text)
duration = extractor.extract_duration(text)
roi = extractor.extract_roi(supply_data, demand_data)

print("Dates:", dates)
print("Letter Numbers:", letter_numbers)
print("First Party:", first_party)
print("First Party Data:", first_party_pic)
print("Second Party:", second_party)
print("Second Party Data:", second_party_pic)
print("Supply Data:", supply_data)
print("Demand Data:", demand_data)
print("Duration:", duration)
print("RoI:", roi)