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
        
        # Remove tabs from the text
        text = text.replace("\t", " ")  # Replace tabs with a single space
        text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
        return text
    
    def extract_entities_with_bert(self, text, entity_type):
        entities = self.nlp(text)
        extracted_entities = []
        current_entity = []

        for entity in entities:
            if entity['entity'] == f'B-{entity_type}':
                if current_entity:
                    extracted_entities.append(' '.join(current_entity))
                current_entity = [entity['word'].replace('##', '')]
            elif entity['entity'] == f'I-{entity_type}':
                current_entity.append(entity['word'].replace('##', ''))
            else:
                if current_entity:
                    extracted_entities.append(' '.join(current_entity))
                    current_entity = []

        if current_entity:
            extracted_entities.append(' '.join(current_entity))

        return extracted_entities
    
    def extract_text_block(self, text, start_markers, end_markers):
        """
        Extracts a block of text between specified start markers and end markers.
        Handles both Indonesian and English start markers.
        """
        for start_marker in start_markers:
            start_pattern = re.escape(start_marker)
            
            # Combine end markers into a single regex pattern
            end_pattern = '|'.join(re.escape(marker) for marker in end_markers)
            
            # Construct the full pattern with a non-capturing lookahead for the end markers
            pattern = rf"{start_pattern}(.*?)(?={end_pattern})"
            
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return None
    
    def extract_date_of_agreement(self, text):
        dates = self.extract_entities_with_bert(text, 'DATE')
        if not dates:
            date_pattern_first_page = r"On this date ([\d\w\s]+), we the undersigned below:"
            date_pattern_chapter_v_english = r"The term of this Memorandum of Understanding is valid for a period of [\w\s(),]+\s*from\s([\d\w\s]+)\sto"
            date_pattern_chapter_v_indonesian = r"Jangka waktu Nota Kesepahaman ini\s*berlaku untuk jangka waktu\s*[\w\s(),]+\s*sejak\s([\d\s\w]+)\s*sampai"
            
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
        letter_number_pattern = r"Number :\s*(\S+)\s*|Nomor :\s*(\S+)\s*"
        
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
        
        second_party_pattern_company_english = r"2\.\s*(?P<company_name>[\w\s,]+), a company"
        second_party_pattern_company_indonesian = r"2\.\s*(?P<company_name>[\w\s,]+), suatu perusahaan"
        
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

    def extract_first_party_pic_block(self, text):
        start_markers = [
            "menunjuk:",  # Indonesian start marker
            "designates:"  # English start marker
        ]
        end_markers = [
            "dan PIHAK",  # Indonesian end marker
            "and the"        # English end marker
        ]
        return self.extract_text_block(text, start_markers, end_markers)

    def extract_second_party_pic_block(self, text):
        start_markers = [
            "KEDUA menunjuk:",  # Indonesian start marker
            "SECOND PARTY designates:"  # English start marker
        ]
        end_markers = [
            "sebagai koordinator",  # Indonesian end marker
            "as the coordinator"        # English end marker
        ]
        return self.extract_text_block(text, start_markers, end_markers)

    def extract_individual_fields(self, text):
        fields = {}
        block_patterns = {
            "Name": {
                "start": ["Name :", "Nama :"],
                "end": ["Position :", "Jabatan :"]
            },
            "Position": {
                "start": ["Position :", "Jabatan :"],
                "end": ["Telp/fax :"]
            },
            "Telp/fax": {
                "start": ["Telp/fax :"],
                "end": ["Email :"]
            },
            "Email": {
                "start": ["Email :"],
                "end": ["Address :", "Alamat :"]
            },
            "Address": {
                "start": ["Address :", "Alamat :"],
                "end": ["sebagai koordinator", "as the coordinator"]
            }
        }

        for field, markers in block_patterns.items():
            extracted_text = self.extract_text_block(text, markers["start"], markers.get("end", [None]))
            if extracted_text:
                fields[field] = extracted_text

        return fields

    def extract_first_party_details(self, text):
        pic_block = self.extract_first_party_pic_block(text)
        return self.extract_individual_fields(pic_block) if pic_block else {}

    def extract_second_party_details(self, text):
        pic_block = self.extract_second_party_pic_block(text)
        return self.extract_individual_fields(pic_block) if pic_block else {}
    
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
        start_markers = [
            "berlaku  untuk  jangka waktu ",  # Indonesian start marker
            "is valid for a period of "  # English start marker
        ]
        end_markers = [
            "sejak",  # Indonesian end marker
            "from"        # English end marker
        ]
        return self.extract_text_block(text, start_markers, end_markers)
    
    def extract_roi(self, supply_data, demand_data):
        # Implement the RoI calculation logic based on supply and demand data
        roi = self.calculate_roi(supply_data, demand_data)
        return roi
    
    def calculate_roi(self, supply_data, demand_data):
        # Placeholder for actual calculation
        total_supply = sum(float(re.findall(r'\d+', s)[0]) for s in supply_data if re.findall(r'\d+', s))
        total_demand = sum(float(re.findall(r'\d+', d)[0]) for d in demand_data if re.findall(r'\d+', d))
        
        if total_supply == 0:
            return 0
        
        roi = (total_demand - total_supply / total_supply) * 100  # ROI as a percentage
        return roi

# Example usage
file_path = "MoU Sample(1).pdf"
extractor = AgreementExtractor()
text = extractor.extract_text_from_pdf(file_path)

# Extract different entities using BERT and regex
dates = extractor.extract_date_of_agreement(text)
letter_numbers = extractor.extract_letter_number(text)
first_party_names = extractor.extract_party_names(text)[0]
second_party_names = extractor.extract_party_names(text)[1]
first_party_pic_raw_data = extractor.extract_first_party_pic_block(text)
second_party_pic_raw_data = extractor.extract_second_party_pic_block(text)
first_party_pic_data = extractor.extract_first_party_details(text)
second_party_pic_data = extractor.extract_second_party_details(text)
supply_data = extractor.extract_supply_data(text)
demand_data = extractor.extract_demand_data(text)
duration = extractor.extract_duration(text)
roi = extractor.extract_roi(supply_data, demand_data)

print("Dates:", dates)
print("Letter Numbers:", letter_numbers)
print("First Party:", first_party_names)
print("First Party Raw Data:", first_party_pic_raw_data)
print("First Party Data:", first_party_pic_data)
print("Second Party:", second_party_names)
print("Second Party Raw Data:", second_party_pic_raw_data)
print("Second Party Data:", second_party_pic_data)
print("Supply Data:", supply_data)
print("Demand Data:", demand_data)
print("Duration:", duration)
print("RoI:", roi)