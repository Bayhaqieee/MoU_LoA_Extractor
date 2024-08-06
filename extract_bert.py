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
    
    def extract_block_data(self, text, start_markers, end_markers, type_patterns):
        blocks = {}
        for type_name, type_pattern in type_patterns.items():
            blocks[type_name] = []

        for start_marker in start_markers:
            start_pattern = re.escape(start_marker)
            end_pattern = '|'.join(re.escape(marker) for marker in end_markers)
            pattern = rf"{start_pattern}(.*?)(?={end_pattern})"
            matches = re.findall(pattern, text, re.DOTALL)

            if matches:
                for match in matches:
                    match = match.strip()
                    for type_name, type_pattern in type_patterns.items():
                        if re.search(type_pattern, match):
                            blocks[type_name].append(match)
                            break
        return blocks
    
    def extract_block_data_deal(self, text, type_patterns):
        blocks = {}
        for type_name, markers in type_patterns.items():
            blocks[type_name] = {'detected': False, 'matches': []}
            start_markers = markers['start']
            end_markers = markers['end']
            pattern = markers['pattern']

            for start_marker in start_markers:
                start_pattern = re.escape(start_marker)
                end_pattern = '|'.join(re.escape(marker) for marker in end_markers)
                search_pattern = rf"{start_pattern}(.*?)(?={end_pattern})"
                matches = re.findall(search_pattern, text, re.DOTALL)

                if matches:
                    for match in matches:
                        match = match.strip()
                        if re.search(pattern, match):
                            merged_text = f"{start_marker}{match}{next((end_marker for end_marker in end_markers if re.search(re.escape(end_marker), text)), '')}"
                            blocks[type_name]['matches'].append(merged_text)
                            blocks[type_name]['detected'] = True
                            break
        return blocks

    def extract_supply_data(self, text):
        supply_type_patterns = {
            "Logo Placement": {
                "start": ["FIRST PARTY responsibilities to place", "PIHAK  PERTAMA  berkewajiban  untuk mencantumkan"],
                "end": ["LPJ internal FIRST PARTY", "LPJ internal PIHAK PERTAMA"],
                "pattern": r"logo|logo"
            },
            "Information Sharing": {
                "start": ["FIRST PARTY responsibilities to", "PIHAK  PERTAMA  berkewajiban  untuk"],
                "end": ["partnership with SECOND PARTY", "kerja sama dengan PIHAK KEDUA"],
                "pattern": r"inform all things|menginformasikan"
            },
            "Cooperation Tracking": {
                "start": ["FIRST PARTY responsibilities to keep track of cooperation", "PIHAK PERTAMA bertanggung jawab untuk memantau kerja sama"],
                "end": ["SECOND PARTY responsibilities", "TANGGUNG JAWAB PIHAK KEDUA", "ARTICLE", "PASAL", "SECTION"],
                "pattern": r"keep track of cooperation|memantau kerja sama"
            },
            "Regulation Compliance": {
                "start": ["FIRST PARTY responsibilities to obey entirely regulation", "PIHAK PERTAMA bertanggung jawab untuk mematuhi sepenuhnya peraturan"],
                "end": ["SECOND PARTY responsibilities", "TANGGUNG JAWAB PIHAK KEDUA", "ARTICLE", "PASAL", "SECTION"],
                "pattern": r"obey entirely regulation|mematuhi sepenuhnya peraturan"
            },
            "Certificate and Newsletter": {
                "start": ["FIRST PARTY responsibilities to give a certificate and newsletter report", "PIHAK PERTAMA bertanggung jawab untuk memberikan sertifikat dan laporan buletin"],
                "end": ["SECOND PARTY responsibilities", "TANGGUNG JAWAB PIHAK KEDUA", "ARTICLE", "PASAL", "SECTION"],
                "pattern": r"give a certificate and newsletter report|memberikan sertifikat dan laporan buletin"
            },
            "Pre-Event Article": {
                "start": ["FIRST PARTY responsibilities to include SECOND PARTY in pre-event article", "PIHAK PERTAMA bertanggung jawab untuk menyertakan PIHAK KEDUA dalam artikel pra-acara"],
                "end": ["SECOND PARTY responsibilities", "TANGGUNG JAWAB PIHAK KEDUA", "ARTICLE", "PASAL", "SECTION"],
                "pattern": r"include SECOND PARTY in pre-event article|menyertakan PIHAK KEDUA dalam artikel pra-acara"
            },
            "Selling Space": {
                "start": ["FIRST PARTY responsibilities to conduct selling space of SECOND PARTY product", "PIHAK PERTAMA bertanggung jawab untuk menyediakan ruang penjualan produk PIHAK KEDUA"],
                "end": ["SECOND PARTY responsibilities", "TANGGUNG JAWAB PIHAK KEDUA", "ARTICLE", "PASAL", "SECTION"],
                "pattern": r"conduct selling space of SECOND PARTY product|menyediakan ruang penjualan produk PIHAK KEDUA"
            },
            "Research Survey": {
                "start": ["FIRST PARTY responsibilities to fulfill SECOND PARTY Research Survey", "PIHAK PERTAMA bertanggung jawab untuk melaksanakan survei riset PIHAK KEDUA"],
                "end": ["SECOND PARTY responsibilities", "TANGGUNG JAWAB PIHAK KEDUA", "ARTICLE", "PASAL", "SECTION"],
                "pattern": r"fulfill SECOND PARTY Research Survey|melaksanakan survei riset PIHAK KEDUA"
            },
            "Ad-Libs": {
                "start": ["FIRST PARTY responsibilities to conduct Ad-Libs of SECOND PARTY", "PIHAK PERTAMA bertanggung jawab untuk melaksanakan Ad-Libs PIHAK KEDUA"],
                "end": ["SECOND PARTY responsibilities", "TANGGUNG JAWAB PIHAK KEDUA", "ARTICLE", "PASAL", "SECTION"],
                "pattern": r"conduct Ad-Libs of SECOND PARTY|melaksanakan Ad-Libs PIHAK KEDUA"
            },
            "Company Video Promotion": {
                "start": ["FIRST PARTY responsibilities to play Company Video Promotion", "PIHAK PERTAMA bertanggung jawab untuk memutar Video Promosi Perusahaan PIHAK KEDUA"],
                "end": ["SECOND PARTY responsibilities", "TANGGUNG JAWAB PIHAK KEDUA", "ARTICLE", "PASAL", "SECTION"],
                "pattern": r"play Company Video Promotion|memutar Video Promosi Perusahaan"
            },
            "Instagram Story Post": {
                "start": ["FIRST PARTY responsibilities to post 1 \\(one\\) Story for SECOND PARTY with 20,000\\+ Account Follower on Instagram", "PIHAK PERTAMA bertanggung jawab untuk memposting 1 \\(satu\\) Cerita untuk PIHAK KEDUA dengan 20.000\\+ Pengikut di Instagram"],
                "end": ["SECOND PARTY responsibilities", "TANGGUNG JAWAB PIHAK KEDUA", "ARTICLE", "PASAL", "SECTION"],
                "pattern": r"post 1 \\(one\\) Story for SECOND PARTY with 20,000\\+ Account Follower on Instagram|memposting 1 \\(satu\\) Cerita untuk PIHAK KEDUA dengan 20.000\\+ Pengikut di Instagram"
            }
        }

        return self.extract_block_data_deal(text, supply_type_patterns)

    def extract_demand_data(self, text):
        demand_type_patterns = {
            # Define demand type patterns here, similar to supply_type_patterns
            # Example:
            "Payment": {
                "start": ["SECOND PARTY responsibilities to pay", "PIHAK KEDUA bertanggung jawab untuk membayar"],
                "end": ["FIRST PARTY responsibilities", "TANGGUNG JAWAB PIHAK PERTAMA", "ARTICLE", "PASAL", "SECTION"],
                "pattern": r"pay [\w\s]+ amount|membayar sejumlah"
            }
        }

        return self.extract_block_data_deal(text, demand_type_patterns)

    def detect_deal(self, text, type_patterns):
        for type_name, markers in type_patterns.items():
            pattern = markers['pattern']
            if re.search(pattern, text):
                return True
        return False
    
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
first_party_pic_data = extractor.extract_first_party_details(text)
second_party_pic_data = extractor.extract_second_party_details(text)
supply_data = extractor.extract_supply_data(text)
demand_data = extractor.extract_demand_data(text)
duration = extractor.extract_duration(text)
roi = extractor.extract_roi(supply_data, demand_data)

print("Dates:", dates)
print("Letter Numbers:", letter_numbers)
print("First Party:", first_party_names)
print("First Party Data:", first_party_pic_data)
print("Second Party:", second_party_names)
print("Second Party Data:", second_party_pic_data)
print("Supply Data:", supply_data)
print("Demand Data:", demand_data)
print("Duration:", duration)
print("RoI:", roi)