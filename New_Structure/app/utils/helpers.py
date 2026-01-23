# helpers.py - Shared utilities and constants

import re

# Year extraction regex
YEAR_REGEX = re.compile(r'\b(19|20)\d{2}\b')

# Comprehensive Educational Keywords Database
EDUCATIONAL_KEYWORDS = {
    # K-12 Core Subjects
    'k12_subjects': ['mathematics', 'math', 'algebra', 'geometry', 'calculus', 'statistics', 'trigonometry',
                     'science', 'biology', 'chemistry', 'physics', 'earth science', 'environmental science',
                     'english', 'language arts', 'literature', 'reading', 'writing', 'composition',
                     'social studies', 'history', 'geography', 'civics', 'government', 'economics',
                     'arts', 'music', 'visual art', 'drama', 'theater', 'dance',
                     'physical education', 'health', 'pe', 'stem', 'steam'],
    
    # STEM & Natural Sciences
    'stem_sciences': ['computer science', 'cs', 'information technology', 'it', 'programming',
                      'data science', 'artificial intelligence', 'machine learning',
                      'astronomy', 'astrophysics', 'biochemistry', 'biophysics', 'biotechnology',
                      'molecular biology', 'genetics', 'microbiology', 'zoology', 'botany',
                      'organic chemistry', 'inorganic chemistry', 'analytical chemistry',
                      'quantum physics', 'nuclear physics', 'applied physics',
                      'pure mathematics', 'applied mathematics', 'discrete mathematics'],
    
    # Engineering Branches
    'engineering': ['engineering', 'chemical engineering', 'civil engineering', 'structural engineering',
                    'electrical engineering', 'electronic engineering', 'mechanical engineering',
                    'aerospace engineering', 'aeronautical engineering', 'automotive engineering',
                    'biomedical engineering', 'industrial engineering', 'manufacturing engineering',
                    'materials engineering', 'metallurgical engineering', 'mining engineering',
                    'petroleum engineering', 'software engineering', 'systems engineering',
                    'environmental engineering', 'agricultural engineering', 'marine engineering',
                    'robotics', 'mechatronics', 'nanotechnology'],
    
    # Medical & Health Sciences
    'medical_health': ['medicine', 'medical', 'nursing', 'dentistry', 'dental', 'pharmacy', 'pharmaceutical',
                       'public health', 'health sciences', 'biomedical sciences', 'clinical sciences',
                       'anesthesiology', 'cardiology', 'dermatology', 'emergency medicine',
                       'endocrinology', 'gastroenterology', 'hematology', 'immunology',
                       'neurology', 'neuroscience', 'obstetrics', 'gynecology', 'oncology',
                       'ophthalmology', 'orthopedics', 'otolaryngology', 'pathology',
                       'pediatrics', 'psychiatry', 'radiology', 'surgery', 'urology',
                       'epidemiology', 'anatomy', 'physiology', 'pharmacology', 'toxicology',
                       'physical therapy', 'occupational therapy', 'medical laboratory',
                       'radiography', 'respiratory therapy', 'nutrition', 'dietetics'],
    
    # Business, Finance & Economics
    'business_economics': ['business', 'business administration', 'management', 'mba',
                           'accounting', 'finance', 'economics', 'econometrics',
                           'marketing', 'supply chain', 'operations', 'logistics',
                           'human resources', 'hr', 'entrepreneurship', 'strategy',
                           'hospitality', 'tourism', 'retail', 'e-commerce',
                           'international business', 'organizational behavior'],
    
    # Humanities
    'humanities': ['english literature', 'comparative literature', 'creative writing',
                   'history', 'ancient history', 'modern history', 'art history',
                   'philosophy', 'ethics', 'logic', 'metaphysics',
                   'religious studies', 'theology', 'divinity',
                   'languages', 'linguistics', 'spanish', 'french', 'german', 'chinese',
                   'japanese', 'arabic', 'latin', 'greek', 'italian', 'russian',
                   'fine arts', 'performing arts', 'film studies', 'media studies'],
    
    # Social Sciences
    'social_sciences': ['psychology', 'clinical psychology', 'cognitive psychology',
                        'sociology', 'anthropology', 'archaeology',
                        'political science', 'international relations', 'public policy',
                        'criminology', 'criminal justice', 'social work',
                        'human geography', 'urban planning', 'demography',
                        'communication', 'journalism', 'public relations'],
    
    # Education & Teaching
    'education': ['education', 'teacher education', 'curriculum', 'instruction',
                  'pedagogy', 'educational psychology', 'special education',
                  'early childhood education', 'elementary education', 'secondary education',
                  'higher education', 'adult education', 'distance learning',
                  'educational technology', 'instructional design'],
    
    # Professional & Applied Fields
    'professional': ['law', 'legal studies', 'jurisprudence',
                     'architecture', 'urban design', 'landscape architecture',
                     'library science', 'information science',
                     'kinesiology', 'sports science', 'exercise science',
                     'agriculture', 'agronomy', 'horticulture', 'veterinary',
                     'forestry', 'fisheries', 'food science',
                     'design', 'graphic design', 'fashion design', 'interior design'],
    
    # Common Educational Metrics
    'metrics': ['enrollment', 'enrolment', 'students', 'pupils', 'learners',
                'attendance', 'graduation', 'dropout', 'retention',
                'gpa', 'grade', 'score', 'marks', 'test', 'exam', 'assessment',
                'faculty', 'staff', 'teachers', 'professors', 'instructors',
                'tuition', 'fees', 'scholarship', 'financial aid',
                'admissions', 'applications', 'acceptance rate',
                'class size', 'student-teacher ratio', 'credits', 'courses'],
    
    # Institution Types
    'institutions': ['school', 'college', 'university', 'institute', 'academy',
                     'elementary', 'primary', 'secondary', 'high school',
                     'undergraduate', 'graduate', 'postgraduate', 'doctoral',
                     'campus', 'department', 'faculty', 'division', 'program']
}

# Flatten all keywords for quick lookup
ALL_EDUCATIONAL_KEYWORDS = set()
for category, keywords in EDUCATIONAL_KEYWORDS.items():
    ALL_EDUCATIONAL_KEYWORDS.update([k.lower() for k in keywords])

def extract_year_from_string(s):
    """Extract first 4-digit year from string."""
    import pandas as pd
    if pd.isna(s):
        return None
    s = str(s)
    m = YEAR_REGEX.search(s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None
