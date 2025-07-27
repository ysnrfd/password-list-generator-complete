import argparse
import string
import re
import math
import json
import os
import time
import random
import hashlib
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import nltk
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
import requests
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein


"""

Developer: YSNRFD
Telegram: @ysnrfd

"""


nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
LANGUAGE_DATA = {
    'en': {
        'name': 'English',
        'common_words': ['love', 'password', 'welcome', 'admin', 'sunshine', 'dragon', 'monkey', 'football', 'baseball', 'letmein',
                        'qwerty', 'trustno1', '123456', '12345678', 'baseball', 'football', '123456789', 'abc123', '1234567', 'monkey',
                        'iloveyou', 'princess', 'admin123', 'welcome1', 'password1', 'qwerty123', '12345', '123123', '111111', 'abc123'],
        'special_chars': ['@', '#', '$', '%', '&', '*', '!', '_', '.', '-'],
        'number_patterns': ['1234', '12345', '123456', '1111', '2023', '2024', '0000', '123123', '7777', '9999', '123', '321', '01', '13', '23', '24', '99', '00'],
        'cultural_events': ['Christmas', 'Halloween', 'Thanksgiving', 'Easter', 'New Year', 'Independence Day', 'Valentine', 'Super Bowl', 'Memorial Day', 'Labor Day', 'Cinco de Mayo', 'St. Patrick\'s Day', 'Mardi Gras', 'Fourth of July'],
        'zodiac_signs': ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces', 'Ophiuchus'],
        'celebrity_names': ['Beyonce', 'Taylor', 'Adele', 'Brad', 'Angelina', 'Elon', 'Oprah', 'Trump', 'Bieber', 'Ariana', 'Kardashian', 'Perry', 'Rihanna', 'Drake', 'Kanye', 'Kim', 'Zendaya', 'Tom', 'Jennifer', 'Leonardo'],
        'sports_teams': ['Yankees', 'Cowboys', 'Lakers', 'Giants', 'Patriots', 'Warriors', 'Cubs', 'Eagles', 'Knicks', 'Rangers', 'Red Sox', 'Jets', 'Dolphins', 'Steelers', 'Packers', 'Broncos', 'Bears', 'Seahawks'],
        'universities': ['Harvard', 'Yale', 'Stanford', 'MIT', 'Princeton', 'Columbia', 'Berkeley', 'UCLA', 'Oxford', 'Cambridge', 'Cornell', 'Duke', 'NYU', 'USC', 'Chicago', 'Penn', 'Brown', 'Dartmouth'],
        'common_dates': ['0101', '1231', '0704', '1031', '1225', '0214', '0911', '1111', '0505', '0317'],
        'leet_mappings': {
            'a': ['@', '4', 'A', 'À', 'Á'],
            'e': ['3', 'E', '&', '€', 'È', 'É'],
            'i': ['1', '!', 'I', '|', 'Ì', 'Í'],
            'o': ['0', 'O', '*', 'Ò', 'Ó'],
            's': ['$', '5', 'S', 'Š', '§'],
            't': ['+', '7', 'T', 'Ţ', 'Ť'],
            'l': ['1', '|', 'L', '£', '₤'],
            'g': ['9', '6', 'G', 'Ģ', 'Ĝ'],
            'b': ['8', 'B', 'b', 'ß'],
            'z': ['2', '7', 'Z', 'Ž']
        },
        'keyboard_patterns': [
            'qwerty', 'asdfgh', 'zxcvbn', '123456', 'qazwsx', '1q2w3e', '123qwe',
            'zaq12wsx', '1qaz2wsx', 'qwerasdf', '1234qwer', '!@#$%^&*()', '1q2w3e4r',
            'qwe123', '123asd', 'qaz123', '1qazxsw', '1q2w3e4', 'qazxsw'
        ],
        'common_suffixes': ['123', '1234', '12345', '007', '2023', '2024', '!', '@', '#', '$', '%', '&', '*', '_', '.'],
        'common_prefixes': ['my', 'i', 'the', 'new', 'old', 'super', 'mega', 'ultra', 'best', 'cool']
    },
    'de': {
        'name': 'German',
        'common_words': ['hallo', 'passwort', 'willkommen', 'admin', 'sonne', 'drache', 'affe', 'fussball', 'baseball', 'einfach',
                        'qwertz', 'geheim1', '123456', '12345678', 'fussball', '123456789', 'abc123', '1234567', 'affe',
                        'liebe', 'prinzessin', 'admin123', 'willkommen1', 'passwort1', 'qwertz123', '12345', '123123', '111111', 'abc123'],
        'special_chars': ['@', '#', '$', '%', '&', '*', '!', '_', '.', '-'],
        'number_patterns': ['1234', '12345', '123456', '1111', '2023', '2024', '0000', '123123', '7777', '9999', '123', '321', '01', '13', '18', '42', '77', '88'],
        'cultural_events': ['Weihnachten', 'Halloween', 'Erntedank', 'Ostern', 'Neujahr', 'Tag der Deutschen Einheit', 'Valentinstag', 'Oktoberfest', 'Karneval', 'Silvester', 'Muttertag', 'Vatertag', 'Schützenfest'],
        'zodiac_signs': ['Widder', 'Stier', 'Zwillinge', 'Krebs', 'Löwe', 'Jungfrau', 'Waage', 'Skorpion', 'Schütze', 'Steinbock', 'Wassermann', 'Fische', 'Schlangenträger'],
        'celebrity_names': ['Angela', 'Merkel', 'Bundesliga', 'Bayern', 'Dortmund', 'Schumi', 'Schumacher', 'Lindemann', 'Rammstein', 'Klum', 'Helene', 'Fischer', 'Thomas', 'Gottschalk', 'Heidi', 'Klum', 'Böhmermann'],
        'sports_teams': ['Bayern', 'Dortmund', 'Schalke', 'BVB', 'FCB', 'Werder', 'Hoffenheim', 'RB Leipzig', 'Bayer', 'Leverkusen', 'Hamburg', 'Frankfurt', 'Wolfsburg', 'Stuttgart', 'Union', 'Köln'],
        'universities': ['LMU', 'TUM', 'Heidelberg', 'Humboldt', 'FU Berlin', 'KIT', 'RWTH', 'Goethe', 'Tübingen', 'Freiburg', 'Jena', 'Konstanz', 'Bonn', 'Halle', 'Marburg', 'Göttingen'],
        'common_dates': ['0101', '1231', '1003', '3110', '2512', '1402', '0911', '1111', '1508', '0310'],
        'leet_mappings': {
            'a': ['@', '4', 'Ä', 'ä', 'Â'],
            'e': ['3', 'E', '&', '€', 'È', 'É'],
            'i': ['1', '!', 'I', '|', 'Ì', 'Í'],
            'o': ['0', 'O', 'Ö', 'ö', 'Ò', 'Ó'],
            's': ['$', '5', 'S', 'ß', 'Š', '§'],
            't': ['+', '7', 'T', 'Ţ', 'Ť'],
            'l': ['1', '|', 'L', '£', '₤'],
            'g': ['9', '6', 'G', 'Ģ', 'Ĝ'],
            'b': ['8', 'B', 'b', 'ß'],
            'u': ['µ', 'ü', 'Ü']
        },
        'keyboard_patterns': [
            'qwertz', 'asdfgh', 'yxcvbn', '123456', 'qaywsx', '1q2w3e', '123qwe',
            'zaq12wsx', '1qaz2wsx', 'qwerasdf', '1234qwer', '!@#$%^&*()', '1q2w3e4r',
            'qwe123', '123asd', 'qaz123', '1qazxsw', '1q2w3e4', 'qazxsw', 'qay123'
        ],
        'common_suffixes': ['123', '1234', '12345', '007', '2023', '2024', '!', '@', '#', '$', '%', '&', '*', '_', '.'],
        'common_prefixes': ['mein', 'meine', 'der', 'die', 'das', 'super', 'mega', 'ultra', 'gut', 'cool']
    },
    'fa': {
        'name': 'Persian',
        'common_words': ['سلام', 'پسورد', 'خوش آمدید', 'مدیر', 'خورشید', 'اژدها', 'میمون', 'فوتبال', 'بیسبال', 'بگذار',
                        '123456', '12345678', 'فوتبال', '123456789', 'abc123', '1234567', 'میمون',
                        'عشق', 'شیرینی', 'admin123', 'خوش آمدید1', 'پسورد1', '12345', '123123', '111111', 'abc123'],
        'special_chars': ['@', '#', '$', '%', '&', '*', '!', '_', '.', '-'],
        'number_patterns': ['1234', '12345', '123456', '1111', '2023', '2024', '0000', '123123', '7777', '9999', '123', '321', '01', '13', '88', '99', '110', '313', '5', '7', '14', '22'],
        'cultural_events': ['نوروز', 'تاسوعا', 'عاشورا', 'یلدا', 'عید نوروز', 'عید فطر', 'عید قربان', 'شب یلدا', 'سیزده به در', 'رحلت پیامبر', 'ولادت علی', 'عید سعید', 'عید ده', 'عید دو'],
        'zodiac_signs': ['حمل', 'ثور', 'جوزا', 'سرطان', 'اسد', 'سنبله', 'میزان', 'عقرب', 'قوس', 'جدی', 'دلو', 'حوت', 'مارپیچ'],
        'celebrity_names': ['محمد', 'رضا', 'احمد', 'علی', 'حسین', 'فاطمه', 'زهرا', 'شادی', 'پوریا', 'سحر', 'سارا', 'مریم', 'رضا', 'صدر', 'حسین', 'پور', 'خان', 'علوی', 'محمدی', 'حسینی'],
        'sports_teams': ['استقلال', 'پرسپولیس', 'تراکتور', 'سپاهان', 'فولاد', 'ذوب آهن', 'سایپا', 'ملی', 'ملی ایران', 'سپاهان', 'ذوب', 'آهن', 'تراکتور', 'ذوب', 'سپاهان'],
        'universities': ['تهران', 'شریف', 'امیرکبیر', 'صنعتی', 'شهید', 'بهشتی', 'فردوسی', 'مشهد', 'اصفهان', 'شیراز', 'عمران', 'مکانیک', 'علوم', 'پزشکی', 'تربیت'],
        'common_dates': ['0101', '1231', '2103', '1302', '2206', '1402', '0911', '1111', '0104', '0102', '1102', '1301'],
        'leet_mappings': {
            'a': ['@', '4', 'آ', 'ا', 'أ'],
            'i': ['1', '!', 'ی', 'ي', 'ئ'],
            'o': ['0', '*', 'او', 'ؤ'],
            's': ['$', '5', 'ث', 'س'],
            'l': ['1', '|', 'ل', 'لـ'],
            'g': ['9', '6', 'گ', 'گـ'],
            'b': ['8', 'ب', 'بـ'],
            'p': ['9', 'پ', 'پـ'],
            't': ['7', 'ط', 'ت'],
            'j': ['7', 'ج', 'چ']
        },
        'keyboard_patterns': [
            'ضثص', 'شیس', 'آکل', '123456', 'ضشی', '1ض2ث3ص',
            'ضشیثص', '1ض2ش3ی', 'ضثشی', '1234ضث', '!@#$%^&*()',
            'ضصثق', 'یسش', 'آکل', '12345', 'ضشی', '1ض2ث3ص4',
            'ضشیثص', '1ض2ش3ی', 'ضثشی', '1234ضث', 'ضثص123'
        ],
        'common_suffixes': ['123', '1234', '12345', '007', '2023', '2024', '!', '@', '#', '$', '%', '&', '*', '_', '.'],
        'common_prefixes': ['من', 'منو', 'عشق', 'دوست', 'دوست دارم', 'مثل', 'خیلی', 'خیلیی', 'عزیزم', 'عزیزمم']
    },
    'fr': {
        'name': 'French',
        'common_words': ['bonjour', 'motdepasse', 'bienvenue', 'admin', 'soleil', 'dragon', 'singe', 'football', 'baseball', 'simple',
                        'azerty', 'secret1', '123456', '12345678', 'football', '123456789', 'abc123', '1234567', 'singe',
                        'amour', 'princesse', 'admin123', 'bienvenue1', 'motdepasse1', 'azerty123', '12345', '123123', '111111', 'abc123'],
        'special_chars': ['@', '#', '$', '%', '&', '*', '!', '_', '.', '-'],
        'number_patterns': ['1234', '12345', '123456', '1111', '2023', '2024', '0000', '123123', '7777', '9999', '123', '321', '01', '13', '14', '75', '89', '42', '18'],
        'cultural_events': ['Noël', 'Halloween', 'Action de Grâce', 'Pâques', 'Nouvel An', 'Fête Nationale', 'Saint-Valentin', 'Tour de France', 'Bastille Day', 'La Fête de la Musique', 'Carnaval', 'Fête du Travail'],
        'zodiac_signs': ['Bélier', 'Taureau', 'Gémeaux', 'Cancer', 'Lion', 'Vierge', 'Balance', 'Scorpion', 'Sagittaire', 'Capricorne', 'Verseau', 'Poissons', 'Ophiuchus'],
        'celebrity_names': ['Zidane', 'Del Piero', 'Depardieu', 'Johnny', 'Hallyday', 'Sarkozy', 'Macron', 'Amelie', 'Poulain', 'Audrey', 'Tautou', 'Gad', 'Elkabbach', 'Deneuve'],
        'sports_teams': ['PSG', 'OM', 'OL', 'Bayern', 'Real', 'Barça', 'Marseille', 'Lyon', 'Paris', 'Monaco', 'Saint-Étienne', 'Nantes', 'Lille', 'Lens'],
        'universities': ['Sorbonne', 'Polytechnique', 'Sciences Po', 'HEC', 'ENS', 'Dauphine', 'Panthéon', 'Aix-Marseille', 'Lyon 2', 'Toulouse 1', 'Grenoble', 'Strasbourg'],
        'common_dates': ['0101', '1231', '1407', '1111', '2512', '1402', '0911', '1111', '0104', '0102', '1107', '0803'],
        'leet_mappings': {
            'a': ['@', '4', 'À', 'Á', 'Â'],
            'e': ['3', 'E', '€', 'È', 'É', 'Ê', 'Ë'],
            'i': ['1', '!', 'I', '|', 'Ì', 'Í'],
            'o': ['0', 'O', 'Ö', 'Ò', 'Ó'],
            's': ['$', '5', 'S', 'Š'],
            't': ['+', '7', 'T', 'Ţ'],
            'l': ['1', '|', 'L', '£'],
            'g': ['9', '6', 'G', 'Ĝ'],
            'b': ['8', 'B', 'b']
        },
        'keyboard_patterns': [
            'azerty', 'qsdfgh', 'wxcvbn', '123456', 'aqwzsx', '1&2é3"', '123&é"',
            '1234az', 'qsdfaz', '1234qwer', '!@#$%^&*()', '1&2é34"',
            'qwertz', '12345', '1q2w3e', '123qwe', 'qaz123', '1qazxsw'
        ],
        'common_suffixes': ['123', '1234', '12345', '007', '2023', '2024', '!', '@', '#', '$', '%', '&', '*', '_', '.'],
        'common_prefixes': ['mon', 'ma', 'mes', 'le', 'la', 'les', 'super', 'mega', 'ultra', 'bon', 'bien']
    },
    'es': {
        'name': 'Spanish',
        'common_words': ['hola', 'contraseña', 'bienvenido', 'admin', 'sol', 'dragón', 'mono', 'fútbol', 'béisbol', 'fácil',
                        'qwerty', 'secreto1', '123456', '12345678', 'fútbol', '123456789', 'abc123', '1234567', 'mono',
                        'amor', 'princesa', 'admin123', 'bienvenido1', 'contraseña1', 'qwerty123', '12345', '123123', '111111', 'abc123'],
        'special_chars': ['@', '#', '$', '%', '&', '*', '!', '_', '.', '-'],
        'number_patterns': ['1234', '12345', '123456', '1111', '2023', '2024', '0000', '123123', '7777', '9999', '123', '321', '01', '13', '15', '80', '21', '99', '00'],
        'cultural_events': ['Navidad', 'Halloween', 'Día de Acción de Gracias', 'Pascua', 'Año Nuevo', 'Día de la Independencia', 'San Valentín', 'Feria de Abril', 'San Fermín', 'Día de los Muertos', 'La Tomatina', 'Semana Santa'],
        'zodiac_signs': ['Aries', 'Tauro', 'Géminis', 'Cáncer', 'Leo', 'Virgo', 'Libra', 'Escorpio', 'Sagitario', 'Capricornio', 'Acuario', 'Piscis', 'Ofiuco'],
        'celebrity_names': ['Messi', 'Ronaldo', 'Beyoncé', 'Shakira', 'Piqué', 'García', 'Martínez', 'Rodríguez', 'Fernández', 'López', 'González', 'Pérez', 'Sánchez', 'Díaz'],
        'sports_teams': ['Barça', 'Madrid', 'Atletico', 'Barcelona', 'Real', 'Madrid', 'Sevilla', 'Valencia', 'Atlético', 'Betis', 'Villarreal', 'Athletic', 'Espanyol', 'Málaga'],
        'universities': ['Complutense', 'Autónoma', 'Politécnica', 'Barcelona', 'Sorbona', 'Salamanca', 'Granada', 'Sevilla', 'Valencia', 'Málaga', 'Santiago', 'Navarra'],
        'common_dates': ['0101', '1231', '1207', '1508', '2512', '1402', '0911', '1111', '0104', '0102', '2802', '0208'],
        'leet_mappings': {
            'a': ['@', '4', 'Á', 'À'],
            'e': ['3', 'E', '€', 'É', 'È'],
            'i': ['1', '!', 'I', '|', 'Í', 'Ì'],
            'o': ['0', 'O', 'Ö', 'Ó', 'Ò'],
            's': ['$', '5', 'S', 'Š'],
            't': ['+', '7', 'T'],
            'l': ['1', '|', 'L', '£'],
            'g': ['9', '6', 'G', 'Ĝ'],
            'b': ['8', 'B', 'b']
        },
        'keyboard_patterns': [
            'qwerty', 'asdfgh', 'zxcvbn', '123456', 'qwaszx', '1q2w3e', '123qwe',
            'zaq12wsx', '1qaz2wsx', 'qwerasdf', '1234qwer', '!@#$%^&*()', '1q2w3e4r',
            'qwe123', '123asd', 'qaz123', '1qazxsw', '1q2w3e4', 'qazxsw'
        ],
        'common_suffixes': ['123', '1234', '12345', '007', '2023', '2024', '!', '@', '#', '$', '%', '&', '*', '_', '.'],
        'common_prefixes': ['mi', 'mis', 'el', 'la', 'los', 'las', 'super', 'mega', 'ultra', 'buen', 'bien']
    }
}
CURRENT_LANGUAGE = 'en'
class EthicalSafeguard:
    def __init__(self):
        self.usage_log = []
        self.authorization_key = None
        self.ethical_agreement = False
        self.geolocation_verified = False
        self.purpose_verified = False
    def verify_ethical_usage(self):
        print("\n🛡️  ETHICAL USAGE VERIFICATION REQUIRED")
        print("This tool is strictly for educational and authorized security testing purposes only.")
        try:
            response = requests.get('https://ipapi.co/json/', timeout=5)
            if response.status_code == 200:
                geo_data = response.json()
                country = geo_data.get('country', '').lower()
                print(f"📍 Detected country: {geo_data.get('country_name', 'Unknown')}")
                restricted_countries = ['cn', 'ru', 'kp', 'iq', 'ir', 'sy']
                if country in restricted_countries:
                    print(f"❌ Usage restricted in {geo_data['country_name']} due to local regulations")
                    return False
                self.geolocation_verified = True
        except:
            print("⚠️  Could not verify geolocation. Proceed with caution.")
        print("\n📜 ETHICAL AGREEMENT:")
        print("1. I confirm I have explicit written authorization to test the target system")
        print("2. I understand that unauthorized access is illegal and unethical")
        print("3. I will not use this tool for any malicious or unauthorized purpose")
        print("4. I accept full responsibility for any consequences of my actions")
        agree = input("\nDo you agree to these terms? (YES/NO): ").strip().upper()
        if agree != "YES":
            print("❌ Ethical agreement not accepted. Exiting...")
            return False
        self.ethical_agreement = True
        print("\n🔍 PURPOSE VERIFICATION:")
        print("Please describe the authorized purpose of this security test:")
        purpose = input("> ").strip()
        valid_purposes = [
            'penetration testing', 'security assessment', 'vulnerability research', 
            'educational purpose', 'authorized security test', 'red team exercise'
        ]
        if not any(p in purpose.lower() for p in valid_purposes):
            print("❌ Purpose does not match authorized security testing. Exiting...")
            return False
        self.purpose_verified = True
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        self.authorization_key = f"AUTH-{timestamp}-{random_str}"
        self.usage_log.append({
            'timestamp': datetime.now().isoformat(),
            'agreement_accepted': True,
            'purpose': purpose,
            'authorization_key': self.authorization_key,
            'geolocation_verified': self.geolocation_verified
        })
        print(f"\n✅ Ethical verification successful!")
        print(f"🔑 Authorization Key: {self.authorization_key}")
        print("⚠️  This key must be documented in your security testing report")
        return True
    def log_usage(self, passwords_generated, target_info):
        usage_record = {
            'timestamp': datetime.now().isoformat(),
            'authorization_key': self.authorization_key,
            'passwords_generated': passwords_generated,
            'target_info_summary': {
                'has_name': bool(target_info.get('first_name')),
                'has_birthdate': bool(target_info.get('birthdate')),
                'has_email': bool(target_info.get('email')),
                'language': target_info.get('language', 'en')
            },
            'ethical_verification': {
                'agreement': self.ethical_agreement,
                'geolocation': self.geolocation_verified,
                'purpose': self.purpose_verified
            }
        }
        log_dir = os.path.join(os.path.expanduser("~"), ".security_tool_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"usage_log_{datetime.now().strftime('%Y%m%d')}.enc")
        encrypted_log = hashlib.sha256(json.dumps(usage_record).encode()).hexdigest()
        with open(log_file, 'a') as f:
            f.write(encrypted_log + "\n")
        return usage_record
class UserBehaviorPredictor:
    def __init__(self, info):
        self.user_info = info
        self.behavior_profile = self._build_behavior_profile()
    def _build_behavior_profile(self):
        profile = {
            'security_awareness': 0.5,
            'password_complexity_preference': 0.5,
            'cultural_influence': 'moderate',
            'emotional_attachment_level': 0.5,
            'tech_savviness': 0.5
        }
        if self.user_info.get('password_change_frequency'):
            try:
                freq = int(self.user_info['password_change_frequency'])
                profile['security_awareness'] = min(1.0, freq / 3)
            except:
                pass
        tech_indicators = 0
        if self.user_info.get('tech_savviness'):
            try:
                profile['tech_savviness'] = min(1.0, int(self.user_info['tech_savviness']) / 10)
                tech_indicators += 1
            except:
                pass
        if self.user_info.get('occupation') in ['developer', 'engineer', 'security', 'it']:
            profile['tech_savviness'] = 0.8
            tech_indicators += 1
        if self.user_info.get('device_models'):
            if any('iphone' in d.lower() or 'android' in d.lower() for d in self.user_info['device_models']):
                profile['tech_savviness'] = max(profile['tech_savviness'], 0.3)
                tech_indicators += 0.5
        if tech_indicators > 1 or profile['tech_savviness'] > 0.6:
            profile['password_complexity_preference'] = 0.7
        else:
            profile['password_complexity_preference'] = 0.3
        nationality = self.user_info.get('nationality', '').lower()
        if any(c in nationality for c in ['iran', 'persia', 'farsi']):
            profile['cultural_influence'] = 'high'
        elif any(c in nationality for c in ['usa', 'uk', 'canada', 'australia']):
            profile['cultural_influence'] = 'low'
        emotional_indicators = 0
        if self.user_info.get('pets'):
            emotional_indicators += len(self.user_info['pets']) * 0.2
        if self.user_info.get('children'):
            emotional_indicators += len(self.user_info['children']) * 0.3
        if self.user_info.get('spouse'):
            emotional_indicators += 0.3
        profile['emotional_attachment_level'] = min(1.0, emotional_indicators)
        return profile
    def predict_password_patterns(self):
        patterns = {
            'structure': [],
            'transformations': [],
            'common_elements': [],
            'avoided_patterns': []
        }
        security_awareness = self.behavior_profile['security_awareness']
        if security_awareness < 0.3:
            patterns['structure'] = [
                '{name}{birth_year}', 
                '{pet}{number}',
                '{favorite}{special}{number}',
                '{name}{birth_day}{birth_month}',
                '{pet}{birth_year}',
                '{child}{number}'
            ]
            patterns['avoided_patterns'] = ['{complex_mixture}', '{random_caps}']
        elif security_awareness < 0.6:
            patterns['structure'] = [
                '{name}{special}{birth_year}',
                '{pet}{number}{special}',
                '{word}{number}{special}',
                '{name}{birth_year}{special}',
                '{pet}{birth_year}',
                '{spouse}{number}',
                '{child}{special}{number}'
            ]
            patterns['transformations'] = ['add_number', 'add_special', 'capitalize', 'simple_leet']
        elif security_awareness < 0.8:
            patterns['structure'] = [
                '{word1}{special}{word2}{number}',
                '{word}{special}{number}{special}',
                '{name}{pet}{number}',
                '{zodiac}{number}',
                '{cultural_event}{number}'
            ]
            patterns['transformations'] = ['leet_speak', 'camel_case', 'random_caps', 'add_special']
        else:
            patterns['structure'] = [
                '{random_mixture}',
                '{complex_pattern}',
                '{word1}{word2}{number}{special}',
                '{cultural_event}{zodiac}{number}'
            ]
            patterns['transformations'] = ['complex_leet', 'random_caps', 'spinal_case', 'hex_encoding']
        
        tech_savviness = self.behavior_profile['tech_savviness']
        if tech_savviness > 0.8:
            patterns['transformations'].extend(['hex_encoding', 'base64_patterns', 'unicode_mixing'])
            patterns['common_elements'].extend(['tech_terms', 'crypto_terms'])
        
        emotional_level = self.behavior_profile['emotional_attachment_level']
        if emotional_level > 0.8:
            patterns['structure'].insert(0, '{pet}{child}{special}{number}')
            patterns['structure'].insert(0, '{spouse}{pet}{number}')
            patterns['structure'].insert(0, '{child}{birth_year}')
        
        cultural_influence = self.behavior_profile['cultural_influence']
        if cultural_influence == 'high':
            patterns['common_elements'].extend(['cultural_events', 'zodiac', 'national_holidays'])
            if self.user_info.get('nationality', '').lower() in ['iran', 'persia', 'farsi']:
                patterns['structure'].extend(['{cultural_event}{number}', '{zodiac}{number}'])
        
        current_year = datetime.now().year
        if self.user_info.get('birth_year'):
            birth_year = int(self.user_info['birth_year'])
            age = current_year - birth_year
            if 13 <= age <= 25:
                patterns['common_elements'].append('pop_culture')
            elif 26 <= age <= 40:
                patterns['common_elements'].append('family_elements')
            elif age > 40:
                patterns['common_elements'].append('nostalgic_elements')
        
        return patterns
    def get_password_generation_weights(self):
        weights = {
            'personal_info': 0.7,
            'dates': 0.8,
            'pets': 0.9,
            'children': 0.95,
            'spouse': 0.92,
            'interests': 0.6,
            'cultural': 0.5,
            'keyboard': 0.3,
            'common': 0.2,
            'tech_terms': 0.4,
            'crypto_terms': 0.3
        }
        if self.behavior_profile['emotional_attachment_level'] > 0.7:
            weights['pets'] = 0.95
            weights['children'] = 0.95
            weights['spouse'] = 0.92
            weights['anniversary'] = 0.93
        if self.behavior_profile['security_awareness'] < 0.4:
            weights['dates'] = 0.9
            weights['personal_info'] = 0.85
        elif self.behavior_profile['security_awareness'] > 0.7:
            weights['keyboard'] = 0.6
            weights['common'] = 0.4
            weights['tech_terms'] = 0.7
        if self.behavior_profile['cultural_influence'] == 'high':
            weights['cultural'] = 0.75
        return weights
class PasswordEntropyAnalyzer:
    def __init__(self, language='en'):
        self.language = language
        self.lang_data = LANGUAGE_DATA.get(language, LANGUAGE_DATA['en'])
        self.common_patterns = [
            r'(?:password|pass|1234|qwerty|admin|login|welcome|123456|111111|iloveyou)',
            r'(\d{4})\1', r'(.)\1{2,}', 
            r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|uvw|vwx|wxy|xyz)'
        ]
        self.dictionary_words = set()
        self.dictionary_words.update(LANGUAGE_DATA[language]['common_words'])
        for w in wn.all_lemma_names():
            if len(w) > 3:
                self.dictionary_words.add(w.lower())
        self.dictionary_words.update([
            'christmas', 'halloween', 'thanksgiving', 'easter', 'newyear', 'valentine',
            'yankees', 'cowboys', 'lakers', 'giants', 'patriots', 'warriors',
            'harvard', 'yale', 'stanford', 'mit', 'princeton', 'columbia',
            'superbowl', 'eagles', 'knicks', 'rangers', 'redsox'
        ])
    def calculate_entropy(self, password):
        if not password:
            return 0
        char_space = 0
        if any(c.islower() for c in password): char_space += 26
        if any(c.isupper() for c in password): char_space += 26
        if any(c.isdigit() for c in password): char_space += 10
        if any(c in string.punctuation for c in password): char_space += len(string.punctuation)
        freq = Counter(password)
        entropy = -sum((count / len(password)) * math.log2(count / len(password)) for count in freq.values())
        for pattern in self.common_patterns:
            if re.search(pattern, password.lower()):
                entropy *= 0.2
        keyboard_walks = self.lang_data['keyboard_patterns']
        for walk in keyboard_walks:
            if walk in password.lower():
                entropy *= 0.3
        for word in self.dictionary_words:
            if word in password.lower() and len(word) > 3:
                entropy *= 0.4
        complexity_bonus = 1.0
        if (any(c.islower() for c in password) and any(c.isupper() for c in password)):
            complexity_bonus += 0.2
        if any(c.isdigit() for c in password):
            complexity_bonus += 0.15
        if any(c in string.punctuation for c in password):
            complexity_bonus += 0.25
        if re.search(r'\d{4}', password) and any(c.isalpha() for c in password):
            complexity_bonus += 0.1
        entropy *= complexity_bonus
        length_bonus = min(1.0, len(password) / 16) * 0.4
        entropy *= (1 + length_bonus)
        return round(entropy * len(password), 2)
    def analyze_password_patterns(self, password):
        analysis = {
            'length': len(password),
            'has_upper': any(c.isupper() for c in password),
            'has_lower': any(c.islower() for c in password),
            'has_digit': any(c.isdigit() for c in password),
            'has_special': any(c in string.punctuation for c in password),
            'digit_count': sum(1 for c in password if c.isdigit()),
            'special_count': sum(1 for c in password if c in string.punctuation),
            'repeated_chars': self._detect_repeated_chars(password),
            'keyboard_patterns': self._detect_keyboard_patterns(password),
            'common_words': self._detect_common_words(password),
            'cultural_patterns': self._detect_cultural_patterns(password),
            'entropy': self.calculate_entropy(password)
        }
        return analysis
    def _detect_repeated_chars(self, password):
        repeats = []
        for match in re.finditer(r'(.)\1{2,}', password):
            repeats.append({
                'char': match.group(1),
                'count': len(match.group(0)),
                'position': match.start()
            })
        return repeats
    def _detect_keyboard_patterns(self, password):
        patterns = []
        password_lower = password.lower()
        keyboard_layouts = {
            'qwerty': [
                'qwertyuiop', 'asdfghjkl', 'zxcvbnm',
                '1234567890', '!@#$%^&*()'
            ],
            'qwertz': [
                'qwertzuiop', 'asdfghjkl', 'yxcvbnm',
                '1234567890', '!@#$%^&*()'
            ],
            'azerty': [
                'azertyuiop', 'qsdfghjklm', 'wxcvbn',
                '1234567890', '&é"\'(-è_çà)'
            ],
            'dvorak': [
                'pyfgcrl', 'aoeuidhtns', 'qjkxbmwvz',
                '1234567890', '!@#$%^&*()'
            ]
        }
        likely_layout = 'qwerty'
        if 'z' in password_lower and 'w' in password_lower and 'x' in password_lower:
            likely_layout = 'qwertz'
        elif 'a' in password_lower and 'z' in password_lower and 'q' in password_lower:
            likely_layout = 'azerty'
        elif 'p' in password_lower and 'y' in password_lower and 'f' in password_lower:
            likely_layout = 'dvorak'
        layout = keyboard_layouts.get(likely_layout, keyboard_layouts['qwerty'])
        for row in layout:
            for i in range(len(row)):
                for j in range(i+3, min(i+10, len(row)+1)):
                    segment = row[i:j]
                    if segment in password_lower:
                        patterns.append({
                            'pattern': segment,
                            'type': f'{likely_layout}_horizontal',
                            'length': len(segment),
                            'layout': likely_layout
                        })
        vertical_sequences = []
        for col_idx in range(min(len(row) for row in layout if len(row) > 0)):
            col_chars = []
            for row in layout:
                if col_idx < len(row):
                    col_chars.append(row[col_idx])
            col_str = ''.join(col_chars)
            if len(col_str) >= 3:
                vertical_sequences.append(col_str)
        for seq in vertical_sequences:
            for i in range(len(seq)):
                for j in range(i+3, min(i+8, len(seq)+1)):
                    segment = seq[i:j]
                    if segment in password_lower:
                        patterns.append({
                            'pattern': segment,
                            'type': f'{likely_layout}_vertical',
                            'length': len(segment),
                            'layout': likely_layout
                        })
        diagonal_sequences = []
        for start_row in range(len(layout)):
            for start_col in range(len(layout[start_row])):
                diag_chars = []
                row, col = start_row, start_col
                while row < len(layout) and col < len(layout[row]):
                    diag_chars.append(layout[row][col])
                    row += 1
                    col += 1
                if len(diag_chars) >= 3:
                    diagonal_sequences.append(''.join(diag_chars))
        for start_row in range(len(layout)):
            for start_col in range(len(layout[start_row])):
                diag_chars = []
                row, col = start_row, start_col
                while row < len(layout) and col >= 0:
                    if col < len(layout[row]):
                        diag_chars.append(layout[row][col])
                    row += 1
                    col -= 1
                if len(diag_chars) >= 3:
                    diagonal_sequences.append(''.join(diag_chars))
        for seq in diagonal_sequences:
            for i in range(len(seq)):
                for j in range(i+3, min(i+8, len(seq)+1)):
                    segment = seq[i:j]
                    if segment in password_lower:
                        patterns.append({
                            'pattern': segment,
                            'type': f'{likely_layout}_diagonal',
                            'length': len(segment),
                            'layout': likely_layout
                        })
        spiral_patterns = [
            'q2we43', 'qaz2sx3ed', '1qaz2wsx', 'zaq12wsx',
            'qscde32', 'qwe321', '123edc', 'asdf432',
            'q1a2z3', 'q12w3e', '1q2w3e4r', '123qwe',
            '1234qwer', 'qwer4321', '123456', '654321'
        ]
        for pattern in spiral_patterns:
            if pattern in password_lower:
                patterns.append({
                    'pattern': pattern,
                    'type': 'spiral',
                    'length': len(pattern),
                    'layout': likely_layout
                })
        return patterns
    def _detect_common_words(self, password):
        matches = []
        password_lower = password.lower()
        for word in self.dictionary_words:
            if word in password_lower and len(word) > 3:
                matches.append({
                    'word': word,
                    'position': password_lower.find(word),
                    'length': len(word)
                })
        return matches
    def _detect_cultural_patterns(self, password):
        patterns = []
        password_lower = password.lower()
        for event in self.lang_data['cultural_events']:
            if event.lower() in password_lower:
                patterns.append({
                    'pattern': event,
                    'type': 'cultural_event',
                    'relevance': 0.8
                })
        for sign in self.lang_data['zodiac_signs']:
            if sign.lower() in password_lower:
                patterns.append({
                    'pattern': sign,
                    'type': 'zodiac',
                    'relevance': 0.7
                })
        for team in self.lang_data['sports_teams']:
            if team.lower() in password_lower:
                patterns.append({
                    'pattern': team,
                    'type': 'sports_team',
                    'relevance': 0.6
                })
        for university in self.lang_data['universities']:
            if university.lower() in password_lower:
                patterns.append({
                    'pattern': university,
                    'type': 'university',
                    'relevance': 0.5
                })
        for pattern in self.lang_data['number_patterns']:
            if pattern in password:
                patterns.append({
                    'pattern': pattern,
                    'type': 'number_pattern',
                    'relevance': 0.4
                })
        if self.language == 'fa':
            persian_patterns = {
                'religious': ['روز', 'عید', 'نماز', 'قرآن', 'حاج', 'سجاد', 'حسین', 'فاطمه', 'زهره', 'عاشورا', 'نوروز', 'یلدا'],
                'national': ['ایران', 'تهران', 'شهید', 'نظام', 'انقلاب', 'آزادی', 'سپاه', 'بدر', 'قادسیه', 'پارس', 'شیراز']
            }
            for category, words in persian_patterns.items():
                for word in words:
                    if word.lower() in password_lower:
                        patterns.append({
                            'pattern': word,
                            'type': f'persian_{category}',
                            'relevance': 0.7 if category == 'religious' else 0.6
                        })
        elif self.language == 'de':
            german_patterns = {
                'cultural': ['oktoberfest', 'bier', 'wurst', 'bayern', 'berlin', 'deutschland', 'karneval'],
                'historical': ['mauer', 'berliner', 'euro', 'dm', 'pfennig', 'reich', 'wiedervereinigung']
            }
            for category, words in german_patterns.items():
                for word in words:
                    if word.lower() in password_lower:
                        patterns.append({
                            'pattern': word,
                            'type': f'german_{category}',
                            'relevance': 0.6
                        })
        elif self.language == 'fr':
            french_patterns = {
                'cultural': ['tour', 'eiffel', 'paris', 'baguette', 'fromage', 'vin', 'bastille'],
                'historical': ['revolution', 'napoleon', 'berlin', 'liberté', 'fraternité', 'egalité']
            }
            for category, words in french_patterns.items():
                for word in words:
                    if word.lower() in password_lower:
                        patterns.append({
                            'pattern': word,
                            'type': f'french_{category}',
                            'relevance': 0.6
                        })
        elif self.language == 'es':
            spanish_patterns = {
                'cultural': ['flamenco', 'paella', 'toro', 'fiesta', 'barça', 'madrid', 'tomatina'],
                'historical': ['inquisicion', 'colombus', 'espana', 'reconquista', 'cervantes']
            }
            for category, words in spanish_patterns.items():
                for word in words:
                    if word.lower() in password_lower:
                        patterns.append({
                            'pattern': word,
                            'type': f'spanish_{category}',
                            'relevance': 0.6
                        })
        return patterns
class ContextualPasswordGenerator:
    def __init__(self, language='en'):
        self.language = language
        self.lang_data = LANGUAGE_DATA.get(language, LANGUAGE_DATA['en'])
        self.entropy_analyzer = PasswordEntropyAnalyzer(language)
        self.context_weights = self._initialize_context_weights()
        self.context_info = {}
    def _initialize_context_weights(self):
        return {
            'personal_info': 0.8,
            'dates': 0.9,
            'pets': 0.7,
            'children': 0.8,
            'spouse': 0.75,
            'interests': 0.6,
            'cultural': 0.5,
            'keyboard': 0.4,
            'common': 0.3,
            'tech_terms': 0.4,
            'crypto_terms': 0.3
        }
    def _calculate_relevance_score(self, info, element, category):
        score = self.context_weights.get(category, 0.5)
        psychological_factors = {
            'emotional_value': 0.0,
            'temporal_relevance': 0.0,
            'cognitive_load': 0.0,
            'length_factor': 0.0
        }
        emotional_keywords = {
            'pet': 0.85, 'child': 0.92, 'spouse': 0.88, 'anniversary': 0.75,
            'favorite': 0.78, 'love': 0.95, 'heart': 0.82, 'baby': 0.90, 'soulmate': 0.93
        }
        if isinstance(element, str):
            element_lower = element.lower()
            for keyword, value in emotional_keywords.items():
                if keyword in element_lower or category == keyword:
                    psychological_factors['emotional_value'] = max(
                        psychological_factors['emotional_value'], value
                    )
            current_year = datetime.now().year
            if re.search(r'\d{4}', element):
                year_match = re.search(r'(\d{4})', element)
                if year_match:
                    year = int(year_match.group(1))
                    if abs(current_year - year) <= 2:
                        psychological_factors['temporal_relevance'] = 0.7
                    elif year == int(info.get('birth_year', 0)):
                        psychological_factors['temporal_relevance'] = 0.9
            if 3 <= len(element) <= 8:
                psychological_factors['cognitive_load'] = 0.6
            elif 9 <= len(element) <= 12:
                psychological_factors['cognitive_load'] = 0.3
            else:
                psychological_factors['cognitive_load'] = 0.1
            if 8 <= len(element) <= 12:
                psychological_factors['length_factor'] = 0.8
            elif 6 <= len(element) <= 14:
                psychological_factors['length_factor'] = 0.6
            else:
                psychological_factors['length_factor'] = 0.2
        emotional_weight = 0.35
        temporal_weight = 0.20
        cognitive_weight = 0.20
        length_weight = 0.25
        psychological_score = (
            psychological_factors['emotional_value'] * emotional_weight +
            psychological_factors['temporal_relevance'] * temporal_weight +
            (1 - psychological_factors['cognitive_load']) * cognitive_weight +
            psychological_factors['length_factor'] * length_weight
        )
        final_score = (score * 0.5) + (psychological_score * 0.5)
        if category in ['pets', 'children', 'spouse', 'favorite_numbers', 'anniversary']:
            final_score *= 1.3
        if category in ['common', 'keyboard'] and info.get('password_patterns') and 'complex' in info['password_patterns']:
            final_score *= 0.6
        return min(1.0, max(0.2, final_score))
    def _apply_leet_transformations(self, text):
        if not text or len(text) < 3:
            return [text]
        results = set([text])
        text_lower = text.lower()
        target_language = self.language
        if 'nationality' in self.context_info:
            nationality_to_lang = {
                'usa': 'en', 'united states': 'en', 'america': 'en',
                'uk': 'en', 'united kingdom': 'en', 'britain': 'en',
                'germany': 'de', 'deutschland': 'de', 'deutsch': 'de',
                'france': 'fr', 'français': 'fr', 'france': 'fr',
                'spain': 'es', 'españa': 'es', 'spanish': 'es',
                'iran': 'fa', 'persian': 'fa', 'farsi': 'fa'
            }
            nationality = self.context_info['nationality'].lower()
            for key, lang in nationality_to_lang.items():
                if key in nationality:
                    target_language = lang
                    break
        language_specific_leet = {
            'en': {'a': ['@', '4'], 'e': ['3', '&'], 'i': ['1', '!'], 'o': ['0', '*'], 's': ['$', '5']},
            'de': {'a': ['@', '4', 'ä'], 'e': ['3', '&'], 'i': ['1', '!'], 'o': ['0', '*'], 's': ['$', '5', 'ß']},
            'fr': {'a': ['@', '4', 'à', 'â'], 'e': ['3', '&', 'é', 'è', 'ê'], 'c': ['(', '©']},
            'es': {'a': ['@', '4', 'á'], 'e': ['3', '&', 'é'], 'o': ['0', '*', 'ó']},
            'fa': {'a': ['@', '4', 'آ', 'ا'], 'i': ['1', '!', 'ی'], 'o': ['0', '*', 'او']}
        }
        leet_mappings = language_specific_leet.get(target_language, 
                                                self.lang_data['leet_mappings'])
        birth_year = self.context_info.get('birth_year', '')
        if birth_year and len(birth_year) == 4:
            year_suffix = birth_year[2:]
            results.add(text + year_suffix)
            results.add(year_suffix + text)
            if len(year_suffix) == 2:
                results.add(text + year_suffix + '!')
                results.add(text + year_suffix + '@')
        email = self.context_info.get('email', '')
        if '@' in email:
            domain = email.split('@')[1].split('.')[0]
            if domain and len(domain) > 2:
                results.add(text + '@' + domain)
                results.add(domain + '@' + text)
        base_transformations = []
        for i, char in enumerate(text_lower):
            if char in leet_mappings:
                position_factor = 0.7 if 1 < i < len(text_lower) - 2 else 0.9
                if random.random() < position_factor:
                    for replacement in leet_mappings[char]:
                        new_text = text_lower[:i] + replacement + text_lower[i+1:]
                        base_transformations.append(new_text)
        if len(text) > 5:
            for _ in range(min(5, len(base_transformations))):
                if len(base_transformations) > 1:
                    base = random.choice(base_transformations)
                    for i, char in enumerate(base):
                        if char.isalpha() and char in leet_mappings and random.random() < 0.4:
                            for replacement in leet_mappings[char]:
                                new_text = base[:i] + replacement + base[i+1:]
                                base_transformations.append(new_text)
                                break
        special_chars = self.lang_data['special_chars']
        if target_language == 'fa':
            special_chars += ['_', 'ـ', '•']
        for char in special_chars[:3]:
            results.add(text + char)
            results.add(char + text)
            if len(text) > 4:
                results.add(text[:len(text)//2] + char + text[len(text)//2:])
        cultural_numbers = {
            'en': ['1', '7', '13', '21', '23', '69', '123', '2023', '2024'],
            'de': ['7', '13', '18', '42', '77', '88', '123', '2023', '2024'],
            'fr': ['7', '13', '17', '21', '42', '89', '123', '2023', '2024'],
            'es': ['7', '10', '13', '21', '99', '123', '2023', '2024'],
            'fa': ['5', '7', '14', '22', '88', '99', '110', '123', '2023', '2024']
        }
        numbers = cultural_numbers.get(target_language, ['1', '7', '13', '21', '99', '123'])
        for num in numbers:
            results.add(text + num)
            results.add(num + text)
            if len(text) > 4:
                results.add(text[:3] + num + text[3:])
        if len(text) > 3:
            results.add(text.capitalize())
            results.add(text.upper())
            results.add(text.lower())
            if len(text) > 5:
                camel_case = text[0].lower() + text[1].upper() + text[2:]
                results.add(camel_case)
        return list(set(results))[:15]
    def _generate_weighted_combinations(self, info, count, min_length, max_length):
        self.context_info = info
        weighted_elements = []
        behavioral_categories = {
            'high_emotional': ['pets', 'children', 'spouse', 'anniversary', 'favorite_things'],
            'medium_emotional': ['hobbies', 'sports', 'music', 'cars', 'food', 'books'],
            'low_emotional': ['job_title', 'employer', 'school', 'uni', 'location'],
            'temporal': ['birth_year', 'grad_year', 'grad_year_uni', 'favorite_numbers', 'current_year'],
            'cultural': ['cultural_events', 'zodiac', 'national_holidays']
        }
        for category, keys in behavioral_categories.items():
            for key in keys:
                if key in info:
                    items = info[key]
                    if not isinstance(items, list):
                        items = [items]
                    for item in items:
                        if item and isinstance(item, str) and len(item) >= 2:
                            if category == 'high_emotional':
                                weight = 0.95
                            elif category == 'medium_emotional':
                                weight = 0.75
                            elif category == 'temporal':
                                weight = 0.8
                                if re.search(r'\d{4}', item):
                                    year = int(re.search(r'\d{4}', item).group())
                                    current_year = datetime.now().year
                                    weight = 0.9 - (current_year - year) * 0.05
                                    weight = max(0.4, weight)
                            elif category == 'cultural':
                                weight = 0.65
                                if 'nationality' in info:
                                    nat = info['nationality'].lower()
                                    if (self.language == 'fa' and ('iran' in nat or 'persia' in nat)) or \
                                       (self.language == 'de' and ('german' in nat or 'germany' in nat)) or \
                                       (self.language == 'fr' and ('french' in nat or 'france' in nat)) or \
                                       (self.language == 'es' and ('spanish' in nat or 'spain' in nat)):
                                        weight = 0.85
                            else:
                                weight = 0.5
                            length_factor = 0.5
                            if min_length <= len(item) <= max_length:
                                length_factor = 1.0
                            elif len(item) < min_length:
                                length_factor = 0.7
                            weight *= length_factor
                            weighted_elements.append((item, category, weight))
        weighted_elements.sort(key=lambda x: (
            x[2] * (1.0 if min_length <= len(x[0]) <= max_length else 0.7),
            -abs(len(x[0]) - (min_length + max_length) / 2)
        ), reverse=True)
        passwords = set()
        for item, category, weight in weighted_elements:
            if weight > 0.6:
                cultural_numbers = self._get_cultural_numbers(info)
                for num in cultural_numbers[:3]:
                    pwd = f"{item}{num}"
                    if min_length <= len(pwd) <= max_length:
                        passwords.add(pwd)
                    pwd = f"{num}{item}"
                    if min_length <= len(pwd) <= max_length:
                        passwords.add(pwd)
                for char in self.lang_data['special_chars'][:2]:
                    pwd = f"{item}{char}"
                    if min_length <= len(pwd) <= max_length:
                        passwords.add(pwd)
                    pwd = f"{char}{item}"
                    if min_length <= len(pwd) <= max_length:
                        passwords.add(pwd)
                    if len(item) > 4:
                        pwd = f"{item[:len(item)//2]}{char}{item[len(item)//2:]}"
                        if min_length <= len(pwd) <= max_length:
                            passwords.add(pwd)
        if len(weighted_elements) >= 2:
            high_weight_elements = [e for e in weighted_elements if e[2] > 0.75]
            emotional_items = [e for e in high_weight_elements if e[1] == 'high_emotional']
            temporal_items = [e for e in weighted_elements if e[1] == 'temporal' and e[2] > 0.5]
            for emo in emotional_items[:3]:
                for temp in temporal_items[:2]:
                    for sep in ['', '_', '@', '#', '!']:
                        pwd = f"{emo[0]}{sep}{temp[0]}"
                        if min_length <= len(pwd) <= max_length:
                            passwords.add(pwd)
                        pwd = f"{temp[0]}{sep}{emo[0]}"
                        if min_length <= len(pwd) <= max_length:
                            passwords.add(pwd)
            for i in range(min(3, len(emotional_items))):
                for j in range(min(3, len(emotional_items))):
                    if i != j:
                        for sep in ['', '_', '@']:
                            pwd = f"{emotional_items[i][0]}{sep}{emotional_items[j][0]}"
                            if min_length <= len(pwd) <= max_length:
                                passwords.add(pwd)
        if info.get('tech_savviness'):
            try:
                tech_level = int(info['tech_savviness'])
                if tech_level >= 7:
                    tech_terms = ['admin', 'root', 'sys', 'dev', 'prod', 'test', 'api', 'db', 'sql', 'http', 'www']
                    for term in tech_terms:
                        for num in ['123', '456', '789', '007', '2023', '2024']:
                            pwd = f"{term}{num}"
                            if min_length <= len(pwd) <= max_length:
                                passwords.add(pwd)
                        for special in self.lang_data['special_chars'][:2]:
                            pwd = f"{term}{special}{datetime.now().year % 100}"
                            if min_length <= len(pwd) <= max_length:
                                passwords.add(pwd)
                elif tech_level <= 3:
                    for item, category, weight in weighted_elements:
                        if weight > 0.5 and min_length <= len(item) <= max_length:
                            passwords.add(item)
        common_structures = [
            "{word}{num}", "{num}{word}", "{word}{special}{num}",
            "{word1}{word2}", "{word}{num}{special}", "{word}{special}{word}"
        ]
        for structure in common_structures:
            if "{word}" in structure:
                for item, category, weight in weighted_elements[:5]:
                    if weight > 0.6:
                        num = random.choice(self.lang_data['number_patterns'][:3])
                        special = random.choice(self.lang_data['special_chars'])
                        pwd = structure.format(word=item, num=num, special=special)
                        if min_length <= len(pwd) <= max_length:
                            passwords.add(pwd)
            elif "{word1}" in structure and len(weighted_elements) >= 2:
                for i in range(min(3, len(weighted_elements))):
                    for j in range(min(3, len(weighted_elements))):
                        if i != j:
                            item1 = weighted_elements[i][0]
                            item2 = weighted_elements[j][0]
                            num = random.choice(self.lang_data['number_patterns'][:2])
                            special = random.choice(self.lang_data['special_chars'])
                            pwd = structure.format(word1=item1, word2=item2, num=num, special=special)
                            if min_length <= len(pwd) <= max_length:
                                passwords.add(pwd)
        if info.get('leaked_passwords'):
            for pwd in info['leaked_passwords']:
                analysis = self.entropy_analyzer.analyze_password_patterns(pwd)
                if analysis['has_digit'] and analysis['has_special']:
                    new_num = random.choice(self.lang_data['number_patterns'][:3])
                    new_special = random.choice(self.lang_data['special_chars'])
                    new_pwd = re.sub(r'\d+', new_num, pwd)
                    new_pwd = re.sub(r'[^\w\s]', new_special, new_pwd)
                    if min_length <= len(new_pwd) <= max_length:
                        passwords.add(new_pwd)
        final_passwords = set()
        for pwd in passwords:
            leet_versions = self._apply_leet_transformations(pwd)
            for leet_pwd in leet_versions:
                if min_length <= len(leet_pwd) <= max_length:
                    final_passwords.add(leet_pwd)
        sorted_passwords = self._rank_passwords_by_probability(list(final_passwords), info, min_length, max_length)
        return sorted_passwords[:count]
    def _get_cultural_numbers(self, info):
        target_language = info.get('language', 'en')
        nationality = info.get('nationality', '').lower()
        cultural_numbers = {
            'en': {
                'usa': ['1776', '13', '7', '42', '21', '69', '123', '2023', '2024'],
                'uk': ['1066', '13', '7', '42', '18', '77', '123', '2023', '2024']
            },
            'de': {
                'germany': ['1989', '13', '7', '42', '18', '88', '123', '2023', '2024'],
                'austria': ['1918', '13', '7', '42', '19', '77', '123', '2023', '2024']
            },
            'fa': {
                'iran': ['1399', '5', '7', '14', '22', '88', '99', '110', '313', '123', '2023', '2024'],
                'afghanistan': ['1399', '7', '14', '22', '88', '99', '123', '2023', '2024']
            },
            'fr': {
                'france': ['1789', '13', '7', '42', '14', '75', '89', '123', '2023', '2024']
            },
            'es': {
                'spain': ['1492', '7', '10', '13', '21', '99', '123', '2023', '2024']
            }
        }
        if nationality in cultural_numbers.get(target_language, {}):
            return cultural_numbers[target_language][nationality]
        default_numbers = {
            'en': ['1', '7', '13', '21', '23', '69', '123', '2023', '2024'],
            'de': ['7', '13', '18', '42', '77', '88', '123', '2023', '2024'],
            'fa': ['5', '7', '14', '22', '88', '99', '110', '123', '2023', '2024'],
            'fr': ['7', '13', '14', '17', '42', '75', '89', '123', '2023', '2024'],
            'es': ['7', '10', '13', '21', '99', '123', '2023', '2024']
        }
        return default_numbers.get(target_language, ['1', '7', '12', '13', '21', '99', '123', '2023', '2024'])
    def _rank_passwords_by_probability(self, passwords, info, min_length, max_length):
        ranked = []
        for pwd in passwords:
            score = 0.0
            length = len(pwd)
            if min_length <= length <= max_length:
                score += 0.4
            elif min_length - 2 <= length <= max_length + 2:
                score += 0.2
            else:
                score -= 0.2
            analysis = self.entropy_analyzer.analyze_password_patterns(pwd)
            if analysis['has_upper'] and analysis['has_lower'] and analysis['has_digit']:
                score += 0.2
            if analysis['has_special']:
                score += 0.1
            high_importance = ['pets', 'children', 'spouse', 'birth_year', 'anniversary']
            for key in high_importance:
                if key in info and info[key]:
                    values = info[key] if isinstance(info[key], list) else [info[key]]
                    for val in values:
                        if val and val.lower() in pwd.lower():
                            score += 0.5
                            break
            for event in self.lang_data['cultural_events']:
                if event.lower() in pwd.lower():
                    score += 0.2
            if analysis['repeated_chars']:
                score *= 0.6
            if len(analysis['keyboard_patterns']) > 1:
                score *= 0.5
            behavior_predictor = UserBehaviorPredictor(info)
            behavior_profile = behavior_predictor.predict_password_patterns()
            for structure in behavior_profile['structure']:
                pattern = structure.replace('{name}', '[a-zA-Z]+')
                pattern = pattern.replace('{pet}', '[a-zA-Z]+')
                pattern = pattern.replace('{child}', '[a-zA-Z]+')
                pattern = pattern.replace('{spouse}', '[a-zA-Z]+')
                pattern = pattern.replace('{birth_year}', '\d{4}')
                pattern = pattern.replace('{number}', '\d+')
                pattern = pattern.replace('{special}', '[!@#$%^&*()_+\-=\[\]{};\'\\:"|,.<>\/?]')
                if re.match(f"^{pattern}$", pwd):
                    score += 0.3
            ranked.append((pwd, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in ranked]
    def _generate_cultural(self, info, count, min_length, max_length):
        passwords = set()
        lang_data = self.lang_data
        relevant_events = []
        relevant_events.extend(lang_data['cultural_events'])
        nationality = info.get('nationality', '').lower()
        if 'iran' in nationality or 'persian' in nationality:
            relevant_events.extend(['Nowruz', 'Tasua', 'Ashura', 'Yalda', 'Sizdah Bedar'])
        elif 'german' in nationality or 'germany' in nationality:
            relevant_events.extend(['Oktoberfest', 'Weihnachten', 'Karneval', 'Silvester'])
        elif 'french' in nationality or 'france' in nationality:
            relevant_events.extend(['Bastille Day', 'Tour de France', 'La Fête de la Musique'])
        elif 'spanish' in nationality or 'spain' in nationality:
            relevant_events.extend(['La Tomatina', 'San Fermin', 'Dia de los Muertos'])
        relevant_events = list(set(relevant_events))
        for event in relevant_events[:5]:
            for year_type in ['current_year', 'birth_year', 'common_years']:
                years = []
                if year_type == 'current_year':
                    years = [str(datetime.now().year), str(datetime.now().year)[-2:]]
                elif year_type == 'birth_year' and info.get('birth_year'):
                    years = [info['birth_year'], info['birth_year'][-2:]]
                else:
                    years = ['2023', '2024', '23', '24', '00', '01', '99']
                for year in years:
                    structures = [
                        '{event}{year}',
                        '{year}{event}',
                        '{event}_{year}',
                        '{event}#{year}',
                        '{event}@{year}',
                        '{event}{special}{year}'
                    ]
                    for structure in structures:
                        pwd = structure.format(event=event, year=year, special=random.choice(lang_data['special_chars']))
                        if min_length <= len(pwd) <= max_length:
                            passwords.add(pwd)
            for num_pattern in lang_data['number_patterns'][:3]:
                pwd = f"{event}{num_pattern}"
                if min_length <= len(pwd) <= max_length:
                    passwords.add(pwd)
                pwd = f"{num_pattern}{event}"
                if min_length <= len(pwd) <= max_length:
                    passwords.add(pwd)
        if info.get('zodiac'):
            zodiac = info['zodiac']
            for num_pattern in lang_data['number_patterns'][:2]:
                pwd = f"{zodiac}{num_pattern}"
                if min_length <= len(pwd) <= max_length:
                    passwords.add(pwd)
                pwd = f"{num_pattern}{zodiac}"
                if min_length <= len(pwd) <= max_length:
                    passwords.add(pwd)
        cultural_numbers = self._get_cultural_numbers(info)
        for event in relevant_events[:3]:
            for num in cultural_numbers[:3]:
                pwd = f"{event}{num}"
                if min_length <= len(pwd) <= max_length:
                    passwords.add(pwd)
        return list(passwords)[:count]
    def _predict_password_patterns(self, info):
        behavior_predictor = UserBehaviorPredictor(info)
        return behavior_predictor.predict_password_patterns()
    def _generate_behavioral(self, info, count, min_length, max_length):
        passwords = set()
        behavior_profile = self._predict_password_patterns(info)
        for structure in behavior_profile['structure'][:3]:
            required_elements = []
            if '{name}' in structure:
                if info.get('first_name'):
                    required_elements.append(info['first_name'])
                elif info.get('nickname'):
                    required_elements.append(info['nickname'])
            if '{pet}' in structure and info.get('pets'):
                required_elements.extend(info['pets'][:3])
            if '{child}' in structure and info.get('children'):
                required_elements.extend(info['children'][:2])
            if '{spouse}' in structure and info.get('spouse'):
                required_elements.append(info['spouse'])
            if '{birth_year}' in structure and info.get('birth_year'):
                required_elements.append(info['birth_year'])
            if '{birth_date}' in structure and info.get('birth_day') and info.get('birth_month'):
                required_elements.append(f"{info['birth_day']}{info['birth_month']}")
            if '{cultural_event}' in structure:
                required_elements.extend(self.lang_data['cultural_events'][:2])
            if '{zodiac}' in structure and info.get('zodiac'):
                required_elements.append(info['zodiac'])
            if required_elements:
                for elem in required_elements[:3]:
                    for num_type in ['favorite_numbers', 'common_numbers', 'birth_year']:
                        numbers = []
                        if num_type == 'favorite_numbers' and info.get('favorite_numbers'):
                            numbers = info['favorite_numbers'][:2]
                        elif num_type == 'birth_year' and info.get('birth_year'):
                            numbers = [info['birth_year']]
                        else:
                            numbers = self.lang_data['number_patterns'][:3]
                        for num in numbers:
                            pwd = structure
                            if '{name}' in pwd:
                                pwd = pwd.replace('{name}', elem, 1)
                            if '{pet}' in pwd:
                                pwd = pwd.replace('{pet}', elem, 1)
                            if '{child}' in pwd:
                                pwd = pwd.replace('{child}', elem, 1)
                            if '{spouse}' in pwd:
                                pwd = pwd.replace('{spouse}', elem, 1)
                            if '{cultural_event}' in pwd:
                                pwd = pwd.replace('{cultural_event}', elem, 1)
                            if '{zodiac}' in pwd:
                                pwd = pwd.replace('{zodiac}', elem, 1)
                            if '{number}' in pwd:
                                pwd = pwd.replace('{number}', num, 1)
                            if '{special}' in pwd:
                                pwd = pwd.replace('{special}', random.choice(self.lang_data['special_chars']), 1)
                            if min_length <= len(pwd) <= max_length:
                                passwords.add(pwd)
        for pwd in list(passwords):
            if 'leet_speak' in behavior_profile['transformations']:
                leet_versions = self._apply_leet_transformations(pwd)
                for leet_pwd in leet_versions:
                    if min_length <= len(leet_pwd) <= max_length:
                        passwords.add(leet_pwd)
            if 'camel_case' in behavior_profile['transformations']:
                camel_case = pwd[0].lower() + pwd[1:].capitalize()
                if min_length <= len(camel_case) <= max_length:
                    passwords.add(camel_case)
            if 'random_caps' in behavior_profile['transformations']:
                random_caps = ''.join(c.upper() if random.random() > 0.7 else c for c in pwd)
                if min_length <= len(random_caps) <= max_length:
                    passwords.add(random_caps)
        if 'cultural_events' in behavior_profile['common_elements']:
            for event in self.lang_data['cultural_events'][:3]:
                for num in self.lang_data['number_patterns'][:2]:
                    pwd = f"{event}{num}"
                    if min_length <= len(pwd) <= max_length:
                        passwords.add(pwd)
                    pwd = f"{num}{event}"
                    if min_length <= len(pwd) <= max_length:
                        passwords.add(pwd)
        if 'pets' in behavior_profile['common_elements'] and info.get('pets'):
            for pet in info['pets'][:3]:
                for num in ['01', '02', '03', '123', '2023', '2024']:
                    pwd = f"{pet}{num}"
                    if min_length <= len(pwd) <= max_length:
                        passwords.add(pwd)
        if 'children' in behavior_profile['common_elements'] and info.get('children'):
            for child in info['children'][:2]:
                for year in [info.get('birth_year', '')[-2:], '01', '2023']:
                    if year:
                        pwd = f"{child}{year}"
                        if min_length <= len(pwd) <= max_length:
                            passwords.add(pwd)
        return list(passwords)[:count]
    def _filter_and_rank_passwords(self, passwords, info, count, min_length, max_length):
        valid_passwords = []
        for pwd in passwords:
            is_valid, _ = self._validate_password(pwd, info, min_length=min_length, max_length=max_length)
            if is_valid:
                valid_passwords.append(pwd)
        ranked_passwords = []
        for pwd in valid_passwords:
            probability_score = self._calculate_password_probability(pwd, info, min_length, max_length)
            ranked_passwords.append((pwd, probability_score))
        ranked_passwords.sort(key=lambda x: x[1], reverse=True)
        return [pwd for pwd, score in ranked_passwords[:count]]
    def _calculate_password_probability(self, password, info, min_length, max_length):
        score = 0.0
        length = len(password)
        if min_length <= length <= max_length:
            score += 0.3
        elif min_length - 2 <= length <= max_length + 2:
            score += 0.1
        else:
            score -= 0.2
        high_value_elements = ['pets', 'children', 'spouse', 'birth_year', 'anniversary']
        for element in high_value_elements:
            if element in info and info[element]:
                values = info[element] if isinstance(info[element], list) else [info[element]]
                for val in values:
                    if val and val.lower() in password.lower():
                        score += 0.9
                        break
        common_structures = [
            r'^[a-zA-Z]+\d+$',
            r'^\d+[a-zA-Z]+$',
            r'^[a-zA-Z]+[!@#$%^&*()_+\-=\[\]{};\'\\:"|,.<>\/?]\d+$',
            r'^[a-zA-Z]+\d+[!@#$%^&*()_+\-=\[\]{};\'\\:"|,.<>\/?]$'
        ]
        for pattern in common_structures:
            if re.match(pattern, password):
                score += 0.3
                break
        if (any(c.islower() for c in password) and any(c.isupper() for c in password)):
            score += 0.1
        if any(c.isdigit() for c in password):
            score += 0.1
        if any(c in string.punctuation for c in password):
            score += 0.1
        if re.search(r'(.)\1{2,}', password):
            score *= 0.5
        behavior_predictor = UserBehaviorPredictor(info)
        behavior_profile = behavior_predictor.predict_password_patterns()
        for structure in behavior_profile['structure']:
            pattern = structure.replace('{name}', '[a-zA-Z]+')
            pattern = pattern.replace('{pet}', '[a-zA-Z]+')
            pattern = pattern.replace('{birth_year}', '\d{4}')
            pattern = pattern.replace('{number}', '\d+')
            pattern = pattern.replace('{special}', '[!@#$%^&*()_+\-=\[\]{};\'\\:"|,.<>\/?]')
            if re.match(f"^{pattern}$", password):
                score += 0.5
        return min(1.0, score)
    def _validate_password(self, password, info, min_length=8, max_length=64, entropy_threshold=45):
        if not password or len(password) < min_length or len(password) > max_length:
            return False, 0
        behavior_predictor = UserBehaviorPredictor(info)
        behavior_profile = behavior_predictor.get_password_generation_weights()
        if behavior_profile['security_awareness'] < 0.4:
            entropy_threshold = 35
        elif behavior_profile['security_awareness'] > 0.7:
            entropy_threshold = 55
        has_upper = any(c in string.ascii_uppercase for c in password)
        has_lower = any(c in string.ascii_lowercase for c in password)
        has_digit = any(c in string.digits for c in password)
        has_special = any(c in string.punctuation for c in password)
        required_types = 3
        if behavior_profile['security_awareness'] < 0.4:
            required_types = 2
        elif behavior_profile['security_awareness'] > 0.7:
            required_types = 4
        if sum([has_upper, has_lower, has_digit, has_special]) < required_types:
            return False, 0
        common_patterns = [
            r'(?:password|pass|1234|qwerty|admin|login|welcome|123456|111111|iloveyou)',
            r'(\d{4})\1',
            r'(.)\1{2,}',
            r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|uvw|vwx|wxy|xyz)'
        ]
        for pattern in common_patterns:
            if re.search(pattern, password.lower()):
                is_personal = False
                personal_elements = []
                for key in ['pets', 'children', 'spouse', 'birth_year', 'favorite_numbers', 'anniversary']:
                    if info.get(key):
                        if isinstance(info[key], str):
                            personal_elements.append(info[key])
                        elif isinstance(info[key], list):
                            personal_elements.extend(info[key])
                for elem in personal_elements:
                    if elem and elem.lower() in password.lower():
                        is_personal = True
                        break
                if not is_personal:
                    return False, 0
        for word in self.entropy_analyzer.dictionary_words:
            if len(word) > 4 and word in password.lower():
                is_personal = False
                for key, value in info.items():
                    if isinstance(value, str) and value.lower() == word:
                        is_personal = True
                        break
                    elif isinstance(value, list):
                        if any(v.lower() == word for v in value if isinstance(v, str)):
                            is_personal = True
                            break
                cultural_data = [
                    *self.lang_data['cultural_events'],
                    *self.lang_data['zodiac_signs'],
                    *self.lang_data['sports_teams'],
                    *self.lang_data['universities']
                ]
                if any(word in item.lower() for item in cultural_data):
                    is_personal = True
                if not is_personal:
                    return False, 0
        entropy = self.entropy_analyzer.calculate_entropy(password)
        if entropy < entropy_threshold:
            return False, 0
        if behavior_profile['tech_savviness'] > 0.7:
            tech_terms = ['admin', 'root', 'sys', 'dev', 'prod', 'test', 'api', 'db', 'sql', 'http']
            has_tech_term = any(term in password.lower() for term in tech_terms)
            if not has_tech_term:
                if entropy < entropy_threshold + 10:
                    return False, 0
        return True, entropy
    def generate_with_context(self, info, count=50, min_length=8, max_length=12, strategy='comprehensive'):
        behavior_predictor = UserBehaviorPredictor(info)
        behavior_profile = behavior_predictor.predict_password_patterns()
        behavior_weights = behavior_predictor.get_password_generation_weights()
        self.context_weights = behavior_weights
        if behavior_profile['structure'] and strategy == 'comprehensive':
            if any('random' in s for s in behavior_profile['structure']):
                strategy = 'behavioral'
            elif any('emotional' in s for s in behavior_profile['structure']):
                strategy = 'smart'
        if strategy == 'comprehensive' or strategy == 'smart':
            behavioral_passwords = self._generate_behavioral(info, count//2, min_length, max_length)
            cultural_passwords = self._generate_cultural(info, count//3, min_length, max_length)
            basic_passwords = self._generate_weighted_combinations(info, count//6, min_length, max_length)
            all_passwords = behavioral_passwords + cultural_passwords + basic_passwords
            return self._filter_and_rank_passwords(all_passwords, info, count, min_length, max_length)
        elif strategy == 'behavioral':
            return self._generate_behavioral(info, count, min_length, max_length)
        elif strategy == 'cultural':
            return self._generate_cultural(info, count, min_length, max_length)
        else:
            return self._generate_weighted_combinations(info, count, min_length, max_length)
def get_extended_user_info():
    print("\n" + "="*60)
    print("🔐 ENTER DETAILED TARGET INFORMATION (PRESS ENTER TO SKIP FIELDS)")
    print("="*60)
    info = {'language': 'en'}
    print("\n👤 PERSONAL INFORMATION")
    info.update({
        'first_name': input("First Name: "),
        'middle_name': input("Middle Name: "),
        'last_name': input("Last Name: "),
        'nickname': input("Nickname: "),
        'maiden_name': input("Mother's Maiden Name (type 'non' if unknown): "),
        'gender': input("Gender: "),
        'language': 'en',
        'nationality': input("Nationality: "),
        'passport': input("Passport Number: "),
        'national_id': input("National ID Number: "),
    })
    print("\n📅 DATES & AGE")
    info.update({
        'birthdate': input("Birthdate (YYYY-MM-DD): "),
        'birth_year': input("Birth Year: "),
        'birth_month': input("Birth Month: "),
        'birth_day': input("Birth Day: "),
        'zodiac': input("Zodiac Sign: "),
        'age': input("Age: "),
        'deceased': input("Is the person deceased? (Y/N): ").lower() in ['y', 'yes'],
        'death_date': input("Date of Death (if applicable): "),
        'birth_time': input("Birth Time (HH:MM): "),
    })
    print("\n📍 LOCATION DETAILS")
    info.update({
        'city': input("City: "),
        'state': input("State/Province: "),
        'country': input("Country: "),
        'zipcode': input("ZIP Code: "),
        'street': input("Street Address: "),
        'house_num': input("House Number: "),
        'neighborhood': input("Neighborhood: "),
        'travel_destinations': input("Frequent Travel Destinations (space separated): ").split(),
    })
    print("\n📞 CONTACT INFORMATION")
    info.update({
        'phone': input("Primary Phone: "),
        'mobile': input("Mobile Number: "),
        'home_phone': input("Home Phone: "),
        'work_phone': input("Work Phone: "),
        'email': input("Primary Email: "),
        'alt_email': input("Alternative Email: "),
        'social_media': input("Social Media Handles (space separated): ").split(),
    })
    print("\n💻 DIGITAL FOOTPRINT")
    info.update({
        'username': input("Primary Username: "),
        'prev_usernames': input("Previous Usernames (space separated): ").split(),
        'gaming_ids': input("Gaming IDs (space separated): ").split(),
        'crypto_wallets': input("Crypto Wallet Addresses (space separated): ").split(),
        'websites': input("Owned Websites (space separated): ").split(),
        'device_models': input("Device Models (space separated): ").split(),
        'wifi_ssids': input("Common WiFi SSIDs (space separated): ").split(),
        'social_media_platforms': input("Social Media Platforms Used (space separated): ").split(),
        'online_gaming_platforms': input("Gaming Platforms Used (space separated): ").split(),
    })
    print("\n🎓 EDUCATION & EMPLOYMENT")
    info.update({
        'school': input("High School: "),
        'uni': input("University: "),
        'grad_year': input("High School Graduation Year: "),
        'major': input("University Major: "),
        'grad_year_uni': input("University Graduation Year: "),
        'job_title': input("Job Title: "),
        'employer': input("Employer: "),
        'employee_id': input("Employee ID: "),
    })
    print("\n👨‍👩‍👧‍👦 FAMILY MEMBERS")
    info.update({
        'spouse': input("Spouse's Name: "),
        'spouse_birth': input("Spouse's Birth Year: "),
        'children': input("Children's Names (space separated): ").split(),
        'children_births': input("Children's Birth Years (space separated): ").split(),
        'parents': input("Parents' Names (space separated): ").split(),
        'in_laws': input("In-Laws' Names (space separated): ").split(),
        'cousins': input("Close Cousins' Names (space separated): ").split(),
        'relatives': input("Close Relatives' Names (space separated): ").split(),
    })
    print("\n❤️ INTERESTS & PREFERENCES")
    info.update({
        'hobbies': input("Hobbies (space separated): ").split(),
        'sports': input("Sports Teams (space separated): ").split(),
        'music': input("Favorite Bands/Artists (space separated): ").split(),
        'movies': input("Favorite Movies (space separated): ").split(),
        'tv_shows': input("Favorite TV Shows (space separated): ").split(),
        'books': input("Favorite Books (space separated): ").split(),
        'colors': input("Favorite Colors (space separated): ").split(),
        'pets': input("Pet Names (space separated): ").split(),
        'pet_types': input("Pet Species/Breeds (space separated): ").split(),
        'cars': input("Car Models (space separated): ").split(),
        'car_plates': input("Car License Plates (space separated): ").split(),
        'brands': input("Favorite Brands (space separated): ").split(),
        'foods': input("Favorite Foods (space separated): ").split(),
        'restaurants': input("Frequent Restaurants (space separated): ").split(),
        'authors': input("Favorite Authors (space separated): ").split(),
        'actors': input("Favorite Actors/Actresses (space separated): ").split(),
        'genres': input("Favorite Music/Movie Genres (space separated): ").split(),
    })
    print("\n🔒 SECURITY INFORMATION")
    info.update({
        'common_passwords': input("Known Common Passwords (space separated): ").split(),
        'leaked_passwords': input("Previously Leaked Passwords (space separated): ").split(),
        'data_breaches': input("Involved Data Breaches (space separated): ").split(),
        'security_questions': input("Security Questions Used (space separated): ").split(),
        'security_answers': input("Security Question Answers (space separated): ").split(),
        'password_patterns': input("Common Password Patterns (space separated): ").split(),
        'special_chars': input("Common Special Characters (space separated): ").split(),
        'number_patterns': input("Common Number Patterns (space separated): ").split(),
        'favorite_numbers': input("Favorite/Lucky Numbers (space separated): ").split(),
    })
    print("\n🧠 BEHAVIORAL PATTERNS")
    info.update({
        'leet_transforms': input("Common Leet Transformations (e.g., '4=a', '3=e') (space separated): ").split(),
        'keyboard_walks': input("Common Keyboard Walks (space separated): ").split(),
        'catchphrases': input("Frequently Used Catchphrases (space separated): ").split(),
        'quotes': input("Favorite Quotes (space separated): ").split(),
        'online_behaviors': input("Notable Online Behaviors (space separated): ").split(),
    })
    print("\n🔍 ADVANCED PROFILING")
    info.update({
        'political_views': input("Political Views (space separated): ").split(),
        'religious_views': input("Religious Views (space separated): ").split(),
        'ethnicity': input("Ethnicity: "),
        'education_level': input("Education Level: "),
        'income_range': input("Income Range: "),
        'relationship_status': input("Relationship Status: "),
        'occupation': input("Occupation: "),
        'industry': input("Industry: "),
        'tech_savviness': input("Tech Savviness (1-10): "),
        'password_change_frequency': input("Password Change Frequency (monthly): "),
    })
    return info
def save_passwords(passwords, info, filename="advanced_passwords.txt", min_length=8, max_length=12):
    if not passwords:
        print("❌ No valid passwords generated")
        return
    entropy_analyzer = PasswordEntropyAnalyzer(info.get('language', 'en'))
    entropy_scores = [entropy_analyzer.calculate_entropy(p) for p in passwords]
    avg_entropy = sum(entropy_scores) / len(entropy_scores)
    password_analysis = []
    for pwd in passwords:
        analysis = entropy_analyzer.analyze_password_patterns(pwd)
        password_analysis.append({
            'password': pwd,
            'length': analysis['length'],
            'entropy': analysis['entropy'],
            'complexity_score': min(10, round(analysis['entropy']/10)),
            'pattern_types': []
        })
        if analysis['repeated_chars']:
            password_analysis[-1]['pattern_types'].append('repeated_chars')
        if analysis['keyboard_patterns']:
            password_analysis[-1]['pattern_types'].append('keyboard_walk')
        if analysis['common_words']:
            password_analysis[-1]['pattern_types'].append('dictionary_word')
        if analysis['cultural_patterns']:
            password_analysis[-1]['pattern_types'].append('cultural_pattern')
    password_analysis.sort(key=lambda x: x['entropy'], reverse=True)
    output_dir = os.path.dirname(os.path.abspath(filename))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# ================================================\n")
        f.write("# Ethically Generated Password List\n")
        f.write("# ================================================\n")
        f.write("# Generated with advanced algorithmic pattern analysis\n")
        f.write("# For authorized security testing and educational purposes only\n")
        f.write("# ================================================\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Language: {LANGUAGE_DATA.get(info.get('language', 'en'), LANGUAGE_DATA['en'])['name']}\n")
        f.write(f"# Length Constraints: {min_length}-{max_length} characters\n")
        f.write(f"# Total Passwords: {len(passwords)}\n")
        f.write(f"# Average Entropy: {avg_entropy:.2f}\n")
        f.write(f"# Security Level: {'High' if avg_entropy > 60 else 'Medium' if avg_entropy > 45 else 'Low'}\n")
        f.write("# ================================================\n")
        f.write("# Target Profile Summary:\n")
        f.write(f"# - Name: {info.get('first_name', '')} {info.get('last_name', '')}\n")
        f.write(f"# - Location: {info.get('city', '')}, {info.get('country', '')}\n")
        f.write(f"# - Age: {info.get('age', 'Unknown')}\n")
        f.write("# ================================================\n")
        f.write("# Password Analysis:\n")
        f.write("# High Entropy (60+): Strong password patterns\n")
        f.write("# Medium Entropy (45-60): Moderate security\n")
        f.write("# Low Entropy (<45): Weak patterns, use with caution\n")
        f.write("# ================================================\n")
        f.write("# Passwords:\n")
        for i, analysis in enumerate(password_analysis, 1):
            pwd = analysis['password']
            entropy = analysis['entropy']
            complexity = analysis['complexity_score']
            patterns = ", ".join(analysis['pattern_types']) if analysis['pattern_types'] else "complex_pattern"
            f.write(f"{i:3}. {pwd}\n")
            f.write(f"     🔍 Entropy: {entropy:.2f} | Length: {len(pwd)} | Complexity: {complexity}/10\n")
            f.write(f"     📌 Pattern Type: {patterns}\n")
            f.write(f"     💡 Security Rating: {'⭐' * complexity}{'☆' * (10-complexity)}\n")
    print(f"\n✅ Saved {len(passwords)} advanced passwords to {filename}")
    print(f"📊 Average entropy: {avg_entropy:.2f}")
    print(f"📏 Length range: {min_length}-{max_length} characters")
    print(f"📈 Security level: {'High' if avg_entropy > 60 else 'Medium' if avg_entropy > 45 else 'Low'}")
    stats_file = filename.replace('.txt', '_stats.json')
    stats = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'language': info.get('language', 'en'),
            'length_constraints': {
                'min': min_length,
                'max': max_length
            },
            'target_summary': {
                'has_name': bool(info.get('first_name') and info.get('last_name')),
                'has_birthdate': bool(info.get('birthdate')),
                'has_email': bool(info.get('email')),
                'country': info.get('country', 'Unknown')
            },
            'password_count': len(passwords),
            'average_entropy': avg_entropy,
            'security_level': 'High' if avg_entropy > 60 else 'Medium' if avg_entropy > 45 else 'Low'
        },
        'password_analysis': password_analysis
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"📊 Detailed statistics saved to {stats_file}")
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""
        🔐 Advanced Password List Generator - Smart Algorithmic Edition
        Generates highly personalized password guesses based on detailed user profiles.
        Designed for ethical security testing and educational use only.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
        Example Usage:
          python advanced_password_generator.py -c 100 --min_p 8 --max_p 12 -o passwords.txt --strategy smart
          python advanced_password_generator.py -h
        """
    )
    parser.add_argument(
        '-c', '--count',
        type=int,
        default=50,
        help="Number of passwords to generate (default: 50)"
    )
    parser.add_argument(
        '--min_p',
        type=int,
        default=8,
        help="Minimum length of generated passwords (default: 8)"
    )
    parser.add_argument(
        '--max_p',
        type=int,
        default=12,
        help="Maximum length of generated passwords (default: 12)"
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='passwords.txt',
        help="Output file name for generated passwords (default: passwords.txt)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Seed value for reproducibility (default: 42)"
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='smart',
        choices=['basic', 'cultural', 'behavioral', 'comprehensive', 'smart'],
        help="Password generation strategy:\n"
             "  basic         : Simple combinations of user info\n"
             "  cultural      : Focus on cultural and language patterns\n"
             "  behavioral    : Focus on typing behavior and habits\n"
             "  comprehensive : Balanced combination of approaches\n"
             "  smart         : Intelligent weighted combinations (default)"
    )
    parser.add_argument(
        '--ethical-verify',
        action='store_true',
        default=True,
        help="Enable ethical usage verification (default: True)"
    )
    parser.add_argument(
        '--no-ethical-verify',
        dest='ethical_verify',
        action='store_false',
        help="Disable ethical usage verification (not recommended)"
    )
    return parser.parse_args()
def main():
    print(r"""
    🔐 Advanced Password List Generator v5.1 (Smart Algorithmic Edition)
    🌐 Intelligent Algorithms • Ethical Safeguards • Comprehensive Profiling
    ⚠️  For authorized security testing ONLY with explicit permission
    """)
    ethical_guard = EthicalSafeguard()
    args = parse_arguments()
    random.seed(args.seed)
    if args.ethical_verify and not ethical_guard.verify_ethical_usage():
        print("❌ Ethical verification failed. Exiting...")
        return
    info = get_extended_user_info()
    print(f"\n🔄 Generating passwords using '{args.strategy}' strategy...")
    generator = ContextualPasswordGenerator(
        language=info.get('language', 'en')
    )
    passwords = generator.generate_with_context(
        info, 
        count=args.count, 
        min_length=args.min_p,
        max_length=args.max_p,
        strategy=args.strategy
    )
    if args.ethical_verify:
        ethical_guard.log_usage(len(passwords), info)
    save_passwords(passwords, info, args.output, args.min_p, args.max_p)
    print("\n" + "="*60)
    print(" PASSWORD GENERATION COMPLETE - SECURITY RECOMMENDATIONS")
    print("="*60)
    print("1. This password list is for authorized security testing ONLY")
    print("2. Always obtain explicit written permission before testing")
    print("3. Securely delete these passwords after your authorized test")
    print("4. Document all activities for your security report")
    print("5. Never use these passwords for unauthorized access")
    print("="*60)
    print("🔒 Remember: With great power comes great responsibility")
    print("="*60)
if __name__ == "__main__":
    main()
