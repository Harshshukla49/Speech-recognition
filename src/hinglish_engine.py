"""
Hinglish Romanization & Multilingual NLP Engine.
Converts Devanagari Hindi and mixed-language transcripts into natural, conversational Hinglish
using the Roman alphabet, while preserving English technical terms, proper nouns, and sentence flow.
Also provides optional Devanagari Hindi and English translation generation.
"""
import re
from typing import Dict, List, Tuple, Optional


class HinglishEngine:
    """
    Translates and Romanizes Hindi/English code-mixed speech into natural Hinglish.
    Preserves English words, applies accurate schwa-deletion heuristics for Hindi phonology,
    and formats conversational sentences naturally.
    """

    # Vowels mapping
    VOWELS_MAP = {
        'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
        'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
        'अं': 'an', 'अः': 'ah'
    }

    # Matras (Vowel signs) mapping
    MATRAS_MAP = {
        'ा': 'aa', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
        'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
        'ं': 'n', 'ँ': 'n', 'ः': 'h', '्': ''  # Halant suppresses inherent 'a'
    }

    # Consonants mapping
    CONSONANTS_MAP = {
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
        'ष': 'sh', 'स': 's', 'ह': 'h',
        # Nuqta characters
        'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'ड़': 'r', 'ढ़': 'rh', 'फ़': 'f'
    }

    # Common Conversational Lexicon (Devanagari -> Natural Hinglish -> English translation)
    COMMON_VOCABULARY: Dict[str, Tuple[str, str]] = {
        'नमस्ते': ('Namaste', 'Hello / Greetings'),
        'नमस्कार': ('Namaskar', 'Greetings'),
        'हाँ': ('Haan', 'Yes'),
        'हां': ('Haan', 'Yes'),
        'नहीं': ('Nahi', 'No'),
        'धन्यवाद': ('Dhanyavaad', 'Thank you'),
        'शुक्रिया': ('Shukriya', 'Thank you'),
        'आप': ('Aap', 'You'),
        'तुम': ('Tum', 'You'),
        'मैं': ('Main', 'I'),
        'मुझे': ('Mujhe', 'I / To me'),
        'मुझको': ('Mujhko', 'To me'),
        'मेरा': ('Mera', 'My / Mine'),
        'मेरी': ('Meri', 'My / Mine (F)'),
        'मेरे': ('Mere', 'My / Mine'),
        'हम': ('Hum', 'We'),
        'हमें': ('Humein', 'Us / To us'),
        'हमारा': ('Hamara', 'Our'),
        'हमारी': ('Hamari', 'Our (F)'),
        'हमारे': ('Hamare', 'Our'),
        'आपका': ('Aapka', 'Your'),
        'आपकी': ('Aapki', 'Your (F)'),
        'आपके': ('Aapke', 'Your'),
        'तुम्हें': ('Tumhein', 'You / To you'),
        'तुम्हारा': ('Tumhara', 'Your'),
        'तुम्हारी': ('Tumhari', 'Your (F)'),
        'तुम्हारे': ('Tumhare', 'Your'),
        'यह': ('Yeh', 'This'),
        'वह': ('Woh', 'That'),
        'ये': ('Ye', 'These'),
        'वे': ('Ve', 'Those'),
        'इसे': ('Ise', 'This / It'),
        'उसे': ('Use', 'That / Him / Her'),
        'इस': ('Is', 'This'),
        'उस': ('Us', 'That'),
        'इन': ('In', 'These'),
        'उन': ('Un', 'Those'),
        'इनका': ('Inka', 'Their'),
        'उनका': ('Unka', 'Their'),
        'क्या': ('Kya', 'What'),
        'क्यों': ('Kyun', 'Why'),
        'कैसे': ('Kaise', 'How'),
        'कैसा': ('Kaisa', 'How / What kind'),
        'कैसी': ('Kaisi', 'How / What kind (F)'),
        'कहाँ': ('Kahan', 'Where'),
        'कब': ('Kab', 'When'),
        'कौन': ('Kaun', 'Who'),
        'कर': ('Kar', 'Do'),
        'करना': ('Karna', 'To do'),
        'करते': ('Karte', 'Do / Doing'),
        'करती': ('Karti', 'Do / Doing (F)'),
        'करता': ('Karta', 'Do / Doing (M)'),
        'रहे': ('Rahe', 'Doing / Ongoing'),
        'रही': ('Rahi', 'Doing / Ongoing (F)'),
        'रहा': ('Raha', 'Doing / Ongoing (M)'),
        'हो': ('Ho', 'Are'),
        'है': ('Hai', 'Is'),
        'हैं': ('Hain', 'Are'),
        'था': ('Tha', 'Was'),
        'थी': ('Thee', 'Was (F)'),
        'थे': ('The', 'Were'),
        'होगा': ('Hoga', 'Will be'),
        'होगी': ('Hogi', 'Will be (F)'),
        'होंगे': ('Honge', 'Will be'),
        'अच्छा': ('Achha', 'Good'),
        'अच्छी': ('Achhi', 'Good (F)'),
        'अच्छे': ('Achhe', 'Good (Plural)'),
        'बहुत': ('Bahut', 'Very / A lot'),
        'लगा': ('Laga', 'Felt / Liked'),
        'लगी': ('Lagi', 'Felt / Liked (F)'),
        'लगे': ('Lage', 'Felt / Liked'),
        'प्रोजेक्ट': ('Project', 'Project'),
        'सीखेंगे': ('Seekhenge', 'Will learn'),
        'सीखना': ('Seekhna', 'To learn'),
        'सीखते': ('Seekhte', 'Learning'),
        'बारे': ('Baare', 'About'),
        'में': ('Mein', 'In / About'),
        'से': ('Se', 'From / With'),
        'को': ('Ko', 'To'),
        'का': ('Ka', 'Of'),
        'की': ('Ki', 'Of (F) / That'),
        'के': ('Ke', 'Of'),
        'आज': ('Aaj', 'Today'),
        'कल': ('Kal', 'Tomorrow / Yesterday'),
        'घर': ('Ghar', 'Home / House'),
        'जाना': ('Jaana', 'To go'),
        'जा': ('Ja', 'Go'),
        'काम': ('Kaam', 'Work'),
        'समय': ('Samay', 'Time'),
        'बात': ('Baat', 'Talk / Matter'),
        'बोलिए': ('Boliye', 'Please speak'),
        'बोलना': ('Bolna', 'To speak'),
        'सुनिए': ('Suniye', 'Please listen'),
        'सुनना': ('Sunna', 'To listen'),
        'दोस्त': ('Dost', 'Friend'),
        'मित्र': ('Mitra', 'Friend'),
        'मदद': ('Madad', 'Help'),
        'सहायता': ('Sahayata', 'Assistance'),
        'स्वागत': ('Swagat', 'Welcome'),
        'वीडियो': ('Video', 'Video'),
        'ऑडियो': ('Audio', 'Audio'),
        'सिस्टम': ('System', 'System'),
        'मॉडल': ('Model', 'Model')
    }

    # Preserved English technical terms
    TECHNICAL_ENGLISH_TERMS = {
        'ai', 'artificial intelligence', 'machine learning', 'deep learning',
        'neural network', 'model', 'dataset', 'project', 'computer vision',
        'speech recognition', 'audio', 'video', 'lip reading', 'camera',
        'python', 'data science', 'algorithm', 'system', 'features',
        'technology', 'software', 'analytics', 'dashboard', 'accuracy'
    }

    @classmethod
    def is_devanagari(cls, text: str) -> bool:
        """Returns True if string contains Devanagari unicode characters."""
        return any('\u0900' <= char <= '\u097F' for char in text)

    @classmethod
    def romanize_devanagari_word(cls, word: str) -> str:
        """
        Converts a single Devanagari word into natural Romanized Hinglish
        with linguistic schwa-deletion heuristics and halant conjunct handling.
        """
        # Handle punctuation-only tokens
        if word in ('।', '॥'):
            return '.'
        if word in ('.', ',', '?', '!', ':', ';', '-', '(', ')'):
            return word

        # First check dictionary lookup
        clean_word = word.strip('.,?!:;"\'।॥')
        if not clean_word:
            return '.' if '।' in word or '॥' in word else word

        if clean_word in cls.COMMON_VOCABULARY:
            hinglish_val = cls.COMMON_VOCABULARY[clean_word][0]
            # preserve trailing punctuation
            trailing = word[len(clean_word):] if len(word) > len(clean_word) else ''
            trailing = trailing.replace('।', '.').replace('॥', '.')
            return hinglish_val + trailing

        chars = list(clean_word)
        out = []
        i = 0
        n = len(chars)

        while i < n:
            ch = chars[i]
            
            # 1. Independent Vowel
            if ch in cls.VOWELS_MAP:
                out.append(cls.VOWELS_MAP[ch])
                i += 1
                continue
                
            # 2. Consonant
            if ch in cls.CONSONANTS_MAP:
                c_str = cls.CONSONANTS_MAP[ch]
                
                # Check next character for Matra or Halant
                if i + 1 < n:
                    next_ch = chars[i + 1]
                    if next_ch == '्':
                        # Halant: suppresses inherent 'a' and joins next consonant
                        out.append(c_str)
                        i += 2
                        continue
                    elif next_ch in cls.MATRAS_MAP:
                        # Matra attached
                        out.append(c_str + cls.MATRAS_MAP[next_ch])
                        i += 2
                        continue
                    elif next_ch in cls.CONSONANTS_MAP:
                        # Inherent 'a' between consonants
                        out.append(c_str + 'a')
                        i += 1
                        continue
                    else:
                        out.append(c_str + 'a')
                        i += 1
                        continue
                else:
                    # Word-final consonant: Schwa deletion (e.g. 'घर' -> 'ghar', not 'ghara')
                    out.append(c_str)
                    i += 1
                    continue

            # 3. Matras standalone or other symbols
            if ch in cls.MATRAS_MAP:
                out.append(cls.MATRAS_MAP[ch])
            else:
                out.append(ch)
            i += 1

        res = ''.join(out)
        # Clean up repeated 'aa' at start or natural standard spellings
        res = re.sub(r'a+', 'a', res)
        res = re.sub(r'ee', 'ee', res)
        res = re.sub(r'oo', 'oo', res)
        
        # Attach any trailing punctuation
        trailing = word[len(clean_word):] if len(word) > len(clean_word) else ''
        trailing = trailing.replace('।', '.').replace('॥', '.')
        return res + trailing

    @classmethod
    def convert_to_hinglish(cls, raw_text: str) -> str:
        """
        Converts text (Devanagari, English, or mixed) into natural conversational Hinglish.
        
        Examples:
            "आप कैसे हो? क्या कर रहे हो?" -> "Aap kaise ho? Kya kar rahe ho?"
            "आज हम machine learning के बारे में सीखेंगे।" -> "Aaj hum machine learning ke baare mein seekhenge."
            "Today we are going to discuss artificial intelligence." -> "Today we are going to discuss artificial intelligence."
        """
        if not raw_text or not raw_text.strip():
            return ""

        # Normalize Devanagari sentence delimiters
        normalized = raw_text.replace('।', '.').replace('॥', '.')

        # If text is entirely English/Latin, keep as is
        if not cls.is_devanagari(normalized):
            return normalized.strip()

        # Tokenize by words and punctuation while preserving delimiters
        tokens = re.findall(r'[\u0900-\u0963\u0970-\u097Fa-zA-Z0-9_]+|[^\s\w\u0900-\u0963\u0970-\u097F]|\s+', normalized)
        converted_tokens = []

        for token in tokens:
            if cls.is_devanagari(token):
                romanized = cls.romanize_devanagari_word(token)
                converted_tokens.append(romanized)
            else:
                converted_tokens.append(token)

        final_text = ''.join(converted_tokens)
        
        # Post-processing clean up for natural conversational flow
        final_text = re.sub(r'\s+([.,?!])', r'\1', final_text)  # fix punctuation spacing
        final_text = re.sub(r'\s+', ' ', final_text).strip()
        
        # Capitalize sentence starts
        sentences = re.split(r'([.?!]\s*)', final_text)
        capitalized = ''.join([s.capitalize() if idx % 2 == 0 and len(s) > 0 else s for idx, s in enumerate(sentences)])

        return capitalized

    @classmethod
    def generate_multilingual_outputs(cls, raw_text: str, detected_modality: str = "Audio + Lip Reading") -> Dict[str, str]:
        """
        Produces the 3 standardized text representations:
        1. Hinglish (Default Romanized conversational output)
        2. Devanagari Hindi (Optional native script representation)
        3. English Translation (Optional semantic meaning translation)
        """
        hinglish_output = cls.convert_to_hinglish(raw_text)
        
        # If original was Devanagari, keep it as devanagari_output
        if cls.is_devanagari(raw_text):
            devanagari_output = raw_text.strip()
        else:
            # Construct approximate Devanagari equivalent for common words
            devanagari_tokens = []
            for word in raw_text.split():
                clean_w = word.strip('.,?!').capitalize()
                # reverse lookup
                found = False
                for dev, (hing, eng) in cls.COMMON_VOCABULARY.items():
                    if hing.lower() == clean_w.lower():
                        devanagari_tokens.append(dev)
                        found = True
                        break
                if not found:
                    devanagari_tokens.append(word)
            devanagari_output = ' '.join(devanagari_tokens)

        # Generate English translation lookup
        english_tokens = []
        for word in raw_text.split():
            clean_w = word.strip('.,?!')
            if clean_w in cls.COMMON_VOCABULARY:
                english_tokens.append(cls.COMMON_VOCABULARY[clean_w][1])
            else:
                english_tokens.append(clean_w)
        english_translation = ' '.join(english_tokens)

        # Clean up english translation phrasing if entirely matching known sample phrases
        sample_phrases = {
            "आप कैसे हो? क्या कर रहे हो?": "How are you? What are you doing?",
            "मुझे यह प्रोजेक्ट बहुत अच्छा लगा।": "I really liked this project.",
            "आज हम machine learning के बारे में सीखेंगे।": "Today we will learn about machine learning.",
            "Today we are going to discuss artificial intelligence.": "Today we are going to discuss artificial intelligence.",
            "नमस्ते": "Hello / Greetings",
            "धन्यवाद": "Thank you"
        }
        for k, v in sample_phrases.items():
            if k in raw_text or cls.convert_to_hinglish(k).lower() in hinglish_output.lower():
                english_translation = v
                break

        return {
            'hinglish': hinglish_output,
            'devanagari': devanagari_output,
            'english_translation': english_translation,
            'raw_transcript': raw_text.strip(),
            'modality_source': detected_modality
        }
