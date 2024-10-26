from bs4 import BeautifulSoup
import unidecode
import re
from emot.emo_unicode import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
from flashtext import KeywordProcessor
from src.data.better_profanity import Profanity
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
from time import sleep

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

class TextToxicityCleaner:
    def __init__(self):
        self.data = None
        
        all_emoji_emoticons = {**EMOTICONS_EMO,**UNICODE_EMOJI_ALIAS, **UNICODE_EMOJI_ALIAS}
        all_emoji_emoticons = {k:v.replace(":","").replace("_"," ").strip() for k,v in all_emoji_emoticons.items()}
        self.kp_all_emoji_emoticons = KeywordProcessor()
        for k,v in all_emoji_emoticons.items():
            self.kp_all_emoji_emoticons.add_keyword(k, v)
        self.contrastion_clearner = KeywordProcessor()
        for k,v in CONTRACTION_MAP.items():
            self.contrastion_clearner.add_keyword(k, v)
        self.profanity=Profanity()
    def removeHTMLTags(self,text):
        '''
        Function to remove the HTML Tags from a given text.
        
        Parameter:
        ---------
        text: str
            Text from which the HTML tags has to be removed.
        '''
        
        # Reference: 'Remove html tags using BeautifulSoup' - https://www.geeksforgeeks.org/remove-all-style-scripts-and-html-tags-using-beautifulsoup/
        
        # Create a BeautifulSoup object to parse the given html text content
        soup = BeautifulSoup(text, 'html.parser')
        
        # Remove the <style> and <script> tags from the html content because they contains the styling sheet and javascript
        # file references and won't give any meaningful context.
        for data in soup(['style', 'script']):
            
            # Remove tag
            data.decompose()
            
        # Return the html tag free content
        return ' '.join(soup.stripped_strings)
    
    def removeAccentedChars(self,text):
        '''
        Function to remove the accented characters from a given text.
        
        Parameter:
        ---------
        text: str
            Text from which the accented character has to be removed.
        '''
        
        # Reference: "remove accented characters python" - https://www.geeksforgeeks.org/how-to-remove-string-accents-using-python-3/
        
        # Remove accents
        return unidecode.unidecode(text)
    def removeIPLinkNum(self,text, ipAddress=True, hyperlink=True, numbers=False):
        '''
        Function to remove IP Address and Number from the given text.
        
        Parameter:
        ---------
        text: str
            Text from which IP Address and number(s) have to be removed.
        '''
        
        # Replace IP Address with empty string.
        # Reference: 'Remove IP Address Python' - https://www.geeksforgeeks.org/extract-ip-address-from-file-using-python/#:~:text=The%20regular%20expression%20for%20valid,%5C.)%7B
        if ipAddress == True:
            
            text = re.sub(r'((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', '', text)
        
        # Remove hyperlinks
        # Reference: 'Regex for hperlinks Python' - https://www.geeksforgeeks.org/python-check-url-string/
        if hyperlink == True:
            
            text = re.sub(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", "", text)
        
        # Remove numbers.
        if numbers == True:
            
            text = re.sub(r'[0-9]', '', text)
        
        # Remove the extra space if any.
        text = re.sub(r'[ ][ ]+', ' ', text)
        
        return text



    def clean_emotes(self,text):
        # Replace emoji and emoticons in the text
        text = self.kp_all_emoji_emoticons.replace_keywords(text)
        return text
    
    def decontract(self,text):
        '''
        Function to decontract a given text.
        
        Parameter:
        ---------
        text: str
            Text to be decontracted.
        '''
        # Replace emoji and emoticons in the text
        text = self.contrastion_clearner.replace_keywords(text)
            # Replace the contracted word with its decontracted form.
        return text
    def processSpecialTokens(self,text):
        '''
        Function to add one space around sentence end markers and remove duplicates.
        
        Parameter:
        ---------
        text: str
            Text in which space has to be added around sentence end tokens.
        '''
            
        text = re.sub(r'[!]+[ ]*[!]*', ' ! ', text) # Add space around ! with exclmrk.
        text = re.sub(r'[?]+[ ]*[?]*', ' ? ', text) # Replace ? with qstmrk.
        text = re.sub(r'[.]+[ ]*[.]*', ' . ', text) # Replace . with eosmkr.

        # Remove the extra space if any.
        text = re.sub(r'[ ][ ]+', ' ', text)
        
        return text
    def clean_swear_words(self,text):
        '''
        Function to clean the swear words from the given text.
        
        Parameter:
        ---------
        text: str
            Text from which the swear words has to be removed.
        '''
        
        # Clean the swear words from the text
        return self.profanity.censor(text)
    
    def process_chunk_clean_swear_words(self,chunk):
        # Replace this with your actual processing logic
        chunk['comment_text'] = chunk['comment_text'].apply(lambda x: self.clean_swear_words(x))
        # print(f"Processed chunk with size {len(chunk)}")
        return chunk
    def process_chunk_clean_clean_emotes(self,chunk):
        # Replace this with your actual processing logic
        chunk['comment_text'] = chunk['comment_text'].apply(lambda x: self.clean_emotes(x))
        # print(f"Processed chunk with size {len(chunk)}")
        return chunk

    def chunkify(self,df, num_chunks):
        chunk_size = len(df) // num_chunks
        return [df[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

    def parallel_process_dataframe_with_progress_multi_function(self,df, num_processes, processing_func):
        with mp.Pool(processes=num_processes) as pool:
            chunks = self.chunkify(df, num_processes)
            with tqdm(total=len(chunks)) as pbar:
                processed_chunks = []
                for result in pool.imap_unordered(self.process_chunk_wrapper, [(chunk, processing_func) for chunk in chunks]):
                    processed_chunks.append(result)
                    pbar.update(1)
            
        result_df = pd.concat(processed_chunks)
        result_df.sort_index(inplace=True)
        result_df.dropna(subset=['comment_text'],inplace=True)
        result_df.reset_index(drop=True,inplace=True)
        return result_df

    def process_chunk_wrapper(self,args):
        chunk, processing_func = args
        return self.process_chunk(chunk, processing_func)

    def process_chunk(self,chunk, processing_func):
        return processing_func(chunk)
    
    def clean(self,df, text_column,sleep_time=8):
        '''
        Function to do a general cleaning.
        
        Parameter:
        ---------
        df: pd.DataFrame
            Dataframe containing the text data.
        text_column: str
            Column name containing the text data.
        sleep_time: int
            Time to sleep for not having an overload.
        '''
        # Example usage
        num_processes = mp.cpu_count()
        if num_processes > 1:
            num_processes -= 1
        df = self.parallel_process_dataframe_with_progress_multi_function(df, num_processes, self.process_chunk_clean_swear_words)
        # df = parallel_process_dataframe_with_progress(df, num_processes)
        df[text_column]= df[text_column].swifter.apply(lambda x: self.removeHTMLTags(x))
        df[text_column]= df[text_column].swifter.apply(lambda x: self.removeAccentedChars(x))
        df[text_column]= df[text_column].swifter.apply(lambda x: self.removeIPLinkNum(x))
        sleep(sleep_time) # Sleep for avoiding cpu overloading
        df = self.parallel_process_dataframe_with_progress_multi_function(df, num_processes, self.process_chunk_clean_clean_emotes)
        # df[text_column]= df[text_column].swifter.apply(lambda x: clean_emotes(x))
        df[text_column]= df[text_column].swifter.apply(lambda x: x.lower())
        df[text_column]= df[text_column].swifter.apply(lambda x: self.decontract(x))
        df[text_column]= df[text_column].swifter.apply(lambda x: self.processSpecialTokens(x))
        df[text_column]= df[text_column].swifter.apply(lambda x: re.sub(r'[^a-zA-Z0-9 .,?!]', '', x))
        df[text_column]= df[text_column].swifter.apply(lambda x: re.sub(r'\n', '', x))
        df[text_column]= df[text_column].swifter.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        return df