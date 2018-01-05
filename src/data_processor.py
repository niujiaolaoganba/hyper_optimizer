# -*- coding: utf-8 -*-
"""
@authon: Charles Liu <guobinliulgb@gmail.com>
@brief: 数据预处理
@todo:
正则表达 regex
"""

import csv
import imp
import nltk
import regex
import pandas as pd
import nultiprocessing
from bs4 import BeautifulSoup
from collections import Counter

import config
from utils import pkl_utils, logging_utils, time_utils
from spelling_check import GoogleQuerySpellingChecker, AutoSpellingChecker

class BaseReplacer:
    def __init__(self, pattern_replace_pair_list = []):
        self.pattern_replace_pair_list = pattern_replace_pair_list

    def transform(self, text):
        for pattern, replace in self.pattern_replace_pair_list:
            try:
                text = regex.sub(pattern, replace, text)
            except:
                pass
        return regex.sub(r"\s+", " ", text).strip()

class LowerCaseConverter(BaseReplacer):
    def transform(self, text):
        return text.lower()

class LowerUpperCaseSplitter(BaseReplacer):
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"(\w)[\.?!]([A-Z])", r"\1 \2"),
            (r"(?<=( ))([a-z]+)([A-Z]+)", r"\2 \3"),
        ]

class WordReplacer(BaseReplacer):
    def __init__(self, replace_fname):
        self.replace_fname = replace_fname
        self.pattern_replace_pair_list = []
        for line in csv.reader(open(self.replace_fname)):
            if len(line) == 1 and line[0].startswith("#"):
                continue
            try:
                pattern = r"(?<=\W|^)%s(?=\W|$)"%line[0]
                replace = line[1]
                self.pattern_replace_pair_list.append((pattern, replace))
            except:
                print(line)
                pass

class LetterLetterSplitter(BaseReplacer):
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"([a-zA-Z]+)[/\-]([a-zA-Z]+)", r"\1 \2"),
        ]

## deal with digits and numbers
class DigitLetterSplitter(BaseReplacer):
    """
    x:
    1x1x1x1x1 -> 1 x 1 x 1 x 1 x 1
    19.875x31.5x1 -> 19.875 x 31.5 x 1

    -:
    1-Gang -> 1 Gang
    48-Light -> 48 Light

    .:
    includes a tile flange to further simplify installation.60 in. L x 36 in. W x 20 in. ->
    includes a tile flange to further simplify installation. 60 in. L x 36 in. W x 20 in.
    """
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"(\d+)[\.\-]*([a-zA-Z]+)", r"\1 \2"),
            (r"([a-zA-Z]+)[\.\-]*(\d+)", r"\1 \2"),
        ]

class DigitCommaDigitMerger(BaseReplacer):
    """
    1,000,000 -> 1000000
    """
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"(?<=\d+),(?=000)", r""),
        ]

class NumberDigitMapper(BaseReplacer):
    """
    one -> 1
    two -> 2
    """
    def __init__(self):
        numbers = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
        ]
        digits = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90
        ]
        self.pattern_replace_pair_list = [
            (r"(?<=\W|^)%s(?=\W|$)"%n, str(d)) for n,d in zip(numbers, digits)
        ]

## deal with unit
class UnitConverter(BaseReplacer):
    """
    shadeMature height: 36 in. - 48 in.Mature width
    PUT one UnitConverter before LowerUpperCaseSplitter
    """
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"([0-9]+)( *)(inches|inch|in|in.|')\.?", r"\1 in. "),
            (r"([0-9]+)( *)(pounds|pound|lbs|lb|lb.)\.?", r"\1 lb. "),
            (r"([0-9]+)( *)(foot|feet|ft|ft.|'')\.?", r"\1 ft. "),
            (r"([0-9]+)( *)(square|sq|sq.) ?\.?(inches|inch|in|in.|')\.?", r"\1 sq.in. "),
            (r"([0-9]+)( *)(square|sq|sq.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 sq.ft. "),
            (r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(inches|inch|in|in.|')\.?", r"\1 cu.in. "),
            (r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 cu.ft. "),
            (r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1 gal. "),
            (r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1 oz. "),
            (r"([0-9]+)( *)(centimeters|cm)\.?", r"\1 cm. "),
            (r"([0-9]+)( *)(milimeters|mm)\.?", r"\1 mm. "),
            (r"([0-9]+)( *)(minutes|minute)\.?", r"\1 min. "),
            (r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1 deg. "),
            (r"([0-9]+)( *)(v|volts|volt)\.?", r"\1 volt. "),
            (r"([0-9]+)( *)(wattage|watts|watt)\.?", r"\1 watt. "),
            (r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amp. "),
            (r"([0-9]+)( *)(qquart|quart)\.?", r"\1 qt. "),
            (r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1 hr "),
            (r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r"\1 gal. per min. "),
            (r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1 gal. per hr "),
        ]

class QuartetCleaner(BaseReplacer):
    def __init__(self):
        self.pattern_replace_pair_list = [
            (r"<.+?>", r""),
            # html codes
            (r"&nbsp;", r" "),
            (r"&amp;", r"&"),
            (r"&#39;", r"'"),
            (r"/>/Agt/>", r""),
            (r"</a<gt/", r""),
            (r"gt/>", r""),
            (r"/>", r""),
            (r"<br", r""),
            # do not remove [".", "/", "-", "%"] as they are useful in numbers, e.g., 1.97, 1-1/2, 10%, etc.
            (r"[ &<>)(_,;:!?\+^~@#\$]+", r" "),
            ("'s\\b", r""),
            (r"[']+", r""),
            (r"[\"]+", r""),
        ]


# ------------------- Extract Product Name -------------------
# 3rd solution in CrowdFlower
color_data = imp.load_source("", config.COLOR_DATA)
COLORS_PATTERN = r"(?<=\W|^)%s(?=\W|$)" % ("|".join(color_data.COLOR_LIST))
UNITS = [" ".join(r.strip().split(" ")[1:]) for p, r in UnitConverter().pattern_replace_pair_list]
UNITS_PATTERN = r"(?:\d+[?:.,]?\d*)(?: %s\.*)?" % ("|".join(UNITS))
DIM_PATTERN_NxNxN = r"%s ?x %s ?x %s" % (UNITS_PATTERN, UNITS_PATTERN, UNITS_PATTERN)
DIM_PATTERN_NxN = r"%s ?x %s" % (UNITS_PATTERN, UNITS_PATTERN)


class ProductNameExtractor(BaseReplacer):
    def __init__(self):
        self.pattern_replace_pair_list = [
            # Remove descriptions (text between paranthesis/brackets)
            ("[ ]?[[(].+?[])]", r""),
            # Remove "made in..."
            ("made in [a-z]+\\b", r""),
            # Remove descriptions (hyphen or comma followed by space then at most 2 words, repeated)
            ("([,-]( ([a-zA-Z0-9]+\\b)){1,2}[ ]?){1,}$", r""),
            # Remove descriptions (prepositions staring with: with, for, by, in )
            ("\\b(with|for|by|in|w/) .+$", r""),
            # colors & sizes
            ("size: .+$", r""),
            ("size [0-9]+[.]?[0-9]+\\b", r""),
            (COLORS_PATTERN, r""),
            # dimensions
            (DIM_PATTERN_NxNxN, r""),
            (DIM_PATTERN_NxN, r""),
            # measurement units
            (UNITS_PATTERN, r""),
            # others
            ("(value bundle|warranty|brand new|excellent condition|one size|new in box|authentic|as is)", r""),
            # stop words
            ("\\b(in)\\b", r""),
            # hyphenated words
            ("([a-zA-Z])-([a-zA-Z])", r"\1\2"),
            # special characters
            ("[ &<>)(_,.;:!?/+#*-]+", r" "),
            # numbers that are not part of a word
            ("\\b[0-9]+\\b", r""),
        ]

    def preprocess(self, text):
        pattern_replace_pair_list = [
            # Remove single & double apostrophes
            ("[\"]+", r""),
            # Remove product codes (long words (>5 characters) that are all caps, numbers or mix pf both)
            # don't use raw string format
            ("[ ]?\\b[0-9A-Z-]{5,}\\b", ""),
        ]
        text = BaseReplacer(pattern_replace_pair_list).transform(text)
        text = LowerCaseConverter().transform(text)
        text = DigitLetterSplitter().transform(text)
        text = UnitConverter().transform(text)
        text = DigitCommaDigitMerger().transform(text)
        text = NumberDigitMapper().transform(text)
        text = UnitConverter().transform(text)
        return text

    def transform(self, text):
        text = super().transform(self.preprocess(text))
        text = Lemmatizer().transform(text)
        text = Stemmer(stemmer_type="snowball").transform(text)
        # last two words in product
        text = " ".join(text.split(" ")[-2:])
        return text


class HtmlCleaner:
    def __init__(self, parse):
        self.parse = parse

    def transfrom(self, text):
        bs = BeautifulSoup(text, self.parse)
        text = bs.get_text(separator = " ")
        return text


class Lemmatizer:
    def __init__(self):
        self.Tokenizer = nltk.tokenize.TreebankWordTokenizer()
        self.Lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    def transform(self, text):
        tokens = [self.Lemmatizer.lemmatize(token) for token in self.Tokenizer.tokenize(text)]
        return " ".join(tokens)

class Stemmer:
    def __init__(self, stemmer_type = "snowball"):
        self.stemmer_type = stemmer_type
        if self.stemmer_type == "porter":
            self.stemmer = nltk.stem.PorterStemmer()
        elif self.stemmer_type == "snowball":
            self.stemmer = nltk.stem.SnowballStemmer("english")

    def transform(self, text):
        tokens = [self.stemmer.stem(token) for token in text.split(" ")]
        return " ".join(tokens)

#------------------- Process Attributes -------------------
def _split_attr_to_text(text):
    attrs = text.split(config.ATTR_SEPARATOR)
    return " ".join(attrs)

def _split_attr_to_list(text):
    attrs = text.split(config.ATTR_SEPARATOR)
    if len(attrs) == 1:
        # missing
        return [[attrs[0], attrs[0]]]
    else:
        return [[n,v] for n,v in zip(attrs[::2], attrs[1::2])]



#----------------------- Processor Wrapper -----------------------
class ProcessorWrapper:
    def __init__(self, processor):
        self.processor = processor

    def transform(self, input):
        if isinstance(input, str):
            out = self.processor.transform(input)
        elif isinstance(input, float) or isinstance(input, int):
            out = self.processor.transform(str(input))
        elif isinstance(input, list):
            # take care when the input is a list
            # currently for a list of attributes
            out = [0]*len(input)
            for i in range(len(input)):
                out[i] = ProcessorWrapper(self.processor).transform(input[i])
        else:
            raise(ValueError("Currently not support type: %s"%type(input).__name__))
        return out


#------------------- List/DataFrame Processor Wrapper -------------------
class ListProcessor:
    """
    WARNING: This class will operate on the original input list itself
    """
    def __init__(self, processors):
        self.processors = processors

    def process(self, lst):
        for i in range(len(lst)):
            for processor in self.processors:
                lst[i] = ProcessorWrapper(processor).transform(lst[i])
        return lst


class DataFrameProcessor:
    """
    WARNING: This class will operate on the original input dataframe itself
    """
    def __init__(self, processors):
        self.processors = processors

    def process(self, df):
        for processor in self.processors:
            df = df.apply(ProcessorWrapper(processor).transform)
        return df


class DataFrameParallelProcessor:
    """
    WARNING: This class will operate on the original input dataframe itself

    https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap
    """
    def __init__(self, processors, n_jobs=4):
        self.processors = processors
        self.n_jobs = n_jobs

    def process(self, dfAll, columns):
        df_processor = DataFrameProcessor(self.processors)
        p = multiprocessing.Pool(self.n_jobs)
        dfs = p.imap(df_processor.process, [dfAll[col] for col in columns])
        for col,df in zip(columns, dfs):
            dfAll[col] = df
        return dfAll


def main():
    logname = "data_processor_%s.log"%now
    logger = logging_utils._get_logger(config.LOG_DIR, logname)

    columns_to_proc = [
        # # product_attribute_list is very time consuming to process
        # # so we just process product_attribute which is of the form
        # # attr_name1 | attr_value1 | attr_name2 | attr_value2 | ...
        # # and split it into a list afterwards
        # "product_attribute_list",
        "product_attribute_concat",
        "product_description",
        "product_brand",
        "product_color",
        "product_title",
        "search_term",
    ]
    if config.PLATFORM == "Linux":
        config.DATA_PROCESSOR_N_JOBS = len(columns_to_proc)

    processors = [
        LowerCaseConverter(),
        LowerUpperCaseSplitter(),
        WordReplacer(replace_fname=config.WORD_REPLACER_DATA),
        LetterLetterSplitter(),
        DigitCommaDigitMerger(),
        NumberDigitMapper(),
        UnitConverter()
        HtmlCleaner(parser="html.parser"),
        Lemmatizer(),
    ]
    stemmers = [
        Stemmer(stemmer_type="snowball"),
        Stemmer(stemmer_type="porter")
    ][0:1]

    dfAll = pkl_utils._load(config.ALL_DATA_RAW)
    columns_to_proc = [col for col in columns_to_proc if col in dfAll.columns]

    ext = ProductNameExtractor()
    dfAll["search_term_product_name"] = dfAll["search_term"].apply(ext.transform)
    dfAll["product_title_product_name"] = dfAll["product_title"].apply(ext.transform)
    if config.TASK == "sample":
        print(dfAll[["search_term", "search_term_product_name", "product_title_product_name"]])


    #clean using a list of processors
    df_processor = DataFrameParallelProcessor(processors, config.DATA_PROCESSOR_N_JOBS)
    df_processor.process(dfAll, columns_to_proc)

    if config.QUERY_EXPANSION:
        list_processor = ListProcessor(processors)
        base_stopwords = set(list_processor.process(list(config.STOP_WORDS)))
        qe = QueryExpansion(dfAll, ngram=3, stopwords_threshold-0.9, base_stopwords=base_stopwords)
        dfAll["search_term_alt"]=qe.build()
        if config.TASK == "sample":
            print(dfAll[["search_term", "search_term_alt"]])

    # save data
    logger.info("Save to %s"%config.ALL_DATA_LEMMATIZED)
    columns_to_save = [col for col in dfAll.columns if col != "product_attribute_concat"]
    pkl_utils._save(config.ALL_DATA_LEMMATIZED, dfAll[columns_to_save])

    ## clean using stemmers
    df_processor = DataFrameParallelProcessor(stemmers, config.DATA_PROCESSOR_N_JOBS)
    df_processor.process(dfAll, columns_to_proc)
    # split product_attribute_concat into product_attribute and product_attribute_list
    dfAll["product_attribute"] = dfAll["product_attribute_concat"].apply(_split_attr_to_text)
    dfAll["product_attribute_list"] = dfAll["product_attribute_concat"].apply(_split_attr_to_list)

    # save data
    logger.info("Save to %s"%config.ALL_DATA_STEMMED)
    columns_to_save = [col for col in dfAll.columns if col != "product_attribute_concat"]
    pkl_utils._save(config.ALL_DATA_STEMMED, dfAll[columns_to_save])


if __name__ == "__main__":
    main()