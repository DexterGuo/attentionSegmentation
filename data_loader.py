from torch.utils.data import Dataset
from text_manipulation import word_model, location_model
from pathlib2 import Path
import re
import os


def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files


def get_cache_path(wiki_folder):
    cache_file_path = wiki_folder / 'paths_cache'
    return cache_file_path


def cache_wiki_filenames(wiki_folder):
    files = Path(wiki_folder).glob('*/*/*/*')
    cache_file_path = get_cache_path(wiki_folder)

    with cache_file_path.open('w') as f:
        for file in files:
            f.write(unicode(file) + u'\n')


def clean_section(section):
    cleaned_section = section.strip('\n')
    return cleaned_section


def get_scections_from_text(txt):
    sentences = txt.strip().split("\n")
    section, all_sections, flag_list = [], [], []
    for line in sentences:
        line = line.strip("\n")
        if line.startswith("===###") and line.endswith("###==="):
            if section: 
                all_sections.append("\n".join(section) )
                section = []
            flag_list.append(line[6:-6])
            continue
        section.append(line)
    if section:
        all_sections.append("\n".join(section) )
    assert len(all_sections) == len(flag_list)

    non_empty_sections, new_flag_list = [], []
    for s,f in zip(all_sections, flag_list) :
        if len(s) > 0:
            non_empty_sections.append(s)
            new_flag_list.append(f)
    return non_empty_sections, new_flag_list

def get_sections(path, high_granularity=True):
    file = open(str(path), "r")
    raw_content = file.read()
    file.close()

    clean_txt = raw_content.decode('utf-8').strip()

    sections, flag_list = get_scections_from_text(clean_txt)
    sections = [clean_section(s) for s in sections ]

    return sections


def read_seg_file(path, word2vec, remove_preface_segment=True, 
                   ignore_list=False, remove_special_tokens=False,
                   return_as_sentences=False, high_granularity=True,only_letters = False):
    data, targets = [], []
    all_sections = get_sections(path, high_granularity)

    for section in all_sections:
        sentences = section.split('\n')
        if not sentences: continue

        for sentence in sentences:
            loc, ocr = sentence.split("\t")
            sentence_words = ocr.split(" ")
            if 1 <= len(sentence_words):
                print len(sentence_words), sentence_words
                offset, token_num = 0, len(sentence_words)
                for i, word in enumerate(sentence_words):
                    word_embed = word_model(word, word2vec)
                    loc_embed = location_model(loc, i, offset, token_num)
                    data.append([word_embed, loc_embed])
                    offset += len(word)
            else:
                logger.info('Sentence in wikipedia file is empty')
        if data:
            targets.append(len(data) - 1)

    return data, targets, path

class SegTextDataSet(Dataset):
    def __init__(self, root, word2vec, train=True, manifesto=False, folder=False, high_granularity=False):

        if (manifesto):
            self.textfiles = list(Path(root).glob('*'))
        else:
            if (folder):
                self.textfiles = get_files(root)
            else:
                root_path = Path(root)
                cache_path = get_cache_path(root_path)
                if not cache_path.exists():
                    cache_wiki_filenames(root_path)
                self.textfiles = cache_path.read_text().splitlines()

        if len(self.textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))
        self.train = train
        self.root = root
        self.word2vec = word2vec
        self.high_granularity = high_granularity

    def __getitem__(self, index):
        path = "../text-segmentation-master/" + self.textfiles[index]

        return read_seg_file(Path(path), self.word2vec, ignore_list=True, remove_special_tokens=True,
                              high_granularity=self.high_granularity)

    def __len__(self):
        return len(self.textfiles)

if __name__ == "__main__":
    import sys
    root = "../text-segmentation-master/data/math_1.11/test"
    segtext = SegTextDataSet(root, None)
    print segtext[0][0][0]

