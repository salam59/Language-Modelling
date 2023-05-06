import re
import os

""" To use this, put the file you want to clean in the data folder as a
.txt file, with _raw at the end of the filename. This will generate a new
file, <filename>_clean.txt, which you can analyze."""

if __name__ == "__main__":
    sentence_delimiter = r'([\.?!]"?) ("?[A-Z])'
    sd_pattern = re.compile(sentence_delimiter)
    for entry in os.scandir("data"):
        if entry.is_file() and entry.path.endswith("raw.txt"):
            raw_path = entry.path
            clean_path = raw_path.replace("raw", "clean")
            if os.path.exists(clean_path):
                continue
            with open(raw_path, "r", encoding="utf-8") as raw:
                with open(clean_path, "w", encoding="utf-8") as clean:
                    raw_toks = []
                    story = ""
                    for line in raw:
                        line = line.strip()
                        if line == "":
                            # Blank line. Skip.
                            continue
                        if not re.search('[a-zA-Z]', line):
                            # Line that only has numbers and symbols. Possible page number line. Skip.
                            continue
                        if line.isupper():
                            # A title. Write the previous story.
                            if story:
                                story = re.sub(sd_pattern, r'\1__\2', story)
                                sentences = story.split("__")
                                clean_toks = []
                                for sentence in sentences:
                                    raw_toks = sentence.split()
                                    for tok in raw_toks:
                                        head = []
                                        tail = []
                                        while tok and not tok[0].isalpha() and not tok[0].isdigit():
                                            head.append(tok[0])
                                            tok = tok[1:]
                                        while tok and not tok[-1].isalpha() and not tok[-1].isdigit():
                                            tail.append(tok[-1])
                                            tok = tok[:-1]
                                        clean_toks.extend(head)
                                        if tok:
                                            clean_toks.append(tok.lower())
                                        clean_toks.extend(reversed(tail))
                                    if clean_toks:
                                        cleaned = " ".join(clean_toks)
                                        clean.write(cleaned)
                                        clean_toks = []
                                        clean.write("\n")
                                story = ""
                            continue
                        story = story + " " + line
