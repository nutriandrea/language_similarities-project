import epitran
import pandas as pd
import Levenshtein as lev

# Language code conversion
languages = {
    'ita': 'ita-Latn',
    'fra': 'fra-Latn',
    'spa': 'spa-Latn',
    'por': 'por-Latn'
}

lex_size = {
    'ita': 250000,
    'fra': 200000,
    'spa': 150000,
    'por': 120000
}

# Caching transliteration results
transliteration_cache = {}
def wordtophonems(word, lang):
    if lang not in languages:
        print(f"Warning: Language code '{lang}' not found in languages mapping.")
        return ''
    cache_key = (word, lang)
    if cache_key in transliteration_cache:
        return transliteration_cache[cache_key]
    try:
        epit = create_epitran(lang)
        phonemes = epit.transliterate(word)
        transliteration_cache[cache_key] = phonemes
        print(f"Transliterated '{word}' ({lang}): {phonemes}")
        return phonemes
    except Exception as e:
        print(f"Error transliterating word '{word}' in language '{lang}': {e}")
        return ''

# Word similarity
def levenshtein_distance(w1, w2):
    if len(w1) < len(w2):
        return levenshtein_distance(w2, w1)

    if len(w2) == 0:
        return len(w1)

    previous_row = range(len(w2) + 1)
    for i, c1 in enumerate(w1):
        current_row = [i + 1]
        for j, c2 in enumerate(w2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def string_similarity(w1, w2):
    lw1, lw2 = len(w1), len(w2)
    max_len = max(lw1, lw2)
    if max_len == 0:
        return 1.0
    return (max_len - levenshtein_distance(w1, w2)) / max_len

def cognate_similarity(cognates, alpha, lexicon_size_A, lexicon_size_B):
    sum_similarity = 0
    for cA, cB in cognates:
        sum_similarity += alpha + (1 - alpha) * string_similarity(cA, cB)

    harmonic_mean = 2 * (lexicon_size_A * lexicon_size_B) / (lexicon_size_A + lexicon_size_B)
    return sum_similarity / harmonic_mean

def calculate_similarity_percentage(pairs):
    total_similarity = 0
    for w1, w2 in pairs:
        total_similarity += string_similarity(w1, w2)
    return (total_similarity / len(pairs)) * 100

def create_epitran(lang):
    return epitran.Epitran(languages[lang])

file_input = 'Mod-CogNet-v2.0.tsv'  # Path to your input file
file_output = 'Output.tsv'  # Path to your output file

print(f"Reading input file: {file_input}")
# Read the TSV file
df = pd.read_csv(file_input, sep='\t')

print(f"Filtering rows based on languages.")
# Filter rows based on the specified languages
filtered_df = df[df['lang 1'].isin(languages.keys()) & df['lang 2'].isin(languages.keys())]
print(f"Filtered rows: {len(filtered_df)}")

print(f"Transliterating words to phonemes.")
filtered_df['translit_phonems_1'] = [None] * len(filtered_df)
filtered_df['translit_phonems_2'] = [None] * len(filtered_df)

for idx, row in filtered_df.iterrows():
    filtered_df.at[idx, 'translit_phonems_1'] = wordtophonems(row['word 1'], row['lang 1'])
    filtered_df.at[idx, 'translit_phonems_2'] = wordtophonems(row['word 2'], row['lang 2'])

print(f"Saving results to output file: {file_output}")
# Save the new dataframe to a new file
filtered_df.to_csv(file_output, sep='\t', index=False)

print(f"Creating cognate pairs for each language pair in text and phonemes.")
cognates_text = {}
cognates_phonems = {}

for _, row in filtered_df.iterrows():
    lang1, lang2 = sorted([row['lang 1'], row['lang 2']])
    if (lang1 == 'por' and lang2 == 'por') or (lang1 == 'fra' and lang2 == 'fra'):
        continue  # Skip 'por-por' pairs
    key_text = f'cognates_text_{lang1}_{lang2}'
    key_phonems = f'cognates_phonems_{lang1}_{lang2}'
    if key_text not in cognates_text:
        cognates_text[key_text] = []
    cognates_text[key_text].append((row['word 1'], row['word 2']))
    if key_phonems not in cognates_phonems:
        cognates_phonems[key_phonems] = []
    cognates_phonems[key_phonems].append((row['translit_phonems_1'], row['translit_phonems_2']))

print("Calculating text similarity percentages:")
for key, pairs in cognates_text.items():
    similarity_percentage = calculate_similarity_percentage(pairs)
    print(f"{key}: {similarity_percentage:.2f}%")

print("Calculating phoneme similarity percentages:")
for key, pairs in cognates_phonems.items():
    similarity_percentage = calculate_similarity_percentage(pairs)
    print(f"{key}: {similarity_percentage:.2f}%")

print("Process completed.")

